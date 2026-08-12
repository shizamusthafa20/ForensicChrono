"""
microbial_model.py ── ForensicChrono (Module 2: Microbial Succession Model)
============================================================================
Trains a leakage-free XGBoost model on human post-mortem microbiome succession data 
(Qiita 13810 16S dataset).

METHODOLOGICAL RIGOR:
  1. Zero Target Leakage: Target (pmi) is NEVER multiplied into features.
  2. 5-Fold Donor GroupKFold Cross-Validation: Prevents donor fingerprint leakage across folds.
  3. Longitudinal Sample-Level Evaluation: Evaluates predictions at the sample level 
     since each donor cadaver is sampled repeatedly across time (1 to 21 days).
  4. In-Fold Taxa Selection & Environmental Encoding.
"""

import os
import json
import pickle
from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ── Configuration ──────────────────────────────────────────────────────────────
METADATA_PATH  = "data/raw/microbiome/metadata.tsv"
OTU_TABLE_PATH = "data/raw/microbiome/otu_table.tsv"

TOP_N_TAXA  = 600   # Top bacterial taxa by variance, selected IN-FOLD
N_FOLDS     = 5
SEED        = 42


# ═══════════════════════════════════════════════════════════════════════════════
# GROUPED CV
# ═══════════════════════════════════════════════════════════════════════════════

def group_kfold_splits(groups, n_splits=5, seed=42):
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    rng = np.random.RandomState(seed)
    unique_groups = unique_groups.copy()
    rng.shuffle(unique_groups)
    splits = np.array_split(unique_groups, n_splits)
    for val_groups in splits:
        val_mask = np.isin(groups, val_groups)
        tr_idx = np.where(~val_mask)[0]
        val_idx = np.where(val_mask)[0]
        yield tr_idx, val_idx


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & TEMPERATURE DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════════

def load_microbial_data():
    print("Loading Microbial metadata...")
    meta = pd.read_csv(METADATA_PATH, sep="\t", low_memory=False)
    
    meta["pmi"] = pd.to_numeric(meta["pmi"], errors="coerce")
    meta = meta.dropna(subset=["pmi", "subject_id", "sample_id"]).reset_index(drop=True)
    meta = meta[meta["pmi"] >= 0.0].reset_index(drop=True)

    # ── Temperature Feature Discovery ──
    temp_col = None
    possible_temp_cols = ["temp_c", "temp_prob", "temperature_station", "temp"]
    for col in possible_temp_cols:
        if col in meta.columns:
            non_nulls = pd.to_numeric(meta[col], errors="coerce").notna().sum()
            print(f"  [Diagnostic] Metadata column '{col}': {non_nulls} / {len(meta)} non-null values.")
            if non_nulls > 0 and temp_col is None:
                temp_col = col

    if temp_col is not None:
        meta["env_temp"] = pd.to_numeric(meta[temp_col], errors="coerce")
        print(f"  -> Using '{temp_col}' for ambient temperature feature.")
    else:
        meta["env_temp"] = np.nan
        print("  -> [Notice] No numerical ambient temperature column found; fallback active.")

    print("Loading OTU abundance table...")
    otu = pd.read_csv(OTU_TABLE_PATH, sep="\t", index_col=0, low_memory=False)

    common_samples = list(set(meta["sample_id"]).intersection(set(otu.columns)))
    meta = meta[meta["sample_id"].isin(common_samples)].reset_index(drop=True)
    
    otu = otu[meta["sample_id"]].T  # samples x taxa
    
    print(f"  -> Aligned {len(common_samples)} samples across {meta['subject_id'].nunique()} unique donors.")

    # Relative Abundance & Log1p Transformation
    row_sums = otu.sum(axis=1).values.copy()
    row_sums[row_sums == 0] = 1.0
    otu_norm = otu.div(row_sums, axis=0)
    otu_log = np.log1p(otu_norm * 10000.0)

    return meta, otu_log


# ═══════════════════════════════════════════════════════════════════════════════
# IN-FOLD FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def make_in_fold_features(train_meta, val_meta, train_otu, val_otu):
    # 1. Taxa Variance Selection (Training Folds Only)
    train_var = train_otu.var(axis=0)
    top_taxa = train_var.sort_values(ascending=False).head(TOP_N_TAXA).index

    X_tr_taxa = train_otu[top_taxa].values
    X_val_taxa = val_otu[top_taxa].values

    # 2. Ambient Temperature (In-Fold Median Imputation)
    temp_median = train_meta["env_temp"].median()
    if pd.isna(temp_median):
        temp_median = 20.0
        
    train_temp = train_meta["env_temp"].fillna(temp_median).values
    val_temp   = val_meta["env_temp"].fillna(temp_median).values

    env_tr_temp = np.column_stack([train_temp, train_temp ** 2])
    env_va_temp = np.column_stack([val_temp, val_temp ** 2])

    # 3. Body Site Categorical One-Hot (In-Fold)
    site_col = "body_site" if "body_site" in train_meta.columns else "host_body_site"
    if site_col in train_meta.columns:
        tr_sites = train_meta[site_col].fillna("unknown").astype(str)
        va_sites = val_meta[site_col].fillna("unknown").astype(str)
        
        site_dummies_tr = pd.get_dummies(tr_sites)
        categories = site_dummies_tr.columns
        
        site_dummies_va = pd.get_dummies(va_sites).reindex(columns=categories, fill_value=0)
        
        env_tr_site = site_dummies_tr.values.astype(np.float64)
        env_va_site = site_dummies_va.values.astype(np.float64)
    else:
        env_tr_site = np.zeros((len(train_meta), 1))
        env_va_site = np.zeros((len(val_meta), 1))

    X_tr  = np.hstack([X_tr_taxa, env_tr_temp, env_tr_site])
    X_val = np.hstack([X_val_taxa, env_va_temp, env_va_site])

    return X_tr, X_val


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(meta, otu_log):
    n_samples = len(meta)
    y_all = meta["pmi"].values
    groups_all = meta["subject_id"].values

    y_log = np.log1p(y_all)
    sample_predictions = np.zeros(n_samples)

    print(f"\nRunning 5-Fold Donor GroupKFold CV...")

    for fold, (tr_idx, val_idx) in enumerate(group_kfold_splits(groups_all, n_splits=N_FOLDS, seed=SEED), 1):
        tr_meta, val_meta = meta.iloc[tr_idx], meta.iloc[val_idx]
        tr_otu,  val_otu  = otu_log.iloc[tr_idx], otu_log.iloc[val_idx]

        X_tr, X_val = make_in_fold_features(tr_meta, val_meta, tr_otu, val_otu)
        y_tr_log = y_log[tr_idx]

        if HAS_XGB:
            dtrain = xgb.DMatrix(X_tr, label=y_tr_log)
            dval   = xgb.DMatrix(X_val)
            params = {
                "max_depth": 6,
                "eta": 0.02,
                "subsample": 0.85,
                "colsample_bytree": 0.75,
                "alpha": 0.3,
                "lambda": 1.0,
                "objective": "reg:squarederror",
                "seed": SEED + fold,
                "verbosity": 0,
            }
            bst = xgb.train(params, dtrain, num_boost_round=500)
            pred_log = bst.predict(dval)
        else:
            rf = RandomForestRegressor(
                n_estimators=400, max_depth=12, min_samples_leaf=2,
                n_jobs=-1, random_state=SEED
            )
            rf.fit(X_tr, y_tr_log)
            pred_log = rf.predict(X_val)

        sample_predictions[val_idx] = np.clip(np.expm1(pred_log), 0.0, None)
        print(f"  Fold {fold}/{N_FOLDS} completed.")

    # ── 1. PRIMARY: SAMPLE-LEVEL EVALUATION (Longitudinal PMI Study) ──
    sample_r2 = r2_score(y_all, sample_predictions)
    sample_mae = mean_absolute_error(y_all, sample_predictions)
    
    baseline_sample = np.full(n_samples, y_all.mean())
    baseline_sample_mae = mean_absolute_error(y_all, baseline_sample)
    sample_improvement = (1.0 - sample_mae / baseline_sample_mae) * 100.0

    print("\n" + "=" * 65)
    print("  M I C R O B I A L   S U C C E S S I O N   R E S U L T S")
    print("=" * 65)
    print(f"  Evaluation Level           : Sample-Level (Longitudinal Timepoints)")
    print(f"  Total Samples              : {n_samples}")
    print(f"  Total Unique Donors        : {meta['subject_id'].nunique()}")
    print(f"  Sample-Level R²            : {sample_r2:.4f}")
    print(f"  Sample-Level MAE           : {sample_mae:.2f} days ({sample_mae*24:.1f} hrs)")
    print(f"  Baseline MAE (Mean Guess)  : {baseline_sample_mae:.2f} days ({baseline_sample_mae*24:.1f} hrs)")
    print(f"  Improvement over Baseline  : {sample_improvement:.1f}%")
    print("=" * 65)

    _save_scatter(y_all, sample_predictions, sample_r2, sample_mae)
    log_experiment_results(sample_r2, sample_mae, baseline_sample_mae, sample_improvement,
                           meta['subject_id'].nunique(), n_samples)


def _save_scatter(y_actual, y_pred, r2, mae, out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(y_actual, y_pred, alpha=0.35, s=18, color="#059669",
               edgecolors="none", label="Held-Out Samples", zorder=3)
    lo = min(y_actual.min(), y_pred.min())
    hi = max(y_actual.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y = x (perfect fit)", zorder=2)

    ax.text(0.05, 0.92, f"Sample R² = {r2:.4f}", transform=ax.transAxes,
            fontsize=13, fontweight="bold", color="#065f46")
    ax.text(0.05, 0.85, f"MAE = {mae:.2f} days ({mae*24:.1f} hrs)", transform=ax.transAxes,
            fontsize=10, color="#374151")

    ax.set_xlabel("Actual Post-Mortem Interval (Days)", fontsize=12)
    ax.set_ylabel("Predicted Post-Mortem Interval (Days)", fontsize=12)
    ax.set_title("Module 2: Microbial Succession PMI Model\n"
                 "(Qiita 13810 16S Taxa, 5-Fold Donor GroupKFold)", fontsize=10)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    for ext in ("png", "svg"):
        path = os.path.join(out_dir, f"microbial_model_predicted_vs_actual.{ext}")
        fig.savefig(path, dpi=150)
        print(f"  Plot saved -> {path}")
    plt.close(fig)


def log_experiment_results(r2_val, mae_val, baseline_mae, improvement, n_donors, n_samples,
                            out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = os.path.join(out_dir, "experiment_summary.csv")
    file_exists = os.path.exists(csv_path)

    summary_dict = {
        "timestamp": timestamp,
        "model_type": "Microbial_Succession_XGBoost",
        "n_donors": n_donors,
        "n_samples": n_samples,
        "sample_r2": round(float(r2_val), 4),
        "sample_mae_days": round(float(mae_val), 2),
        "sample_mae_hrs": round(float(mae_val) * 24.0, 2),
        "baseline_mae_hrs": round(float(baseline_mae) * 24.0, 2),
        "improvement_pct": round(float(improvement), 1),
        "n_var_taxa": TOP_N_TAXA,
        "cv_strategy": "donor_5fold_group_kfold_sample_level_eval",
    }

    summary_row = pd.DataFrame([summary_dict])
    try:
        summary_row.to_csv(csv_path, mode="a", header=not file_exists, index=False)
        print(f"  Appended CSV row -> {csv_path}")
    except Exception as e:
        print(f"  Could not write to CSV: {e}")


def main():
    meta, otu_log = load_microbial_data()
    train_and_evaluate(meta, otu_log)


if __name__ == "__main__":
    main()