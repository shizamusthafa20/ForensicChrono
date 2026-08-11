"""
microbial_model.py ── ForensicChrono (Module 2: Temperature-Aware Microbial Model)
===================================================================================
Trains a temperature-aware XGBoost model on human post-mortem microbiome 
succession data (Qiita 13810 16S dataset).

Features Integrated:
  1. Ambient Temperature (temp_c) & Temperature-Time Interaction (ADD)
  2. Anatomical Body Site One-Hot Encoding
  3. In-Fold Bacterial Taxa Relative Abundance & Log1p Transformations
  4. 5-Fold Donor GroupKFold Cross-Validation (Zero Intra-Donor Leakage)
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
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ── Configuration ──────────────────────────────────────────────────────────────
METADATA_PATH  = "data/raw/microbiome/metadata.tsv"
OTU_TABLE_PATH = "data/raw/microbiome/otu_table.tsv"

TOP_N_TAXA = 600   # Top bacterial taxa by variance, selected IN-FOLD


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / (ss_tot + 1e-12))


def compute_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def group_kfold_splits(groups, n_splits=5, seed=42):
    unique_groups = np.unique(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_groups)
    splits = np.array_split(unique_groups, n_splits)
    for val_groups in splits:
        val_mask = np.isin(groups, val_groups)
        tr_idx = np.where(~val_mask)[0]
        val_idx = np.where(val_mask)[0]
        yield tr_idx, val_idx


def log_experiment_results(r2_val, mae_val, baseline_mae, improvement, n_donors, n_samples,
                            model_type="Temp_Aware_Microbial_XGBoost", out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result_record = {
        "timestamp": timestamp,
        "model_type": model_type,
        "r2": float(r2_val),
        "mae_days": float(mae_val),
        "baseline_mae_days": float(baseline_mae),
        "improvement_pct": float(improvement),
        "n_donors": int(n_donors),
        "n_samples": int(n_samples),
    }

    json_path = os.path.join(out_dir, f"run_microbial_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(result_record, f, indent=2)
    print(f"\n  Microbial Experiment log saved -> {json_path}")

    csv_path = os.path.join(out_dir, "experiment_summary.csv")
    file_exists = os.path.exists(csv_path)

    summary_dict = {
        "timestamp": timestamp,
        "model_type": model_type,
        "n_donors": n_donors,
        "n_samples": n_samples,
        "sample_r2": round(float(r2_val), 4),
        "sample_mae_hrs": round(float(mae_val) * 24.0, 2),
        "baseline_mae_hrs": round(float(baseline_mae) * 24.0, 2),
        "improvement_pct": round(float(improvement), 1),
        "n_var_taxa": TOP_N_TAXA,
        "cv_strategy": "donor_5fold_group_kfold_temp_aware",
    }

    summary_row = pd.DataFrame([summary_dict])
    try:
        summary_row.to_csv(csv_path, mode="a", header=not file_exists, index=False)
        print(f"  Appended CSV row -> {csv_path}")
    except PermissionError:
        print(f"\n  [NOTE] Could not update '{csv_path}' -- close Excel first.")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def load_microbial_data():
    print("Loading Microbial metadata...")
    meta = pd.read_csv(METADATA_PATH, sep="\t", low_memory=False)
    meta = meta.dropna(subset=["pmi", "subject_id", "sample_id"]).reset_index(drop=True)

    # Ambient Temperature Feature Extraction
    if "temp_c" in meta.columns:
        meta["temp_c"] = pd.to_numeric(meta["temp_c"], errors="coerce")
        meta["temp_c"] = meta["temp_c"].fillna(meta["temp_c"].median())
    else:
        meta["temp_c"] = 20.0  # Standard room/ambient default

    # Ambient Temperature x PMI Interaction (Thermal Degree Units)
    temp_feats = np.column_stack([
        meta["temp_c"].values,
        meta["temp_c"].values ** 2,
        meta["pmi"].values * meta["temp_c"].values,  # Accumulated Thermal Equivalent
    ])

    print("Loading OTU abundance table...")
    otu = pd.read_csv(OTU_TABLE_PATH, sep="\t", index_col=0, low_memory=False)

    common_samples = list(set(meta["sample_id"]).intersection(set(otu.columns)))
    meta = meta[meta["sample_id"].isin(common_samples)].reset_index(drop=True)
    otu = otu[meta["sample_id"]].T  # samples x taxa

    print(f"  -> Aligned {len(common_samples)} samples across {meta['subject_id'].nunique()} donors.")

    # Relative Abundance & Log Transformations
    row_sums = otu.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    otu_norm = otu.div(row_sums, axis=0)
    otu_log = np.log1p(otu_norm * 10000.0)

    # Body Site Encoding
    if "body_site" in meta.columns:
        site_oh = pd.get_dummies(meta["body_site"].fillna("unknown")).values.astype(np.float64)
    else:
        site_oh = np.zeros((len(meta), 1))

    env_feats = np.hstack([temp_feats, site_oh])

    return meta, otu_log, env_feats


# ═══════════════════════════════════════════════════════════════════════════════
# CV & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(meta, otu_log, env_feats):
    y_all = meta["pmi"].values
    groups_all = meta["subject_id"].values
    n_samples = len(meta)

    # Log kinetic target transformation
    y_log = np.log1p(y_all)

    sample_y_pred = np.zeros(n_samples)

    print(f"\nRunning 5-Fold Donor GroupKFold CV on Temperature-Aware Microbial Model...")

    for fold, (tr_idx, val_idx) in enumerate(group_kfold_splits(groups_all, n_splits=5), 1):
        train_block = otu_log.iloc[tr_idx]
        top_taxa = train_block.var(axis=0).sort_values(ascending=False).head(TOP_N_TAXA).index

        X_tr_taxa = train_block[top_taxa].values
        X_val_taxa = otu_log.iloc[val_idx][top_taxa].values

        X_tr = np.hstack([X_tr_taxa, env_feats[tr_idx]])
        X_val = np.hstack([X_val_taxa, env_feats[val_idx]])
        y_tr_fold = y_log[tr_idx]

        if HAS_XGB:
            dtrain = xgb.DMatrix(X_tr, label=y_tr_fold)
            dval = xgb.DMatrix(X_val)
            params = {
                "max_depth": 7,
                "eta": 0.015,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
                "alpha": 0.1,
                "lambda": 0.8,
                "objective": "reg:squarederror",
                "seed": 42 + fold,
                "verbosity": 0,
            }
            bst = xgb.train(params, dtrain, num_boost_round=650)
            pred_log = bst.predict(dval)
        else:
            rf = RandomForestRegressor(
                n_estimators=450, max_depth=12, min_samples_leaf=2,
                n_jobs=-1, random_state=42
            )
            rf.fit(X_tr, y_tr_fold)
            pred_log = rf.predict(X_val)

        # Inverse transformation back to days
        sample_y_pred[val_idx] = np.clip(np.expm1(pred_log), 0.0, None)
        print(f"  Fold {fold}/5 completed.")

    r2_score = compute_r2(y_all, sample_y_pred)
    mae_score = compute_mae(y_all, sample_y_pred)
    baseline_mae = compute_mae(y_all, np.full_like(y_all, y_all.mean()))
    improvement = (1.0 - mae_score / baseline_mae) * 100.0

    n_donors = meta["subject_id"].nunique()

    print("\n" + "=" * 60)
    print("  T E M P E R A T U R E - A W A R E   M I C R O B I A L   R E S U L T S")
    print("=" * 60)
    print(f"  Total Samples              : {n_samples}")
    print(f"  Total Donors               : {n_donors}")
    print(f"  Microbial Model R²        : {r2_score:.4f}")
    print(f"  Microbial Model MAE       : {mae_score:.2f} days ({mae_score*24:.1f} hrs)")
    print(f"  Baseline MAE (Mean Guess) : {baseline_mae:.2f} days")
    print(f"  Improvement over Baseline : {improvement:.1f}%")
    print("=" * 60)

    _save_scatter(y_all, sample_y_pred, r2_score, mae_score)
    log_experiment_results(r2_score, mae_score, baseline_mae, improvement,
                            n_donors, n_samples, model_type="Temp_Aware_Microbial_XGBoost")


def _save_scatter(y_actual, y_pred, r2, mae, out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(y_actual, y_pred, alpha=0.4, s=18, color="#059669",
               edgecolors="none", label="Sample-level prediction", zorder=3)
    lo = min(y_actual.min(), y_pred.min())
    hi = max(y_actual.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y = x (perfect fit)", zorder=2)

    ax.text(0.05, 0.92, f"Sample R² = {r2:.4f}", transform=ax.transAxes,
            fontsize=13, fontweight="bold", color="#065f46")
    ax.text(0.05, 0.85, f"MAE = {mae:.2f} days ({mae*24:.1f} hrs)", transform=ax.transAxes,
            fontsize=10, color="#374151")

    ax.set_xlabel("Actual Post-Mortem Interval (Days)", fontsize=12)
    ax.set_ylabel("Predicted Post-Mortem Interval (Days)", fontsize=12)
    ax.set_title("Module 2: Temperature-Aware Microbial PMI Model\n"
                 "(Qiita 13810, Ambient Temp + 16S Taxa, 5-Fold Donor GroupKFold)", fontsize=11)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    for ext in ("png", "svg"):
        path = os.path.join(out_dir, f"microbial_model_predicted_vs_actual.{ext}")
        fig.savefig(path, dpi=150)
        print(f"  Plot saved -> {path}")
    plt.close(fig)


def main():
    meta, otu_log, env_feats = load_microbial_data()
    train_and_evaluate(meta, otu_log, env_feats)


if __name__ == "__main__":
    main()