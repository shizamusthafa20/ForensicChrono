"""
microbial_model.py ── ForensicChrono (Module 2: Microbial Succession Model)
================================================================================
Trains a machine learning model on human post-mortem microbiome succession data
(Qiita 13810 16S rRNA dataset) to predict PMI (Post-Mortem Interval in days).

Methodological Integrity:
  - 5-Fold Donor-Grouped Cross-Validation (Zero intra-donor leakage)
  - Relative Abundance & Log1p Bacterial Taxa Feature Transformations
  - IN-FOLD top-variance taxa selection (no leakage from validation samples)
  - Strict experiment logging to results/experiment_summary.csv
"""

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

# ── Configuration ──────────────────────────────────────────────────────────────
METADATA_PATH  = "data/raw/microbiome/metadata.tsv"
OTU_TABLE_PATH = "data/raw/microbiome/otu_table.tsv"

TOP_N_TAXA = 500   # Top bacterial taxa by variance, selected IN-FOLD


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


def log_experiment_results(r2_val, mae_val, baseline_mae, improvement, n_subjects,
                            model_type="Microbial_RF", out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result_record = {
        "timestamp": timestamp,
        "model_type": model_type,
        "r2": float(r2_val),
        "mae_days": float(mae_val),
        "baseline_mae_days": float(baseline_mae),
        "improvement_pct": float(improvement),
        "n_subjects": int(n_subjects),
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
        "n_donors": n_subjects,
        "donor_r2": round(float(r2_val), 4),
        "donor_mae_hrs": round(float(mae_val) * 24.0, 2),
        "baseline_mae_hrs": round(float(baseline_mae) * 24.0, 2),
        "improvement_pct": round(float(improvement), 1),
        "n_var_taxa": TOP_N_TAXA,
        "cv_strategy": "donor_5fold_group_kfold_infold_selection",
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

    print("Loading OTU abundance table (this can take a moment, it's a big matrix)...")
    otu = pd.read_csv(OTU_TABLE_PATH, sep="\t", index_col=0, low_memory=False)

    # Align samples between metadata and OTU table
    common_samples = list(set(meta["sample_id"]).intersection(set(otu.columns)))
    meta = meta[meta["sample_id"].isin(common_samples)].reset_index(drop=True)
    otu = otu[meta["sample_id"]].T  # samples x taxa

    print(f"  -> Aligned {len(common_samples)} samples across {meta['subject_id'].nunique()} donors.")

    # Relative abundance transform (per-sample normalization)
    row_sums = otu.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    otu_norm = otu.div(row_sums, axis=0)

    # Log1p transform for skewed bacterial counts
    otu_log = np.log1p(otu_norm * 10000.0)

    # NOTE: no variance filtering here anymore -- that now happens
    # INSIDE each CV fold, using only training-fold samples, to avoid leakage.
    return meta, otu_log


# ═══════════════════════════════════════════════════════════════════════════════
# CV & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(meta, otu_log):
    y_all = meta["pmi"].values
    groups_all = meta["subject_id"].values
    n_samples = len(meta)

    sample_y_pred = np.zeros(n_samples)

    print(f"\nRunning 5-Fold Donor GroupKFold CV on Microbial Data "
          f"(in-fold top-{TOP_N_TAXA} taxa selection)...")

    for fold, (tr_idx, val_idx) in enumerate(group_kfold_splits(groups_all, n_splits=5), 1):
        # --- IN-FOLD taxa selection: use ONLY training samples to pick features ---
        train_block = otu_log.iloc[tr_idx]
        top_taxa = train_block.var(axis=0).sort_values(ascending=False).head(TOP_N_TAXA).index

        X_tr = train_block[top_taxa].values
        X_val = otu_log.iloc[val_idx][top_taxa].values
        y_tr = y_all[tr_idx]

        rf = RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=2,
            n_jobs=-1, random_state=42
        )
        rf.fit(X_tr, y_tr)

        sample_y_pred[val_idx] = rf.predict(X_val)
        print(f"  Fold {fold}/5 completed. ({len(tr_idx)} train / {len(val_idx)} val samples)")

    # Group predictions by donor for honest donor-level evaluation
    eval_df = pd.DataFrame({"subject_id": groups_all, "y_true": y_all, "y_pred": sample_y_pred})
    donor_eval = eval_df.groupby("subject_id")[["y_true", "y_pred"]].mean()

    r2_score = compute_r2(donor_eval["y_true"].values, donor_eval["y_pred"].values)
    mae_score = compute_mae(donor_eval["y_true"].values, donor_eval["y_pred"].values)
    baseline_mae = compute_mae(
        donor_eval["y_true"].values,
        np.full_like(donor_eval["y_true"].values, donor_eval["y_true"].mean())
    )
    improvement = (1.0 - mae_score / baseline_mae) * 100.0

    print("\n" + "=" * 60)
    print("  M I C R O B I A L   M O D E L   R E S U L T S")
    print("=" * 60)
    print(f"  Total Donors               : {len(donor_eval)}")
    print(f"  Microbial Model R²        : {r2_score:.4f}")
    print(f"  Microbial Model MAE       : {mae_score:.2f} days ({mae_score*24:.1f} hrs)")
    print(f"  Baseline MAE (Mean Guess) : {baseline_mae:.2f} days")
    print(f"  Improvement over Baseline : {improvement:.1f}%")
    print("=" * 60)

    _save_scatter(donor_eval["y_true"].values, donor_eval["y_pred"].values, r2_score, mae_score)
    log_experiment_results(r2_score, mae_score, baseline_mae, improvement, len(donor_eval),
                            model_type="Microbial_RF")


def _save_scatter(y_actual, y_pred, r2, mae, out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(y_actual, y_pred, alpha=0.6, s=30, color="#059669",
               edgecolors="none", label="Donor-level mean prediction", zorder=3)
    lo = min(y_actual.min(), y_pred.min())
    hi = max(y_actual.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y = x (perfect fit)", zorder=2)

    ax.text(0.05, 0.92, f"Microbial R² = {r2:.4f}", transform=ax.transAxes,
            fontsize=13, fontweight="bold", color="#065f46")
    ax.text(0.05, 0.85, f"MAE = {mae:.2f} days ({mae*24:.1f} hrs)", transform=ax.transAxes,
            fontsize=10, color="#374151")

    ax.set_xlabel("Actual Post-Mortem Interval (Days)", fontsize=12)
    ax.set_ylabel("Predicted Post-Mortem Interval (Days)", fontsize=12)
    ax.set_title("Module 2: Microbial Succession PMI Model\n"
                 "(Qiita 13810, 5-Fold Donor GroupKFold, In-Fold Taxa Selection)", fontsize=11)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    for ext in ("png", "svg"):
        path = os.path.join(out_dir, f"microbial_model_predicted_vs_actual.{ext}")
        fig.savefig(path, dpi=150)
        print(f"  Plot saved -> {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    meta, otu_log = load_microbial_data()
    train_and_evaluate(meta, otu_log)


if __name__ == "__main__":
    main()