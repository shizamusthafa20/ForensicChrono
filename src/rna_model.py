"""
rna_model.py ── ForensicChrono (High-Kinetic Window Stacking Model)
================================================================================
Trains a multi-tissue RNA degradation model on the primary 0-12 hour (720 min) 
forensic window using 5-Fold Donor GroupKFold Cross-Validation.

Key Enhancements for High Precision (R² >= 0.80 target):
  1. Primary Kinetic Window: 0 - 720 minutes (retains ample donors for robust CV)
  2. Pairwise Ratio Interaction Kinetics & Decay Velocity Features
  3. Optimized Weighted Stacking (XGBoost + Random Forest + Ridge)
"""

import os
import gzip
import json
import pickle
from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from sklearn.ensemble import RandomForestRegressor
    HAS_RF = True
except ImportError:
    HAS_RF = False

# ── Configuration ──────────────────────────────────────────────────────────────
SAMPLE_ATTR_PATH   = "data/raw/rna/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
SUBJECT_PHENO_PATH = "data/raw/rna/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt"
GENE_READS_PATH    = "data/raw/rna/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct.gz"

TARGET_TISSUES = [
    "Muscle - Skeletal",
    "Lung",
    "Skin - Sun Exposed (Lower leg)",
    "Nerve - Tibial",
]

MAX_TIME_MIN = 720    # 0 - 12 Hours: Focus on active decay window where RNA is forensic-grade
N_VAR_GENES  = 1200
N_DEGRADERS  = 35
N_STABLE     = 12
N_PCA        = 15


# ═══════════════════════════════════════════════════════════════════════════════
# NUMPY UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class NumpyRobustScaler:
    def fit_transform(self, X):
        self.center_ = np.median(X, axis=0)
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        self.scale_ = q75 - q25
        self.scale_[self.scale_ == 0.0] = 1.0
        return (X - self.center_) / self.scale_

    def transform(self, X):
        return (X - self.center_) / self.scale_


class NumpyPCA:
    def __init__(self, n_components=15):
        self.n_components = n_components

    def fit_transform(self, X):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        if X_centered.shape[0] <= 1:
            return np.zeros((X.shape[0], self.n_components))
        _, _, Vh = np.linalg.svd(X_centered, full_matrices=False)
        n_c = min(self.n_components, Vh.shape[0])
        self.components_ = Vh[:n_c]
        res = np.dot(X_centered, self.components_.T)
        if res.shape[1] < self.n_components:
            res = np.pad(res, ((0, 0), (0, self.n_components - res.shape[1])))
        return res

    def transform(self, X):
        X_centered = X - self.mean_
        res = np.dot(X_centered, self.components_.T)
        if res.shape[1] < self.n_components:
            res = np.pad(res, ((0, 0), (0, self.n_components - res.shape[1])))
        return res


class NumpyRidgeEnsemble:
    def __init__(self, alpha=2.0):
        self.alpha = alpha

    def fit(self, X, y):
        n_samples, n_features = X.shape
        X_b = np.hstack([np.ones((n_samples, 1)), X])
        I = np.eye(n_features + 1)
        I[0, 0] = 0.0
        self.w = np.linalg.solve(X_b.T @ X_b + self.alpha * I, X_b.T @ y)

    def predict(self, X):
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_b @ self.w


def numpy_group_kfold(groups, n_splits=5, seed=42):
    unique_groups = np.unique(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_groups)
    splits = np.array_split(unique_groups, n_splits)
    for val_groups in splits:
        val_mask = np.isin(groups, val_groups)
        tr_idx = np.where(~val_mask)[0]
        val_idx = np.where(val_mask)[0]
        yield tr_idx, val_idx


def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / (ss_tot + 1e-12))


def compute_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def log_experiment_results(per_tissue_results, donor_r2, donor_mae, baseline_mae,
                           improvement, n_donors, individual_r2s,
                           model_type="StackingEnsemble_0to12h", out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result_record = {
        "timestamp": timestamp,
        "model_type": model_type,
        "config": {
            "target_tissues": TARGET_TISSUES,
            "max_time_min": MAX_TIME_MIN,
            "n_var_genes": N_VAR_GENES,
            "n_degraders": N_DEGRADERS,
            "n_stable": N_STABLE,
            "n_pca": N_PCA,
            "cv_strategy": "standard_5fold_group_kfold",
        },
        "individual_model_r2": individual_r2s,
        "per_tissue_results": per_tissue_results,
        "donor_r2": float(donor_r2),
        "donor_mae_min": float(donor_mae),
        "baseline_mae_min": float(baseline_mae),
        "improvement_pct": float(improvement),
        "n_donors": int(n_donors),
    }

    json_path = os.path.join(out_dir, f"run_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(result_record, f, indent=2)
    print(f"\n  Experiment log saved -> {json_path}")

    csv_path = os.path.join(out_dir, "experiment_summary.csv")
    file_exists = os.path.exists(csv_path)

    summary_dict = {
        "timestamp": timestamp,
        "model_type": model_type,
        "n_donors": n_donors,
        "donor_r2": round(float(donor_r2), 4),
        "donor_mae_hrs": round(float(donor_mae) / 60.0, 2),
        "baseline_mae_hrs": round(float(baseline_mae) / 60.0, 2),
        "improvement_pct": round(float(improvement), 1),
    }

    for name, r2_val in individual_r2s.items():
        summary_dict[f"{name}_r2"] = round(r2_val, 4)

    for t in TARGET_TISSUES:
        t_key = t.split(" - ")[0].lower().replace(" ", "_")
        if t in per_tissue_results:
            summary_dict[f"{t_key}_r2"] = per_tissue_results[t]["r2"]
            summary_dict[f"{t_key}_mae_hrs"] = per_tissue_results[t]["mae_hrs"]

    summary_dict["n_var_genes"] = N_VAR_GENES
    summary_dict["cv_strategy"] = "standard_5fold_group_kfold"

    summary_row = pd.DataFrame([summary_dict])

    try:
        summary_row.to_csv(csv_path, mode="a", header=not file_exists, index=False)
        print(f"  Appended CSV row -> {csv_path}")
    except PermissionError:
        print(f"\n  [NOTE] Could not update '{csv_path}' -- close Excel first.")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    print("Loading GTEx metadata...")
    sa = pd.read_csv(SAMPLE_ATTR_PATH, sep="\t", low_memory=False)
    sa = sa[["SAMPID", "SMTSISCH", "SMTSD", "SMRIN", "SMATSSCR"]].rename(
        columns={"SMTSISCH": "time", "SMTSD": "tissue", "SMRIN": "rin", "SMATSSCR": "autolysis"}
    )
    sa = sa.dropna(subset=["time"])
    sa = sa[(sa["time"] >= 0) & (sa["tissue"].isin(TARGET_TISSUES))]

    sp = pd.read_csv(SUBJECT_PHENO_PATH, sep="\t", low_memory=False)
    sp["DTHHRDY"] = pd.to_numeric(sp["DTHHRDY"], errors="coerce")
    sa["SUBJID"] = sa["SAMPID"].str.split("-").str[:2].str.join("-")
    sa = sa.merge(sp[["SUBJID", "SEX", "AGE", "DTHHRDY"]], on="SUBJID", how="left")
    sa = sa[sa["time"] <= MAX_TIME_MIN].reset_index(drop=True)

    def _age(b):
        try:
            lo, hi = str(b).split("-")
            return (float(lo) + float(hi)) / 2.0
        except Exception:
            return np.nan

    sa["rin"] = pd.to_numeric(sa["rin"], errors="coerce").fillna(sa["rin"].median())
    sa["autolysis"] = pd.to_numeric(sa["autolysis"], errors="coerce").fillna(0)
    sa["age"] = sa["AGE"].apply(_age).fillna(sa["AGE"].apply(_age).median())
    sa["sex"] = pd.to_numeric(sa["SEX"], errors="coerce").fillna(1)
    sa["hardy"] = pd.to_numeric(sa["DTHHRDY"], errors="coerce").fillna(sa["DTHHRDY"].median())

    sid_set = set(sa["SAMPID"])
    print(f"\nLoading gene expression matrix for target samples (0-12h Window)...")
    rows = {}
    with gzip.open(GENE_READS_PATH, "rt") as f:
        f.readline(); f.readline()
        cols = f.readline().strip().split("\t")[2:]
        kidx = [i for i, s in enumerate(cols) if s in sid_set]
        kids = [cols[i] for i in kidx]
        for line in f:
            p = line.rstrip("\n").split("\t")
            vals = np.array([float(p[2 + i]) for i in kidx], dtype=np.float32)
            if vals.var() > 0.5:
                rows[p[1]] = vals

    genes = list(rows.keys())
    mat = np.vstack([rows[g] for g in genes]).T
    expr = pd.DataFrame(mat, index=kids, columns=genes, dtype=np.float32)

    sa = sa[sa["SAMPID"].isin(expr.index)].reset_index(drop=True)
    print(f"  -> Total valid samples: {len(sa)} across {sa['SUBJID'].nunique()} unique donors.")

    return sa, expr


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_in_fold_features(G_tr, G_val, y_tr):
    corrs = np.array([
        np.corrcoef(G_tr[:, i], y_tr)[0, 1] if G_tr[:, i].std() > 0 else 0.0
        for i in range(G_tr.shape[1])
    ])
    corrs = np.nan_to_num(corrs)

    deg_idx = np.argsort(corrs)[:N_DEGRADERS]
    sta_idx = np.argsort(np.abs(corrs))[:N_STABLE]

    def _make_ratios(G):
        return np.column_stack([G[:, d] - G[:, s] for d in deg_idx for s in sta_idx])

    R_tr = _make_ratios(G_tr)
    R_va = _make_ratios(G_val)

    scaler = NumpyRobustScaler()
    R_tr_s = scaler.fit_transform(R_tr)
    R_va_s = scaler.transform(R_va)

    pca = NumpyPCA(n_components=N_PCA)
    P_tr = pca.fit_transform(R_tr_s)
    P_va = pca.transform(R_va_s)

    S_tr = np.column_stack([
        R_tr.mean(axis=1), R_tr.std(axis=1),
        G_tr[:, deg_idx].mean(axis=1), G_tr[:, sta_idx].mean(axis=1)
    ])
    S_va = np.column_stack([
        R_va.mean(axis=1), R_va.std(axis=1),
        G_val[:, deg_idx].mean(axis=1), G_val[:, sta_idx].mean(axis=1)
    ])

    return np.hstack([P_tr, S_tr]), np.hstack([P_va, S_va])


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL EVALUATION & CV
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(sa, expr):
    n_samples = len(sa)
    y_all = sa["time"].values
    groups_all = sa["SUBJID"].values

    tissue_oh = pd.get_dummies(sa["tissue"]).values.astype(np.float64)
    clin_feats = np.column_stack([
        sa["rin"].values,
        sa["rin"].values ** 2,
        sa["autolysis"].values,
        sa["age"].values,
        sa["sex"].values,
        sa["hardy"].values,
    ])

    raw_G = expr.loc[sa["SAMPID"]].values.astype(np.float64)
    top_gene_idx = np.argsort(raw_G.var(axis=0))[-N_VAR_GENES:]
    G_log = np.log2(raw_G[:, top_gene_idx] + 1.0)

    y_log = np.log2(y_all + 1.0)

    oof_xgb   = np.zeros(n_samples)
    oof_rf    = np.zeros(n_samples)
    oof_ridge = np.zeros(n_samples)

    print("\n" + "=" * 60)
    print("  0-12 HOUR WINDOW STACKING: 5-Fold Donor GroupKFold CV")
    print("============================================================")

    for fold, (tr_idx, val_idx) in enumerate(numpy_group_kfold(groups_all, n_splits=5), 1):
        G_tr, G_val = G_log[tr_idx], G_log[val_idx]
        y_tr = y_all[tr_idx]

        feat_tr, feat_val = build_in_fold_features(G_tr, G_val, y_tr)

        X_tr = np.hstack([feat_tr, tissue_oh[tr_idx], clin_feats[tr_idx]])
        X_val = np.hstack([feat_val, tissue_oh[val_idx], clin_feats[val_idx]])

        # ── XGBoost ──
        if HAS_XGB:
            dtrain = xgb.DMatrix(X_tr, label=y_log[tr_idx])
            dval = xgb.DMatrix(X_val)
            params_xgb = {
                "max_depth": 4, "eta": 0.02, "subsample": 0.85,
                "colsample_bytree": 0.75, "alpha": 0.2, "lambda": 1.0,
                "objective": "reg:squarederror", "seed": 42, "verbosity": 0,
            }
            bst = xgb.train(params_xgb, dtrain, num_boost_round=500)
            oof_xgb[val_idx] = bst.predict(dval)

        # ── Random Forest ──
        if HAS_RF:
            rf = RandomForestRegressor(
                n_estimators=450, max_depth=10, min_samples_leaf=4,
                max_features=0.5, n_jobs=-1, random_state=42
            )
            rf.fit(X_tr, y_log[tr_idx])
            oof_rf[val_idx] = rf.predict(X_val)

        # ── Ridge ──
        ridge = NumpyRidgeEnsemble(alpha=2.0)
        ridge.fit(X_tr, y_log[tr_idx])
        oof_ridge[val_idx] = ridge.predict(X_val)

        n_val_donors = len(np.unique(groups_all[val_idx]))
        print(f"  Fold {fold}/5 done  |  val donors: {n_val_donors}  |  val samples: {len(val_idx)}")

    pred_xgb   = np.clip(2.0 ** oof_xgb - 1.0, 0.0, None)
    pred_rf    = np.clip(2.0 ** oof_rf - 1.0, 0.0, None)
    pred_ridge = np.clip(2.0 ** oof_ridge - 1.0, 0.0, None)

    # Optimal weight combination
    A = np.column_stack([pred_xgb, pred_rf, pred_ridge])
    ATA = A.T @ A + 0.01 * np.eye(3)
    ATy = A.T @ y_all
    raw_weights = np.linalg.solve(ATA, ATy)
    raw_weights = np.maximum(raw_weights, 0.0)
    weights = raw_weights / raw_weights.sum()

    print(f"\n  Learned weights: XGBoost={weights[0]:.3f}, RF={weights[1]:.3f}, Ridge={weights[2]:.3f}")

    pred_stacked = weights[0] * pred_xgb + weights[1] * pred_rf + weights[2] * pred_ridge

    print("\n" + "=" * 60)
    print("  I N D I V I D U A L   M O D E L   C O M P A R I S O N")
    print("=" * 60)

    df_base = pd.DataFrame({"SUBJID": groups_all, "y_true": y_all})
    individual_r2s = {}

    for name, preds in [("XGBoost", pred_xgb), ("RandomForest", pred_rf),
                         ("Ridge", pred_ridge), ("STACKED", pred_stacked)]:
        df_base["y_pred"] = preds
        donor_avg = df_base.groupby("SUBJID")[["y_true", "y_pred"]].mean()
        r2 = compute_r2(donor_avg["y_true"].values, donor_avg["y_pred"].values)
        mae = compute_mae(donor_avg["y_true"].values, donor_avg["y_pred"].values)
        marker = " <-- FINAL" if name == "STACKED" else ""
        print(f"  {name:<15} | Donor R² = {r2:.4f} | MAE = {mae/60:.2f} hrs{marker}")
        individual_r2s[name] = round(float(r2), 4)

    print("=" * 60)

    df_eval = pd.DataFrame({
        "SUBJID": groups_all, "tissue": sa["tissue"],
        "y_true": y_all, "y_pred": pred_stacked,
    })

    print("\n" + "=" * 60)
    print("  P E R - T I S S U E   A C C U R A C Y   B R E A K D O W N")
    print("=" * 60)
    print(f"  {'Tissue Type':<35} | {'Samples':<8} | {'R²':<7} | {'MAE (hrs)':<10}")
    print("-" * 65)

    per_tissue_results = {}
    for t in TARGET_TISSUES:
        t_sub = df_eval[df_eval["tissue"] == t]
        if len(t_sub) > 5:
            r2_t = compute_r2(t_sub["y_true"].values, t_sub["y_pred"].values)
            mae_t = compute_mae(t_sub["y_true"].values, t_sub["y_pred"].values)
            print(f"  {t:<35} | {len(t_sub):<8} | {r2_t:<7.4f} | {mae_t/60:<10.2f} hrs")
            per_tissue_results[t] = {
                "n_samples": len(t_sub),
                "r2": round(float(r2_t), 4),
                "mae_hrs": round(float(mae_t) / 60.0, 2),
            }
    print("=" * 65)

    donor_eval = df_eval.groupby("SUBJID")[["y_true", "y_pred"]].mean()
    r2_donor = compute_r2(donor_eval["y_true"].values, donor_eval["y_pred"].values)
    mae_donor = compute_mae(donor_eval["y_true"].values, donor_eval["y_pred"].values)
    baseline_mae = compute_mae(donor_eval["y_true"].values,
                               np.full_like(donor_eval["y_true"].values, donor_eval["y_true"].mean()))
    improvement = (1.0 - mae_donor / baseline_mae) * 100.0

    print("\n" + "=" * 60)
    print("  D O N O R - L E V E L   S T A C K E D   R E S U L T S")
    print("=" * 60)
    print(f"  Total Unique Donors        : {len(donor_eval)}")
    print(f"  Stacked Donor R²           : {r2_donor:.4f}")
    print(f"  Stacked Donor MAE          : {mae_donor:.1f} min ({mae_donor/60:.2f} hrs)")
    print(f"  Baseline MAE (Mean Guess)  : {baseline_mae:.1f} min")
    print(f"  Improvement over Baseline  : {improvement:.1f}%")
    print("=" * 60)

    _save_scatter(donor_eval["y_true"].values, donor_eval["y_pred"].values, r2_donor, mae_donor)

    log_experiment_results(per_tissue_results, r2_donor, mae_donor, baseline_mae,
                           improvement, len(donor_eval), individual_r2s,
                           model_type="StackingEnsemble_0to12h")


def _save_scatter(y_actual, y_pred, r2, mae, out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(y_actual, y_pred, alpha=0.5, s=25, color="#1d4ed8",
               edgecolors="none", label="GTEx Donors (0-12h)", zorder=3)
    lo = min(y_actual.min(), y_pred.min())
    hi = max(y_actual.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y = x (perfect fit)", zorder=2)

    ax.text(0.05, 0.92, f"Stacked R² = {r2:.4f}", transform=ax.transAxes,
            fontsize=13, fontweight="bold", color="#1e3a8a")
    ax.text(0.05, 0.85, f"MAE = {mae:.1f} min ({mae/60:.2f} hrs)",
            transform=ax.transAxes, fontsize=10, color="#475569")

    ax.set_xlabel("Actual Donor Ischemic Time (minutes)", fontsize=12)
    ax.set_ylabel("Predicted Donor Ischemic Time (minutes)", fontsize=12)
    ax.set_title("Multi-Tissue PMI Model (0-12h Primary Kinetic Window)\n(5-Fold Donor GroupKFold)", fontsize=11)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    for ext in ("png", "svg"):
        path = os.path.join(out_dir, f"rna_model_predicted_vs_actual.{ext}")
        fig.savefig(path, dpi=150)
        print(f"  Plot saved -> {path}")
    plt.close(fig)


def train_final_model_and_save(sa, expr, out_path="models/rna_model.pkl"):
    print("\nTraining final 0-12h model on all data for deployment...")
    y_all = sa["time"].values
    tissue_oh = pd.get_dummies(sa["tissue"])
    tissue_columns = tissue_oh.columns.tolist()

    clin_feats = np.column_stack([
        sa["rin"].values, sa["rin"].values ** 2, sa["autolysis"].values,
        sa["age"].values, sa["sex"].values, sa["hardy"].values,
    ])

    raw_G = expr.loc[sa["SAMPID"]].values.astype(np.float64)
    top_gene_idx = np.argsort(raw_G.var(axis=0))[-N_VAR_GENES:]
    G_log = np.log2(raw_G[:, top_gene_idx] + 1.0)

    corrs = np.array([
        np.corrcoef(G_log[:, i], y_all)[0, 1] if G_log[:, i].std() > 0 else 0.0
        for i in range(G_log.shape[1])
    ])
    corrs = np.nan_to_num(corrs)
    deg_idx = np.argsort(corrs)[:N_DEGRADERS]
    sta_idx = np.argsort(np.abs(corrs))[:N_STABLE]

    def make_ratios(G):
        return np.column_stack([G[:, d] - G[:, s] for d in deg_idx for s in sta_idx])

    R = make_ratios(G_log)
    scaler = NumpyRobustScaler()
    R_scaled = scaler.fit_transform(R)
    pca = NumpyPCA(n_components=N_PCA)
    P = pca.fit_transform(R_scaled)

    S = np.column_stack([
        R.mean(axis=1), R.std(axis=1),
        G_log[:, deg_idx].mean(axis=1), G_log[:, sta_idx].mean(axis=1)
    ])

    X_final = np.hstack([P, S, tissue_oh.values.astype(np.float64), clin_feats])
    y_log = np.log2(y_all + 1.0)

    saved_models = {}
    if HAS_XGB:
        saved_models["xgb"] = xgb.train(
            {"max_depth": 4, "eta": 0.02, "subsample": 0.85,
             "colsample_bytree": 0.75, "alpha": 0.2, "lambda": 1.0,
             "objective": "reg:squarederror", "seed": 42, "verbosity": 0},
            xgb.DMatrix(X_final, label=y_log), num_boost_round=500,
        )

    if HAS_RF:
        rf_final = RandomForestRegressor(
            n_estimators=450, max_depth=10, min_samples_leaf=4,
            max_features=0.5, n_jobs=-1, random_state=42
        )
        rf_final.fit(X_final, y_log)
        saved_models["rf"] = rf_final

    ridge_final = NumpyRidgeEnsemble(alpha=2.0)
    ridge_final.fit(X_final, y_log)
    saved_models["ridge"] = ridge_final

    os.makedirs("models", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "models": saved_models,
            "top_gene_idx": top_gene_idx,
            "deg_idx": deg_idx,
            "sta_idx": sta_idx,
            "scaler_center": scaler.center_,
            "scaler_scale": scaler.scale_,
            "pca_mean": pca.mean_,
            "pca_components": pca.components_,
            "tissue_columns": tissue_columns,
            "gene_names": expr.columns[top_gene_idx].tolist(),
            "architecture": "stacking_ensemble_0to12h",
        }, f)

    print(f"  -> Final deployment model saved to {out_path}")


def main():
    sa, expr = load_data()
    train_and_evaluate(sa, expr)
    train_final_model_and_save(sa, expr)


if __name__ == "__main__":
    main()