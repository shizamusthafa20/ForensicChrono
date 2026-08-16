"""
ForensicChrono - RNA Degradation Model
======================================

MODEL:
    Leakage-Controlled Nested OOF Multi-Model Stacking

BASE MODELS:
    1. XGBoost - direct PMI
    2. XGBoost - log PMI
    3. Random Forest - direct PMI
    4. Random Forest - log PMI
    5. ExtraTrees - direct PMI
    6. ExtraTrees - log PMI

META MODEL:
    Ridge Regression

VALIDATION:
    Outer 5-fold donor-level CV
    Inner 4-fold donor-level OOF stacking

OUTPUTS:
    results/rna_oof_predictions.csv
    results/rna_donor_predictions.csv
    results/experiment_summary.csv
    results/experiment_summary.xlsx
    results/rna_results.json

    reports/rna_model_predicted_vs_actual.png
    reports/rna_model_predicted_vs_actual.svg

    models/rna_model.pkl
"""

import os
import gzip
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor
)

from sklearn.linear_model import Ridge

from sklearn.metrics import (
    r2_score,
    mean_absolute_error
)


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_ATTR_PATH = (
    "data/raw/rna/"
    "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
)

SUBJECT_PHENO_PATH = (
    "data/raw/rna/"
    "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt"
)

GENE_READS_PATH = (
    "data/raw/rna/"
    "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct.gz"
)

TISSUES = [
    "Muscle - Skeletal",
    "Lung",
    "Skin - Sun Exposed (Lower leg)",
    "Nerve - Tibial",
]

MAX_TIME_MIN = 1200

N_GENES = 1500
N_DEGRADERS = 75
N_STABLE = 30

# Compact tree ensemble used consistently for BOTH validation and deployment.
XGB_TREES = 500
RF_TREES = 300
EXTRA_TREES = 300

OUTER_FOLDS = 5
INNER_FOLDS = 4

SEED = 42


# ============================================================
# GROUPED CROSS-VALIDATION
# ============================================================

def grouped_folds(groups, n_splits, seed=42):

    groups = np.asarray(groups)

    unique_groups = np.unique(groups)

    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Only {len(unique_groups)} donors available, "
            f"but {n_splits} folds requested."
        )

    rng = np.random.RandomState(seed)

    unique_groups = unique_groups.copy()
    rng.shuffle(unique_groups)

    folds = np.array_split(
        unique_groups,
        n_splits
    )

    for fold_groups in folds:

        val_mask = np.isin(
            groups,
            fold_groups
        )

        train_idx = np.where(
            ~val_mask
        )[0]

        val_idx = np.where(
            val_mask
        )[0]

        yield train_idx, val_idx


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading GTEx metadata...")

    sa = pd.read_csv(
        SAMPLE_ATTR_PATH,
        sep="\t",
        low_memory=False
    )

    sa = sa[
        [
            "SAMPID",
            "SMTSISCH",
            "SMTSD",
            "SMRIN",
            "SMATSSCR"
        ]
    ].rename(
        columns={
            "SMTSISCH": "time",
            "SMTSD": "tissue",
            "SMRIN": "rin",
            "SMATSSCR": "autolysis"
        }
    )

    sa = sa.dropna(
        subset=["time"]
    )

    sa = sa[
        (sa["time"] >= 0) &
        (sa["time"] <= MAX_TIME_MIN) &
        (sa["tissue"].isin(TISSUES))
    ]

    sp = pd.read_csv(
        SUBJECT_PHENO_PATH,
        sep="\t",
        low_memory=False
    )

    sp["DTHHRDY"] = pd.to_numeric(
        sp["DTHHRDY"],
        errors="coerce"
    )

    sa["SUBJID"] = (
        sa["SAMPID"]
        .str.split("-")
        .str[:2]
        .str.join("-")
    )

    sa = sa.merge(
        sp[
            [
                "SUBJID",
                "SEX",
                "AGE",
                "DTHHRDY"
            ]
        ],
        on="SUBJID",
        how="left"
    )

    def age_mid(x):

        try:
            lo, hi = str(x).split("-")
            return (
                float(lo) +
                float(hi)
            ) / 2.0

        except Exception:
            return np.nan

    sa["age"] = sa["AGE"].apply(
        age_mid
    )

    sa["rin"] = pd.to_numeric(
        sa["rin"],
        errors="coerce"
    )

    sa["autolysis"] = pd.to_numeric(
        sa["autolysis"],
        errors="coerce"
    )

    sa["sex"] = pd.to_numeric(
        sa["SEX"],
        errors="coerce"
    )

    sa["hardy"] = pd.to_numeric(
        sa["DTHHRDY"],
        errors="coerce"
    )

    wanted = set(
        sa["SAMPID"]
    )

    print("Loading gene expression...")

    rows = {}

    with gzip.open(
        GENE_READS_PATH,
        "rt"
    ) as f:

        f.readline()
        f.readline()

        columns = (
            f.readline()
            .strip()
            .split("\t")[2:]
        )

        indices = [
            i
            for i, sample in enumerate(columns)
            if sample in wanted
        ]

        sample_ids = [
            columns[i]
            for i in indices
        ]

        for line in f:

            parts = line.rstrip(
                "\n"
            ).split("\t")

            values = np.array(
                [
                    float(
                        parts[2 + i]
                    )
                    for i in indices
                ],
                dtype=np.float32
            )

            if values.var() > 0.1:
                rows[
                    parts[1]
                ] = values

    genes = list(
        rows.keys()
    )

    matrix = np.vstack(
        [
            rows[g]
            for g in genes
        ]
    ).T

    expr = pd.DataFrame(
        matrix,
        index=sample_ids,
        columns=genes
    )

    sa = sa[
        sa["SAMPID"].isin(
            expr.index
        )
    ].reset_index(
        drop=True
    )

    expr = expr.loc[
        sa["SAMPID"]
    ]

    print(
        f"Samples : {len(sa)}"
    )

    print(
        f"Donors  : "
        f"{sa['SUBJID'].nunique()}"
    )

    print(
        f"Genes   : {expr.shape[1]}"
    )

    return sa, expr


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def make_features(
    train_df,
    val_df,
    train_expr,
    val_expr
):

    train_log = np.log2(
        train_expr + 1.0
    )

    val_log = np.log2(
        val_expr + 1.0
    )

    variance = train_log.var(
        axis=0
    )

    selected = (
        variance
        .sort_values(
            ascending=False
        )
        .head(N_GENES)
        .index
    )

    train_g = train_log[
        selected
    ].values

    val_g = val_log[
        selected
    ].values

    y = train_df[
        "time"
    ].values

    correlations = np.array([
        np.corrcoef(
            train_g[:, i],
            y
        )[0, 1]
        if train_g[:, i].std() > 0
        else 0.0
        for i in range(
            train_g.shape[1]
        )
    ])

    correlations = np.nan_to_num(
        correlations
    )

    degrader_idx = np.argsort(
        correlations
    )[:N_DEGRADERS]

    stable_idx = np.argsort(
        np.abs(correlations)
    )[:N_STABLE]

    def make_ratios(G):

        return np.column_stack([
            G[:, d] - G[:, s]
            for d in degrader_idx
            for s in stable_idx
        ])

    train_ratio = make_ratios(
        train_g
    )

    val_ratio = make_ratios(
        val_g
    )

    train_summary = np.column_stack([
        train_ratio.mean(axis=1),
        train_ratio.std(axis=1),
        train_ratio.min(axis=1),
        train_ratio.max(axis=1),
        train_g[:, degrader_idx].mean(axis=1),
        train_g[:, stable_idx].mean(axis=1)
    ])

    val_summary = np.column_stack([
        val_ratio.mean(axis=1),
        val_ratio.std(axis=1),
        val_ratio.min(axis=1),
        val_ratio.max(axis=1),
        val_g[:, degrader_idx].mean(axis=1),
        val_g[:, stable_idx].mean(axis=1)
    ])

    numeric_cols = [
        "rin",
        "autolysis",
        "age",
        "sex",
        "hardy"
    ]

    train_clean = train_df.copy()
    val_clean = val_df.copy()

    for col in numeric_cols:

        median = train_clean[
            col
        ].median()

        if pd.isna(median):
            median = 0.0

        train_clean[col] = (
            train_clean[col]
            .fillna(median)
        )

        val_clean[col] = (
            val_clean[col]
            .fillna(median)
        )

    def clinical_features(df):

        rin = df["rin"].values
        aut = df["autolysis"].values
        age = df["age"].values
        sex = df["sex"].values
        hardy = df["hardy"].values

        return np.column_stack([
            rin,
            rin ** 2,
            aut,
            aut ** 2,
            age,
            sex,
            hardy,
            rin * aut,
            rin / (aut + 1.0)
        ])

    train_clinical = clinical_features(
        train_clean
    )

    val_clinical = clinical_features(
        val_clean
    )

    X_train = np.hstack([
        train_g,
        train_ratio,
        train_summary,
        train_clinical
    ])

    X_val = np.hstack([
        val_g,
        val_ratio,
        val_summary,
        val_clinical
    ])

    return X_train, X_val


def build_full_features(df, expr):
    """
    Build the final feature matrix using the COMPLETE training
    dataset for one tissue.

    Returns:
        X
        feature_info
    """

    log_expr = np.log2(
        expr + 1.0
    )

    # --------------------------------------------------------
    # VARIABLE GENES
    # --------------------------------------------------------

    variance = log_expr.var(
        axis=0
    )

    selected = (
        variance
        .sort_values(
            ascending=False
        )
        .head(N_GENES)
        .index
    )

    G = log_expr[
        selected
    ].values

    y = df[
        "time"
    ].values

    # --------------------------------------------------------
    # PMI-CORRELATED GENES
    # --------------------------------------------------------

    correlations = np.array([
        np.corrcoef(
            G[:, i],
            y
        )[0, 1]
        if G[:, i].std() > 0
        else 0.0
        for i in range(
            G.shape[1]
        )
    ])

    correlations = np.nan_to_num(
        correlations
    )

    degrader_idx = np.argsort(
        correlations
    )[:N_DEGRADERS]

    stable_idx = np.argsort(
        np.abs(correlations)
    )[:N_STABLE]

    # --------------------------------------------------------
    # LOG-RATIO FEATURES
    # --------------------------------------------------------

    ratio_features = np.column_stack([
        G[:, d] - G[:, s]
        for d in degrader_idx
        for s in stable_idx
    ])

    summary_features = np.column_stack([
        ratio_features.mean(axis=1),
        ratio_features.std(axis=1),
        ratio_features.min(axis=1),
        ratio_features.max(axis=1),
        G[:, degrader_idx].mean(axis=1),
        G[:, stable_idx].mean(axis=1)
    ])

    # --------------------------------------------------------
    # CLINICAL / QC FEATURES
    # --------------------------------------------------------

    numeric_cols = [
        "rin",
        "autolysis",
        "age",
        "sex",
        "hardy"
    ]

    clean = df.copy()
    medians = {}

    for col in numeric_cols:

        median = clean[
            col
        ].median()

        if pd.isna(median):
            median = 0.0

        medians[col] = float(
            median
        )

        clean[col] = (
            clean[col]
            .fillna(median)
        )

    rin = clean["rin"].values
    aut = clean["autolysis"].values
    age = clean["age"].values
    sex = clean["sex"].values
    hardy = clean["hardy"].values

    clinical = np.column_stack([
        rin,
        rin ** 2,
        aut,
        aut ** 2,
        age,
        sex,
        hardy,
        rin * aut,
        rin / (aut + 1.0)
    ])

    X = np.hstack([
        G,
        ratio_features,
        summary_features,
        clinical
    ])

    feature_info = {
        "selected_genes":
            list(selected),

        "degrader_indices":
            degrader_idx.tolist(),

        "stable_indices":
            stable_idx.tolist(),

        "clinical_medians":
            medians,

        "n_genes":
            N_GENES,

        "n_degraders":
            N_DEGRADERS,

        "n_stable":
            N_STABLE
    }

    return X, feature_info


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def make_xgb():

    return xgb.XGBRegressor(
        n_estimators=XGB_TREES,
        max_depth=5,
        learning_rate=0.02,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.75,
        gamma=0.05,
        reg_alpha=0.2,
        reg_lambda=2.0,
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=-1
    )


def make_rf():

    return RandomForestRegressor(
        n_estimators=RF_TREES,
        max_depth=16,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=SEED
    )


def make_extra():

    return ExtraTreesRegressor(
        n_estimators=EXTRA_TREES,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=SEED
    )


# ============================================================
# BASE MODEL TRAIN + PREDICT
# ============================================================

def train_predict(
    model_type,
    target_mode,
    X_train,
    y_train,
    X_val
):

    if model_type == "xgb":
        model = make_xgb()
    elif model_type == "rf":
        model = make_rf()
    elif model_type == "extra":
        model = make_extra()
    else:
        raise ValueError("Unknown model type.")

    if target_mode == "direct":
        model.fit(X_train, y_train)
        prediction = model.predict(X_val)
    else:
        model.fit(X_train, np.log1p(y_train))
        prediction = np.expm1(model.predict(X_val))

    return np.clip(prediction, 0, MAX_TIME_MIN)


MODEL_CONFIGS = [
    ("xgb_direct", "xgb", "direct"),
    ("xgb_log", "xgb", "log"),
    ("rf_direct", "rf", "direct"),
    ("rf_log", "rf", "log"),
    ("extra_direct", "extra", "direct"),
    ("extra_log", "extra", "log")
]


# ============================================================
# NESTED OOF EVALUATION
# ============================================================

def evaluate_nested_oof(sa, expr):

    final_predictions = np.zeros(len(sa))
    all_oof_base_predictions = np.zeros((len(sa), len(MODEL_CONFIGS)))
    tissue_results = {}

    for tissue in TISSUES:

        print("\n")
        print("=" * 70)
        print(f"TISSUE: {tissue}")
        print("=" * 70)

        tissue_indices = np.where(sa["tissue"].values == tissue)[0]
        tissue_sa = sa.iloc[tissue_indices].reset_index(drop=True)
        tissue_expr = expr.loc[tissue_sa["SAMPID"]]
        groups = tissue_sa["SUBJID"].values

        tissue_predictions = np.zeros(len(tissue_sa))

        for outer_fold, (outer_train_idx, outer_val_idx) in enumerate(
            grouped_folds(groups, OUTER_FOLDS, SEED), 1
        ):

            print(f"\nOuter fold {outer_fold}/{OUTER_FOLDS}")

            outer_train_df = tissue_sa.iloc[outer_train_idx]
            outer_val_df = tissue_sa.iloc[outer_val_idx]
            outer_train_expr = tissue_expr.iloc[outer_train_idx]
            outer_val_expr = tissue_expr.iloc[outer_val_idx]

            inner_groups = outer_train_df["SUBJID"].values
            inner_oof = np.zeros((len(outer_train_df), len(MODEL_CONFIGS)))

            print("  Creating inner OOF predictions...")

            for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
                grouped_folds(inner_groups, INNER_FOLDS, SEED + outer_fold), 1
            ):

                inner_train_df = outer_train_df.iloc[inner_train_idx]
                inner_val_df = outer_train_df.iloc[inner_val_idx]
                inner_train_expr = outer_train_expr.iloc[inner_train_idx]
                inner_val_expr = outer_train_expr.iloc[inner_val_idx]

                X_train, X_val = make_features(
                    inner_train_df, inner_val_df,
                    inner_train_expr, inner_val_expr
                )
                y_train = inner_train_df["time"].values

                for col, (name, model_type, target_mode) in enumerate(MODEL_CONFIGS):
                    inner_oof[inner_val_idx, col] = train_predict(
                        model_type, target_mode, X_train, y_train, X_val
                    )

                print(f"    Inner fold {inner_fold}/{INNER_FOLDS}")

            meta_model = Ridge(alpha=10.0)
            meta_model.fit(inner_oof, outer_train_df["time"].values)

            X_train, X_val = make_features(
                outer_train_df, outer_val_df,
                outer_train_expr, outer_val_expr
            )
            y_train = outer_train_df["time"].values

            outer_predictions = np.zeros((len(outer_val_df), len(MODEL_CONFIGS)))

            for col, (name, model_type, target_mode) in enumerate(MODEL_CONFIGS):
                outer_predictions[:, col] = train_predict(
                    model_type, target_mode, X_train, y_train, X_val
                )

            stacked_prediction = meta_model.predict(outer_predictions)
            stacked_prediction = np.clip(stacked_prediction, 0, MAX_TIME_MIN)

            tissue_predictions[outer_val_idx] = stacked_prediction

            all_oof_base_predictions[tissue_indices[outer_val_idx], :] = outer_predictions
            print("  Outer fold completed.")

        actual = tissue_sa["time"].values
        r2 = r2_score(actual, tissue_predictions)
        mae = mean_absolute_error(actual, tissue_predictions)

        tissue_results[tissue] = {
            "r2": float(r2),
            "mae_minutes": float(mae)
        }

        final_predictions[tissue_indices] = tissue_predictions

        print(f"\n{tissue}")
        print(f"R²  : {r2:.4f}")
        print(f"MAE : {mae:.1f} min ({mae / 60:.2f} hrs)")

    return final_predictions, all_oof_base_predictions, tissue_results


# ============================================================
# FINAL DEPLOYABLE MODEL
# ============================================================

def train_final_deployable_model(
    sa,
    expr,
    all_oof_base_predictions
):
    """
    Train the final deployable model after OOF evaluation.

    The six base models are trained on ALL available samples
    within each tissue.

    The Ridge meta-model is trained on the already-generated
    OUT-OF-FOLD predictions from the evaluation stage. This
    avoids using in-sample base-model predictions to train the
    meta-model.

    A separate Ridge meta-model is stored for each tissue,
    matching the tissue-specific validation architecture.
    """

    print("\n" + "=" * 70)
    print("TRAINING FINAL DEPLOYABLE MODEL")
    print("=" * 70)

    deployable_models = {}

    for tissue in TISSUES:

        print(f"\nFinal training: {tissue}")

        tissue_indices = np.where(
            sa["tissue"].values == tissue
        )[0]

        tissue_sa = sa.iloc[
            tissue_indices
        ].reset_index(
            drop=True
        )

        tissue_expr = expr.loc[
            tissue_sa["SAMPID"]
        ]

        # ----------------------------------------------------
        # Build final features on ALL data for this tissue
        # ----------------------------------------------------

        X, feature_info = build_full_features(
            tissue_sa,
            tissue_expr
        )

        y = tissue_sa[
            "time"
        ].values

        # ----------------------------------------------------
        # Six final base models
        # ----------------------------------------------------

        models = {}

        for name, model_type, target_mode in MODEL_CONFIGS:

            print(
                f"  Training {name}..."
            )

            if model_type == "xgb":
                model = make_xgb()

            elif model_type == "rf":
                model = make_rf()

            else:
                model = make_extra()

            if target_mode == "direct":
                model.fit(
                    X,
                    y
                )
            else:
                model.fit(
                    X,
                    np.log1p(y)
                )

            models[name] = model

        # ----------------------------------------------------
        # Final Ridge
        #
        # Use ONLY OOF predictions generated during the
        # evaluation stage. These predictions are already
        # out-of-fold for every sample.
        # ----------------------------------------------------

        tissue_oof = all_oof_base_predictions[
            tissue_indices,
            :
        ]

        ridge = Ridge(
            alpha=10.0
        )

        ridge.fit(
            tissue_oof,
            y
        )

        print(
            "  Ridge meta-model trained on "
            "out-of-fold predictions."
        )

        deployable_models[
            tissue
        ] = {
            "base_models": models,
            "meta_model": ridge,
            "feature_info": feature_info
        }

    # --------------------------------------------------------
    # Complete package
    # --------------------------------------------------------

    package = {

        "model_type":
            "Nested OOF "
            "XGB_RF_ExtraTrees_Ridge",

        "base_models":
            deployable_models,

        "tissues":
            TISSUES,

        "max_time_min":
            MAX_TIME_MIN,

        "n_genes":
            N_GENES,

        "n_degraders":
            N_DEGRADERS,

        "n_stable":
            N_STABLE,

        "seed":
            SEED,

        "base_model_configs": MODEL_CONFIGS,

        "tree_counts": {
            "xgb": XGB_TREES,
            "random_forest": RF_TREES,
            "extra_trees": EXTRA_TREES
        },

        "description":
            "Final deployable tissue-specific "
            "RNA PMI stacking model. Base models "
            "are trained on all available data; "
            "Ridge meta-models are trained only "
            "on out-of-fold predictions."
    }

    os.makedirs(
        "models",
        exist_ok=True
    )

    model_path = (
        "models/"
        "rna_model.pkl"
    )

    with open(
        model_path,
        "wb"
    ) as f:

        pickle.dump(
            package,
            f
        )

    print(
        f"\nDeployable model saved -> "
        f"{model_path}"
    )


# ============================================================
# REGRESSION GRAPH
# ============================================================

def save_regression_graph(donor, r2, mae, out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)

    actual    = donor["actual"].values
    predicted = donor["predicted"].values

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(
        actual, predicted,
        alpha=0.45, s=24,
        color="#2563eb", edgecolors="none",
        label="Held-out donors"
    )

    # ONE line only — fitted predicted-vs-actual regression line
    slope, intercept = np.polyfit(actual, predicted, 1)
    x = np.linspace(actual.min(), actual.max(), 100)
    regression_line = slope * x + intercept
    ax.plot(x, regression_line, linewidth=2.0, label="Fitted regression line")

    ax.text(0.05, 0.93, f"R² = {r2:.4f}",
            transform=ax.transAxes, fontsize=13, fontweight="bold")
    ax.text(0.05, 0.87, f"MAE = {mae:.1f} min ({mae / 60:.2f} hrs)",
            transform=ax.transAxes, fontsize=11)

    ax.set_xlabel("Actual Donor Ischemic Time (minutes)")
    ax.set_ylabel("Predicted Donor Ischemic Time (minutes)")
    ax.set_title("ForensicChrono RNA Model\nNested OOF Multi-Model Stacking")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()

    png_path = os.path.join(out_dir, "rna_model_predicted_vs_actual.png")
    svg_path = os.path.join(out_dir, "rna_model_predicted_vs_actual.svg")

    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)

    print(f"Regression graph saved -> {png_path}")


# ============================================================
# EXCEL + CSV LOGGING
# ============================================================

def save_results(summary, sample_predictions, donor_predictions):
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    sample_predictions.to_csv("results/rna_oof_predictions.csv", index=False)
    donor_predictions.to_csv("results/rna_donor_predictions.csv")

    summary_row = {
        "Timestamp": timestamp,
        "Model": "Compact XGB + RF + ExtraTrees Direct/Log + Ridge",
        "Donors": summary["n_donors"],
        "R2": round(summary["donor_r2"], 4),
        "MAE (hours)": round(summary["donor_mae_hours"], 2),
        "Baseline MAE (hours)": round(summary["baseline_mae_minutes"] / 60, 2),
        "Improvement (%)": round(summary["improvement_percent"], 1),
        "Outer CV": "5-fold donor-level",
        "Inner CV": "4-fold donor-level",
        "Leakage": "None"
    }

    summary_df = pd.DataFrame([summary_row])

    csv_path = "results/experiment_summary.csv"
    file_exists = os.path.exists(csv_path)

    csv_dict = {
        "timestamp": timestamp,
        "model_type": summary_row["Model"],
        "n_donors": summary["n_donors"],
        "donor_r2": round(summary["donor_r2"], 4),
        "donor_mae_hrs": round(summary["donor_mae_hours"], 2),
        "baseline_mae_hrs": round(summary["baseline_mae_minutes"] / 60.0, 2),
        "improvement_pct": round(summary["improvement_percent"], 1),
    }

    tissue_res = summary["tissue_results"]
    for t in TISSUES:
        t_key = t.split(" - ")[0].lower().replace(" ", "_")
        if t in tissue_res:
            csv_dict[f"{t_key}_r2"] = round(tissue_res[t]["r2"], 4)
            csv_dict[f"{t_key}_mae_hrs"] = round(tissue_res[t]["mae_minutes"] / 60.0, 2)

    csv_df = pd.DataFrame([csv_dict])
    try:
        csv_df.to_csv(csv_path, mode="a", header=not file_exists, index=False)
        print(f"Summary appended -> {csv_path}")
    except Exception as e:
        print(f"Could not write CSV: {e}")

    try:
        excel_path = "results/experiment_summary.xlsx"
        tissue_rows = []
        for tissue, values in summary["tissue_results"].items():
            tissue_rows.append({
                "Tissue": tissue,
                "R2": round(values["r2"], 4),
                "MAE (hours)": round(values["mae_minutes"] / 60, 2)
            })
        tissue_df = pd.DataFrame(tissue_rows)

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            tissue_df.to_excel(writer, sheet_name="Tissue Results", index=False)
            sample_predictions.to_excel(writer, sheet_name="OOF Predictions", index=False)
            donor_predictions.reset_index().to_excel(writer, sheet_name="Donor Predictions", index=False)
        print(f"Excel saved -> {excel_path}")
    except Exception as e:
        print(f"[NOTE] Excel export skipped ({e}). CSV logged cleanly.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("FORENSICCHRONO RNA MODEL — FINAL EVALUATION + DEPLOYMENT")
    print("=" * 70)
    print(f"N_GENES     : {N_GENES}")
    print(f"N_DEGRADERS : {N_DEGRADERS}")
    print(f"N_STABLE    : {N_STABLE}")
    print("Nested donor-level OOF evaluation + final deployable model")
    print("=" * 70)

    # ------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------
    sa, expr = load_data()

    # ------------------------------------------------------------
    # 1. LEAKAGE-CONTROLLED NESTED OOF EVALUATION
    # ------------------------------------------------------------
    (
        final_predictions,
        all_oof_base_predictions,
        tissue_results
    ) = evaluate_nested_oof(sa, expr)

    # ------------------------------------------------------------
    # SAMPLE-LEVEL OOF RESULTS
    # ------------------------------------------------------------
    sample_results = pd.DataFrame({
        "SUBJID": sa["SUBJID"].values,
        "tissue": sa["tissue"].values,
        "actual": sa["time"].values,
        "predicted": final_predictions
    })

    # ------------------------------------------------------------
    # DONOR-LEVEL RESULTS
    # ------------------------------------------------------------
    donor_results = (
        sample_results
        .groupby("SUBJID")
        .agg(
            actual=("actual", "mean"),
            predicted=("predicted", "mean")
        )
    )

    donor_r2 = r2_score(
        donor_results["actual"],
        donor_results["predicted"]
    )

    donor_mae = mean_absolute_error(
        donor_results["actual"],
        donor_results["predicted"]
    )

    baseline_mae = mean_absolute_error(
        donor_results["actual"],
        np.full(
            len(donor_results),
            donor_results["actual"].mean()
        )
    )

    improvement_percent = (
        100.0 * (baseline_mae - donor_mae) / baseline_mae
        if baseline_mae > 0 else 0.0
    )

    summary = {
        "n_donors": int(len(donor_results)),
        "donor_r2": float(donor_r2),
        "donor_mae_hours": float(donor_mae / 60.0),
        "baseline_mae_minutes": float(baseline_mae),
        "improvement_percent": float(improvement_percent),
        "tissue_results": tissue_results
    }

    # ------------------------------------------------------------
    # PRINT FINAL EVALUATION RESULTS
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL NESTED OOF RNA RESULT")
    print("=" * 70)
    print(f"Samples : {len(sample_results)}")
    print(f"Donors  : {len(donor_results)}")
    print(f"R²      : {donor_r2:.4f}")
    print(f"MAE     : {donor_mae:.1f} min ({donor_mae / 60:.2f} hrs)")

    print("\nPer-tissue results:")
    for tissue, values in tissue_results.items():
        print(
            f"{tissue}: "
            f"R²={values['r2']:.4f}, "
            f"MAE={values['mae_minutes']/60:.2f} hrs"
        )

    # ------------------------------------------------------------
    # 2. SAVE OOF RESULTS + EXCEL/CSV
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SAVING EVALUATION RESULTS")
    print("=" * 70)

    save_results(
        summary,
        sample_results,
        donor_results
    )

    # ------------------------------------------------------------
    # 3. SAVE REGRESSION GRAPH
    # ------------------------------------------------------------
    save_regression_graph(
        donor_results,
        donor_r2,
        donor_mae
    )

    # ------------------------------------------------------------
    # 4. SAVE JSON SUMMARY
    # ------------------------------------------------------------
    os.makedirs("results", exist_ok=True)

    json_summary = {
        "model": "Compact XGB + RF + ExtraTrees Direct/Log + Ridge",
        "validation": {
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "grouping": "donor-level",
            "leakage_controlled": True
        },
        "samples": int(len(sample_results)),
        "donors": int(len(donor_results)),
        "r2": float(donor_r2),
        "mae_minutes": float(donor_mae),
        "mae_hours": float(donor_mae / 60.0),
        "baseline_mae_minutes": float(baseline_mae),
        "improvement_percent": float(improvement_percent),
        "tissue_results": tissue_results,
        "configuration": {
            "n_genes": N_GENES,
            "n_degraders": N_DEGRADERS,
            "n_stable": N_STABLE,
            "xgb_trees": XGB_TREES,
            "rf_trees": RF_TREES,
            "extra_trees": EXTRA_TREES,
            "seed": SEED
        }
    }

    with open("results/rna_results.json", "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2)

    print("JSON saved -> results/rna_results.json")

    # ------------------------------------------------------------
    # 5. TRAIN + SAVE FINAL DEPLOYABLE MODEL
    # ------------------------------------------------------------
    train_final_deployable_model(
        sa,
        expr,
        all_oof_base_predictions
    )

    # ------------------------------------------------------------
    # FINAL STATUS
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL RNA PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Validated donor-level R² : {donor_r2:.4f}")
    print(f"Validated donor-level MAE: {donor_mae:.1f} min ({donor_mae / 60:.2f} hrs)")
    print("\nCreated files:")
    print("  results/rna_oof_predictions.csv")
    print("  results/rna_donor_predictions.csv")
    print("  results/experiment_summary.csv")
    print("  results/experiment_summary.xlsx")
    print("  results/rna_results.json")
    print("  reports/rna_model_predicted_vs_actual.png")
    print("  reports/rna_model_predicted_vs_actual.svg")
    print("  models/rna_model.pkl")
    print("=" * 70)


if __name__ == "__main__":
    main()