"""
===============================================================================
FORENSICCHRONO - MODULE 2
MICROBIAL SUCCESSION PMI MODEL V3
===============================================================================

Dataset:
    Qiita 13810 16S microbiome dataset

Goal:
    Predict Post-Mortem Interval (PMI) using:

        1. Microbial community composition
        2. Accumulated Degree Days (ADD)
        3. Indoor / outdoor environment
        4. Body site

IMPORTANT:
    This model does NOT use:
        PMI * temperature
        PMI * ADD
        PMI-derived environmental features

    PMI is used only inside TRAINING-FOLD feature selection.

Environmental features actually present in the metadata:
    - indoor_add
    - outdoor_add
    - indoor_outdoor
    - body_site

Microbiome features:
    - Relative abundance
    - CLR transformation
    - Log relative abundance
    - Prevalence filtering
    - Variance-based taxa selection
    - PMI-associated taxa selection (training fold only)
    - Shannon diversity
    - Simpson diversity
    - Richness
    - Evenness
    - Dominance
    - Top-1 / Top-3 / Top-5 / Top-10 abundance

Models:
    - XGBoost direct PMI
    - XGBoost log PMI
    - ExtraTrees
    - Random Forest
    - PLS Regression

Final prediction:
    Weighted ensemble of the base models.

Evaluation:
    5-Fold DONOR-GROUPED cross-validation.

This means samples from the same donor NEVER appear in both
training and validation in an outer fold.

===============================================================================
"""

import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARNING] XGBoost is not installed.")


# =============================================================================
# CONFIGURATION
# =============================================================================

METADATA_PATH = "data/raw/microbiome/metadata.tsv"
OTU_TABLE_PATH = "data/raw/microbiome/otu_table.tsv"

SEED = 42
N_FOLDS = 5

# -------------------------------------------------------------------------
# Microbiome feature selection
# -------------------------------------------------------------------------

MIN_PREVALENCE = 0.05

N_SUPERVISED_TAXA = 500

N_VARIANCE_TAXA = 300

MAX_FINAL_TAXA = 750


# =============================================================================
# PRINT UTILITIES
# =============================================================================

def print_header(text):

    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


# =============================================================================
# METRICS
# =============================================================================

def safe_r2(y_true, y_pred):

    try:
        return r2_score(
            y_true,
            y_pred
        )

    except Exception:
        return np.nan


def compute_mae(y_true, y_pred):

    return mean_absolute_error(
        y_true,
        y_pred
    )


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():

    print_header(
        "LOADING QIITA MICROBIOME DATA"
    )

    print(
        f"Metadata path : {METADATA_PATH}"
    )

    print(
        f"OTU path      : {OTU_TABLE_PATH}"
    )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    meta = pd.read_csv(
        METADATA_PATH,
        sep="\t",
        low_memory=False
    )

    print(
        f"\nMetadata rows before cleaning: {len(meta)}"
    )

    required_columns = [
        "sample_id",
        "subject_id",
        "pmi"
    ]

    missing = [
        c for c in required_columns
        if c not in meta.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required metadata columns: {missing}"
        )

    # -------------------------------------------------------------------------
    # Convert required columns
    # -------------------------------------------------------------------------

    meta["sample_id"] = (
        meta["sample_id"]
        .astype(str)
    )

    meta["subject_id"] = (
        meta["subject_id"]
        .astype(str)
    )

    meta["pmi"] = pd.to_numeric(
        meta["pmi"],
        errors="coerce"
    )

    # -------------------------------------------------------------------------
    # Remove invalid PMI
    # -------------------------------------------------------------------------

    meta = meta.dropna(
        subset=[
            "sample_id",
            "subject_id",
            "pmi"
        ]
    )

    meta = meta[
        meta["pmi"] >= 0
    ]

    # Remove duplicate sample IDs
    meta = meta.drop_duplicates(
        subset=["sample_id"],
        keep="first"
    )

    meta = meta.reset_index(
        drop=True
    )

    # -------------------------------------------------------------------------
    # Check environmental columns
    # -------------------------------------------------------------------------

    environmental_columns = [
        "indoor_add",
        "outdoor_add",
        "indoor_outdoor",
        "body_site"
    ]

    print("\nEnvironmental metadata:")

    for col in environmental_columns:

        if col in meta.columns:

            print(
                f"  {col:<20} FOUND"
            )

        else:

            print(
                f"  {col:<20} NOT FOUND"
            )

    # -------------------------------------------------------------------------
    # Load OTU
    # -------------------------------------------------------------------------

    print(
        "\nLoading OTU table..."
    )

    otu = pd.read_csv(
        OTU_TABLE_PATH,
        sep="\t",
        index_col=0,
        low_memory=False
    )

    otu.columns = (
        otu.columns
        .astype(str)
    )

    # Make sure counts are numeric
    otu = otu.apply(
        pd.to_numeric,
        errors="coerce"
    ).fillna(0.0)

    # -------------------------------------------------------------------------
    # Find common samples
    # -------------------------------------------------------------------------

    common_samples = [
        s
        for s in meta["sample_id"]
        if s in otu.columns
    ]

    if len(common_samples) == 0:

        raise ValueError(
            "No common samples found between metadata and OTU table."
        )

    meta = meta[
        meta["sample_id"]
        .isin(common_samples)
    ].copy()

    meta = meta.reset_index(
        drop=True
    )

    sample_order = (
        meta["sample_id"]
        .tolist()
    )

    otu = otu[
        sample_order
    ].T

    otu.index = (
        otu.index
        .astype(str)
    )

    print(
        f"\nAligned samples : {len(meta)}"
    )

    print(
        f"Unique donors   : "
        f"{meta['subject_id'].nunique()}"
    )

    print(
        f"Initial taxa    : "
        f"{otu.shape[1]}"
    )

    print(
        f"PMI minimum     : "
        f"{meta['pmi'].min():.3f} days"
    )

    print(
        f"PMI maximum     : "
        f"{meta['pmi'].max():.3f} days"
    )

    return meta, otu


# =============================================================================
# MICROBIOME TRANSFORMATIONS
# =============================================================================

def relative_abundance(
    counts
):

    X = np.asarray(
        counts,
        dtype=np.float64
    )

    row_sums = X.sum(
        axis=1
    )

    row_sums[
        row_sums <= 0
    ] = 1.0

    return (
        X /
        row_sums[:, None]
    )


def clr_transform(
    relative
):

    X = np.asarray(
        relative,
        dtype=np.float64
    )

    positive = X[
        X > 0
    ]

    if len(positive) > 0:

        pseudocount = max(
            np.min(positive) * 0.5,
            1e-8
        )

    else:

        pseudocount = 1e-8

    X = X + pseudocount

    log_X = np.log(
        X
    )

    center = np.mean(
        log_X,
        axis=1,
        keepdims=True
    )

    return (
        log_X -
        center
    )


def log_relative_abundance(
    relative
):

    return np.log1p(
        relative * 10000.0
    )


# =============================================================================
# MICROBIOME DIVERSITY FEATURES
# =============================================================================

def compute_diversity_features(
    relative
):

    X = np.asarray(
        relative,
        dtype=np.float64
    )

    eps = 1e-12

    # -------------------------------------------------------------------------
    # Shannon
    # -------------------------------------------------------------------------

    shannon = -np.sum(
        X * np.log(
            X + eps
        ),
        axis=1
    )

    # -------------------------------------------------------------------------
    # Simpson
    # -------------------------------------------------------------------------

    simpson = 1.0 - np.sum(
        X ** 2,
        axis=1
    )

    # -------------------------------------------------------------------------
    # Richness
    # -------------------------------------------------------------------------

    richness = np.sum(
        X > 0,
        axis=1
    )

    # -------------------------------------------------------------------------
    # Evenness
    # -------------------------------------------------------------------------

    evenness = (
        shannon /
        (
            np.log(
                richness + 1.0
            ) + eps
        )
    )

    # -------------------------------------------------------------------------
    # Dominance
    # -------------------------------------------------------------------------

    dominance = np.max(
        X,
        axis=1
    )

    # -------------------------------------------------------------------------
    # Largest abundances
    # -------------------------------------------------------------------------

    sorted_X = np.sort(
        X,
        axis=1
    )[:, ::-1]

    top1 = sorted_X[
        :, 0
    ]

    top3 = np.sum(
        sorted_X[:, :3],
        axis=1
    )

    top5 = np.sum(
        sorted_X[:, :5],
        axis=1
    )

    top10 = np.sum(
        sorted_X[:, :10],
        axis=1
    )

    features = np.column_stack([
        shannon,
        simpson,
        richness,
        evenness,
        dominance,
        top1,
        top3,
        top5,
        top10
    ])

    names = [
        "shannon",
        "simpson",
        "richness",
        "evenness",
        "dominance",
        "top1_abundance",
        "top3_abundance",
        "top5_abundance",
        "top10_abundance"
    ]

    return (
        features,
        names
    )


# =============================================================================
# TAXA SELECTION
# =============================================================================

def select_taxa(
    train_counts,
    y_train
):

    """
    Select informative taxa using ONLY the training fold.

    Selection mechanisms:

        1. Prevalence filtering
        2. Variance ranking
        3. Absolute correlation with PMI

    PMI is NOT included as a final feature.

    It is only used for selecting taxa inside the training fold.
    """

    X = np.asarray(
        train_counts,
        dtype=np.float64
    )

    y = np.asarray(
        y_train,
        dtype=np.float64
    )

    # -------------------------------------------------------------------------
    # Prevalence
    # -------------------------------------------------------------------------

    prevalence = np.mean(
        X > 0,
        axis=0
    )

    valid = (
        prevalence >=
        MIN_PREVALENCE
    )

    if np.sum(valid) == 0:

        valid[:] = True

    valid_indices = np.where(
        valid
    )[0]

    X_filtered = X[
        :,
        valid
    ]

    # -------------------------------------------------------------------------
    # Relative abundance and CLR
    # -------------------------------------------------------------------------

    relative = relative_abundance(
        X_filtered
    )

    clr = clr_transform(
        relative
    )

    # -------------------------------------------------------------------------
    # Variance
    # -------------------------------------------------------------------------

    variance = np.var(
        clr,
        axis=0
    )

    n_variance = min(
        N_VARIANCE_TAXA,
        len(variance)
    )

    variance_indices = np.argsort(
        variance
    )[::-1][:n_variance]

    # -------------------------------------------------------------------------
    # PMI correlation
    # -------------------------------------------------------------------------

    y_centered = (
        y -
        np.mean(y)
    )

    X_centered = (
        clr -
        np.mean(
            clr,
            axis=0,
            keepdims=True
        )
    )

    numerator = np.sum(
        X_centered *
        y_centered[:, None],
        axis=0
    )

    x_norm = np.sqrt(
        np.sum(
            X_centered ** 2,
            axis=0
        )
    )

    y_norm = np.sqrt(
        np.sum(
            y_centered ** 2
        )
    )

    denominator = (
        x_norm *
        y_norm +
        1e-12
    )

    correlations = np.abs(
        numerator /
        denominator
    )

    correlations[
        ~np.isfinite(
            correlations
        )
    ] = 0.0

    n_supervised = min(
        N_SUPERVISED_TAXA,
        len(correlations)
    )

    supervised_indices = np.argsort(
        correlations
    )[::-1][:n_supervised]

    # -------------------------------------------------------------------------
    # Combine
    # -------------------------------------------------------------------------

    selected_local = np.unique(
        np.concatenate([
            variance_indices,
            supervised_indices
        ])
    )

    # -------------------------------------------------------------------------
    # If too many, rank using both signals
    # -------------------------------------------------------------------------

    if len(selected_local) > MAX_FINAL_TAXA:

        selected_variance = (
            variance[
                selected_local
            ]
        )

        max_variance = (
            np.max(
                selected_variance
            ) + 1e-12
        )

        combined_score = (
            correlations[
                selected_local
            ]
            +
            selected_variance /
            max_variance
        )

        ranking = np.argsort(
            combined_score
        )[::-1]

        selected_local = (
            selected_local[
                ranking[
                    :MAX_FINAL_TAXA
                ]
            ]
        )

    selected_local = np.sort(
        selected_local
    )

    selected_global = (
        valid_indices[
            selected_local
        ]
    )

    return selected_global


# =============================================================================
# ENVIRONMENTAL FEATURES
# =============================================================================

def build_environment_features(
    train_meta,
    val_meta
):

    """
    Build environmental features from the ACTUAL metadata.

    Uses:

        indoor_add
        outdoor_add
        indoor_outdoor
        body_site

    No PMI-derived features are created.
    """

    # =========================================================================
    # ADD FEATURES
    # =========================================================================

    def numeric_column(
        dataframe,
        column
    ):

        if column in dataframe.columns:

            return pd.to_numeric(
                dataframe[column],
                errors="coerce"
            )

        return pd.Series(
            np.nan,
            index=dataframe.index
        )

    train_indoor = numeric_column(
        train_meta,
        "indoor_add"
    )

    val_indoor = numeric_column(
        val_meta,
        "indoor_add"
    )

    train_outdoor = numeric_column(
        train_meta,
        "outdoor_add"
    )

    val_outdoor = numeric_column(
        val_meta,
        "outdoor_add"
    )

    # -------------------------------------------------------------------------
    # Training-fold medians
    # -------------------------------------------------------------------------

    indoor_median = train_indoor.median()

    outdoor_median = train_outdoor.median()

    if not np.isfinite(
        indoor_median
    ):
        indoor_median = 0.0

    if not np.isfinite(
        outdoor_median
    ):
        outdoor_median = 0.0

    train_indoor = (
        train_indoor
        .fillna(
            indoor_median
        )
        .values
    )

    val_indoor = (
        val_indoor
        .fillna(
            indoor_median
        )
        .values
    )

    train_outdoor = (
        train_outdoor
        .fillna(
            outdoor_median
        )
        .values
    )

    val_outdoor = (
        val_outdoor
        .fillna(
            outdoor_median
        )
        .values
    )

    # -------------------------------------------------------------------------
    # Center ADD using training statistics
    # -------------------------------------------------------------------------

    indoor_center = np.mean(
        train_indoor
    )

    outdoor_center = np.mean(
        train_outdoor
    )

    train_indoor_c = (
        train_indoor -
        indoor_center
    )

    val_indoor_c = (
        val_indoor -
        indoor_center
    )

    train_outdoor_c = (
        train_outdoor -
        outdoor_center
    )

    val_outdoor_c = (
        val_outdoor -
        outdoor_center
    )

    # -------------------------------------------------------------------------
    # ADD difference
    # -------------------------------------------------------------------------

    train_add_difference = (
        train_indoor -
        train_outdoor
    )

    val_add_difference = (
        val_indoor -
        val_outdoor
    )

    # -------------------------------------------------------------------------
    # ADD ratio
    # -------------------------------------------------------------------------

    train_add_ratio = (
        train_indoor /
        (
            np.abs(
                train_outdoor
            ) + 1.0
        )
    )

    val_add_ratio = (
        val_indoor /
        (
            np.abs(
                val_outdoor
            ) + 1.0
        )
    )

    # -------------------------------------------------------------------------
    # Log ADD
    # -------------------------------------------------------------------------

    train_indoor_log = np.log1p(
        np.maximum(
            train_indoor,
            0.0
        )
    )

    val_indoor_log = np.log1p(
        np.maximum(
            val_indoor,
            0.0
        )
    )

    train_outdoor_log = np.log1p(
        np.maximum(
            train_outdoor,
            0.0
        )
    )

    val_outdoor_log = np.log1p(
        np.maximum(
            val_outdoor,
            0.0
        )
    )

    # -------------------------------------------------------------------------
    # ADD feature matrix
    # -------------------------------------------------------------------------

    train_env = np.column_stack([
        train_indoor_c,
        val_dummy_safe(
            train_indoor_c
        ),
        train_outdoor_c,
        train_add_difference,
        train_add_ratio,
        train_indoor_log,
        train_outdoor_log
    ])

    val_env = np.column_stack([
        val_indoor_c,
        val_dummy_safe(
            val_indoor_c
        ),
        val_outdoor_c,
        val_add_difference,
        val_add_ratio,
        val_indoor_log,
        val_outdoor_log
    ])

    env_names = [
        "indoor_ADD_centered",
        "indoor_ADD_squared",
        "outdoor_ADD_centered",
        "ADD_indoor_minus_outdoor",
        "ADD_indoor_outdoor_ratio",
        "indoor_ADD_log",
        "outdoor_ADD_log"
    ]

    # =========================================================================
    # INDOOR / OUTDOOR CATEGORY
    # =========================================================================

    if "indoor_outdoor" in train_meta.columns:

        train_category = (
            train_meta[
                "indoor_outdoor"
            ]
            .fillna("unknown")
            .astype(str)
        )

        val_category = (
            val_meta[
                "indoor_outdoor"
            ]
            .fillna("unknown")
            .astype(str)
        )

        train_dummies = pd.get_dummies(
            train_category
        )

        categories = (
            train_dummies.columns
        )

        val_dummies = pd.get_dummies(
            val_category
        ).reindex(
            columns=categories,
            fill_value=0
        )

        train_cat = (
            train_dummies
            .values
            .astype(np.float64)
        )

        val_cat = (
            val_dummies
            .values
            .astype(np.float64)
        )

        train_env = np.hstack([
            train_env,
            train_cat
        ])

        val_env = np.hstack([
            val_env,
            val_cat
        ])

        env_names.extend([
            f"environment_{c}"
            for c in categories
        ])

    # =========================================================================
    # BODY SITE
    # =========================================================================

    site_column = None

    if "body_site" in train_meta.columns:

        site_column = "body_site"

    elif "host_body_site" in train_meta.columns:

        site_column = "host_body_site"

    if site_column is not None:

        train_sites = (
            train_meta[
                site_column
            ]
            .fillna("unknown")
            .astype(str)
        )

        val_sites = (
            val_meta[
                site_column
            ]
            .fillna("unknown")
            .astype(str)
        )

        train_site_dummies = pd.get_dummies(
            train_sites
        )

        site_categories = (
            train_site_dummies.columns
        )

        val_site_dummies = (
            pd.get_dummies(
                val_sites
            )
            .reindex(
                columns=site_categories,
                fill_value=0
            )
        )

        train_site = (
            train_site_dummies
            .values
            .astype(np.float64)
        )

        val_site = (
            val_site_dummies
            .values
            .astype(np.float64)
        )

        train_env = np.hstack([
            train_env,
            train_site
        ])

        val_env = np.hstack([
            val_env,
            val_site
        ])

        env_names.extend([
            f"body_site_{c}"
            for c in site_categories
        ])

    return (
        train_env,
        val_env,
        env_names
    )


def val_dummy_safe(
    x
):

    """
    Adds the squared centered ADD feature.

    Kept as a separate function so the feature construction is explicit.
    """

    return x ** 2


# =============================================================================
# COMPLETE FEATURE ENGINEERING
# =============================================================================

def make_features(
    train_meta,
    val_meta,
    train_otu,
    val_otu,
    y_train
):

    train_counts = (
        train_otu
        .values
        .astype(np.float64)
    )

    val_counts = (
        val_otu
        .values
        .astype(np.float64)
    )

    # =========================================================================
    # TAXA SELECTION
    # =========================================================================

    selected_taxa = select_taxa(
        train_counts,
        y_train
    )

    train_selected = (
        train_counts[
            :,
            selected_taxa
        ]
    )

    val_selected = (
        val_counts[
            :,
            selected_taxa
        ]
    )

    # =========================================================================
    # RELATIVE ABUNDANCE
    # =========================================================================

    train_relative = relative_abundance(
        train_selected
    )

    val_relative = relative_abundance(
        val_selected
    )

    # =========================================================================
    # CLR
    # =========================================================================

    train_clr = clr_transform(
        train_relative
    )

    val_clr = clr_transform(
        val_relative
    )

    # =========================================================================
    # LOG RELATIVE ABUNDANCE
    # =========================================================================

    train_log = log_relative_abundance(
        train_relative
    )

    val_log = log_relative_abundance(
        val_relative
    )

    # =========================================================================
    # DIVERSITY
    # =========================================================================

    train_diversity, diversity_names = (
        compute_diversity_features(
            train_relative
        )
    )

    val_diversity, _ = (
        compute_diversity_features(
            val_relative
        )
    )

    # =========================================================================
    # ENVIRONMENT
    # =========================================================================

    train_environment, val_environment, environment_names = (
        build_environment_features(
            train_meta,
            val_meta
        )
    )

    # =========================================================================
    # COMBINE
    # =========================================================================

    X_train = np.hstack([
        train_clr,
        train_log,
        train_diversity,
        train_environment
    ])

    X_val = np.hstack([
        val_clr,
        val_log,
        val_diversity,
        val_environment
    ])

    feature_names = []

    feature_names.extend([
        f"CLR_taxon_{i}"
        for i in selected_taxa
    ])

    feature_names.extend([
        f"LOG_taxon_{i}"
        for i in selected_taxa
    ])

    feature_names.extend(
        diversity_names
    )

    feature_names.extend(
        environment_names
    )

    # =========================================================================
    # SAFETY
    # =========================================================================

    X_train = np.nan_to_num(
        X_train,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    X_val = np.nan_to_num(
        X_val,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return (
        X_train,
        X_val,
        feature_names,
        selected_taxa
    )


# =============================================================================
# MODELS
# =============================================================================

def create_models():

    models = {}

    # =========================================================================
    # XGBOOST
    # =========================================================================

    if HAS_XGB:

        models["xgb_direct"] = xgb.XGBRegressor(

            n_estimators=1600,

            learning_rate=0.015,

            max_depth=4,

            min_child_weight=2,

            subsample=0.90,

            colsample_bytree=0.65,

            gamma=0.0,

            reg_alpha=0.05,

            reg_lambda=3.0,

            objective="reg:squarederror",

            eval_metric="mae",

            random_state=SEED,

            n_jobs=-1
        )

        models["xgb_log"] = xgb.XGBRegressor(

            n_estimators=1600,

            learning_rate=0.015,

            max_depth=4,

            min_child_weight=2,

            subsample=0.90,

            colsample_bytree=0.65,

            gamma=0.0,

            reg_alpha=0.05,

            reg_lambda=3.0,

            objective="reg:squarederror",

            eval_metric="mae",

            random_state=SEED + 100,

            n_jobs=-1
        )

    # =========================================================================
    # EXTRA TREES
    # =========================================================================

    models["extra_trees"] = ExtraTreesRegressor(

        n_estimators=800,

        max_depth=None,

        min_samples_leaf=2,

        max_features=0.55,

        bootstrap=False,

        n_jobs=-1,

        random_state=SEED
    )

    # =========================================================================
    # RANDOM FOREST
    # =========================================================================

    models["random_forest"] = RandomForestRegressor(

        n_estimators=700,

        max_depth=20,

        min_samples_leaf=2,

        max_features=0.45,

        bootstrap=True,

        n_jobs=-1,

        random_state=SEED
    )

    return models


# =============================================================================
# MODEL TRAINING
# =============================================================================

def fit_predict_model(
    name,
    model,
    X_train,
    y_train,
    X_val
):

    # =========================================================================
    # LOG XGBOOST
    # =========================================================================

    if name == "xgb_log":

        y_log = np.log1p(
            np.maximum(
                y_train,
                0
            )
        )

        model.fit(
            X_train,
            y_log
        )

        prediction_log = model.predict(
            X_val
        )

        prediction = np.expm1(
            prediction_log
        )

    # =========================================================================
    # NORMAL MODELS
    # =========================================================================

    else:

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(
            X_val
        )

    prediction = np.asarray(
        prediction,
        dtype=float
    )

    prediction = np.clip(
        prediction,
        0.0,
        None
    )

    return prediction


# =============================================================================
# PLS
# =============================================================================

def fit_predict_pls(
    X_train,
    y_train,
    X_val
):

    n_components = min(
        20,
        X_train.shape[1],
        max(
            2,
            X_train.shape[0] - 1
        )
    )

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "pls",
            PLSRegression(
                n_components=n_components,
                scale=False,
                max_iter=1000
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    prediction = (
        model
        .predict(
            X_val
        )
        .ravel()
    )

    prediction = np.clip(
        prediction,
        0.0,
        None
    )

    return prediction


# =============================================================================
# OUTER DONOR-GROUPED CROSS VALIDATION
# =============================================================================

def train_and_evaluate(
    meta,
    otu
):

    print_header(
        "STRICT 5-FOLD DONOR-GROUPED CROSS-VALIDATION"
    )

    y = (
        meta["pmi"]
        .values
        .astype(float)
    )

    groups = (
        meta["subject_id"]
        .values
    )

    n_samples = len(meta)

    n_donors = (
        meta["subject_id"]
        .nunique()
    )

    print(
        f"Samples : {n_samples}"
    )

    print(
        f"Donors  : {n_donors}"
    )

    print(
        f"Folds   : {N_FOLDS}"
    )

    # =========================================================================
    # OUTER GROUP K-FOLD
    # =========================================================================

    gkf = GroupKFold(
        n_splits=N_FOLDS
    )

    final_predictions = np.zeros(
        n_samples
    )

    all_predictions = []

    fold_results = []

    # =========================================================================
    # FOLDS
    # =========================================================================

    for fold, (
        train_idx,
        val_idx
    ) in enumerate(
        gkf.split(
            otu,
            y,
            groups
        ),
        1
    ):

        print_header(
            f"FOLD {fold}/{N_FOLDS}"
        )

        train_meta = (
            meta
            .iloc[train_idx]
            .copy()
        )

        val_meta = (
            meta
            .iloc[val_idx]
            .copy()
        )

        train_otu = (
            otu
            .iloc[train_idx]
            .copy()
        )

        val_otu = (
            otu
            .iloc[val_idx]
            .copy()
        )

        y_train = y[
            train_idx
        ]

        y_val = y[
            val_idx
        ]

        print(
            f"Training samples : "
            f"{len(train_idx)}"
        )

        print(
            f"Validation samples : "
            f"{len(val_idx)}"
        )

        print(
            f"Training donors : "
            f"{train_meta['subject_id'].nunique()}"
        )

        print(
            f"Validation donors : "
            f"{val_meta['subject_id'].nunique()}"
        )

        # =========================================================================
        # FEATURES
        # =========================================================================

        print(
            "\nBuilding leakage-free features..."
        )

        (
            X_train,
            X_val,
            feature_names,
            selected_taxa
        ) = make_features(
            train_meta,
            val_meta,
            train_otu,
            val_otu,
            y_train
        )

        print(
            f"Selected taxa : "
            f"{len(selected_taxa)}"
        )

        print(
            f"Total features : "
            f"{X_train.shape[1]}"
        )

        # =========================================================================
        # TRAIN BASE MODELS
        # =========================================================================

        models = create_models()

        validation_predictions = {}

        # -------------------------------------------------------------------------
        # Tree models
        # -------------------------------------------------------------------------

        for name, model in models.items():

            print(
                f"\nTraining {name}..."
            )

            prediction = fit_predict_model(
                name,
                model,
                X_train,
                y_train,
                X_val
            )

            validation_predictions[
                name
            ] = prediction

            model_r2 = safe_r2(
                y_val,
                prediction
            )

            model_mae = compute_mae(
                y_val,
                prediction
            )

            print(
                f"  R²  = "
                f"{model_r2:.4f}"
            )

            print(
                f"  MAE = "
                f"{model_mae:.3f} days"
            )

        # -------------------------------------------------------------------------
        # PLS
        # -------------------------------------------------------------------------

        print(
            "\nTraining PLS..."
        )

        pls_prediction = fit_predict_pls(
            X_train,
            y_train,
            X_val
        )

        validation_predictions[
            "pls"
        ] = pls_prediction

        pls_r2 = safe_r2(
            y_val,
            pls_prediction
        )

        pls_mae = compute_mae(
            y_val,
            pls_prediction
        )

        print(
            f"  R²  = "
            f"{pls_r2:.4f}"
        )

        print(
            f"  MAE = "
            f"{pls_mae:.3f} days"
        )

        # =========================================================================
        # ENSEMBLE
        # =========================================================================

        print(
            "\nCreating ensemble..."
        )

        prediction_matrix = np.column_stack([
            validation_predictions[name]
            for name in validation_predictions
        ])

        model_names = list(
            validation_predictions.keys()
        )

        print(
            "\nModels included:"
        )

        for name in model_names:

            print(
                f"  - {name}"
            )

        # =========================================================================
        # ROBUST ENSEMBLE
        #
        # XGBoost models receive more weight because they generally work well
        # for nonlinear high-dimensional microbiome data.
        #
        # ExtraTrees / RF provide complementary tree structures.
        #
        # PLS receives a smaller weight because it is useful for linear
        # high-dimensional structure but can underperform nonlinear models.
        # =========================================================================

        raw_weights = {}

        for name in model_names:

            if name == "xgb_direct":

                raw_weights[name] = 0.30

            elif name == "xgb_log":

                raw_weights[name] = 0.25

            elif name == "extra_trees":

                raw_weights[name] = 0.20

            elif name == "random_forest":

                raw_weights[name] = 0.15

            elif name == "pls":

                raw_weights[name] = 0.10

            else:

                raw_weights[name] = 0.10

        # Normalize only over models actually available
        weight_sum = sum(
            raw_weights[name]
            for name in model_names
        )

        weights = np.array([
            raw_weights[name] /
            weight_sum
            for name in model_names
        ])

        print(
            "\nEnsemble weights:"
        )

        for name, weight in zip(
            model_names,
            weights
        ):

            print(
                f"  {name:<20} "
                f"{weight:.3f}"
            )

        # =========================================================================
        # FINAL PREDICTION
        # =========================================================================

        final_prediction = (
            prediction_matrix *
            weights
        ).sum(
            axis=1
        )

        final_prediction = np.clip(
            final_prediction,
            0.0,
            None
        )

        final_predictions[
            val_idx
        ] = final_prediction

        # =========================================================================
        # FOLD RESULT
        # =========================================================================

        fold_r2 = safe_r2(
            y_val,
            final_prediction
        )

        fold_mae = compute_mae(
            y_val,
            final_prediction
        )

        print(
            "\n"
            + "-" * 65
        )

        print(
            f"FOLD {fold} ENSEMBLE RESULT"
        )

        print(
            f"R²  : "
            f"{fold_r2:.4f}"
        )

        print(
            f"MAE : "
            f"{fold_mae:.3f} days"
        )

        print(
            f"MAE : "
            f"{fold_mae * 24:.1f} hours"
        )

        print(
            "-" * 65
        )

        fold_results.append({

            "fold": fold,

            "n_train": len(train_idx),

            "n_validation": len(val_idx),

            "train_donors":
                train_meta[
                    "subject_id"
                ].nunique(),

            "validation_donors":
                val_meta[
                    "subject_id"
                ].nunique(),

            "selected_taxa":
                len(selected_taxa),

            "ensemble_r2":
                float(fold_r2),

            "ensemble_mae_days":
                float(fold_mae),

            "ensemble_mae_hours":
                float(
                    fold_mae * 24
                )
        })

        # -------------------------------------------------------------------------
        # Save fold predictions
        # -------------------------------------------------------------------------

        fold_prediction_df = pd.DataFrame({

            "sample_id":
                val_meta[
                    "sample_id"
                ].values,

            "subject_id":
                val_meta[
                    "subject_id"
                ].values,

            "actual_pmi":
                y_val,

            "ensemble_prediction":
                final_prediction
        })

        for name in model_names:

            fold_prediction_df[
                f"pred_{name}"
            ] = validation_predictions[
                name
            ]

        all_predictions.append(
            fold_prediction_df
        )

    # =========================================================================
    # FINAL EVALUATION
    # =========================================================================

    print_header(
        "FINAL MICROBIOME SUCCESSION RESULTS"
    )

    sample_r2 = safe_r2(
        y,
        final_predictions
    )

    sample_mae = compute_mae(
        y,
        final_predictions
    )

    # =========================================================================
    # BASELINE
    # =========================================================================

    baseline_prediction = np.full(
        n_samples,
        np.mean(y)
    )

    baseline_mae = compute_mae(
        y,
        baseline_prediction
    )

    improvement = (
        1.0 -
        sample_mae /
        baseline_mae
    ) * 100.0

    print(
        f"Evaluation Level        : "
        f"Sample-Level / Longitudinal"
    )

    print(
        f"Total Samples           : "
        f"{n_samples}"
    )

    print(
        f"Total Unique Donors     : "
        f"{n_donors}"
    )

    print(
        f"Sample-Level R²         : "
        f"{sample_r2:.4f}"
    )

    print(
        f"Sample-Level MAE        : "
        f"{sample_mae:.3f} days"
    )

    print(
        f"Sample-Level MAE        : "
        f"{sample_mae * 24:.1f} hours"
    )

    print(
        f"Baseline MAE            : "
        f"{baseline_mae:.3f} days"
    )

    print(
        f"Improvement over Baseline: "
        f"{improvement:.1f}%"
    )

    # =========================================================================
    # DONOR-LEVEL DIAGNOSTIC
    # =========================================================================

    complete_prediction_df = pd.DataFrame({

        "sample_id":
            meta[
                "sample_id"
            ].values,

        "subject_id":
            meta[
                "subject_id"
            ].values,

        "actual_pmi":
            y,

        "predicted_pmi":
            final_predictions,

        "absolute_error":
            np.abs(
                y -
                final_predictions
            )
    })

    donor_predictions = (
        complete_prediction_df
        .groupby(
            "subject_id"
        )
        .agg({
            "actual_pmi": "mean",
            "predicted_pmi": "mean"
        })
        .reset_index()
    )

    donor_r2 = safe_r2(
        donor_predictions[
            "actual_pmi"
        ].values,

        donor_predictions[
            "predicted_pmi"
        ].values
    )

    donor_mae = compute_mae(
        donor_predictions[
            "actual_pmi"
        ].values,

        donor_predictions[
            "predicted_pmi"
        ].values
    )

    print(
        "\nDONOR-LEVEL DIAGNOSTIC"
    )

    print(
        f"Donor-level R² : "
        f"{donor_r2:.4f}"
    )

    print(
        f"Donor-level MAE: "
        f"{donor_mae:.3f} days"
    )

    # =========================================================================
    # FOLD RESULTS
    # =========================================================================

    fold_df = pd.DataFrame(
        fold_results
    )

    print(
        "\nFOLD RESULTS"
    )

    print(
        fold_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    os.makedirs(
        "results",
        exist_ok=True
    )

    os.makedirs(
        "reports",
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    # -------------------------------------------------------------------------
    # All predictions
    # -------------------------------------------------------------------------

    all_predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    predictions_path = (
        "results/"
        f"microbial_predictions_"
        f"{timestamp}.csv"
    )

    all_predictions_df.to_csv(
        predictions_path,
        index=False
    )

    print(
        f"\nPredictions saved -> "
        f"{predictions_path}"
    )

    # -------------------------------------------------------------------------
    # Fold results
    # -------------------------------------------------------------------------

    fold_path = (
        "results/"
        f"microbial_fold_results_"
        f"{timestamp}.csv"
    )

    fold_df.to_csv(
        fold_path,
        index=False
    )

    print(
        f"Fold results saved -> "
        f"{fold_path}"
    )

    # -------------------------------------------------------------------------
    # Summary JSON
    # -------------------------------------------------------------------------

    summary = {

        "project":
            "ForensicChrono",

        "module":
            "Microbial Succession PMI Model",

        "dataset":
            "Qiita 13810",

        "timestamp":
            timestamp,

        "samples":
            int(n_samples),

        "donors":
            int(n_donors),

        "sample_r2":
            float(sample_r2),

        "sample_mae_days":
            float(sample_mae),

        "sample_mae_hours":
            float(
                sample_mae * 24
            ),

        "baseline_mae_days":
            float(
                baseline_mae
            ),

        "improvement_percent":
            float(
                improvement
            ),

        "donor_level_r2":
            float(
                donor_r2
            ),

        "donor_level_mae_days":
            float(
                donor_mae
            ),

        "cv":
            "5-fold donor-grouped",

        "microbiome_features": [

            "relative_abundance",

            "CLR",

            "log_relative_abundance",

            "prevalence_filtering",

            "variance_taxa_selection",

            "PMI_associated_taxa_selection_in_training_fold",

            "Shannon",

            "Simpson",

            "richness",

            "evenness",

            "dominance",

            "top_abundance_features"
        ],

        "environmental_features": [

            "indoor_add",

            "outdoor_add",

            "indoor_ADD_squared",

            "ADD_indoor_minus_outdoor",

            "ADD_indoor_outdoor_ratio",

            "indoor_ADD_log",

            "outdoor_ADD_log",

            "indoor_outdoor"
        ],

        "anatomical_features": [

            "body_site"
        ],

        "models": [

            "XGBoost_direct",

            "XGBoost_log",

            "ExtraTrees",

            "RandomForest",

            "PLS",

            "weighted_ensemble"
        ],

        "target_leakage": False,

        "pmi_as_input_feature": False,

        "pmi_times_temperature": False,

        "pmi_times_add": False
    }

    summary_path = (
        "results/"
        f"microbial_summary_"
        f"{timestamp}.json"
    )

    with open(
        summary_path,
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    print(
        f"Summary saved -> "
        f"{summary_path}"
    )

    # =========================================================================
    # HUMAN READABLE REPORT
    # =========================================================================

    report_path = (
        "reports/"
        f"microbial_model_report_"
        f"{timestamp}.txt"
    )

    with open(
        report_path,
        "w"
    ) as f:

        f.write(
            "FORENSICCHRONO\n"
        )

        f.write(
            "MICROBIAL SUCCESSION PMI MODEL\n"
        )

        f.write(
            "=" * 65 +
            "\n\n"
        )

        f.write(
            f"Samples: {n_samples}\n"
        )

        f.write(
            f"Donors: {n_donors}\n"
        )

        f.write(
            f"Sample R2: {sample_r2:.4f}\n"
        )

        f.write(
            f"Sample MAE: "
            f"{sample_mae:.4f} days\n"
        )

        f.write(
            f"Sample MAE hours: "
            f"{sample_mae * 24:.2f}\n"
        )

        f.write(
            f"Baseline MAE: "
            f"{baseline_mae:.4f} days\n"
        )

        f.write(
            f"Improvement: "
            f"{improvement:.2f}%\n"
        )

        f.write(
            f"Donor R2: "
            f"{donor_r2:.4f}\n"
        )

        f.write(
            f"Donor MAE: "
            f"{donor_mae:.4f} days\n"
        )

        f.write(
            "\nEnvironmental variables used:\n"
        )

        f.write(
            "  indoor_add\n"
        )

        f.write(
            "  outdoor_add\n"
        )

        f.write(
            "  indoor_outdoor\n"
        )

        f.write(
            "  body_site\n"
        )

        f.write(
            "\nNo PMI-derived input features were used.\n"
        )

        f.write(
            "Evaluation uses donor-grouped CV.\n"
        )

    print(
        f"Report saved -> "
        f"{report_path}"
    )

    # =========================================================================
    # FINAL INTERPRETATION
    # =========================================================================

    print_header(
        "FINAL INTERPRETATION"
    )

    if sample_r2 >= 0.90:

        print(
            "R² >= 0.90"
        )

        print(
            "Very strong result under strict donor-grouped evaluation."
        )

    elif sample_r2 >= 0.80:

        print(
            "R² >= 0.80"
        )

        print(
            "Strong result under strict donor-grouped evaluation."
        )

    elif sample_r2 >= 0.60:

        print(
            "R² >= 0.60"
        )

        print(
            "Substantial predictive signal."
        )

    elif sample_r2 >= 0.40:

        print(
            "R² >= 0.40"
        )

        print(
            "Moderate predictive signal."
        )

    else:

        print(
            "R² < 0.40"
        )

        print(
            "Limited predictive signal under unseen-donor evaluation."
        )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Do NOT use PMI * temperature or PMI * ADD."
    )

    print(
        "The ADD variables are used as environmental information."
    )

    print(
        "PMI is never supplied directly to the prediction model."
    )

    print(
        "\nMicrobial model training complete."
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "\n"
        + "=" * 78
    )

    print(
        "FORENSICCHRONO"
    )

    print(
        "MICROBIAL SUCCESSION MODEL V3"
    )

    print(
        "=" * 78
    )

    print(
        "\nEnvironmental variables:"
    )

    print(
        "  ✓ indoor_add"
    )

    print(
        "  ✓ outdoor_add"
    )

    print(
        "  ✓ indoor_outdoor"
    )

    print(
        "  ✓ body_site"
    )

    print(
        "\nMicrobiome:"
    )

    print(
        "  ✓ CLR"
    )

    print(
        "  ✓ Log relative abundance"
    )

    print(
        "  ✓ Diversity"
    )

    print(
        "  ✓ In-fold taxa selection"
    )

    print(
        "\nLeakage protection:"
    )

    print(
        "  ✓ Donor GroupKFold"
    )

    print(
        "  ✓ No PMI × ADD"
    )

    print(
        "  ✓ No PMI × temperature"
    )

    print(
        "  ✓ No PMI as model feature"
    )

    # =========================================================================
    # Check dependencies
    # =========================================================================

    if not HAS_XGB:

        print(
            "\nWARNING: XGBoost is not installed."
        )

        print(
            "Install using:"
        )

        print(
            "pip install xgboost"
        )

    # =========================================================================
    # Load
    # =========================================================================

    meta, otu = load_data()

    # =========================================================================
    # Train
    # =========================================================================

    train_and_evaluate(
        meta,
        otu
    )


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    main()