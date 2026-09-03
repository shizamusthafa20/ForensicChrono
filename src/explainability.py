"""
===============================================================
FORENSICCHRONO — SHAP EXPLAINABILITY
===============================================================

Purpose:
    Explain the trained ForensicChrono RNA and Microbiome models
    using SHAP.

IMPORTANT:
    - Does NOT retrain models.
    - Loads existing .pkl files.
    - RNA:
        Explains the XGBoost direct PMI base model
        for each tissue.
    - Microbiome:
        Explains the XGBoost direct PMI base model.
    - The complete RNA model is actually:
        6 base models + Ridge stacking.
      Therefore SHAP is reported specifically for the
      XGBoost direct component, not the entire stacked model.

Outputs:
    reports/
        shap/
            RNA/
                muscle_skeletal/
                lung/
                skin_sun_exposed_lower_leg/
                nerve_tibial/
            Microbiome/

        Each model gets:
            - SHAP bar plot
            - SHAP beeswarm plot
            - CSV containing mean absolute SHAP importance

Run from project root:

    python src/explainability.py
===============================================================
"""

import os
import sys
import pickle
import warnings

import numpy as np
import pandas as pd
import shap
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SRC_DIR = os.path.join(
    PROJECT_ROOT,
    "src"
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Make relative paths used by the original model files work
os.chdir(PROJECT_ROOT)


# ============================================================
# PATHS
# ============================================================

RNA_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "rna_model.pkl"
)

MICROBIAL_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "microbial_model.pkl"
)

REPORTS_DIR = os.path.join(
    PROJECT_ROOT,
    "reports",
    "shap"
)

RNA_REPORT_DIR = os.path.join(
    REPORTS_DIR,
    "RNA"
)

MICROBIAL_REPORT_DIR = os.path.join(
    REPORTS_DIR,
    "Microbiome"
)


# ============================================================
# SHAP CONFIGURATION
# ============================================================

# We do NOT need to explain every sample.
# A representative subset is enough for the report.
RNA_MAX_SAMPLES = 250
MICROBIAL_MAX_SAMPLES = 300

# Number of features displayed in the plots.
MAX_DISPLAY = 20

RANDOM_SEED = 42

warnings.filterwarnings(
    "ignore"
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_name(text):
    """
    Convert a tissue/model name into a safe folder name.
    """
    return (
        str(text)
        .lower()
        .replace(" - ", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def make_output_dir(path):
    """
    Create output directory if it does not exist.
    """
    os.makedirs(
        path,
        exist_ok=True
    )


def choose_sample_indices(
    n_samples,
    max_samples,
    seed=42
):
    """
    Select a reproducible random subset of samples.
    """
    rng = np.random.RandomState(seed)

    if n_samples <= max_samples:
        return np.arange(n_samples)

    return np.sort(
        rng.choice(
            n_samples,
            size=max_samples,
            replace=False
        )
    )


# ============================================================
# LOAD RNA MODEL
# ============================================================

def load_rna_model():
    """
    Load the already-trained RNA deployable model.
    """
    print("\nLoading RNA model...")

    if not os.path.exists(RNA_MODEL_PATH):
        raise FileNotFoundError(
            f"RNA model not found:\n{RNA_MODEL_PATH}"
        )

    with open(
        RNA_MODEL_PATH,
        "rb"
    ) as f:
        package = pickle.load(f)

    print("RNA model loaded successfully.")

    print(
        "Available tissues:",
        package.get("tissues", [])
    )

    return package


# ============================================================
# LOAD MICROBIOME MODEL
# ============================================================

def load_microbiome_model():
    """
    Load the already-trained microbiome deployable model.
    """
    print("\nLoading microbiome model...")

    if not os.path.exists(MICROBIAL_MODEL_PATH):
        raise FileNotFoundError(
            f"Microbiome model not found:\n"
            f"{MICROBIAL_MODEL_PATH}"
        )

    with open(
        MICROBIAL_MODEL_PATH,
        "rb"
    ) as f:
        package = pickle.load(f)

    print(
        "Microbiome model loaded successfully."
    )

    print(
        "Available models:",
        list(
            package.get(
                "fitted_models",
                {}
            ).keys()
        )
    )

    return package


# ============================================================
# RNA FEATURE RECONSTRUCTION
# ============================================================

def build_rna_features_from_saved_info(
    df,
    expr,
    feature_info
):
    """
    Reconstruct the EXACT feature representation expected by
    the saved RNA XGBoost model.

    The deployable RNA model stores:
        - selected_genes
        - degrader_indices
        - stable_indices
        - clinical_medians

    We use these saved values instead of performing new feature
    selection.

    This prevents the SHAP script from accidentally creating a
    different feature matrix.
    """

    print("  Reconstructing RNA features...")

    # --------------------------------------------------------
    # Saved feature information
    # --------------------------------------------------------

    selected_genes = feature_info[
        "selected_genes"
    ]

    degrader_indices = np.asarray(
        feature_info[
            "degrader_indices"
        ],
        dtype=int
    )

    stable_indices = np.asarray(
        feature_info[
            "stable_indices"
        ],
        dtype=int
    )

    clinical_medians = feature_info.get(
        "clinical_medians",
        {}
    )

    # --------------------------------------------------------
    # Make sure selected genes exist
    # --------------------------------------------------------

    missing_genes = [
        gene
        for gene in selected_genes
        if gene not in expr.columns
    ]

    if len(missing_genes) > 0:

        print(
            f"  WARNING: {len(missing_genes)} "
            f"selected genes are missing from expression data."
        )

        # Only use genes that are actually available
        selected_genes = [
            gene
            for gene in selected_genes
            if gene in expr.columns
        ]

    if len(selected_genes) == 0:
        raise ValueError(
            "None of the saved RNA genes were found "
            "in the expression matrix."
        )

    # --------------------------------------------------------
    # log2(expression + 1)
    # --------------------------------------------------------

    log_expr = np.log2(
        expr + 1.0
    )

    G = log_expr[
        selected_genes
    ].values

    # --------------------------------------------------------
    # Ratio features
    #
    # Same mathematical operation as training:
    #
    # degrader - stable
    # --------------------------------------------------------

    ratio_features = np.column_stack([
        G[:, d] - G[:, s]
        for d in degrader_indices
        for s in stable_indices
    ])

    # --------------------------------------------------------
    # Summary features
    # --------------------------------------------------------

    summary_features = np.column_stack([
        ratio_features.mean(axis=1),
        ratio_features.std(axis=1),
        ratio_features.min(axis=1),
        ratio_features.max(axis=1),
        G[:, degrader_indices].mean(axis=1),
        G[:, stable_indices].mean(axis=1)
    ])

    # --------------------------------------------------------
    # Clinical / QC features
    # --------------------------------------------------------

    numeric_cols = [
        "rin",
        "autolysis",
        "age",
        "sex",
        "hardy"
    ]

    clean = df.copy()

    for col in numeric_cols:

        median = clinical_medians.get(
            col,
            0.0
        )

        clean[col] = pd.to_numeric(
            clean[col],
            errors="coerce"
        )

        clean[col] = clean[col].fillna(
            median
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

    # --------------------------------------------------------
    # Final feature matrix
    # --------------------------------------------------------

    X = np.hstack([
        G,
        ratio_features,
        summary_features,
        clinical
    ])

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    feature_names = []

    # Gene features
    feature_names.extend(
        [
            f"Gene_{gene}"
            for gene in selected_genes
        ]
    )

    # Ratio features
    for d in degrader_indices:

        for s in stable_indices:

            feature_names.append(
                f"Degrader_{d}_minus_Stable_{s}"
            )

    # Summary features
    feature_names.extend([
        "Ratio_mean",
        "Ratio_std",
        "Ratio_min",
        "Ratio_max",
        "Degrader_mean",
        "Stable_mean"
    ])

    # Clinical features
    feature_names.extend([
        "RIN",
        "RIN_squared",
        "Autolysis",
        "Autolysis_squared",
        "Age",
        "Sex",
        "Hardy_death_scale",
        "RIN_x_Autolysis",
        "RIN_div_Autolysis_plus_1"
    ])

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if X.shape[1] != len(feature_names):

        raise ValueError(
            "RNA feature-name mismatch:\n"
            f"X has {X.shape[1]} columns\n"
            f"Feature names have {len(feature_names)}"
        )

    print(
        f"  RNA feature matrix: {X.shape}"
    )

    return X, feature_names


# ============================================================
# RNA SHAP EXPLANATION
# ============================================================

def explain_rna_tissue(
    package,
    tissue
):
    """
    Generate SHAP explanations for one RNA tissue.

    IMPORTANT:
        We explain only:
            xgb_direct

        This is one of the six base models.

        We are NOT claiming that this is a SHAP explanation
        of the entire Ridge stacking model.
    """

    print("\n" + "=" * 70)
    print(
        f"RNA SHAP — {tissue}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Import original RNA module
    # --------------------------------------------------------

    try:

        import rna_model

    except Exception as e:

        raise ImportError(
            "Could not import src/rna_model.py\n"
            f"Error: {e}"
        )

    # --------------------------------------------------------
    # Load original RNA data
    # --------------------------------------------------------

    print(
        "Loading GTEx data using the original RNA pipeline..."
    )

    sa, expr = rna_model.load_data()

    # --------------------------------------------------------
    # Select tissue
    # --------------------------------------------------------

    tissue_mask = (
        sa["tissue"] == tissue
    )

    tissue_df = sa.loc[
        tissue_mask
    ].reset_index(
        drop=True
    )

    tissue_expr = expr.loc[
        tissue_df["SAMPID"]
    ]

    print(
        f"Samples available: {len(tissue_df)}"
    )

    # --------------------------------------------------------
    # Retrieve saved tissue model
    # --------------------------------------------------------

    tissue_package = package[
        "base_models"
    ][tissue]

    base_models = tissue_package[
        "base_models"
    ]

    feature_info = tissue_package[
        "feature_info"
    ]

    if "xgb_direct" not in base_models:

        raise KeyError(
            f"xgb_direct not found for tissue: {tissue}"
        )

    xgb_model = base_models[
        "xgb_direct"
    ]

    print(
        "Using saved XGBoost direct model."
    )

    # --------------------------------------------------------
    # Reconstruct exact features
    # --------------------------------------------------------

    X, feature_names = (
        build_rna_features_from_saved_info(
            tissue_df,
            tissue_expr,
            feature_info
        )
    )

    # --------------------------------------------------------
    # Select representative samples
    # --------------------------------------------------------

    indices = choose_sample_indices(
        len(X),
        RNA_MAX_SAMPLES,
        RANDOM_SEED
    )

    X_sample = X[
        indices
    ]

    print(
        f"Samples used for SHAP: "
        f"{len(X_sample)}"
    )

    # --------------------------------------------------------
    # SHAP TreeExplainer
    # --------------------------------------------------------

    print(
        "Calculating SHAP values..."
    )

    explainer = shap.TreeExplainer(
        xgb_model
    )

    shap_values = explainer.shap_values(
        X_sample
    )

    shap_values = np.asarray(
        shap_values
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    tissue_dir = os.path.join(
        RNA_REPORT_DIR,
        safe_name(tissue)
    )

    make_output_dir(
        tissue_dir
    )

    # ========================================================
    # SHAP BAR PLOT
    # ========================================================

    print(
        "Saving SHAP bar plot..."
    )

    plt.figure(
        figsize=(10, 8)
    )

    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        plot_type="bar",
        max_display=MAX_DISPLAY,
        show=False
    )

    plt.title(
        f"RNA SHAP Feature Importance\n"
        f"{tissue}\n"
        f"XGBoost Direct PMI Model"
    )

    plt.tight_layout()

    bar_path = os.path.join(
        tissue_dir,
        "shap_feature_importance_bar.png"
    )

    plt.savefig(
        bar_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # SHAP BEESWARM PLOT
    # ========================================================

    print(
        "Saving SHAP beeswarm plot..."
    )

    plt.figure(
        figsize=(10, 8)
    )

    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        max_display=MAX_DISPLAY,
        show=False
    )

    plt.title(
        f"RNA SHAP Summary\n"
        f"{tissue}\n"
        f"XGBoost Direct PMI Model"
    )

    plt.tight_layout()

    beeswarm_path = os.path.join(
        tissue_dir,
        "shap_summary_beeswarm.png"
    )

    plt.savefig(
        beeswarm_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # SHAP IMPORTANCE CSV
    # ========================================================

    mean_abs_shap = np.mean(
        np.abs(shap_values),
        axis=0
    )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    })

    importance_df = (
        importance_df
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    csv_path = os.path.join(
        tissue_dir,
        "shap_feature_importance.csv"
    )

    importance_df.to_csv(
        csv_path,
        index=False
    )

    # --------------------------------------------------------
    # Print top features
    # --------------------------------------------------------

    print(
        "\nTop RNA SHAP features:"
    )

    print(
        importance_df.head(20).to_string(
            index=False
        )
    )

    print(
        f"\nSaved:"
        f"\n  {bar_path}"
        f"\n  {beeswarm_path}"
        f"\n  {csv_path}"
    )


# ============================================================
# MICROBIOME SHAP EXPLANATION
# ============================================================

def explain_microbiome(
    package
):
    """
    Generate SHAP explanation for the direct XGBoost
    microbiome model.

    The complete microbiome model is an ensemble.

    SHAP here explains specifically:
        xgb_direct
    """

    print("\n" + "=" * 70)
    print(
        "MICROBIOME SHAP"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Import original microbiome module
    # --------------------------------------------------------

    try:

        import microbial_model

    except Exception as e:

        raise ImportError(
            "Could not import src/microbial_model.py\n"
            f"Error: {e}"
        )

    # --------------------------------------------------------
    # Load microbiome data using original pipeline
    # --------------------------------------------------------

    print(
        "Loading microbiome data..."
    )

    meta, otu = (
        microbial_model.load_data()
    )

    print(
        f"Microbiome samples: {len(meta)}"
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y = meta[
        "pmi"
    ].values.astype(
        np.float64
    )

    # --------------------------------------------------------
    # Recreate exact full-data feature matrix
    #
    # This uses the same make_features() function as the
    # original model.
    # --------------------------------------------------------

    print(
        "Reconstructing microbiome features..."
    )

    X, _, feature_names, selected_taxa = (
        microbial_model.make_features(
            meta,
            meta,
            otu,
            otu,
            y
        )
    )

    X = np.asarray(
        X,
        dtype=np.float64
    )

    print(
        f"Microbiome feature matrix: {X.shape}"
    )

    # --------------------------------------------------------
    # Compare with saved package information
    # --------------------------------------------------------

    saved_feature_names = package.get(
        "feature_names",
        None
    )

    if saved_feature_names is not None:

        if len(saved_feature_names) != X.shape[1]:

            raise ValueError(
                "Microbiome feature count does not match "
                "the saved model.\n"
                f"Saved: {len(saved_feature_names)}\n"
                f"Current: {X.shape[1]}"
            )

        feature_names = list(
            saved_feature_names
        )

    # --------------------------------------------------------
    # Retrieve direct XGBoost model
    # --------------------------------------------------------

    fitted_models = package[
        "fitted_models"
    ]

    if "xgb_direct" not in fitted_models:

        raise KeyError(
            "xgb_direct was not found in "
            "microbial_model.pkl"
        )

    xgb_model = fitted_models[
        "xgb_direct"
    ]

    print(
        "Using saved microbiome XGBoost direct model."
    )

    # --------------------------------------------------------
    # Select representative samples
    # --------------------------------------------------------

    indices = choose_sample_indices(
        len(X),
        MICROBIAL_MAX_SAMPLES,
        RANDOM_SEED
    )

    X_sample = X[
        indices
    ]

    print(
        f"Samples used for SHAP: "
        f"{len(X_sample)}"
    )

    # --------------------------------------------------------
    # SHAP TreeExplainer
    # --------------------------------------------------------

    print(
        "Calculating microbiome SHAP values..."
    )

    explainer = shap.TreeExplainer(
        xgb_model
    )

    shap_values = explainer.shap_values(
        X_sample
    )

    shap_values = np.asarray(
        shap_values
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    make_output_dir(
        MICROBIAL_REPORT_DIR
    )

    # ========================================================
    # BAR PLOT
    # ========================================================

    print(
        "Saving microbiome SHAP bar plot..."
    )

    plt.figure(
        figsize=(10, 8)
    )

    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        plot_type="bar",
        max_display=MAX_DISPLAY,
        show=False
    )

    plt.title(
        "Microbiome SHAP Feature Importance\n"
        "XGBoost Direct PMI Model"
    )

    plt.tight_layout()

    bar_path = os.path.join(
        MICROBIAL_REPORT_DIR,
        "shap_feature_importance_bar.png"
    )

    plt.savefig(
        bar_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # BEESWARM PLOT
    # ========================================================

    print(
        "Saving microbiome SHAP beeswarm plot..."
    )

    plt.figure(
        figsize=(10, 8)
    )

    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        max_display=MAX_DISPLAY,
        show=False
    )

    plt.title(
        "Microbiome SHAP Summary\n"
        "XGBoost Direct PMI Model"
    )

    plt.tight_layout()

    beeswarm_path = os.path.join(
        MICROBIAL_REPORT_DIR,
        "shap_summary_beeswarm.png"
    )

    plt.savefig(
        beeswarm_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # CSV
    # ========================================================

    mean_abs_shap = np.mean(
        np.abs(shap_values),
        axis=0
    )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    })

    importance_df = (
        importance_df
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    csv_path = os.path.join(
        MICROBIAL_REPORT_DIR,
        "shap_feature_importance.csv"
    )

    importance_df.to_csv(
        csv_path,
        index=False
    )

    # --------------------------------------------------------
    # Print top features
    # --------------------------------------------------------

    print(
        "\nTop microbiome SHAP features:"
    )

    print(
        importance_df.head(20).to_string(
            index=False
        )
    )

    print(
        f"\nSaved:"
        f"\n  {bar_path}"
        f"\n  {beeswarm_path}"
        f"\n  {csv_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 78)
    print("FORENSICCHRONO — SHAP EXPLAINABILITY")
    print("=" * 78)
    print(
        "No model retraining."
    )
    print(
        "Existing .pkl models are used."
    )
    print("=" * 78)

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    make_output_dir(
        REPORTS_DIR
    )

    make_output_dir(
        RNA_REPORT_DIR
    )

    make_output_dir(
        MICROBIAL_REPORT_DIR
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    rna_package = load_rna_model()

    microbial_package = (
        load_microbiome_model()
    )

    # ========================================================
    # RNA SHAP
    # ========================================================

    print("\n")
    print("=" * 78)
    print("STARTING RNA SHAP")
    print("=" * 78)

    tissues = rna_package.get(
        "tissues",
        []
    )

    for tissue in tissues:

        try:

            explain_rna_tissue(
                rna_package,
                tissue
            )

        except Exception as e:

            print(
                f"\nERROR in RNA tissue "
                f"'{tissue}':"
            )

            print(
                repr(e)
            )

    # ========================================================
    # MICROBIOME SHAP
    # ========================================================

    print("\n")
    print("=" * 78)
    print("STARTING MICROBIOME SHAP")
    print("=" * 78)

    try:

        explain_microbiome(
            microbial_package
        )

    except Exception as e:

        print(
            "\nERROR in microbiome SHAP:"
        )

        print(
            repr(e)
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 78)
    print("SHAP EXPLAINABILITY COMPLETE")
    print("=" * 78)

    print(
        f"\nAll SHAP outputs are in:"
    )

    print(
        REPORTS_DIR
    )

    print("\nExpected structure:")
    print(
        "reports/shap/"
    )
    print(
        "├── RNA/"
    )
    print(
        "│   ├── muscle_skeletal/"
    )
    print(
        "│   ├── lung/"
    )
    print(
        "│   ├── skin_sun_exposed_lower_leg/"
    )
    print(
        "│   └── nerve_tibial/"
    )
    print(
        "│"
    )
    print(
        "└── Microbiome/"
    )

    print("\nDone.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()