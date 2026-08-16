"""
======================================================================
FORENSICCHRONO
RELIABILITY-WEIGHTED LATE FUSION - CORRECTED VERSION
======================================================================

RNA:
    OOF predictions:
        results/rna_oof_predictions.csv

    Columns:
        SUBJID
        tissue
        actual
        predicted

    Unit:
        minutes

MICROBIOME:
    OOF predictions:
        results/microbial_oof_predictions.csv

    Columns:
        sample_id
        subject_id
        actual_pmi
        ensemble_prediction

    Unit:
        days

IMPORTANT
---------
DO NOT aggregate the microbiome samples to 27 donors.

The original microbiome model reports its main OOF performance at
sample/longitudinal level:

    R²  ≈ 0.9478
    MAE ≈ 0.987 days = 23.7 hours

The donor-level R² of -3.8548 is a separate diagnostic and must NOT
be used as the main microbiome model performance in fusion.

Calibration DOES use GroupKFold by donor to prevent donor leakage.

FUSION:
    1. Load OOF predictions
    2. Convert both modalities to hours
    3. Evaluate original OOF performance
    4. Cross-fitted Ridge calibration using GroupKFold
    5. Estimate uncertainty from held-out calibration residuals
    6. Compute reliability using inverse variance
    7. Perform reliability-weighted late fusion
    8. Save deployable fusion model

There is intentionally NO fusion R² because RNA and microbiome
datasets contain different/unpaired donors.

======================================================================
"""

import os
import json
import pickle

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor


# ======================================================================
# CONFIGURATION
# ======================================================================

SEED = 42

CALIBRATION_FOLDS = 5
RIDGE_ALPHA = 10.0

MIN_SIGMA_HOURS = 1.0
CI_Z = 1.96

RNA_OOF_PATH = "results/rna_oof_predictions.csv"
MICROBIAL_OOF_PATH = "results/microbial_oof_predictions.csv"

MODEL_DIR = "models"
RESULTS_DIR = "results"

FUSION_MODEL_PATH = (
    f"{MODEL_DIR}/fusion_model.pkl"
)

FUSION_SUMMARY_PATH = (
    f"{RESULTS_DIR}/fusion_summary.json"
)

FUSION_COMPONENT_PATH = (
    f"{RESULTS_DIR}/fusion_component_results.csv"
)


# ======================================================================
# UTILITIES
# ======================================================================

def print_header(text):

    print()
    print("=" * 75)
    print(text)
    print("=" * 75)


def calculate_metrics(y_true, y_pred):

    return {
        "r2": float(
            r2_score(
                y_true,
                y_pred
            )
        ),
        "mae_hours": float(
            mean_absolute_error(
                y_true,
                y_pred
            )
        )
    }


# ======================================================================
# LOAD RNA OOF
# ======================================================================

def load_rna_oof():

    print_header(
        "LOADING RNA OOF PREDICTIONS"
    )

    if not os.path.exists(RNA_OOF_PATH):

        raise FileNotFoundError(
            f"RNA OOF file not found:\n"
            f"{RNA_OOF_PATH}"
        )

    df = pd.read_csv(
        RNA_OOF_PATH
    )

    required = [
        "SUBJID",
        "tissue",
        "actual",
        "predicted"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"RNA OOF file is missing: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[
        [
            "SUBJID",
            "tissue",
            "actual",
            "predicted"
        ]
    ].copy()

    df["actual"] = pd.to_numeric(
        df["actual"],
        errors="coerce"
    )

    df["predicted"] = pd.to_numeric(
        df["predicted"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "SUBJID",
            "actual",
            "predicted"
        ]
    )

    # --------------------------------------------------------------
    # RNA: minutes -> hours
    # --------------------------------------------------------------

    df["actual_hours"] = (
        df["actual"] / 60.0
    )

    df["prediction_hours"] = (
        df["predicted"] / 60.0
    )

    df["donor_id"] = (
        df["SUBJID"]
        .astype(str)
    )

    print(
        f"RNA samples : {len(df)}"
    )

    print(
        f"RNA donors  : "
        f"{df['donor_id'].nunique()}"
    )

    return df.reset_index(
        drop=True
    )


# ======================================================================
# LOAD MICROBIOME OOF
# ======================================================================

def load_microbiome_oof():

    print_header(
        "LOADING MICROBIOME OOF PREDICTIONS"
    )

    if not os.path.exists(
        MICROBIAL_OOF_PATH
    ):

        raise FileNotFoundError(
            f"Microbiome OOF file not found:\n"
            f"{MICROBIAL_OOF_PATH}"
        )

    df = pd.read_csv(
        MICROBIAL_OOF_PATH
    )

    required = [
        "sample_id",
        "subject_id",
        "actual_pmi",
        "ensemble_prediction"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Microbiome OOF file is missing: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[
        [
            "sample_id",
            "subject_id",
            "actual_pmi",
            "ensemble_prediction"
        ]
    ].copy()

    df["actual_pmi"] = pd.to_numeric(
        df["actual_pmi"],
        errors="coerce"
    )

    df["ensemble_prediction"] = pd.to_numeric(
        df["ensemble_prediction"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "subject_id",
            "actual_pmi",
            "ensemble_prediction"
        ]
    )

    # --------------------------------------------------------------
    # Microbiome: days -> hours
    # --------------------------------------------------------------

    df["actual_hours"] = (
        df["actual_pmi"] * 24.0
    )

    df["prediction_hours"] = (
        df["ensemble_prediction"] * 24.0
    )

    df["donor_id"] = (
        df["subject_id"]
        .astype(str)
    )

    print(
        f"Microbiome samples : {len(df)}"
    )

    print(
        f"Microbiome donors  : "
        f"{df['donor_id'].nunique()}"
    )

    return df.reset_index(
        drop=True
    )


# ======================================================================
# CROSS-FITTED GROUPED CALIBRATION
# ======================================================================

def cross_fitted_group_calibration(
    predictions,
    actual,
    groups
):

    predictions = np.asarray(
        predictions,
        dtype=float
    )

    actual = np.asarray(
        actual,
        dtype=float
    )

    groups = np.asarray(
        groups
    )

    unique_groups = np.unique(
        groups
    )

    n_groups = len(
        unique_groups
    )

    if n_groups < CALIBRATION_FOLDS:

        raise ValueError(
            f"Only {n_groups} donors available. "
            f"Need at least {CALIBRATION_FOLDS}."
        )

    gkf = GroupKFold(
        n_splits=CALIBRATION_FOLDS
    )

    calibrated_oof = np.zeros(
        len(predictions),
        dtype=float
    )

    # --------------------------------------------------------------
    # Cross-fitted calibration
    #
    # Every validation fold contains donors that were not used
    # to train that fold's calibration model.
    # --------------------------------------------------------------

    for fold, (
        train_idx,
        val_idx
    ) in enumerate(
        gkf.split(
            predictions,
            actual,
            groups
        ),
        start=1
    ):

        calibrator = Ridge(
            alpha=RIDGE_ALPHA
        )

        calibrator.fit(
            predictions[
                train_idx
            ].reshape(-1, 1),

            actual[
                train_idx
            ]
        )

        calibrated_oof[
            val_idx
        ] = calibrator.predict(
            predictions[
                val_idx
            ].reshape(-1, 1)
        )

        print(
            f"  Calibration fold "
            f"{fold}/{CALIBRATION_FOLDS}"
        )

    # --------------------------------------------------------------
    # Final calibration model
    #
    # Used for future deployment.
    # --------------------------------------------------------------

    final_calibrator = Ridge(
        alpha=RIDGE_ALPHA
    )

    final_calibrator.fit(
        predictions.reshape(-1, 1),
        actual
    )

    return (
        calibrated_oof,
        final_calibrator
    )


# ======================================================================
# UNCERTAINTY MODEL
# ======================================================================

def fit_uncertainty_model(
    calibrated_predictions,
    absolute_errors
):

    x = np.asarray(
        calibrated_predictions,
        dtype=float
    )

    y = np.asarray(
        absolute_errors,
        dtype=float
    )

    # KNN estimates typical error magnitude
    # around a particular predicted PMI.

    n_neighbors = min(
        15,
        len(x)
    )

    n_neighbors = max(
        3,
        n_neighbors
    )

    model = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights="distance"
    )

    model.fit(
        x.reshape(-1, 1),
        y
    )

    return model


# ======================================================================
# PREPARE ONE COMPONENT
# ======================================================================

def prepare_component(
    name,
    df
):

    print_header(
        f"{name.upper()} CALIBRATION + UNCERTAINTY"
    )

    y = df[
        "actual_hours"
    ].values

    raw_prediction = df[
        "prediction_hours"
    ].values

    groups = df[
        "donor_id"
    ].values

    # --------------------------------------------------------------
    # ORIGINAL MODEL OOF PERFORMANCE
    #
    # IMPORTANT:
    # This is calculated at the same sample/longitudinal level
    # as the original model's main reported result.
    # --------------------------------------------------------------

    raw_metrics = calculate_metrics(
        y,
        raw_prediction
    )

    # --------------------------------------------------------------
    # GROUPED CROSS-FITTED CALIBRATION
    # --------------------------------------------------------------

    (
        calibrated_oof,
        calibrator
    ) = cross_fitted_group_calibration(
        raw_prediction,
        y,
        groups
    )

    calibrated_metrics = (
        calculate_metrics(
            y,
            calibrated_oof
        )
    )

    # --------------------------------------------------------------
    # HELD-OUT CALIBRATION RESIDUALS
    # --------------------------------------------------------------

    residuals = (
        y -
        calibrated_oof
    )

    absolute_errors = np.abs(
        residuals
    )

    global_mae = float(
        np.mean(
            absolute_errors
        )
    )

    global_sigma = float(
        np.std(
            residuals,
            ddof=1
        )
    )

    global_sigma = max(
        global_sigma,
        MIN_SIGMA_HOURS
    )

    # --------------------------------------------------------------
    # LOCAL UNCERTAINTY MODEL
    # --------------------------------------------------------------

    uncertainty_model = (
        fit_uncertainty_model(
            calibrated_oof,
            absolute_errors
        )
    )

    print()
    print(
        f"{name} RAW OOF:"
    )

    print(
        f"  R²  = "
        f"{raw_metrics['r2']:.4f}"
    )

    print(
        f"  MAE = "
        f"{raw_metrics['mae_hours']:.2f} h"
    )

    print()
    print(
        f"{name} CALIBRATED OOF:"
    )

    print(
        f"  R²  = "
        f"{calibrated_metrics['r2']:.4f}"
    )

    print(
        f"  MAE = "
        f"{calibrated_metrics['mae_hours']:.2f} h"
    )

    print()
    print(
        f"Residual SD = "
        f"{global_sigma:.2f} h"
    )

    return {

        "name":
            name,

        "calibrator":
            calibrator,

        "uncertainty_model":
            uncertainty_model,

        "global_sigma_hours":
            global_sigma,

        "raw_metrics":
            raw_metrics,

        "calibrated_metrics":
            calibrated_metrics,

        "actual_hours":
            y,

        "raw_prediction_hours":
            raw_prediction,

        "calibrated_oof_hours":
            calibrated_oof,

        "residuals":
            residuals,

        "groups":
            groups
    }


# ======================================================================
# ESTIMATE UNCERTAINTY FOR NEW PREDICTION
# ======================================================================

def estimate_uncertainty(
    component,
    prediction_hours
):

    prediction_hours = float(
        prediction_hours
    )

    local_mae = float(
        component[
            "uncertainty_model"
        ]
        .predict(
            np.array(
                [[prediction_hours]]
            )
        )[0]
    )

    # Approximate conversion:
    #
    # MAE ≈ sigma * sqrt(2/pi)
    #
    # sigma ≈ MAE / sqrt(2/pi)

    local_sigma = (
        local_mae /
        np.sqrt(
            2.0 / np.pi
        )
    )

    # Prevent uncertainty from becoming
    # unrealistically tiny.

    local_sigma = max(
        local_sigma,
        component[
            "global_sigma_hours"
        ] * 0.50,
        MIN_SIGMA_HOURS
    )

    return local_sigma


# ======================================================================
# CALIBRATE NEW PREDICTION
# ======================================================================

def calibrate_new_prediction(
    component,
    raw_prediction_hours
):

    calibrated = component[
        "calibrator"
    ].predict(
        np.array(
            [[raw_prediction_hours]]
        )
    )[0]

    return float(
        max(
            0.0,
            calibrated
        )
    )


# ======================================================================
# RELIABILITY-WEIGHTED FUSION
# ======================================================================

def fuse_predictions(
    rna_component,
    microbiome_component,
    rna_raw_hours,
    microbiome_raw_hours
):

    # --------------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------------

    rna_calibrated = (
        calibrate_new_prediction(
            rna_component,
            rna_raw_hours
        )
    )

    microbiome_calibrated = (
        calibrate_new_prediction(
            microbiome_component,
            microbiome_raw_hours
        )
    )

    # --------------------------------------------------------------
    # UNCERTAINTY
    # --------------------------------------------------------------

    rna_sigma = estimate_uncertainty(
        rna_component,
        rna_calibrated
    )

    microbiome_sigma = estimate_uncertainty(
        microbiome_component,
        microbiome_calibrated
    )

    # --------------------------------------------------------------
    # PRECISION
    #
    # precision = 1 / variance
    # --------------------------------------------------------------

    rna_precision = (
        1.0 /
        (rna_sigma ** 2)
    )

    microbiome_precision = (
        1.0 /
        (microbiome_sigma ** 2)
    )

    total_precision = (
        rna_precision +
        microbiome_precision
    )

    # --------------------------------------------------------------
    # DYNAMIC RELIABILITY WEIGHTS
    # --------------------------------------------------------------

    rna_weight = (
        rna_precision /
        total_precision
    )

    microbiome_weight = (
        microbiome_precision /
        total_precision
    )

    # --------------------------------------------------------------
    # FINAL PMI
    # --------------------------------------------------------------

    final_pmi = (
        rna_weight *
        rna_calibrated
        +
        microbiome_weight *
        microbiome_calibrated
    )

    # --------------------------------------------------------------
    # COMBINED UNCERTAINTY
    # --------------------------------------------------------------

    final_sigma = np.sqrt(
        1.0 /
        total_precision
    )

    # --------------------------------------------------------------
    # APPROXIMATE 95% INTERVAL
    # --------------------------------------------------------------

    lower = (
        final_pmi -
        CI_Z *
        final_sigma
    )

    upper = (
        final_pmi +
        CI_Z *
        final_sigma
    )

    lower = max(
        0.0,
        lower
    )

    upper = max(
        0.0,
        upper
    )

    # --------------------------------------------------------------
    # PRIMARY EVIDENCE
    # --------------------------------------------------------------

    if rna_weight > microbiome_weight:

        primary_evidence = "RNA"

    elif microbiome_weight > rna_weight:

        primary_evidence = "Microbiome"

    else:

        primary_evidence = "Balanced"

    return {

        "rna_raw_hours":
            float(rna_raw_hours),

        "microbiome_raw_hours":
            float(microbiome_raw_hours),

        "rna_calibrated_hours":
            float(rna_calibrated),

        "microbiome_calibrated_hours":
            float(microbiome_calibrated),

        "rna_uncertainty_hours":
            float(rna_sigma),

        "microbiome_uncertainty_hours":
            float(microbiome_sigma),

        "rna_weight":
            float(rna_weight),

        "microbiome_weight":
            float(microbiome_weight),

        "final_pmi_hours":
            float(final_pmi),

        "final_uncertainty_hours":
            float(final_sigma),

        "ci95_lower_hours":
            float(lower),

        "ci95_upper_hours":
            float(upper),

        "primary_evidence":
            primary_evidence
    }


# ======================================================================
# SAVE DEPLOYABLE FUSION MODEL
# ======================================================================

def save_fusion_model(
    rna_component,
    microbiome_component
):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    package = {

        "model_type":
            "ReliabilityWeightedLateFusion",

        "pmi_unit":
            "hours",

        "rna": {

            "calibrator":
                rna_component[
                    "calibrator"
                ],

            "uncertainty_model":
                rna_component[
                    "uncertainty_model"
                ],

            "global_sigma_hours":
                rna_component[
                    "global_sigma_hours"
                ]
        },

        "microbiome": {

            "calibrator":
                microbiome_component[
                    "calibrator"
                ],

            "uncertainty_model":
                microbiome_component[
                    "uncertainty_model"
                ],

            "global_sigma_hours":
                microbiome_component[
                    "global_sigma_hours"
                ]
        },

        "method":
            "GroupKFold calibration + "
            "uncertainty-aware "
            "precision-weighted late fusion",

        "fusion_r2":
            None,

        "fusion_mae_hours":
            None,

        "fusion_metric_note":
            "Direct fusion R2/MAE cannot be "
            "estimated because RNA and "
            "microbiome cohorts are unpaired."
    }

    with open(
        FUSION_MODEL_PATH,
        "wb"
    ) as f:

        pickle.dump(
            package,
            f
        )

    print()
    print(
        f"Fusion model saved -> "
        f"{FUSION_MODEL_PATH}"
    )


# ======================================================================
# SAVE RESULTS
# ======================================================================

def save_results(
    rna_component,
    microbiome_component
):

    rows = []

    for component in [
        rna_component,
        microbiome_component
    ]:

        rows.append({

            "Model":
                component["name"],

            "Raw OOF R2":
                component[
                    "raw_metrics"
                ]["r2"],

            "Raw OOF MAE (hours)":
                component[
                    "raw_metrics"
                ]["mae_hours"],

            "Calibrated OOF R2":
                component[
                    "calibrated_metrics"
                ]["r2"],

            "Calibrated OOF MAE (hours)":
                component[
                    "calibrated_metrics"
                ]["mae_hours"],

            "Residual SD (hours)":
                component[
                    "global_sigma_hours"
                ]
        })

    result_df = pd.DataFrame(
        rows
    )

    result_df.to_csv(
        FUSION_COMPONENT_PATH,
        index=False
    )

    summary = {

        "project":
            "ForensicChrono",

        "fusion_type":
            "Reliability-weighted late fusion",

        "rna_raw_r2":
            rna_component[
                "raw_metrics"
            ]["r2"],

        "rna_raw_mae_hours":
            rna_component[
                "raw_metrics"
            ]["mae_hours"],

        "rna_calibrated_r2":
            rna_component[
                "calibrated_metrics"
            ]["r2"],

        "rna_calibrated_mae_hours":
            rna_component[
                "calibrated_metrics"
            ]["mae_hours"],

        "microbiome_raw_r2":
            microbiome_component[
                "raw_metrics"
            ]["r2"],

        "microbiome_raw_mae_hours":
            microbiome_component[
                "raw_metrics"
            ]["mae_hours"],

        "microbiome_calibrated_r2":
            microbiome_component[
                "calibrated_metrics"
            ]["r2"],

        "microbiome_calibrated_mae_hours":
            microbiome_component[
                "calibrated_metrics"
            ]["mae_hours"],

        "fusion_r2":
            None,

        "fusion_mae_hours":
            None,

        "fusion_validation":
            "Not directly estimable because "
            "RNA and microbiome cohorts are "
            "unpaired.",

        "important_microbiome_note":
            "Microbiome main performance is "
            "sample/longitudinal OOF performance. "
            "The donor-level R2 diagnostic is "
            "not used as the component performance."
    }

    with open(
        FUSION_SUMMARY_PATH,
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    print(
        f"Component results saved -> "
        f"{FUSION_COMPONENT_PATH}"
    )

    print(
        f"Summary saved -> "
        f"{FUSION_SUMMARY_PATH}"
    )

    return result_df


# ======================================================================
# DEPLOYMENT
# ======================================================================

def load_fusion_model():

    if not os.path.exists(
        FUSION_MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Fusion model not found:\n"
            f"{FUSION_MODEL_PATH}\n\n"
            f"Run fusion_model.py first."
        )

    with open(
        FUSION_MODEL_PATH,
        "rb"
    ) as f:

        return pickle.load(f)


def deploy_fusion(
    rna_prediction_minutes,
    microbiome_prediction_days
):
    """
    Generate one final PMI estimate.

    INPUTS
    ------
    rna_prediction_minutes:
        Prediction produced by the trained RNA model.

    microbiome_prediction_days:
        Prediction produced by the trained microbiome model.

    OUTPUT
    ------
    Dictionary containing:

        RNA PMI
        microbiome PMI
        uncertainties
        reliability weights
        final PMI
        95% confidence interval
        primary evidence
    """

    package = load_fusion_model()

    rna_component = {

        "calibrator":
            package[
                "rna"
            ]["calibrator"],

        "uncertainty_model":
            package[
                "rna"
            ]["uncertainty_model"],

        "global_sigma_hours":
            package[
                "rna"
            ]["global_sigma_hours"]
    }

    microbiome_component = {

        "calibrator":
            package[
                "microbiome"
            ]["calibrator"],

        "uncertainty_model":
            package[
                "microbiome"
            ]["uncertainty_model"],

        "global_sigma_hours":
            package[
                "microbiome"
            ]["global_sigma_hours"]
    }

    # --------------------------------------------------------------
    # Convert deployment predictions to hours
    # --------------------------------------------------------------

    rna_hours = (
        float(
            rna_prediction_minutes
        ) / 60.0
    )

    microbiome_hours = (
        float(
            microbiome_prediction_days
        ) * 24.0
    )

    return fuse_predictions(
        rna_component,
        microbiome_component,
        rna_hours,
        microbiome_hours
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    print_header(
        "FORENSICCHRONO"
    )

    print(
        "RELIABILITY-WEIGHTED LATE FUSION"
    )

    print()
    print(
        "RNA + MICROBIOME"
    )

    print(
        "No simulation."
    )

    print(
        "No supervised cross-modal meta-model."
    )

    print(
        "No artificial fusion R²."
    )

    # --------------------------------------------------------------
    # LOAD
    # --------------------------------------------------------------

    rna_df = load_rna_oof()

    microbiome_df = (
        load_microbiome_oof()
    )

    # --------------------------------------------------------------
    # CALIBRATION + UNCERTAINTY
    # --------------------------------------------------------------

    rna_component = (
        prepare_component(
            "RNA",
            rna_df
        )
    )

    microbiome_component = (
        prepare_component(
            "Microbiome",
            microbiome_df
        )
    )

    # --------------------------------------------------------------
    # SAVE DEPLOYABLE MODEL
    # --------------------------------------------------------------

    save_fusion_model(
        rna_component,
        microbiome_component
    )

    # --------------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------------

    save_results(
        rna_component,
        microbiome_component
    )

    # --------------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------------

    print_header(
        "FUSION TRAINING COMPLETE"
    )

    print()

    print(
        "RNA"
    )

    print(
        f"  Samples : "
        f"{len(rna_df)}"
    )

    print(
        f"  Donors  : "
        f"{rna_df['donor_id'].nunique()}"
    )

    print(
        f"  OOF R²  : "
        f"{rna_component['raw_metrics']['r2']:.4f}"
    )

    print(
        f"  OOF MAE : "
        f"{rna_component['raw_metrics']['mae_hours']:.2f} h"
    )

    print()

    print(
        "MICROBIOME"
    )

    print(
        f"  Samples : "
        f"{len(microbiome_df)}"
    )

    print(
        f"  Donors  : "
        f"{microbiome_df['donor_id'].nunique()}"
    )

    print(
        f"  OOF R²  : "
        f"{microbiome_component['raw_metrics']['r2']:.4f}"
    )

    print(
        f"  OOF MAE : "
        f"{microbiome_component['raw_metrics']['mae_hours']:.2f} h"
    )

    print()

    print(
        "FUSION"
    )

    print(
        "  Type    : "
        "Reliability-weighted late fusion"
    )

    print(
        "  R²      : "
        "NOT ESTIMABLE"
    )

    print(
        "  Reason  : "
        "RNA and microbiome cohorts are unpaired."
    )

    print()

    print(
        "Deployable fusion model ready."
    )


# ======================================================================
# EXECUTE
# ======================================================================

if __name__ == "__main__":

    main()