"""
======================================================================
FORENSICCHRONO
RELIABILITY-WEIGHTED LATE FUSION
======================================================================

PURPOSE
-------
Combine the independently validated RNA and microbiome PMI models.

RNA:
    OOF prediction -> minutes -> hours
    Nested donor-level OOF validation

Microbiome:
    OOF prediction -> days -> hours
    Donor-grouped OOF validation

FUSION
------
    RNA prediction
          |
      Calibration
          |
      Uncertainty
          |
          |\
          | \
          |  \
          |   > Reliability-weighted late fusion
          |  /
          | /
    Uncertainty
          |
    Calibration
          |
    Microbiome prediction

OUTPUT
------
    ONE PMI estimate in hours
    95% confidence interval
    RNA reliability
    Microbiome reliability
    Primary evidence source

IMPORTANT
---------
Because RNA and microbiome datasets are UNPAIRED, this script does
NOT report a fusion R² or fusion MAE.

The individual model OOF metrics are real and are reported separately.

This is NOT a conventional supervised cross-modal meta-model.
It is a reliability-weighted late-fusion engine.
======================================================================
"""

import os
import json
import pickle

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor


# ======================================================================
# CONFIGURATION
# ======================================================================

SEED = 42

# Calibration
CALIBRATION_FOLDS = 5
RIDGE_ALPHA = 10.0

# Uncertainty
MIN_SIGMA_HOURS = 1.0
CI_Z = 1.96

# Existing OOF files
RNA_OOF_PATH = "results/rna_oof_predictions.csv"
MICROBIAL_OOF_PATH = "results/microbial_oof_predictions.csv"

# Output files
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
        "actual",
        "predicted"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "RNA OOF file is missing columns: "
            f"{missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[
        [
            "SUBJID",
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
            "actual",
            "predicted"
        ]
    )

    # --------------------------------------------------------------
    # RNA is predicted in MINUTES.
    # Convert to HOURS.
    # --------------------------------------------------------------

    df["actual_hours"] = (
        df["actual"] / 60.0
    )

    df["prediction_hours"] = (
        df["predicted"] / 60.0
    )

    # --------------------------------------------------------------
    # Donor-level aggregation
    #
    # The RNA model has multiple samples per donor.
    # Aggregate before calibration so donors contribute equally.
    # --------------------------------------------------------------

    donor_df = (
        df
        .groupby(
            "SUBJID",
            as_index=False
        )
        .agg(
            actual_hours=(
                "actual_hours",
                "mean"
            ),
            prediction_hours=(
                "prediction_hours",
                "mean"
            )
        )
    )

    print(
        f"RNA samples : {len(df)}"
    )

    print(
        f"RNA donors  : {len(donor_df)}"
    )

    return donor_df


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
            "Microbiome OOF file is missing columns: "
            f"{missing}\n"
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
            "actual_pmi",
            "ensemble_prediction"
        ]
    )

    # --------------------------------------------------------------
    # Microbiome is predicted in DAYS.
    # Convert to HOURS.
    # --------------------------------------------------------------

    df["actual_hours"] = (
        df["actual_pmi"] * 24.0
    )

    df["prediction_hours"] = (
        df["ensemble_prediction"] * 24.0
    )

    # --------------------------------------------------------------
    # Donor-level aggregation
    # --------------------------------------------------------------

    donor_df = (
        df
        .groupby(
            "subject_id",
            as_index=False
        )
        .agg(
            actual_hours=(
                "actual_hours",
                "mean"
            ),
            prediction_hours=(
                "prediction_hours",
                "mean"
            )
        )
    )

    print(
        f"Microbiome samples : {len(df)}"
    )

    print(
        f"Microbiome donors  : {len(donor_df)}"
    )

    return donor_df


# ======================================================================
# CROSS-FITTED RIDGE CALIBRATION
# ======================================================================

def cross_fitted_calibration(
    predictions,
    actual
):

    predictions = np.asarray(
        predictions,
        dtype=float
    )

    actual = np.asarray(
        actual,
        dtype=float
    )

    n = len(predictions)

    if n < CALIBRATION_FOLDS:

        raise ValueError(
            f"Only {n} donors available. "
            f"Need at least {CALIBRATION_FOLDS} "
            f"for calibration."
        )

    kfold = KFold(
        n_splits=CALIBRATION_FOLDS,
        shuffle=True,
        random_state=SEED
    )

    calibrated_oof = np.zeros(
        n,
        dtype=float
    )

    # --------------------------------------------------------------
    # Generate honest calibration OOF predictions
    # --------------------------------------------------------------

    for fold, (
        train_idx,
        val_idx
    ) in enumerate(
        kfold.split(predictions),
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
    # This is the model used during deployment.
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

    # Prevent KNN from failing on very small datasets.
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
# CALIBRATE ONE MODEL
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

    # --------------------------------------------------------------
    # Original OOF performance
    # --------------------------------------------------------------

    raw_metrics = calculate_metrics(
        y,
        raw_prediction
    )

    # --------------------------------------------------------------
    # Cross-fitted calibration
    # --------------------------------------------------------------

    (
        calibrated_oof,
        calibrator
    ) = cross_fitted_calibration(
        raw_prediction,
        y
    )

    calibrated_metrics = (
        calculate_metrics(
            y,
            calibrated_oof
        )
    )

    # --------------------------------------------------------------
    # OOF residuals
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
    # Local uncertainty model
    #
    # This allows reliability to change depending on
    # where the prediction lies in the PMI range.
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
            residuals
    }


# ======================================================================
# GET UNCERTAINTY FOR A NEW PREDICTION
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
    # therefore:
    #
    # sigma ≈ MAE / sqrt(2/pi)

    local_sigma = (
        local_mae /
        np.sqrt(
            2.0 / np.pi
        )
    )

    # Don't allow local estimate to become
    # unrealistically smaller than the global
    # residual uncertainty.

    minimum_allowed = (
        component[
            "global_sigma_hours"
        ] * 0.50
    )

    local_sigma = max(
        local_sigma,
        minimum_allowed,
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

    micro_calibrated = (
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

    micro_sigma = estimate_uncertainty(
        microbiome_component,
        micro_calibrated
    )

    # --------------------------------------------------------------
    # PRECISION
    #
    # Higher precision = lower uncertainty
    # Higher precision = greater reliability
    # --------------------------------------------------------------

    rna_precision = (
        1.0 /
        (
            rna_sigma ** 2
        )
    )

    micro_precision = (
        1.0 /
        (
            micro_sigma ** 2
        )
    )

    total_precision = (
        rna_precision +
        micro_precision
    )

    # --------------------------------------------------------------
    # DYNAMIC RELIABILITY WEIGHTS
    # --------------------------------------------------------------

    rna_weight = (
        rna_precision /
        total_precision
    )

    micro_weight = (
        micro_precision /
        total_precision
    )

    # --------------------------------------------------------------
    # FINAL PMI
    # --------------------------------------------------------------

    fused_pmi = (
        rna_weight *
        rna_calibrated
        +
        micro_weight *
        micro_calibrated
    )

    # --------------------------------------------------------------
    # COMBINED UNCERTAINTY
    # --------------------------------------------------------------

    fused_sigma = np.sqrt(
        1.0 /
        total_precision
    )

    # --------------------------------------------------------------
    # 95% CONFIDENCE INTERVAL
    # --------------------------------------------------------------

    lower = (
        fused_pmi -
        CI_Z *
        fused_sigma
    )

    upper = (
        fused_pmi +
        CI_Z *
        fused_sigma
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

    if rna_weight > micro_weight:

        primary_evidence = "RNA"

    elif micro_weight > rna_weight:

        primary_evidence = "Microbiome"

    else:

        primary_evidence = "Balanced"

    return {

        "rna_raw_hours":
            rna_raw_hours,

        "microbiome_raw_hours":
            microbiome_raw_hours,

        "rna_calibrated_hours":
            rna_calibrated,

        "microbiome_calibrated_hours":
            micro_calibrated,

        "rna_uncertainty_hours":
            rna_sigma,

        "microbiome_uncertainty_hours":
            micro_sigma,

        "rna_weight":
            rna_weight,

        "microbiome_weight":
            micro_weight,

        "final_pmi_hours":
            fused_pmi,

        "final_uncertainty_hours":
            fused_sigma,

        "ci95_lower_hours":
            lower,

        "ci95_upper_hours":
            upper,

        "primary_evidence":
            primary_evidence
    }


# ======================================================================
# SAVE MODEL
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
            "Precision-weighted "
            "uncertainty-aware "
            "late fusion",

        "validation":
            "RNA and microbiome were "
            "independently evaluated using "
            "donor-level OOF predictions.",

        "fusion_r2":
            None,

        "fusion_mae_hours":
            None,

        "fusion_metric_note":
            "Direct fusion R²/MAE cannot be "
            "estimated because the source "
            "datasets are unpaired."
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
# SAVE COMPONENT RESULTS
# ======================================================================

def save_component_results(
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

    print(
        f"Component results saved -> "
        f"{FUSION_COMPONENT_PATH}"
    )

    return result_df


# ======================================================================
# MAIN TRAINING
# ======================================================================

def main():

    print_header(
        "FORENSICCHRONO"
    )

    print(
        "RELIABILITY-WEIGHTED "
        "LATE FUSION"
    )

    print()
    print(
        "RNA + MICROBIOME"
    )

    print(
        "No simulation."
    )

    print(
        "No supervised cross-modal "
        "meta-model."
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
    # CALIBRATION
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
    # SAVE
    # --------------------------------------------------------------

    save_fusion_model(
        rna_component,
        microbiome_component
    )

    component_df = (
        save_component_results(
            rna_component,
            microbiome_component
        )
    )

    # --------------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------------

    summary = {

        "project":
            "ForensicChrono",

        "fusion_type":
            "Reliability-weighted "
            "late fusion",

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

        "deployment_output":
            [
                "RNA PMI",
                "Microbiome PMI",
                "RNA reliability",
                "Microbiome reliability",
                "Final PMI",
                "95% confidence interval",
                "Primary evidence"
            ]
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
        f"Summary saved -> "
        f"{FUSION_SUMMARY_PATH}"
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
        "RNA and microbiome datasets are unpaired."
    )

    print()
    print(
        "The fusion model is ready for deployment."
    )


# ======================================================================
# OPTIONAL DEPLOYMENT FUNCTION
# ======================================================================

def load_fusion_model():

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
    Generate ONE final ForensicChrono PMI estimate.

    Parameters
    ----------
    rna_prediction_minutes:
        PMI predicted by the RNA model, in minutes.

    microbiome_prediction_days:
        PMI predicted by the microbiome model, in days.

    Returns
    -------
    Dictionary containing:
        final PMI
        95% CI
        model weights
        uncertainties
        primary evidence
    """

    package = load_fusion_model()

    # --------------------------------------------------------------
    # Reconstruct components
    # --------------------------------------------------------------

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
    # Convert units
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

    # --------------------------------------------------------------
    # Fuse
    # --------------------------------------------------------------

    result = fuse_predictions(
        rna_component,
        microbiome_component,
        rna_hours,
        microbiome_hours
    )

    return result


# ======================================================================
# RUN
# ======================================================================

if __name__ == "__main__":

    main()