"""
rna_model.py
------------
ForensicChrono - RNA-based PMI (postmortem interval) prediction model.
"""

import pandas as pd
import numpy as np
import gzip
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.dummy import DummyRegressor

SAMPLE_ATTRIBUTES_PATH = "data/raw/rna/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
SUBJECT_PHENOTYPES_PATH = "data/raw/rna/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt"
GENE_READS_PATH = "data/raw/rna/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct.gz"

TARGET_TISSUE = "Muscle - Skeletal"

REFERENCE_GENES = ["GAPDH", "ACTB", "RPS29", "RPS18", "RPL13A", "B2M"]


def load_sample_metadata():
    print("Loading sample metadata...")
    df = pd.read_csv(SAMPLE_ATTRIBUTES_PATH, sep="\t", low_memory=False)
    df = df[["SAMPID", "SMTSISCH", "SMTSD"]].copy()
    df = df.rename(columns={"SMTSISCH": "ischemic_time_min", "SMTSD": "tissue"})
    df = df.dropna(subset=["ischemic_time_min"])
    print(f"  -> {len(df)} samples have a recorded ischemic time.")

    if TARGET_TISSUE is not None:
        df = df[df["tissue"] == TARGET_TISSUE]
        print(f"  -> Filtered to tissue '{TARGET_TISSUE}': {len(df)} samples remain.")

    return df


def load_reference_gene_expression():
    print("Scanning gene expression file for reference genes (this may take a minute)...")
    rows = []
    header = None

    with gzip.open(GENE_READS_PATH, "rt") as f:
        f.readline()
        f.readline()
        header = f.readline().strip().split("\t")

        for line in f:
            if any(gene in line for gene in REFERENCE_GENES):
                parts = line.strip().split("\t")
                gene_name = parts[1]
                if gene_name in REFERENCE_GENES:
                    rows.append(parts)

    expr_df = pd.DataFrame(rows, columns=header)
    expr_df = expr_df.set_index("Description").drop(columns=["Name"])
    expr_df = expr_df.astype(float)

    print(f"  -> Found expression rows for: {list(expr_df.index)}")
    return expr_df


def build_features(expr_df, meta_df):
    print("Building features...")
    expr_t = expr_df.T
    expr_t.index.name = "SAMPID"
    expr_t = expr_t.reset_index()

    merged = pd.merge(expr_t, meta_df, on="SAMPID", how="inner")

    gene_cols = [g for g in REFERENCE_GENES if g in merged.columns]
    merged["ref_gene_mean"] = merged[gene_cols].mean(axis=1)

    for gene in gene_cols:
        merged[f"{gene}_ratio"] = merged[gene] / merged["ref_gene_mean"]

    feature_cols = gene_cols + [f"{g}_ratio" for g in gene_cols] + ["ref_gene_mean"]

    print(f"  -> {len(merged)} samples with complete features.")
    return merged, feature_cols


def save_scatter_plot_svg(y_actual, y_predicted, out_path="reports/rna_model_predicted_vs_actual.svg"):
    y_actual = np.array(y_actual)
    y_predicted = np.array(y_predicted)

    width, height = 600, 600
    margin = 60
    plot_size = width - 2 * margin

    lo = min(y_actual.min(), y_predicted.min())
    hi = max(y_actual.max(), y_predicted.max())
    span = hi - lo if hi > lo else 1

    def to_svg_x(val):
        return margin + (val - lo) / span * plot_size

    def to_svg_y(val):
        return height - margin - (val - lo) / span * plot_size

    circles = "\n".join(
        f'<circle cx="{to_svg_x(a):.1f}" cy="{to_svg_y(p):.1f}" r="3" '
        f'fill="#2563eb" fill-opacity="0.45" />'
        for a, p in zip(y_actual, y_predicted)
    )

    line_x1, line_y1 = to_svg_x(lo), to_svg_y(lo)
    line_x2, line_y2 = to_svg_x(hi), to_svg_y(hi)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"
     viewBox="0 0 {width} {height}" font-family="Arial, sans-serif">
  <rect width="{width}" height="{height}" fill="white" />
  <text x="{width/2}" y="30" text-anchor="middle" font-size="18" font-weight="bold">
    RNA Model: Predicted vs Actual PMI
  </text>
  <line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}"
        stroke="black" stroke-width="1" />
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}"
        stroke="black" stroke-width="1" />
  <text x="{width/2}" y="{height-15}" text-anchor="middle" font-size="13">
    Actual ischemic time (minutes)
  </text>
  <text x="20" y="{height/2}" text-anchor="middle" font-size="13"
        transform="rotate(-90, 20, {height/2})">
    Predicted ischemic time (minutes)
  </text>
  <line x1="{line_x1:.1f}" y1="{line_y1:.1f}" x2="{line_x2:.1f}" y2="{line_y2:.1f}"
        stroke="red" stroke-width="1.5" stroke-dasharray="6,4" />
  {circles}
</svg>'''

    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(svg)

    print(f"\nSaved scatter plot to {out_path} (open it in any web browser)")


def train_and_evaluate(merged, feature_cols):
    print("Training model...")

    X = merged[feature_cols]
    y = merged["ischemic_time_min"]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    y_pred = cross_val_predict(model, X, y, cv=kf)

    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    baseline = DummyRegressor(strategy="mean")
    baseline_pred = cross_val_predict(baseline, X, y, cv=kf)
    baseline_mae = mean_absolute_error(y, baseline_pred)

    print("\n--- RESULTS ---")
    print(f"Our model's Mean Absolute Error : {mae:.2f} minutes ({mae/60:.2f} hours)")
    print(f"R^2 score                       : {r2:.3f}  (closer to 1 = better)")
    print(f"Baseline (always guess average) : {baseline_mae:.2f} minutes ({baseline_mae/60:.2f} hours)")
    improvement = (1 - mae / baseline_mae) * 100
    print(f"Improvement over baseline        : {improvement:.1f}%")

    save_scatter_plot_svg(y, y_pred)

    model.fit(X, y)
    return model


def main():
    meta_df = load_sample_metadata()
    expr_df = load_reference_gene_expression()
    merged, feature_cols = build_features(expr_df, meta_df)
    train_and_evaluate(merged, feature_cols)


if __name__ == "__main__":
    main()