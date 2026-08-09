"""
merge_biom.py — ForensicChrono (Microbiome data prep)
========================================================
Merges 8 per-prep BIOM feature tables into one combined OTU table,
and reshapes the Qiita sample metadata into the format microbial_model.py expects.

Outputs:
  data/raw/microbiome/otu_table.tsv   (taxa x samples, tab-separated)
  data/raw/microbiome/metadata.tsv    (sample_id, pmi, subject_id, ...)
"""

import glob
import biom
import pandas as pd

BIOM_DIR = "data/raw/microbiome"
METADATA_IN = "data/raw/microbiome/sample_metadata.tsv"

OTU_OUT = "data/raw/microbiome/otu_table.tsv"
METADATA_OUT = "data/raw/microbiome/metadata.tsv"


def merge_biom_tables():
    biom_files = sorted(glob.glob(f"{BIOM_DIR}/*_reference-hit.biom"))
    print(f"Found {len(biom_files)} BIOM files:")
    for f in biom_files:
        print(f"  - {f}")

    dfs = []
    for f in biom_files:
        table = biom.load_table(f)
        df = table.to_dataframe(dense=True)  # features x samples
        dfs.append(df)
        print(f"  Loaded {f}: {df.shape[0]} features x {df.shape[1]} samples")

    # Outer join on feature IDs (rows), fill missing with 0
    # Different preps may not share every exact sequence variant
    merged = pd.concat(dfs, axis=1, join="outer").fillna(0.0)

    # Sanity check: no duplicate sample columns across preps
    n_dupe_cols = merged.columns.duplicated().sum()
    if n_dupe_cols > 0:
        print(f"  WARNING: {n_dupe_cols} duplicate sample IDs found across preps!")

    print(f"\nMerged table shape: {merged.shape[0]} features x {merged.shape[1]} samples")
    merged.to_csv(OTU_OUT, sep="\t")
    print(f"Saved -> {OTU_OUT}")
    return merged


def prep_metadata(otu_sample_ids):
    meta = pd.read_csv(METADATA_IN, sep="\t", low_memory=False)

    keep_cols = [
        "sample_name", "donorid", "days_since_placement",
        "indoor_add_0", "outdoor_add_0", "indoor_outdoor", "body_site"
    ]
    meta = meta[keep_cols].copy()
    meta.columns = ["sample_id", "subject_id", "pmi", "indoor_add", "outdoor_add", "indoor_outdoor", "body_site"]

    meta["pmi"] = pd.to_numeric(meta["pmi"], errors="coerce")
    meta = meta.dropna(subset=["pmi"])

    # Only keep metadata rows that actually have a matching BIOM sample
    before = len(meta)
    meta = meta[meta["sample_id"].isin(otu_sample_ids)].reset_index(drop=True)
    print(f"\nMetadata: kept {len(meta)}/{before} rows with matching BIOM samples")
    print(f"Unique donors: {meta['subject_id'].nunique()}")

    meta.to_csv(METADATA_OUT, sep="\t", index=False)
    print(f"Saved -> {METADATA_OUT}")


if __name__ == "__main__":
    merged = merge_biom_tables()
    prep_metadata(set(merged.columns))