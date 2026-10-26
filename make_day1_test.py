import pandas as pd
import os

# ============================================================
# FORensicChrono — Create Day-1 Microbiome Test File
# ============================================================

METADATA_PATH = "data/raw/microbiome/metadata.tsv"
OTU_PATH = "data/raw/microbiome/otu_table.tsv"

OUTPUT_PATH = "data/raw/microbiome/forensicchrono_day1_test.csv"

# ------------------------------------------------------------
# 1. Load metadata
# ------------------------------------------------------------

print("Loading metadata...")

metadata = pd.read_csv(
    METADATA_PATH,
    sep="\t",
    dtype=str
)

# Convert PMI to numeric
metadata["pmi_numeric"] = pd.to_numeric(
    metadata["pmi"],
    errors="coerce"
)

# Find actual Day-1 samples
day1 = metadata[
    metadata["pmi_numeric"] == 1.0
].copy()

print(f"Day-1 samples found: {len(day1)}")

if len(day1) == 0:
    raise ValueError("No PMI = 1.0 samples found in metadata.")

# Pick the first real Day-1 sample
sample_id = day1.iloc[0]["sample_id"]

print()
print("Selected Day-1 sample:")
print(sample_id)

# ------------------------------------------------------------
# 2. Load OTU table
# ------------------------------------------------------------

print()
print("Loading OTU table...")

otu = pd.read_csv(
    OTU_PATH,
    sep="\t",
    index_col=0
)

print("OTU table shape:", otu.shape)

# ------------------------------------------------------------
# 3. Check whether selected sample exists
# ------------------------------------------------------------

if sample_id not in otu.columns:

    # Sometimes sample IDs may have slightly different formatting.
    matches = [
        col for col in otu.columns
        if str(col).strip() == str(sample_id).strip()
    ]

    if len(matches) == 0:
        raise ValueError(
            f"Sample {sample_id} was found in metadata "
            "but not found in OTU table columns."
        )

    sample_column = matches[0]

else:
    sample_column = sample_id

# ------------------------------------------------------------
# 4. Extract the complete real Day-1 microbiome profile
# ------------------------------------------------------------

test_sample = otu[[sample_column]].copy()

# Transpose:
# rows = features/taxa
# one column = sample
test_sample = test_sample.T

# Name the sample
test_sample.index.name = "sample_id"

# ------------------------------------------------------------
# 5. Save CSV
# ------------------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

test_sample.to_csv(OUTPUT_PATH)

print()
print("=" * 60)
print("DAY-1 TEST FILE CREATED")
print("=" * 60)

print("Sample:", sample_id)
print("PMI:", day1.iloc[0]["pmi_numeric"], "days")
print("Features:", test_sample.shape[1])
print()
print("Saved to:")
print(os.path.abspath(OUTPUT_PATH))
print("=" * 60)