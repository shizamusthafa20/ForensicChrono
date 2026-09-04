import pandas as pd

# Original OTU table
otu_path = "data/raw/microbiome/otu_table.tsv"

# Real sample we want to test
sample_id = "13810.CH.SHED.D14.2021.T1"

print("Reading OTU table...")
otu = pd.read_csv(otu_path, sep="\t", index_col=0)

print("OTU table shape:", otu.shape)

if sample_id not in otu.columns:
    raise ValueError(
        f"Sample {sample_id} was not found.\n"
        f"First few sample IDs:\n{list(otu.columns[:10])}"
    )

# Extract exactly one real sample
test_sample = otu[[sample_id]].copy()

# Save in the format expected by the Streamlit uploader
output_path = "forensicchrono_test_microbiome.csv"
test_sample.to_csv(output_path)

print("\nSUCCESS!")
print("Created:", output_path)
print("Shape:", test_sample.shape)
print("Sample:", sample_id)