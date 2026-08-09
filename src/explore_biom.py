import biom
import pandas as pd

# Load a single BIOM file to inspect structure
table = biom.load_table("data/raw/microbiome/sr38_reference-hit.biom")

print("Shape (features x samples):", table.shape)
print("\nFirst 5 sample IDs:", table.ids('sample')[:5])
print("\nFirst 5 feature IDs (these will be DNA sequences for deblur):", table.ids('observation')[:5])

# Convert to a pandas DataFrame - features as rows, samples as columns
df = table.to_dataframe(dense=True)
print("\nDataFrame shape:", df.shape)
print(df.iloc[:5, :5])  # peek at first 5 features x first 5 samples