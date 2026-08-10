import pandas as pd

# Load the dataset containing site information
# Adjust the path if the CSV is located elsewhere
csv_path = "../data/processed/operations_data.csv"

df = pd.read_csv(csv_path)

# Identify unique site identifiers
site_ids = df['site_id'].unique()

# Create a variable for each site (e.g., SITE_001 -> site_001)
for site_id in site_ids:
    # Convert to a valid Python variable name (lowercase, replace non‑alphanumeric with underscore)
    var_name = site_id.lower().replace("-", "_")
    globals()[var_name] = df[df['site_id'] == site_id].copy()

# Example usage:
# print(site_001.head())
