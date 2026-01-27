"""
Script temporaneo per pulire il CSV household_2025_01.csv
Elimina tutte le colonne tranne time e building (non zero)
"""
import pandas as pd
from pathlib import Path

# Input file
input_file = Path(__file__).parent.parent / "data" / "raw" / "1sec_new" / "household_2025_01.csv"
output_file = Path(__file__).parent.parent / "data" / "raw" / "1sec_new" / "household_2025_01_clean.csv"

print(f"Loading: {input_file}")
df = pd.read_csv(input_file)
print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

# Find building column that is not all zeros
building_cols = [col for col in df.columns if col.startswith('building_')]

building_col = None
for col in building_cols:
    col_sum = df[col].sum()
    print(f"  {col}: sum={col_sum:.2f}")
    if col_sum != 0:
        building_col = col
        break

if not building_col:
    raise ValueError("No non-zero building_* column found")

print(f"\n✅ Using building column: {building_col}")

# Keep only time and building
df_clean = df[['time', building_col]].copy()

# Rename building column to 'power_total' for compatibility
df_clean = df_clean.rename(columns={building_col: 'aggregate_kw'})

print(f"  Cleaned dataframe: {len(df_clean):,} rows, {len(df_clean.columns)} columns")
print(f"  Columns: {list(df_clean.columns)}")

# Save
print(f"\nSaving to: {output_file}")
df_clean.to_csv(output_file, index=False)

print(f"✅ Done! File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
