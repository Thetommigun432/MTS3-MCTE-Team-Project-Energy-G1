"""One-time conversion: Excel to CSV for faster loading."""
import pandas as pd

excel_path = "data/raw/15min/influxdb_query_20251020_074134.xlsx"
csv_path = "data/raw/15min/influxdb_query_20251020_074134.csv"

print(f"Converting {excel_path}...")
df = pd.read_excel(excel_path)
df.to_csv(csv_path, index=False)
print(f"Saved to {csv_path} ({len(df):,} rows, {len(df.columns)} columns)")
