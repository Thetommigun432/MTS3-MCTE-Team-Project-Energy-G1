import pandas as pd
try:
    df = pd.read_parquet('data/processed/1sec_new/nilm_ready_1sec_new.parquet')
    print("Columns:", df.columns)
    print("Index:", df.index)
    print("Index dtype:", df.index.dtype)
    if 'timestamp' in df.columns:
        print("Timestamp column head:", df['timestamp'].head())
except Exception as e:
    print(e)
