
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import os

# CONFIG
RAW_PATH = Path(r'c:\Users\gamek\School\TeamProject\MTS3-MCTE-Team-Project-Energy-G1\data\raw\1sec')
OUT_PATH = Path(r'c:\Users\gamek\School\TeamProject\MTS3-MCTE-Team-Project-Energy-G1\data\processed\10sec')
OUT_FILE = OUT_PATH / 'nilm_10sec_mar_may.parquet'

# Ensure output dir exists
OUT_PATH.mkdir(parents=True, exist_ok=True)

# Files to process (The Golden Quarter)
FILES = [
    'samengevoegd_2024-03.csv',
    'samengevoegd_2024-04.csv',
    'samengevoegd_2024-05.csv'
]

def load_and_resample(filename):
    path = RAW_PATH / filename
    print(f"Loading {filename}...")
    
    # 1sec files are CSVs. Checking separators.
    # Usually they are comma separated? Or semicolon?
    # Let's assume standard CSV read first
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

    # Determine Timestamp column
    # Often 'Time' or 'timestamp' or 'date'
    # Based on previous analysis, likely 'Time' or similar. 
    # Let's inspect columns dynamically or assume standard
    
    # Standardize column naming implies finding the time column
    time_col = None
    for col in df.columns:
        if 'time' in col.lower() or 'date' in col.lower():
            time_col = col
            break
            
    if not time_col:
        # Maybe index is time?
        print(f"❌ No time column found in {filename}. Columns: {df.columns}")
        return None
        
    print(f"   Time column: {time_col}")
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.set_index(time_col)
    
    # Resample to 10s
    # Rule: Aggregation = Mean
    print(f"   Resampling to 10s...")
    df_10s = df.resample('10s').mean()
    
    # Drop Smappee if exists (it is corrupted/null in this period)
    cols_to_drop = [c for c in df_10s.columns if 'smappee' in c.lower() or 'laadpaal' in c.lower()]
    if cols_to_drop:
        print(f"   Dropping {cols_to_drop}")
        df_10s = df_10s.drop(columns=cols_to_drop)
        
    return df_10s

def main():
    dfs = []
    
    for f in FILES:
        df = load_and_resample(f)
        if df is not None:
            dfs.append(df)
            
    if not dfs:
        print("No data loaded!")
        return
        
    print("Concatenating...")
    full_df = pd.concat(dfs).sort_index()
    
    # Renaming for consistency with Training Code
    # We need strictly: 'Aggregate' + Appliance Names
    # Let's look at what we have
    print(f"Columns before rename: {full_df.columns.tolist()}")
    
    # Mapping based on 1sec_problems_analysis.md analysis
    # Need to match standard names: 
    # 'Building'/'Totaal' -> 'Aggregate'
    # 'Fornuis' -> 'Stove' ? No, keep Dutch or Map?
    # V3 used: HeatPump, SmappeeCharger, ChargingStation_Socket, Dishwasher, WashingMachine, Dryer, Stove, GarageCabinet, RangeHood, WaterPump, HeatPump_Controller
    
    # Let's map Dutch to Standard English to reuse V3 code easily
    mapping = {
        'Building': 'Aggregate',
        'Totaal': 'Aggregate',
        'Fornuis': 'Stove',
        'Oven': 'Oven', # New for high freq? Or mapped to Stove? V3 had Stove. 
        # Wait, V3 "Stove" might be "Fornuis".
        'Vaatwasser': 'Dishwasher',
        'Wasmachine': 'WashingMachine',
        'Droogkast': 'Dryer',
        'Dampkap': 'RangeHood',
        'Warmtepomp': 'HeatPump',
        'Regenwaterpomp': 'WaterPump'
        # 'Kast garage' missing in 1sec
        # 'Laadpaal' missing/dropped
    }
    
    new_cols = {}
    for col in full_df.columns:
        # Simple fuzzy match or direct
        for k, v in mapping.items():
            if k.lower() in col.lower():
                new_cols[col] = v
    
    # Rename known cols
    print(f"Renaming map: {new_cols}")
    full_df = full_df.rename(columns=new_cols)
    
    # Final cleanup
    # 1. Fill small gaps? 
    # Just generic fillna(0) for safety in simple approach?
    # Ideally linear interpolation for small gaps, but mean resampling handles some.
    # Let's drop rows where Aggregate is NaN
    full_df = full_df.dropna(subset=['Aggregate'])
    
    # Clip Negatives
    print("Clipping negatives...")
    num_cols = full_df.select_dtypes(include=[np.number]).columns
    full_df[num_cols] = full_df[num_cols].clip(lower=0)
    
    full_df = full_df.reset_index()
    full_df = full_df.rename(columns={'index': 'Time', '_time': 'Time'}) # Handle both cases
    
    # Fix renaming issue if 'index' vs original name
    if 'Time' not in full_df.columns:
         # Find the datetime column
         for col in full_df.columns:
             if pd.api.types.is_datetime64_any_dtype(full_df[col]):
                 full_df = full_df.rename(columns={col: 'Time'})
                 break

    print(f"Final Shape: {full_df.shape}")
    print(f"Saving to {OUT_FILE}...")
    full_df.to_parquet(OUT_FILE)
    print("Done.")

if __name__ == '__main__':
    main()
