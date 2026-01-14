"""Analyze data quality and completeness for 1-second resolution CSV files."""
import pandas as pd
import numpy as np
import json
from pathlib import Path
import datetime as dt


def analyze_file(path):
    """Analyze single CSV file."""
    df = pd.read_csv(path)
    df['_time'] = pd.to_datetime(df['_time'], format='mixed', errors='coerce')
    
    # Time analysis
    time_span = (df['_time'].max() - df['_time'].min()).total_seconds() / 3600
    time_diffs = df['_time'].diff().dropna().apply(lambda x: x.total_seconds())
    resolution = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else None
    
    # Gap analysis
    gaps = time_diffs[time_diffs > resolution * 2] if resolution else pd.Series([])
    gap_time = (gaps - resolution).sum() / 3600 if len(gaps) > 0 and resolution else 0
    
    # Missing data
    missing_pct = df.drop(columns='_time').isna().mean().mean() * 100
    
    # Completeness score
    expected_rows = time_span * 3600 / resolution if resolution else len(df)
    score = min(100, (len(df) / expected_rows * 100)) * (1 - missing_pct / 100)
    score = max(0, score - (gap_time / time_span * 100)) if time_span > 0 else score
    
    return {
        'file': path.name,
        'rows': len(df),
        'days': time_span / 24,
        'resolution_sec': resolution,
        'completeness': score,
        'gaps': len(gaps),
        'gap_hours': gap_time
    }

def main():
    data_dir = Path("data/raw/1sec")
    csv_files = sorted(data_dir.glob("*.csv"))
    
    print(f"Analyzing {len(csv_files)} files...\n")
    results = [analyze_file(f) for f in csv_files]
    
    # Summary
    summary = pd.DataFrame([{
        'Month': r['file'].replace('samengevoegd_', '').replace('.csv', ''),
        'Rows': r['rows'],
        'Days': f"{r['days']:.1f}",
        'Score': f"{r['completeness']:.1f}%",
        'Gaps': r['gaps']
    } for r in results])
    
    print(summary.to_string(index=False))
    
    # Usable months (score >= 70%, gaps < 48h, days >= 20)
    usable = [r['file'].replace('samengevoegd_', '').replace('.csv', '') 
              for r in results 
              if r['completeness'] >= 70 and r['gap_hours'] < 48 and r['days'] >= 20]
    
    print(f"\n{'='*60}\nUsable months ({len(usable)}): {', '.join(usable) or 'None'}")
    
    # Save results
    with open("data/reports/1sec_data_analysis.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    summary.to_csv("data/reports/1sec_data_summary.csv", index=False)
    print("Results saved to data/")

if __name__ == "__main__":
    main()
