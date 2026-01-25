
import pandas as pd
import numpy as np
import os

def generate_test_data(output_path="fixtures/test_data.parquet"):
    """
    Generates a deterministic dataset for E2E testing.
    Format: timestamp (datetime), aggregate_kw (float)
    """
    # 1 hour of data at 1s resolution
    dates = pd.date_range(start="2025-01-01 12:00:00", periods=3600, freq="1s")
    
    # Deterministic pattern: Sine wave + Step
    # 2kW base + 1kW sine wave (period 10 min) + 3kW step at 30 mins
    t = np.arange(3600)
    power = 2000 + 1000 * np.sin(2 * np.pi * t / 600)
    power[1800:] += 3000
    
    df = pd.DataFrame({
        "timestamp": dates,
        "power_total": power, # Watts
        "voltage": 230.0,
        "current": power / 230.0
    })
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Generating {output_path} with {len(df)} rows...")
    df.to_parquet(output_path)
    print("Done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="tests/fixtures/test_data.parquet")
    args = parser.parse_args()
    generate_test_data(args.output)
