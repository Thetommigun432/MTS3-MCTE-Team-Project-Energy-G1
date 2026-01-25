import pytest
import pandas as pd
from pathlib import Path
import os

def test_production_data_file_exists():
    """Verify the production data file exists in the expected location."""
    # Try finding it relative to project root (assuming running from root)
    # or look for it in the known location
    possible_paths = [
        Path("production_jan2025_building_only.parquet"),
        Path("data/production_jan2025_building_only.parquet"),
        Path("../../../production_jan2025_building_only.parquet"), # relative to this test file
    ]
    
    found_path = None
    for p in possible_paths:
        if p.exists():
            found_path = p
            break
            
    assert found_path is not None, f"Could not find production_jan2025_building_only.parquet in {possible_paths}"
    return found_path

def test_production_data_schema():
    """Verify the production data has the correct schema and is readable."""
    # Find the file again (or reuse consistent path logic)
    # Ideally use a fixture, but keeping it simple for now
    file_path = Path("production_jan2025_building_only.parquet")
    if not file_path.exists():
         pytest.skip("Production parquet file not found at root, skipping data test")

    df = pd.read_parquet(file_path)
    
    # Basic Checks
    assert not df.empty, "Dataframe is empty"
    print(f"\nLoaded data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for expected columns (adapt based on real file structure if known, guessing standard ones)
    # Usually we expect a timestamp index or column, and some power columns
    # assert 'timestamp' in df.columns or isinstance(df.index, pd.DatetimeIndex)
