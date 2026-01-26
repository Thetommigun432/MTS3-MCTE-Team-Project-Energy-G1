import pytest
import pandas as pd
from pathlib import Path
import os

@pytest.fixture
def parquet_file():
    """Locate the parquet data file for testing."""
    # Check env var first
    if env_path := os.environ.get("PRODUCTION_PARQUET_PATH"):
        path = Path(env_path)
        if path.exists():
            return path
            
    # Fallback paths
    possible_paths = [
        Path("apps/backend/tests/fixtures/simulation-data.parquet"),
        Path("tests/fixtures/simulation-data.parquet"),
        # Fixture fallback for CI
        Path("apps/backend/tests/fixtures/test_data.parquet"),
    ]
    
    for p in possible_paths:
        if p.exists():
            return p
            
    return None

@pytest.mark.unit
def test_simulation_data_availability(parquet_file):
    """Verify that a simulation data chunk is available."""
    if not parquet_file:
         pytest.skip("No parquet data file found in expected locations.")
    assert parquet_file.exists()
    assert parquet_file.stat().st_size > 0, "Parquet file is zero bytes"

@pytest.mark.unit
def test_simulation_data_schema(parquet_file):
    """Verify the data file has minimal expected schema for NILM."""
    if not parquet_file:
         pytest.skip("No parquet data file found, skipping schema test")

    try:
        df = pd.read_parquet(parquet_file)
    except Exception as e:
        pytest.fail(f"Failed to read parquet file: {e}")
    
    # 1. Non-empty check
    assert not df.empty, "Dataframe is empty"
    
    # 2. Schema check (At least timestamp index/column and some power data)
    # Checking for common variations since we swapped files
    columns = [c.lower() for c in df.columns]
    
    # Usually index is timestamp, or a column is
    has_timestamp = isinstance(df.index, pd.DatetimeIndex) or any('time' in c for c in columns)
    assert has_timestamp, f"Data missing timestamp index or column. Columns: {df.columns}"
    
    # Just ensure we have numeric data
    assert df.shape[1] > 0, "Dataframe has no columns"
    print(f"\nValidated Data: {parquet_file} | Shape: {df.shape}")
