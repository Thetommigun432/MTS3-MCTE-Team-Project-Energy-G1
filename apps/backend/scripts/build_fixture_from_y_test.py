#!/usr/bin/env python3
"""
Generate test fixtures from y_test.npy.

Creates:
- aggregate_kw_sample.npy (N, 1024) - aggregate power windows in kW
- y_midpoint_kw_sample.npy (N, 11) - expected midpoint outputs in kW

Usage:
    python scripts/build_fixture_from_y_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from scripts._asset_finder import find_y_test_npy

# Normalization constant from training (P_MAX_kW)
P_MAX_KW = 13.5118

# Number of samples to extract for fixtures
N_SAMPLES = 20


def main() -> None:
    print("=" * 60)
    print("Fixture Generation from y_test.npy")
    print("=" * 60)

    # 1. Find y_test.npy
    y_path = find_y_test_npy()
    print(f"Loading: {y_path}")

    # 2. Load with mmap to avoid memory issues (file can be large)
    y = np.load(y_path, mmap_mode="r")
    print(f"Shape: {y.shape}, dtype: {y.dtype}")

    # Expected shape: (N, 1024, 11) - normalized values [0, 1]
    if y.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {y.shape}")

    total_samples, window_size, n_appliances = y.shape
    print(f"Total samples: {total_samples}")
    print(f"Window size: {window_size}")
    print(f"Appliances: {n_appliances}")

    # 3. Extract subset
    n_extract = min(N_SAMPLES, total_samples)
    y_subset = np.array(y[:n_extract])  # Copy to regular array
    print(f"Extracting {n_extract} samples")

    # 4. Compute aggregate (sum across appliances) and convert to kW
    # y_subset shape: (N, 1024, 11) - normalized per-appliance power
    aggregate_norm = y_subset.sum(axis=-1)  # (N, 1024)
    aggregate_kw = aggregate_norm * P_MAX_KW

    print(f"Aggregate range: [{aggregate_kw.min():.3f}, {aggregate_kw.max():.3f}] kW")

    # 5. Extract midpoint outputs (model is seq2point, predicts center)
    midpoint_idx = window_size // 2
    y_midpoint_norm = y_subset[:, midpoint_idx, :]  # (N, 11)
    y_midpoint_kw = y_midpoint_norm * P_MAX_KW

    print(f"Midpoint outputs shape: {y_midpoint_kw.shape}")
    print(f"Midpoint range: [{y_midpoint_kw.min():.3f}, {y_midpoint_kw.max():.3f}] kW")

    # 6. Save fixtures
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    aggregate_path = fixtures_dir / "aggregate_kw_sample.npy"
    midpoint_path = fixtures_dir / "y_midpoint_kw_sample.npy"

    np.save(aggregate_path, aggregate_kw.astype(np.float32))
    np.save(midpoint_path, y_midpoint_kw.astype(np.float32))

    print(f"\nSaved fixtures to: {fixtures_dir}")
    print(f"  {aggregate_path.name}: shape {aggregate_kw.shape}")
    print(f"  {midpoint_path.name}: shape {y_midpoint_kw.shape}")

    # 7. Quick sanity check
    print("\nSanity check (first sample):")
    print(f"  Aggregate sum: {aggregate_kw[0].sum():.3f} kW (over {window_size} timesteps)")
    print(f"  Midpoint per-appliance: {y_midpoint_kw[0]}")

    print("=" * 60)
    print("SUCCESS")
    print("=" * 60)


if __name__ == "__main__":
    main()
