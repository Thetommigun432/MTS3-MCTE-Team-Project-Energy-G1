#!/usr/bin/env python3
"""
Test Inference Flow - Standalone test without Redis
====================================================

Simulates the complete production flow:
1. Read parquet data
2. Preprocess (same as training)
3. Buffer to window
4. Run inference
5. Show predictions

Usage:
    python test_inference_flow.py --samples 2000
    python test_inference_flow.py --samples 2000 --appliance HeatPump
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nilm.causal.wavenilm_v3 import WaveNILM_v3


class DataPreprocessor:
    """
    Preprocesses raw sensor data into model-ready features.
    EXACT COPY of training preprocessing!
    
    Features (7 total):
        0: Aggregate (normalized by P_MAX)
        1: hour_sin
        2: hour_cos
        3: dow_sin
        4: dow_cos
        5: month_sin
        6: month_cos
    
    NOTE: P_MAX is in kW (from metadata), so we convert input W to kW first!
    """
    
    def __init__(self, P_MAX_kw: float):
        """
        Args:
            P_MAX_kw: Maximum power in kW (from training metadata)
        """
        self.P_MAX = P_MAX_kw  # kW
        print(f"  Preprocessor initialized with P_MAX={P_MAX_kw:.2f} kW")
        
    def process_sample(self, power_watts: float, timestamp: float) -> np.ndarray:
        """Process a single raw sample into 7-feature vector."""
        # 1. Convert W to kW, then normalize: same as training!
        power_kw = power_watts / 1000.0
        aggregate_norm = np.clip(power_kw / self.P_MAX, 0, 1)
        
        # 2. Time features
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        day_of_week = dt.weekday()
        month = dt.month - 1 + dt.day / 31.0
        
        # 3. Cyclical encoding
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin = np.sin(2 * np.pi * day_of_week / 7)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        return np.array([
            aggregate_norm, hour_sin, hour_cos,
            dow_sin, dow_cos, month_sin, month_cos
        ], dtype=np.float32)


def load_model(checkpoint_path: str, n_features: int = 7, device: str = 'cuda'):
    """Load trained WaveNILM model - auto-detect architecture from checkpoint."""
    print(f"  Loading model: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract architecture from checkpoint
    hidden_channels = ckpt.get('hidden_channels', 64)
    n_blocks = ckpt.get('n_blocks', 9)
    
    # Count actual blocks in state_dict to determine n_stacks
    state = ckpt['model']
    actual_blocks = sum(1 for k in state if k.startswith('blocks.') and '.conv.conv.weight' in k)
    
    # n_stacks = actual_blocks / n_blocks (or 1 if they match)
    n_stacks = max(1, actual_blocks // n_blocks) if n_blocks > 0 else 1
    
    print(f"  Architecture: hidden={hidden_channels}, n_blocks={n_blocks}, actual_blocks={actual_blocks}, n_stacks={n_stacks}")
    
    model = WaveNILM_v3(
        n_input_features=n_features,
        hidden_channels=hidden_channels,
        n_blocks=actual_blocks,  # Use actual block count!
        n_stacks=1,  # Don't duplicate blocks
        use_attention=False,
        use_mtl=True,
        dropout=0.15
    )
    
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    
    print(f"  ✓ Model loaded (epoch {ckpt.get('epoch', '?')}, F1={ckpt.get('f1', 0):.4f})")
    return model


def main():
    parser = argparse.ArgumentParser(description='Test NILM Inference Flow')
    parser.add_argument('--parquet', type=str, 
                        default='production_jan2025_building_only.parquet')
    parser.add_argument('--checkpoint-dir', type=str, default='../checkpoints')
    parser.add_argument('--metadata', type=str, 
                        default='/home/jovyan/mts3-mcte-team-project-g1-model-v1-datavol-1/data/processed/1sec_new/model_ready_1536/metadata.pkl')
    parser.add_argument('--appliance', type=str, default='HeatPump',
                        choices=['HeatPump', 'WashingMachine', 'Dishwasher'])
    parser.add_argument('--samples', type=int, default=2000,
                        help='Number of samples to process')
    parser.add_argument('--skip', type=int, default=0,
                        help='Skip first N samples')
    parser.add_argument('--window', type=int, default=1536)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("  NILM INFERENCE FLOW TEST")
    print("="*70)
    print(f"Device: {device}")
    
    # =========================================================================
    # 1. LOAD METADATA
    # =========================================================================
    print(f"\n📊 Loading metadata...")
    with open(args.metadata, 'rb') as f:
        meta = pickle.load(f)
    
    P_MAX = meta['scaling']['P_MAX']  # In kW!
    P_MAX_unit = meta['scaling'].get('P_MAX_unit', 'kW')
    appliance_idx = meta['appliance_indices'][args.appliance]
    print(f"  P_MAX: {P_MAX:.2f} {P_MAX_unit}")
    print(f"  Appliance: {args.appliance} (idx={appliance_idx})")
    
    # =========================================================================
    # 2. LOAD MODEL
    # =========================================================================
    print(f"\n🧠 Loading model...")
    script_dir = Path(__file__).parent
    ckpt_dir = script_dir / args.checkpoint_dir
    
    # Find checkpoint
    ckpt_pattern = f"wavenilm_v3_SOTA_{args.appliance}_best.pth"
    ckpt_path = ckpt_dir / ckpt_pattern
    
    if not ckpt_path.exists():
        print(f"  ERROR: Checkpoint not found: {ckpt_path}")
        print(f"  Available checkpoints:")
        for f in ckpt_dir.glob("*.pth"):
            print(f"    - {f.name}")
        return 1
    
    model = load_model(str(ckpt_path), n_features=7, device=str(device))
    
    # =========================================================================
    # 3. LOAD PARQUET DATA
    # =========================================================================
    print(f"\n📁 Loading parquet data...")
    parquet_path = script_dir / args.parquet
    df = pd.read_parquet(parquet_path)
    print(f"  Total rows: {len(df):,}")
    print(f"  Columns: {df.columns.tolist()}")
    
    # =========================================================================
    # 4. INITIALIZE PREPROCESSOR
    # =========================================================================
    print(f"\n⚙️ Initializing preprocessor...")
    preprocessor = DataPreprocessor(P_MAX_kw=P_MAX)
    
    # =========================================================================
    # 5. PROCESS DATA AND FILL BUFFER
    # =========================================================================
    print(f"\n📊 Processing {args.samples} samples (skip={args.skip})...")
    
    buffer = []  # Will hold feature vectors
    powers = []  # Will hold raw power values
    
    # Determine columns
    time_col = 'Time' if 'Time' in df.columns else 'timestamp'
    power_col = 'Aggregate' if 'Aggregate' in df.columns else 'power_total'
    
    # Check if kW or W
    is_kw = df[power_col].max() < 100
    print(f"  Power unit: {'kW' if is_kw else 'W'}")
    
    # Process samples
    start_idx = args.skip
    end_idx = min(args.skip + args.samples, len(df))
    
    for i in range(start_idx, end_idx):
        row = df.iloc[i]
        
        # Get timestamp
        ts = row[time_col].timestamp() if hasattr(row[time_col], 'timestamp') else float(row[time_col])
        
        # Get power in WATTS
        power_w = row[power_col] * 1000 if is_kw else row[power_col]
        
        # Preprocess
        features = preprocessor.process_sample(power_w, ts)
        buffer.append(features)
        powers.append(power_w)
        
        # Progress
        if (i - start_idx + 1) % 500 == 0:
            print(f"  Processed {i - start_idx + 1:,} / {args.samples:,}")
    
    print(f"  ✓ Buffer filled: {len(buffer)} samples")
    
    # =========================================================================
    # 6. RUN INFERENCE
    # =========================================================================
    print(f"\n🔮 Running inference...")
    
    if len(buffer) < args.window:
        print(f"  ERROR: Not enough samples ({len(buffer)} < {args.window})")
        return 1
    
    # Use last window_size samples
    window_features = np.array(buffer[-args.window:])
    print(f"  Window shape: {window_features.shape}")
    print(f"  Features range: [{window_features.min():.4f}, {window_features.max():.4f}]")
    
    # Check feature validity
    print(f"\n  Feature check:")
    feature_names = ['Aggregate', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
    for i, name in enumerate(feature_names):
        col = window_features[:, i]
        print(f"    {name}: min={col.min():.4f}, max={col.max():.4f}, mean={col.mean():.4f}")
    
    # Prepare tensor
    x = torch.from_numpy(window_features).float().unsqueeze(0).to(device)
    print(f"  Input tensor shape: {x.shape}")
    
    # Inference
    with torch.no_grad():
        with torch.autocast(str(device), dtype=torch.float16, enabled=device.type=='cuda'):
            power_pred, prob_pred = model(x)
    
    # Extract prediction (model outputs single value per window)
    # Shape: (batch, 1) -> scalar
    power_norm = power_pred[0, 0].item()
    prob = prob_pred[0, 0].item()
    
    # Denormalize power: model output * P_MAX (kW) * 1000 = Watts
    power_watts = power_norm * P_MAX * 1000
    
    # =========================================================================
    # 7. SHOW RESULTS
    # =========================================================================
    print(f"\n" + "="*70)
    print(f"  PREDICTION RESULTS - {args.appliance}")
    print("="*70)
    
    # Context
    avg_power = np.mean(powers[-60:]) if len(powers) >= 60 else np.mean(powers)
    max_power = np.max(powers[-60:]) if len(powers) >= 60 else np.max(powers)
    
    print(f"\n📊 Context (last minute):")
    print(f"   Avg building power: {avg_power:.0f} W")
    print(f"   Max building power: {max_power:.0f} W")
    
    print(f"\n🔮 Prediction:")
    print(f"   {args.appliance} Power: {power_watts:.1f} W")
    print(f"   {args.appliance} Probability: {prob:.4f} ({prob*100:.1f}%)")
    print(f"   Status: {'🟢 ON' if prob > 0.5 else '⚫ OFF'}")
    
    # Interpretation
    print(f"\n📝 Interpretation:")
    if prob > 0.5:
        print(f"   The model predicts {args.appliance} is RUNNING")
        print(f"   Consuming approximately {power_watts:.0f} W")
    else:
        print(f"   The model predicts {args.appliance} is OFF")
        if power_watts > 50:
            print(f"   (Residual power estimate: {power_watts:.0f} W, likely noise)")
    
    print("\n✅ Test completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
