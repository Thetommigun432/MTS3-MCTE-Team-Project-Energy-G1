"""
Diagnose Training Data for NILMFormer
======================================
Verifies appliance activity rates and identifies class imbalance issues.

Usage:
    python diagnose_training_data.py
    python diagnose_training_data.py --data_path path/to/model_ready
"""

import numpy as np
import json
from pathlib import Path
import argparse


def diagnose_training_data(data_path: Path):
    """Analyze training data for class imbalance issues."""
    
    print("=" * 70)
    print("NILMFormer Training Data Diagnosis")
    print("=" * 70)
    print(f"Data path: {data_path}")
    
    # Load metadata
    metadata_path = data_path / "metadata.json"
    if not metadata_path.exists():
        print(f"ERROR: metadata.json not found at {metadata_path}")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    appliances = metadata.get('target_appliances', [])
    scaling = metadata.get('scaling', {})
    P_MAX_kW = scaling.get('P_MAX', 13.5118)
    
    print(f"\nAppliances: {len(appliances)}")
    print(f"P_MAX: {P_MAX_kW} kW")
    
    # Load training targets
    print("\nLoading training data...")
    y_train = np.load(data_path / "y_train.npy")
    y_val = np.load(data_path / "y_val.npy")
    y_test = np.load(data_path / "y_test.npy")
    
    print(f"  y_train: {y_train.shape}")
    print(f"  y_val: {y_val.shape}")
    print(f"  y_test: {y_test.shape}")
    
    # Combine for full analysis
    y_all = np.concatenate([y_train, y_val, y_test], axis=0)
    n_samples, seq_len, n_appliances = y_all.shape
    total_timesteps = n_samples * seq_len
    
    print(f"\nTotal samples: {n_samples:,}")
    print(f"Total timesteps: {total_timesteps:,}")
    
    # ON thresholds in normalized units
    ON_THRESHOLDS_WATTS = {
        'HeatPump': 100,
        'Dishwasher': 30,
        'WashingMachine': 50,
        'Dryer': 50,
        'Oven': 100,
        'Stove': 50,
        'RangeHood': 20,
        'EVCharger': 100,
        'EVSocket': 100,
        'GarageCabinet': 25,
        'RainwaterPump': 50,
    }
    
    # Analysis per appliance
    print("\n" + "=" * 70)
    print("Per-Appliance Activity Analysis")
    print("=" * 70)
    print(f"{'Appliance':<18} {'Activity%':>10} {'ON Count':>12} {'Mean(W)':>10} {'Max(W)':>10} {'Status':>10}")
    print("-" * 70)
    
    issues = []
    
    for i, app in enumerate(appliances):
        app_data = y_all[:, :, i].flatten()  # (total_timesteps,)
        
        # Convert to Watts
        app_watts = app_data * P_MAX_kW * 1000
        
        # Get ON threshold
        threshold_W = ON_THRESHOLDS_WATTS.get(app, 10)
        threshold_norm = threshold_W / (P_MAX_kW * 1000)
        
        # Count ON samples
        on_mask = app_data > threshold_norm
        on_count = np.sum(on_mask)
        activity_pct = 100 * on_count / total_timesteps
        
        # Statistics
        mean_W = np.mean(app_watts)
        max_W = np.max(app_watts)
        mean_on_W = np.mean(app_watts[on_mask]) if on_count > 0 else 0
        
        # Status
        if activity_pct < 1:
            status = "❌ SPARSE"
            issues.append((app, activity_pct, "< 1% activity - will likely fail"))
        elif activity_pct < 5:
            status = "⚠️ LOW"
            issues.append((app, activity_pct, "< 5% activity - may struggle"))
        elif activity_pct < 20:
            status = "OK"
        else:
            status = "✅ GOOD"
        
        print(f"{app:<18} {activity_pct:>9.2f}% {on_count:>12,} {mean_W:>10.1f} {max_W:>10.1f} {status:>10}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    if issues:
        print("\n⚠️  PROBLEMATIC APPLIANCES:")
        for app, pct, reason in sorted(issues, key=lambda x: x[1]):
            print(f"   - {app}: {pct:.2f}% ({reason})")
        
        print("\n💡 RECOMMENDATIONS:")
        print("   1. Use sparsity-aware loss: python train_nilmformer_paper.py --loss sparsity")
        print("   2. Consider removing appliances with <1% activity from training")
        print("   3. Or use per-appliance models for rare appliances")
    else:
        print("✅ All appliances have sufficient activity for training")
    
    # Save report
    report = {
        'data_path': str(data_path),
        'n_samples': n_samples,
        'total_timesteps': total_timesteps,
        'appliances': {},
    }
    
    for i, app in enumerate(appliances):
        app_data = y_all[:, :, i].flatten()
        threshold_W = ON_THRESHOLDS_WATTS.get(app, 10)
        threshold_norm = threshold_W / (P_MAX_kW * 1000)
        on_count = np.sum(app_data > threshold_norm)
        
        report['appliances'][app] = {
            'on_threshold_W': threshold_W,
            'on_count': int(on_count),
            'activity_pct': float(100 * on_count / total_timesteps),
            'mean_W': float(np.mean(app_data) * P_MAX_kW * 1000),
            'max_W': float(np.max(app_data) * P_MAX_kW * 1000),
        }
    
    report_path = data_path / "activity_diagnosis.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Report saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diagnose training data')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to model_ready folder')
    args = parser.parse_args()
    
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        # Default paths to check
        candidates = [
            Path.cwd() / "model_ready",
            Path.cwd() / "data" / "processed" / "1sec_new" / "model_ready",
            Path(__file__).parent.parent / "data" / "processed" / "1sec_new" / "model_ready",
        ]
        
        data_path = None
        for c in candidates:
            if c.exists():
                data_path = c
                break
        
        if data_path is None:
            print("ERROR: Could not find model_ready folder")
            print("Please specify --data_path")
            exit(1)
    
    diagnose_training_data(data_path)
