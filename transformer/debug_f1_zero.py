"""Debug why F1 is always 0 in training."""

import numpy as np
import json
from pathlib import Path

# Load data
data_path = Path("data/processed/1sec_new/model_ready")
with open(data_path / 'metadata.json') as f:
    metadata = json.load(f)

y_train = np.load(data_path / 'y_train.npy')
P_MAX_kW = metadata.get('scaling', {}).get('P_MAX', 13.5118)
appliances = metadata['target_appliances']

print(f"y_train shape: {y_train.shape}")
print(f"P_MAX: {P_MAX_kW} kW")
print(f"Appliances: {appliances}")

# Check thresholds
print("\n=== Threshold Analysis ===")
for i, app in enumerate(appliances):
    app_data = y_train[:, :, i].flatten()
    
    # Thresholds in normalized and Watts
    thresholds_W = {
        'HeatPump': 100, 'Dishwasher': 30, 'WashingMachine': 50,
        'Dryer': 50, 'Oven': 100, 'Stove': 50, 'RangeHood': 20,
        'EVCharger': 100, 'EVSocket': 100, 'GarageCabinet': 25, 'RainwaterPump': 50
    }
    
    thresh_W = thresholds_W.get(app, 50)
    thresh_norm = thresh_W / (P_MAX_kW * 1000)
    
    # Stats
    max_val = np.max(app_data)
    mean_val = np.mean(app_data)
    on_count = np.sum(app_data > thresh_norm)
    on_pct = 100 * on_count / len(app_data)
    
    # Convert to Watts for clarity
    max_W = max_val * P_MAX_kW * 1000
    mean_W = mean_val * P_MAX_kW * 1000
    
    print(f"\n{app}:")
    print(f"  Threshold: {thresh_W}W (normalized: {thresh_norm:.6f})")
    print(f"  Data range: 0 to {max_W:.1f}W (normalized: 0 to {max_val:.6f})")
    print(f"  Mean: {mean_W:.1f}W (normalized: {mean_val:.6f})")
    print(f"  ON samples: {on_count:,} ({on_pct:.2f}%)")
    
    # CRITICAL CHECK: is the threshold > max value?
    if thresh_norm > max_val:
        print(f"  ⚠️ THRESHOLD > MAX VALUE! All samples will be classified as OFF!")
    elif on_pct < 0.1:
        print(f"  ⚠️ Very low activity - F1 may be unstable")
    else:
        print(f"  ✓ OK")

# Also check if data is in correct range
print("\n=== Data Range Check ===")
print(f"y_train min: {y_train.min():.6f}")
print(f"y_train max: {y_train.max():.6f}")
print(f"y_train mean: {y_train.mean():.6f}")

if y_train.max() < 0.01:
    print("\n⚠️ DATA VALUES ARE VERY SMALL!")
    print("Model predictions might be reasonable but thresholds are too high")
