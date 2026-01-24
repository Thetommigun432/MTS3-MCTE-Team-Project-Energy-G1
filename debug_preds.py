"""Debug model predictions."""
import torch
import numpy as np
import json
from pathlib import Path
import sys

# Import directly without __init__
sys.path.insert(0, 'transformer')
from nilmformer_sgn import create_nilmformer_sgn

# Load data
data_path = Path('data/processed/1sec_new/model_ready')
X_val = np.load(data_path / 'X_val.npy')
y_val = np.load(data_path / 'y_val.npy')
with open(data_path / 'metadata.json') as f:
    metadata = json.load(f)
appliances = metadata['target_appliances']

print(f"X_val: {X_val.shape}")
print(f"y_val: {y_val.shape}")

# Create model
model = create_nilmformer_sgn(appliances, c_embedding=6)

# Quick forward pass
x = torch.FloatTensor(X_val[:4])
with torch.no_grad():
    outputs = model(x)

print('\n=== Model Prediction Ranges (UNTRAINED) ===')
for app in ['HeatPump', 'GarageCabinet', 'Dishwasher']:
    idx = appliances.index(app)
    pred_gated = outputs[app]['gated'].numpy()
    pred_power = outputs[app]['power'].numpy()
    state_logits = outputs[app]['state_logits'].numpy()
    gate = 1 / (1 + np.exp(-state_logits))  # sigmoid
    
    target = y_val[:4, :, idx]
    
    print(f'\n{app}:')
    print(f'  Power (pre-gate): min={pred_power.min():.6f}, max={pred_power.max():.6f}, mean={pred_power.mean():.6f}')
    print(f'  Gate sigmoid: min={gate.min():.4f}, max={gate.max():.4f}, mean={gate.mean():.4f}')
    print(f'  Gated (final): min={pred_gated.min():.6f}, max={pred_gated.max():.6f}, mean={pred_gated.mean():.6f}')
    print(f'  Target:         min={target.min():.6f}, max={target.max():.6f}, mean={target.mean():.6f}')
    
    # Compare scale
    scale_ratio = target.mean() / (pred_gated.mean() + 1e-10)
    print(f'  --> Target/Pred ratio: {scale_ratio:.2f}x')

# Also check a trained model if exists
ckpt_path = Path('transformer/checkpoints/sgn_best.pth')
if ckpt_path.exists():
    print('\n\n=== Model Prediction Ranges (TRAINED) ===')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    
    with torch.no_grad():
        outputs = model(x)
    
    for app in ['HeatPump', 'GarageCabinet', 'Dishwasher']:
        idx = appliances.index(app)
        pred_gated = outputs[app]['gated'].numpy()
        pred_power = outputs[app]['power'].numpy()
        
        target = y_val[:4, :, idx]
        
        # Threshold
        thresh = 0.007 if app == 'HeatPump' else 0.002 if app == 'Dishwasher' else 0.002
        
        pred_on = (pred_gated.flatten() > thresh).sum()
        true_on = (target.flatten() > thresh).sum()
        
        print(f'\n{app}:')
        print(f'  Pred gated: min={pred_gated.min():.6f}, max={pred_gated.max():.6f}, mean={pred_gated.mean():.6f}')
        print(f'  Target:     min={target.min():.6f}, max={target.max():.6f}, mean={target.mean():.6f}')
        print(f'  Pred ON (>{thresh:.4f}): {pred_on}')
        print(f'  True ON (>{thresh:.4f}): {true_on}')
else:
    print('\nNo trained model found at', ckpt_path)
