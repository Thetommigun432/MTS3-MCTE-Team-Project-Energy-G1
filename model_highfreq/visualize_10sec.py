
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import sys
import os
# Force append the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from train_transformer_10sec import NILMTransformer, Config, load_data, NILMDataset10Sec

# Reuse Config and Model from training script
def main():
    appliance = 'Dishwasher'
    print(f"Vizualizing {appliance} (10sec)...")
    
    cfg = Config()
    checkpoint_path = cfg.SAVE_PATH / f'transformer_10sec_{appliance.lower()}_best.pth'
    
    # Load Data
    X, y = load_data(cfg.DATA_PATH, appliance)
    
    # Load Checkpoint -> Scalers
    checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE, weights_only=False)
    scaler_X = checkpoint['scaler_X']
    scaler_y = checkpoint['scaler_y']
    
    # Scale
    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten()
    
    # Test Split (Last 15%)
    n = len(X)
    val_end = int(n * 0.85)
    X_test = X_scaled[val_end:]
    y_test = y_scaled[val_end:]
    
    # Dataset
    test_ds = NILMDataset10Sec(X_test, y_test, cfg.WINDOW_SIZE)
    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    # Model
    model = NILMTransformer(n_features=X.shape[1], window_size=cfg.WINDOW_SIZE).to(cfg.DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    preds = []
    actuals = []
    
    print("Running Inference...")
    with torch.no_grad():
        for i, (bx, by) in enumerate(loader):
            bx = bx.to(cfg.DEVICE)
            out = model(bx)
            
            p_inv = scaler_y.inverse_transform(out.cpu().numpy()).flatten()
            y_inv = scaler_y.inverse_transform(by.numpy()).flatten()
            
            preds.extend(p_inv)
            actuals.extend(y_inv)
            
    preds = np.array(preds)
    actuals = np.array(actuals)
    preds = np.maximum(preds, 0)
    
    # Find active chunk
    chunk_size = 400 # 400 * 10s = 4000s = ~1 hour
    best_idx = 0
    max_activity = 0
    
    # Search for max activity in actuals
    for i in range(0, len(actuals) - chunk_size, chunk_size // 2):
        activity = np.sum(actuals[i:i+chunk_size])
        if activity > max_activity:
            max_activity = activity
            best_idx = i
            
    # Polting
    start = best_idx
    end = start + chunk_size
    
    plt.figure(figsize=(12, 5))
    plt.plot(actuals[start:end], label='Actual (10s)', color='black', alpha=0.8)
    plt.plot(preds[start:end], label='Predicted (Transformer)', color='tab:red', alpha=0.8, linewidth=2)
    
    plt.title(f"Dishwasher High-Precision (10sec) - Transformer\nWindow: 1 Hour snapshot")
    plt.ylabel("Power (kW)")
    plt.xlabel("Time Steps (10s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = Path(r'C:\Users\gamek\.gemini\antigravity\brain\80ceaaca-73c7-48e9-81fc-98ecd37ccaa4') / f'viz_10sec_{appliance}.png'
    plt.savefig(save_path)
    print(f"Saved to {save_path}")

if __name__ == '__main__':
    main()
