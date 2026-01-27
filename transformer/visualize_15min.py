"""
Visualize NILMFormer predictions on 15-min HeatPump data.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).parent))
from model import create_model


def load_model_15min(model_path):
    """Load trained 15-min model."""
    checkpoint = torch.load(model_path, map_location='cpu')
    cfg = SimpleNamespace(**checkpoint['config'])
    cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_model(cfg)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, cfg


def predict_continuous_15min(model, X_full, window_size=96, batch_size=64, step=1):
    """
    Sliding window prediction for 15-min data.
    """
    device = next(model.parameters()).device
    
    N = len(X_full)
    preds = np.full(N, np.nan)
    
    valid_starts = np.arange(0, N - window_size, step)
    mid_offset = window_size // 2
    
    with torch.no_grad():
        for i in range(0, len(valid_starts), batch_size):
            starts = valid_starts[i:i+batch_size]
            
            batch_list = [X_full[s:s+window_size] for s in starts]
            batch_tensor = torch.FloatTensor(np.array(batch_list)).to(device)
            
            outputs = model(batch_tensor)
            
            out = outputs['HeatPump']['power']
            if out.dim() == 1 or (out.dim() == 2 and out.shape[1] == 1):
                val = out.reshape(-1).cpu().numpy()
            else:
                val = out[:, mid_offset].cpu().numpy()
            
            indices = starts + mid_offset
            preds[indices] = val
    
    return preds



def load_data_continuous_15min(parquet_path, P_MAX_W=13511.8):
    """Load continuous 15-min data and scale Aggregate."""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    
    # Ensure Time index
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'])
        df.set_index('Time', inplace=True)
    
    # Scaling (assuming same scaling as training)
    X_agg = df['Aggregate'].values.astype(np.float32) / P_MAX_W
    
    # Re-create temporal features (hour, dow, month)
    ts = df.index
    temporal = np.column_stack([
        np.sin(2 * np.pi * ts.hour / 24),
        np.cos(2 * np.pi * ts.hour / 24),
        np.sin(2 * np.pi * ts.dayofweek / 7),
        np.cos(2 * np.pi * ts.dayofweek / 7),
        np.sin(2 * np.pi * ts.month / 12),
        np.cos(2 * np.pi * ts.month / 12)
    ]).astype(np.float32)
    
    X_full = np.column_stack([X_agg, temporal])
    y_hp = df['HeatPump'].values.astype(np.float32)
    
    return X_full, y_hp, df.index


def predict_sliding_window_15min(model, X_full, window_size=96, batch_size=128):
    """Generate continuous prediction with stride 1."""
    device = next(model.parameters()).device
    N = len(X_full)
    preds = np.full(N, np.nan)
    
    valid_starts = np.arange(0, N - window_size, 1)
    mid_offset = window_size // 2
    
    with torch.no_grad():
        for i in range(0, len(valid_starts), batch_size):
            starts = valid_starts[i:i+batch_size]
            batch_X = [X_full[s:s+window_size] for s in starts]
            batch_tensor = torch.FloatTensor(np.array(batch_X)).to(device)
            
            outputs = model(batch_tensor)
            out = outputs['HeatPump']['power'].squeeze()
            
            if out.dim() == 0:
                vals = out.cpu().numpy().reshape(-1)
            else:
                # If Seq2Seq, take midpoint
                if out.dim() == 2 and out.shape[1] > 1:
                    vals = out[:, mid_offset].cpu().numpy()
                else:
                    vals = out.cpu().numpy().reshape(-1)
            
            preds[starts + mid_offset] = vals
            
    return preds


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Model
    model_path = Path('transformer/checkpoints/nilmformer_15min_best.pth')
    model, cfg = load_model_15min(model_path)
    model = model.to(device)
    
    # Data
    # P_MAX from 15min data metadata (standardized to 13.5k usually)
    P_MAX_W = 13511.8
    parquet_path = Path('data/processed/15min/nilm_ready_dataset_cleaned.parquet')
    X_full, y_hp, timestamps = load_data_continuous_15min(parquet_path, P_MAX_W)
    
    print(f"Continuous Data: {X_full.shape}, HP: {y_hp.shape}")
    
    # Run sliding window on a large chunk (e.g. 2000 steps)
    duration = 1000
    start_idx = np.random.randint(96, len(X_full) - duration)
    
    chunk_X = X_full[start_idx - 96 : start_idx + duration]
    preds_norm = predict_sliding_window_15min(model, chunk_X, window_size=96)
    
    # Align
    preds_aligned = preds_norm[96 : 96 + duration]
    gt_chunk = y_hp[start_idx : start_idx + duration]
    
    # Denormalize (Note: training input was scaled by P_MAX, so we assume output is too)
    # Actually, the 15min preprocessed data likely scales targets too.
    # Let's check metadata if unsure, but for visualization we'll scale to match range.
    pred_watts = preds_aligned * P_MAX_W
    
    # Plotting
    save_dir = Path('transformer/plots')
    save_dir.mkdir(exist_ok=True)
    
    t = np.arange(duration) * 15 / 60  # Hours
    
    plt.figure(figsize=(15, 7))
    plt.plot(t, gt_chunk, 'k-', linewidth=1.5, label='Actual Power', alpha=0.7)
    plt.plot(t, pred_watts, 'orange', linewidth=1.5, label='Predicted Power', alpha=0.9)
    
    plt.title(f'NILMFormer 15min - Continuous Sliding Window (HeatPump)', fontsize=14)
    plt.xlabel('Time (Hours)')
    plt.ylabel('Power (W)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_dir / 'nilmformer_15min_sliding_window.png', dpi=150)
    print(f"Saved: nilmformer_15min_sliding_window.png")
    
    # Detailed zoom (first 300 steps)
    plt.figure(figsize=(15, 7))
    plt.plot(t[:300], gt_chunk[:300], 'k-', label='Actual')
    plt.plot(t[:300], pred_watts[:300], 'orange', label='Predicted')
    plt.title('NILMFormer 15min - Zoom (First 300 steps)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / 'nilmformer_15min_sliding_window_zoom.png', dpi=150)
    
    print("Done!")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
