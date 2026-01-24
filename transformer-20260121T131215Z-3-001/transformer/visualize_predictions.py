"""
Visualize model predictions vs ground truth for NILM Transformer.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

# Add transformer to path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
from types import SimpleNamespace
from config import Config
from model import HybridCNNTransformer, create_model
from dataset import load_pretrained_data


def compute_derivative_features(X: np.ndarray) -> np.ndarray:
    """
    Add derivative features (dP/dt, rolling_mean, rolling_std) to input data.
    
    Args:
        X: Input data (n_samples, window_size, n_features)
        
    Returns:
        X_augmented: Input with derivative features (n_samples, window_size, n_features + 3)
    """
    n_samples, window_size, n_features = X.shape
    X_augmented = np.zeros((n_samples, window_size, n_features + 3), dtype=np.float32)
    X_augmented[:, :, :n_features] = X
    
    for i in range(n_samples):
        agg = X[i, :, 0]  # First feature is aggregate power
        
        # dP/dt (derivative)
        dP_dt = np.zeros(window_size, dtype=np.float32)
        dP_dt[1:] = agg[1:] - agg[:-1]
        
        # Rolling mean (window=8 for 5sec data = 40 seconds)
        window = 8
        rolling_mean = np.convolve(agg, np.ones(window)/window, mode='same').astype(np.float32)
        
        # Rolling std
        rolling_std = np.zeros(window_size, dtype=np.float32)
        for j in range(window, window_size):
            rolling_std[j] = np.std(agg[j-window:j])
        rolling_std[:window] = rolling_std[window] if window < window_size else 0
        
        X_augmented[i, :, n_features] = dP_dt
        X_augmented[i, :, n_features + 1] = rolling_mean
        X_augmented[i, :, n_features + 2] = rolling_std
    
    return X_augmented




def load_continuous_data(data_path, appliances, P_MAX_W=13511.8):
    """
    Load continuous data from parquet and prepare for inference.
    Replicates the preprocessing steps (P_MAX scaling) used for training.
    """
    import pandas as pd
    
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    


    # Ensure timestamp index
    if 'timestamp' in df.columns:
        df.index = pd.to_datetime(df['timestamp'])
    elif 'Time' in df.columns:
        df.index = pd.to_datetime(df['Time'])
    elif not isinstance(df.index, pd.DatetimeIndex):
        # Try converting index to datetime if it's not already
        df.index = pd.to_datetime(df.index)
        
    print(f"Index type: {type(df.index)}")
    
    # Select a chunk of data (e.g., last 2 days for testing)
    # Assuming data is sorted. 
    # taking last 50,000 points (approx 14 hours at 1sec, or more if 5sec? 
    # Wait, the model is trained on 1-sec or 5-sec?
    # Config says resolution_sec = 1.
    # 50,000 seconds is ~14 hours. Good enough for visualization.
    df_test = df.iloc[-100000:].copy() # Last ~27 hours
    
    # 1. Scaling (Normalize by P_MAX)
    print(f"Scaling data by P_MAX = {P_MAX_W} W")
    
    # Aggregate
    if 'Aggregate' in df_test.columns:
        agg_col = 'Aggregate'
    elif 'aggregate' in df_test.columns:
        agg_col = 'aggregate'
    else:
        raise ValueError("Aggregate column not found")
        
    X_continuous = df_test[agg_col].values.astype(np.float32) / P_MAX_W
    
    # Temporal features
    # Re-create temporal features as in dataset.py
    ts = df_test.index

    temporal = np.column_stack([
        np.sin(2 * np.pi * ts.hour / 24),
        np.cos(2 * np.pi * ts.hour / 24),
        # np.sin(2 * np.pi * ts.minute / 60), # Removed to match likely training features (Agg+6temp+3deriv=10)
        # np.cos(2 * np.pi * ts.minute / 60),
        np.sin(2 * np.pi * ts.dayofweek / 7),
        np.cos(2 * np.pi * ts.dayofweek / 7),
        np.sin(2 * np.pi * ts.month / 12),
        np.cos(2 * np.pi * ts.month / 12)
    ]).astype(np.float32)
    
    # Combine Aggregate + Temporal
    # Shape: (n_samples, 1 + n_temporal)
    X_combined = np.column_stack([X_continuous, temporal])
    
    # Ground Truth Targets
    y_continuous = {}
    for app in appliances:
        if app in df_test.columns:
            y_continuous[app] = df_test[app].values.astype(np.float32) # Keep in Watts for comparing? 
            # Or normalize to match model output then denormalize? 
            # Let's keep GT in Watts for plotting ease, but model predicts normalized.
            pass
        else:
            print(f"Warning: {app} not in dataframe")
            y_continuous[app] = np.zeros(len(df_test))
            
    return X_combined, y_continuous, df_test.index, X_continuous # Return X_agg separately for plotting

def predict_sliding_window(model, X_full, window_size=512, step_size=16, batch_size=64, appliance_idx=0):
    """
    Perform sliding window inference.
    
    Args:
        X_full: (N, n_features) continuous input
        window_size: Model input window size
        step_size: Stride for sliding window (1 = dense, >1 = faster approx)
        appliance_idx: Index of appliance output to keep
        
    Returns:
        predictions: (N,) array of predictions (aligned with center of windows)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    N = len(X_full)
    # We can only predict where we have a full window
    # Valid indices for window START: 0 to N - window_size
    valid_starts = np.arange(0, N - window_size, step_size)
    
    # Midpoint offset
    mid_offset = window_size // 2
    
    pred_values = np.zeros(N) # Initialize with zeros
    counts = np.zeros(N)
    
    # Batch processing
    with torch.no_grad():
        for i in range(0, len(valid_starts), batch_size):
            batch_starts = valid_starts[i : i+batch_size]
            
            # Prepare batch
            batch_X = []
            for start in batch_starts:
                batch_X.append(X_full[start : start + window_size])
            
            batch_X = torch.FloatTensor(np.array(batch_X)).to(device)
            
            # Predict
            outputs = model(batch_X)
            
            # Extract HeatPump (or specific appliance) prediction
            # Model output dict keys depend on training. Assuming single target or list.
            # We need the key corresponding to appliance_idx. 
            # But the model outputs a dict keyed by name.
            # We assume the caller knows the name or we handle it.
            pass # TODO: Handle appliance name lookup inside
            
            # For now, return the raw dictionary list effectively? 
            # No, let's assume we extract the first key if not specified, or pass name
            
    return valid_starts # Placeholder return, logic implemented in main flow

def predict_continuous(model, X_full, appliance_name, window_size=512, batch_size=128, step=1):
    """
    Generate continuous prediction for a specific appliance.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    N = len(X_full)
    preds = np.full(N, np.nan) # Initialize with NaNs
    
    valid_starts = np.arange(0, N - window_size, step)
    mid_offset = window_size // 2
    
    # Add derivative features if needed (assuming logic matches training)
    # If trained with derivatives, X_full needs them.
    # Our load_continuous_data only added Aggregate+Temporal. 
    # We should add derivatives here if model expects them.
    # Check model features first? handled in main.
    
    with torch.no_grad():
        for i in range(0, len(valid_starts), batch_size):
            starts = valid_starts[i:i+batch_size]
            
            batch_list = [X_full[s:s+window_size] for s in starts]
            batch_tensor = torch.FloatTensor(np.array(batch_list)).to(device)
            
            outputs = model(batch_tensor)
            
            # Get appliance output
            if appliance_name in outputs:
                out = outputs[appliance_name]['power']
                # Seq2Point: (B, 1) or (B,) -> take value
                # Seq2Seq: (B, L) -> take midpoint
                if out.dim() == 1 or (out.dim() == 2 and out.shape[1] == 1):
                    val = out.reshape(-1).cpu().numpy()
                else:
                    val = out[:, mid_offset].cpu().numpy()
                
                # Assign to midpoint index
                # indices = starts + mid_offset
                # preds[indices] = val
                
                # Slicing assignment
                indices = starts + mid_offset
                preds[indices] = val
            else:
                pass # Key missing
                
    return preds

def load_model_custom(model_path):
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    config_dict = checkpoint['config']
    cfg = SimpleNamespace(**config_dict)
    cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Compatibility
    if not hasattr(cfg, 'input_features'):
        cfg.input_features = config_dict.get('n_features', 7) # Default 7
        
    model = create_model(cfg)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--appliance', type=str, default='HeatPump', help='Appliance to predict')
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pth file')

    parser.add_argument('--duration', type=int, default=2000, help='Number of steps to visualize')
    parser.add_argument('--offset', type=int, default=0, help='Offset from end of data')
    parser.add_argument('--random', action='store_true', help='Select random start index')
    args = parser.parse_args()

    # Config constants
    # P_MAX_W = 13511.8 # From user provided config/metadata
    # Let's try to load P_MAX from the model checkpoint if possible, or use default
    P_MAX_W = 13511.8
    
    # 1. Load Model
    model, cfg = load_model_custom(args.model_path)
    print(f"Model loaded. Input features: {cfg.input_features}")
    
    # 2. Load Continuous Data
    # Path to parquet
    data_path = Path('data/processed/1sec_new/nilm_ready_1sec_new.parquet')
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return

    # Appliances list needed for loading ground truth
    appliances = getattr(cfg, 'appliances', [args.appliance])
    X_raw, y_raw_dict, timestamps, X_agg_raw = load_continuous_data(data_path, appliances, P_MAX_W)
    
    # 3. Feature Engineering (Derivatives)
    # Check if we need to add derivatives
    # Base features: 1 (Agg) + 8 (Temporal) = 9
    # If model expects 12, we need +3 derivatives
    
    expected_features = cfg.input_features
    current_features = X_raw.shape[1]
    
    if expected_features > current_features:
        print(f"Adding derivative features... ({current_features} -> {expected_features})")
        # Need to compute derivatives on the whole sequence to ensure continuity
        # First column is Aggregate (normalized)
        agg = X_raw[:, 0]
        
        # dP/dt
        dP_dt = np.zeros_like(agg)
        dP_dt[1:] = agg[1:] - agg[:-1]
        
        # Rolling stats (window=8 as per original)
        win = 8
        roll_mean = pd.Series(agg).rolling(window=win, min_periods=1, center=False).mean().values
        roll_mean = np.nan_to_num(roll_mean)
        
        roll_std = pd.Series(agg).rolling(window=win, min_periods=1, center=False).std().values
        roll_std = np.nan_to_num(roll_std)
        
        X_full = np.column_stack([X_raw, dP_dt, roll_mean, roll_std])
    else:
        X_full = X_raw
        
    print(f"Input data shape: {X_full.shape}")
    

    # 4. Predict Sliding Window
    window_size = getattr(cfg, 'window_size', 512)
    
    # Determine segment
    total_len = len(X_full)
    req_len = args.duration
    
    if req_len > total_len:
        print(f"Warning: Requested duration {req_len} > total data {total_len}. Using full data.")
        req_len = total_len
        start_idx = 0
    elif args.random:
        # Pick a random start point, ensuring enough padding for window
        # Need window_size before start
        valid_range_start = window_size
        valid_range_end = total_len - req_len
        if valid_range_end > valid_range_start:
            start_idx = np.random.randint(valid_range_start, valid_range_end)
            print(f"Selected random start index: {start_idx}")
        else:
            start_idx = window_size
    else:
        # Default to end keys
        if args.offset > 0:
            start_idx = total_len - req_len - args.offset
        else:
            start_idx = total_len - req_len
    
    if start_idx < window_size:
        start_idx = window_size # Ensure context
        
    end_idx = start_idx + req_len
    
    # Context handling
    pad_start = start_idx - window_size
    segment_X = X_full[pad_start : end_idx]
    
    print(f"Predicting continuous sequence for {args.appliance} (Duration: {req_len}, Start: {start_idx})...")
    
    preds_segment = predict_continuous(
        model, 
        segment_X, 
        args.appliance, 
        window_size=window_size, 
        step=1
    )
    
    # Extract valid part matching [start_idx : end_idx]
    # Local index window_size corresponds to start_idx
    
    valid_slice = slice(window_size, window_size + req_len)
    final_preds = preds_segment[valid_slice]
    
    # 5. Plotting
    save_dir = Path("transformer/plots")
    save_dir.mkdir(exist_ok=True)
    
    gt_values = y_raw_dict[args.appliance][start_idx:end_idx]
    agg_values = X_agg_raw[start_idx:end_idx] * P_MAX_W 
    pred_watts = final_preds * P_MAX_W
    
    # Time axis
    # If duration is long (> 3 hours), use Hours
    if req_len > 3 * 3600:
        t = np.arange(len(gt_values)) / 3600.0 # Hours
        t_label = "Time (Hours)"
    else:
        t = np.arange(len(gt_values)) / 60.0 # Minutes
        t_label = "Time (Minutes)"
    
    plt.figure(figsize=(15, 8))
    
    # Plot Aggregate (Context)
    plt.plot(t, agg_values, color='black', alpha=0.15, label='Building (Aggregate)')
    
    # Plot Ground Truth
    plt.plot(t, gt_values, color='black', alpha=0.7, linewidth=1.5, label='Actual Power')
    
    # Plot Prediction
    plt.plot(t, pred_watts, color='orange', alpha=0.9, linewidth=1.5, label='Predicted Power')
    
    plt.title(f"{args.appliance} - Continuous Prediction ({len(gt_values)} steps)", fontsize=14)
    plt.xlabel(t_label)
    plt.ylabel("Power (W)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add timestamp info to title if available
    if len(timestamps) > start_idx:
        ts_start = timestamps[start_idx]
        plt.title(f"{args.appliance} - {ts_start} (+{req_len}s)", fontsize=14)
    
    out_file = save_dir / f"continuous_{args.appliance}_24h.png"
    plt.savefig(out_file)
    print(f"Saved plot to {out_file}")
    
    # Zoom Plot (Random 20% slice)
    zoom_len = int(len(gt_values) * 0.1) # 10% zoom
    if zoom_len > 100:
        zoom_start = np.random.randint(0, len(gt_values) - zoom_len)
        
        plt.figure(figsize=(15, 8))
        plt.plot(t[zoom_start:zoom_start+zoom_len], gt_values[zoom_start:zoom_start+zoom_len], 'k-', alpha=0.7, label='Actual')
        plt.plot(t[zoom_start:zoom_start+zoom_len], pred_watts[zoom_start:zoom_start+zoom_len], 'g-', alpha=0.8, label='Predicted')
        plt.title(f"{args.appliance} - Zoom Segment", fontsize=14)
        plt.xlabel(t_label)
        plt.ylabel("Power (W)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_dir / f"continuous_{args.appliance}_24h_zoom.png")
        print(f"Saved zoom plot")

if __name__ == "__main__":
    main()
