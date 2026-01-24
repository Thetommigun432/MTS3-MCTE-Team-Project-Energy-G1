"""
Visualize NILM predictions vs ground truth as time series.
Uses sliding window inference to reconstruct full time series from Seq2Point model.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
import argparse
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))

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


def load_model_and_data(appliance_filter=None):
    """Load trained model and test data."""
    config = Config()
    
    # Load data
    print("Loading test data...")
    model_ready_path = config.data_path / config.model_ready_subdir
    train_data, val_data, test_data, metadata = load_pretrained_data(model_ready_path)
    X_test = test_data['X']
    y_test = test_data['y']
    
    # Get P_MAX for denormalization
    P_MAX_kW = metadata['scaling']['P_MAX']
    P_MAX_W = P_MAX_kW * 1000
    
    appliances = metadata['target_appliances']
    if appliance_filter:
        appliances = [appliance_filter]
    
    # Load model
    print("Loading model...")
    checkpoint_path = Path(__file__).parent / "checkpoints" / "hybrid_nilm_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get model config from checkpoint
    model_config = checkpoint.get('config', {})
    print(f"Loaded config keys: {model_config.keys()}")
    
    # Create config object for create_model
    cfg_obj = SimpleNamespace(**model_config)
    cfg_obj.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Backwards compatibility for feature names
    if not hasattr(cfg_obj, 'input_features'):
        cfg_obj.input_features = model_config.get('n_features', X_test.shape[2])
    
    n_features = cfg_obj.input_features
    print(f"Using n_features={n_features} from checkpoint")
    
    # Add derivative features if needed
    if n_features > X_test.shape[2]:
        print(f"Adding derivative features: {X_test.shape[2]} → {n_features}")
        X_test = compute_derivative_features(X_test)
    
    # Instantiate model using factory
    model = create_model(cfg_obj)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, X_test, y_test, appliances, P_MAX_W


def sliding_window_inference(model, X_continuous, appliances, window_size=1024, stride=64):
    """
    Run sliding window inference to get predictions for entire time series.
    
    Args:
        model: trained model
        X_continuous: continuous input data [time, features]
        appliances: list of appliance names
        window_size: size of each window
        stride: step between windows
    
    Returns:
        predictions: [time, n_appliances] predicted power
        valid_mask: [time] boolean mask of valid predictions
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    n_time = X_continuous.shape[0]
    n_appliances = len(appliances)
    midpoint = window_size // 2
    
    # Arrays to accumulate predictions
    pred_sum = np.zeros((n_time, n_appliances))
    pred_count = np.zeros(n_time)
    
    n_windows = (n_time - window_size) // stride + 1
    print(f"Running inference on {n_windows} windows (stride={stride})...")
    
    with torch.no_grad():
        for i in range(0, n_time - window_size + 1, stride):
            window = X_continuous[i:i+window_size]
            x = torch.FloatTensor(window).unsqueeze(0).to(device)
            
            outputs = model(x)
            
            # Get midpoint prediction
            mid_idx = i + midpoint
            for j, app_name in enumerate(appliances):
                power = outputs[app_name]['power'].cpu().numpy().squeeze()
                pred_sum[mid_idx, j] += power
            pred_count[mid_idx] += 1
            
            if (i // stride) % 100 == 0:
                print(f"  Window {i // stride + 1}/{n_windows}")
    
    # Average overlapping predictions
    valid_mask = pred_count > 0
    predictions = np.zeros((n_time, n_appliances))
    predictions[valid_mask] = pred_sum[valid_mask] / pred_count[valid_mask, np.newaxis]
    
    return predictions, valid_mask


def plot_timeseries_comparison(time_minutes, gt, pred, valid_mask, appliances, P_MAX_W, save_dir):
    """Plot ground truth vs prediction time series."""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Select key appliances to plot
    key_appliances = ['HeatPump', 'GarageCabinet', 'Dishwasher', 'Dryer', 'WashingMachine']
    plot_indices = [i for i, a in enumerate(appliances) if a in key_appliances]
    
    n_plots = len(plot_indices)
    
    # --- Plot 1: Multi-panel comparison ---
    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 3*n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    
    for ax_idx, app_idx in enumerate(plot_indices):
        app_name = appliances[app_idx]
        
        # Denormalize to Watts
        gt_watts = gt[:, app_idx] * P_MAX_W
        pred_watts = pred[:, app_idx] * P_MAX_W
        
        # Only plot where we have valid predictions
        time_valid = time_minutes[valid_mask]
        gt_valid = gt_watts[valid_mask]
        pred_valid = pred_watts[valid_mask]
        
        axes[ax_idx].plot(time_valid, gt_valid, 'b-', label='Ground Truth', alpha=0.8, linewidth=1)
        axes[ax_idx].plot(time_valid, pred_valid, 'r-', label='Prediction', alpha=0.7, linewidth=1)
        axes[ax_idx].fill_between(time_valid, gt_valid, pred_valid, alpha=0.2, color='purple')
        
        # Calculate metrics
        mae = np.mean(np.abs(gt_valid - pred_valid))
        
        axes[ax_idx].set_ylabel(f'{app_name}\n(Watts)', fontsize=10)
        axes[ax_idx].legend(loc='upper right', fontsize=9)
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].set_title(f'{app_name} — MAE: {mae:.1f}W', fontsize=11)
        
        # Set y limit based on max value
        max_val = max(gt_valid.max(), pred_valid.max()) if len(gt_valid) > 0 else 100
        axes[ax_idx].set_ylim(0, max_val * 1.1)
    
    axes[-1].set_xlabel('Time (minutes)', fontsize=11)
    plt.suptitle('NILM Transformer: Ground Truth vs Prediction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'timeseries_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: timeseries_comparison.png")
    plt.close()
    
    # --- Plot 2: Aggregate comparison ---
    fig, ax = plt.subplots(figsize=(16, 5))
    
    # Sum all appliances
    gt_sum = np.sum(gt * P_MAX_W, axis=1)
    pred_sum = np.sum(pred * P_MAX_W, axis=1)
    
    ax.plot(time_minutes[valid_mask], gt_sum[valid_mask], 'b-', label='GT Sum (all appliances)', linewidth=1.5, alpha=0.8)
    ax.plot(time_minutes[valid_mask], pred_sum[valid_mask], 'r-', label='Pred Sum (all appliances)', linewidth=1.5, alpha=0.7)
    ax.fill_between(time_minutes[valid_mask], gt_sum[valid_mask], pred_sum[valid_mask], alpha=0.2, color='purple')
    
    mae_total = np.mean(np.abs(gt_sum[valid_mask] - pred_sum[valid_mask]))
    ax.set_title(f'Total Disaggregated Power — MAE: {mae_total:.1f}W', fontsize=12)
    ax.set_xlabel('Time (minutes)', fontsize=11)
    ax.set_ylabel('Power (Watts)', fontsize=11)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'aggregate_timeseries.png', dpi=150, bbox_inches='tight')
    print(f"Saved: aggregate_timeseries.png")
    plt.close()
    
    # --- Plot 3: Individual appliance detailed plots ---
    for app_idx in plot_indices:
        app_name = appliances[app_idx]
        
        fig, ax = plt.subplots(figsize=(14, 4))
        
        gt_watts = gt[:, app_idx] * P_MAX_W
        pred_watts = pred[:, app_idx] * P_MAX_W
        
        time_valid = time_minutes[valid_mask]
        gt_valid = gt_watts[valid_mask]
        pred_valid = pred_watts[valid_mask]
        
        ax.plot(time_valid, gt_valid, 'b-', label='Ground Truth', linewidth=1.2, alpha=0.9)
        ax.plot(time_valid, pred_valid, 'r--', label='Prediction', linewidth=1.2, alpha=0.8)
        
        mae = np.mean(np.abs(gt_valid - pred_valid))
        rmse = np.sqrt(np.mean((gt_valid - pred_valid)**2))
        
        ax.set_title(f'{app_name} — MAE: {mae:.1f}W, RMSE: {rmse:.1f}W', fontsize=12)
        ax.set_xlabel('Time (minutes)', fontsize=11)
        ax.set_ylabel('Power (Watts)', fontsize=11)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'timeseries_{app_name}.png', dpi=150, bbox_inches='tight')
        print(f"Saved: timeseries_{app_name}.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--appliance', type=str, default=None, help='Specific appliance to verify')
    args = parser.parse_args()

    print("=" * 60)
    print("NILM Transformer - Time Series Visualization")
    print("=" * 60)
    
    # Load model and data
    model, X_test, y_test, appliances, P_MAX_W = load_model_and_data(args.appliance)
    
    print(f"\nTest data: {X_test.shape[0]} windows")
    print(f"Appliances: {appliances}")
    print(f"P_MAX: {P_MAX_W:.1f} W")
    
    if args.appliance and args.appliance not in appliances:
         print(f"⚠️ Warning: Appliance {args.appliance} not in filtered appliances list.")
    
    # Take a continuous chunk of test data
    # Concatenate 50 consecutive windows to form a long time series
    n_windows_to_use = 50
    window_size = 1024
    stride_original = 256  # Original stride used in preprocessing
    
    print(f"\nReconstructing continuous time series from {n_windows_to_use} windows...")
    
    # We need to reconstruct the continuous signal
    # Each window overlaps by (window_size - stride) samples
    # Total length = window_size + (n_windows - 1) * stride
    
    total_length = window_size + (n_windows_to_use - 1) * stride_original
    print(f"Total time series length: {total_length} samples = {total_length * 5 / 60:.1f} minutes")
    
    # Build continuous X and y from overlapping windows
    X_continuous = np.zeros((total_length, X_test.shape[2]))
    # y_test has shape (N, 1024, n_all_appliances) usually, need to slice it carefully if filtered
    
    # NOTE: load_pretrained_data returns y_test with ALL appliances usually.
    # But appliances list is filtered.
    # We need to find the indices of our filtered appliances in the original y_test
    # For simplicity, if we filtered, we assume y_test matches?
    # No, load_pretrained_data returns y_test based on metadata['target_appliances'].
    # The 'appliances' variable we returned from load_model_and_data is FILTERED.
    
    # We need to get the index of the interesting appliance from the metadata list
    # However, 'load_pretrained_data' is a black box here. 
    # Let's assume y_test corresponds to metadata['target_appliances'].
    
    # Quick fix: Re-load metadata to map indices
    config = Config()
    _, _, _, metadata = load_pretrained_data(config.data_path / config.model_ready_subdir)
    all_appliances = metadata['target_appliances']
    
    # Map filtered appliances to indices in y_test
    app_indices = [all_appliances.index(app) for app in appliances if app in all_appliances]
    
    # y_continuous needs to be [total_length, len(appliances)]
    y_continuous = np.zeros((total_length, len(appliances)))
    count = np.zeros(total_length)
    
    start_idx = 100  # Start from window 100 to avoid edge effects
    
    for i in range(n_windows_to_use):
        window_idx = start_idx + i
        if window_idx >= len(X_test):
            break
            
        start = i * stride_original
        end = start + window_size
        
        X_continuous[start:end] += X_test[window_idx]
        
        # Select specific appliances from y_test
        y_window = y_test[window_idx][:, app_indices]
        y_continuous[start:end] += y_window
        
        count[start:end] += 1
    
    # Average overlapping regions
    valid = count > 0
    X_continuous[valid] /= count[valid, np.newaxis]
    y_continuous[valid] /= count[valid, np.newaxis]
    
    # Run sliding window inference
    predictions, valid_mask = sliding_window_inference(
        model, X_continuous, appliances, 
        window_size=window_size, 
        stride=64  # Use smaller stride for smoother predictions
    )
    
    # Create time axis in minutes
    time_minutes = np.arange(total_length) * 5 / 60  # 5 sec resolution
    
    # Create plots
    save_dir = Path(__file__).parent / "plots"
    print("\nCreating visualization plots...")
    plot_timeseries_comparison(time_minutes, y_continuous, predictions, valid_mask, appliances, P_MAX_W, save_dir)
    
    print(f"\nAll plots saved to: {save_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
