"""
NILM Utilities
==============
Helper functions for training, evaluation, metrics, and post-processing.

Zero Hallucinations Post-Processing:
- median_filter: Remove impulsive noise
- minimum_duration_filter: Enforce HP physics constraints
- calibrate_threshold: Optimize τ for F1 with minimum Recall
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import f1_score, precision_score, recall_score
import time


class EarlyStopping:
    """
    Early stopping callback to prevent overfitting.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        
    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False
            
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
            
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
                
        return False


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_metrics(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    states_pred: Optional[Dict[str, np.ndarray]] = None,
    states_true: Optional[Dict[str, np.ndarray]] = None,
    scalers: Optional[Dict] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate NILM evaluation metrics.
    
    Metrics:
    - MAE: Mean Absolute Error (Watts)
    - RMSE: Root Mean Square Error (Watts)
    - SAE: Signal Aggregate Error (relative)
    - F1: F1-score for state detection (if states provided)
    
    Args:
        predictions: {appliance: [n_samples] predictions}
        targets: {appliance: [n_samples] ground truth}
        states_pred: Optional predicted states
        states_true: Optional true states
        scalers: Optional scalers for inverse transform
        
    Returns:
        {appliance: {metric: value}}
    """
    metrics = {}
    
    for name in predictions.keys():
        if name not in targets:
            continue
            
        pred = predictions[name].flatten()
        tgt = targets[name].flatten()
        
        # Inverse transform if scalers provided
        if scalers and name in scalers:
            pred = scalers[name].inverse_transform(pred.reshape(-1, 1)).flatten()
            tgt = scalers[name].inverse_transform(tgt.reshape(-1, 1)).flatten()
        
        # Clip negative predictions
        pred = np.maximum(pred, 0)
        
        # Regression metrics
        mae = np.mean(np.abs(pred - tgt))
        rmse = np.sqrt(np.mean((pred - tgt) ** 2))
        sae = np.abs(np.sum(pred) - np.sum(tgt)) / max(np.sum(tgt), 1e-6)
        
        metrics[name] = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'SAE': float(sae)
        }
        
        # Classification metrics if states available
        if states_pred and states_true and name in states_pred and name in states_true:
            s_pred = (states_pred[name].flatten() > 0.5).astype(int)
            s_true = (states_true[name].flatten() > 0.5).astype(int)
            
            metrics[name]['F1'] = float(f1_score(s_true, s_pred, zero_division=0))
            metrics[name]['Precision'] = float(precision_score(s_true, s_pred, zero_division=0))
            metrics[name]['Recall'] = float(recall_score(s_true, s_pred, zero_division=0))
    
    return metrics


def calculate_aggregate_metrics(metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate aggregate metrics across all appliances.
    
    Args:
        metrics: Per-appliance metrics from calculate_metrics
        
    Returns:
        Averaged metrics
    """
    if not metrics:
        return {}
        
    agg = {}
    metric_names = list(next(iter(metrics.values())).keys())
    
    for metric_name in metric_names:
        values = [m[metric_name] for m in metrics.values() if metric_name in m]
        if values:
            agg[f'avg_{metric_name}'] = float(np.mean(values))
    
    return agg


def format_metrics(metrics: Dict[str, Dict[str, float]], title: str = "Metrics") -> str:
    """Format metrics as a readable string."""
    lines = [f"\n{'='*60}", f"{title}", '-'*60]
    
    for name, values in metrics.items():
        line = f"{name:25s}"
        for metric, val in values.items():
            if metric in ['MAE', 'RMSE']:
                line += f" | {metric}: {val:7.2f}W"
            elif metric in ['SAE', 'F1', 'Precision', 'Recall']:
                line += f" | {metric}: {val:.4f}"
        lines.append(line)
    
    # Aggregates
    agg = calculate_aggregate_metrics(metrics)
    if agg:
        lines.append('-'*60)
        line = "AVERAGE                  "
        for metric, val in agg.items():
            short_name = metric.replace('avg_', '')
            if short_name in ['MAE', 'RMSE']:
                line += f" | {short_name}: {val:7.2f}W"
            else:
                line += f" | {short_name}: {val:.4f}"
        lines.append(line)
    
    lines.append('='*60)
    return '\n'.join(lines)


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start = None
        self.elapsed = None
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        print(f"{self.name} completed in {self.elapsed:.2f}s")


def count_parameters(model) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }


def print_model_summary(model, config):
    """Print model architecture summary."""
    params = count_parameters(model)
    
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("-"*60)
    print(f"Architecture: Hybrid CNN-Transformer NILM")
    print(f"Input features: {config.input_features}")
    print(f"Window size: {config.window_size}")
    print(f"d_model: {config.d_model}")
    print(f"Layers: {config.n_layers}")
    print(f"Heads: {config.n_heads}")
    print(f"Appliances: {len(config.appliances)}")
    print("-"*60)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print("="*60 + "\n")


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    val_loss: float,
    metrics: Dict,
    path,
    config
):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
        'metrics': metrics,
        'config': {k: str(v) if hasattr(v, '__fspath__') else v 
                   for k, v in vars(config).items() if not k.startswith('_')}
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load training checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


# =============================================================================
# POST-PROCESSING FOR ZERO HALLUCINATIONS
# =============================================================================

def median_filter(signal: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply median filter to remove impulsive noise/spikes.
    
    Critical for Zero Hallucinations:
    - Removes brief false activations (1-2 samples)
    - Smooths predictions without distorting true transitions
    
    Args:
        signal: 1D power signal [n_samples]
        kernel_size: Filter window size (default: 5 samples)
                    For 1Hz data: 5s window
                    For 5s data: 25s window
    
    Returns:
        Filtered signal with same shape
    """
    from scipy.ndimage import median_filter as scipy_median
    return scipy_median(signal, size=kernel_size, mode='reflect')


def minimum_duration_filter(
    signal: np.ndarray,
    min_duration: int,
    threshold: float = 0.0
) -> np.ndarray:
    """
    Remove activations shorter than minimum duration.
    
    Zero Hallucinations: Heat Pumps cannot cycle in seconds (compressor protection).
    Any activation < min_duration is physically impossible = hallucination.
    
    Args:
        signal: 1D power signal [n_samples]
        min_duration: Minimum samples for valid activation
                     For HP at 1Hz: 300 (5 minutes)
                     For HP at 5s: 60 (5 minutes)
        threshold: Power threshold to consider "ON" (default: 0)
        
    Returns:
        Filtered signal with short activations zeroed out
    """
    signal = signal.copy()
    is_on = signal > threshold
    
    # Find contiguous ON regions
    changes = np.diff(is_on.astype(int), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    
    # Zero out short activations
    for start, end in zip(starts, ends):
        duration = end - start
        if duration < min_duration:
            signal[start:end] = 0.0
    
    return signal


def minimum_off_duration_filter(
    signal: np.ndarray,
    min_off_duration: int,
    threshold: float = 0.0
) -> np.ndarray:
    """
    Merge activations separated by very short OFF periods.
    
    HP compressors have minimum off time (typically 3 minutes).
    Short OFF gaps between activations are likely measurement noise.
    
    Args:
        signal: 1D power signal [n_samples]
        min_off_duration: Minimum OFF samples between activations
        threshold: Power threshold to consider "ON"
        
    Returns:
        Signal with short OFF gaps filled
    """
    signal = signal.copy()
    is_on = signal > threshold
    is_off = ~is_on
    
    # Find contiguous OFF regions
    changes = np.diff(is_off.astype(int), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    
    # Fill short OFF gaps (interpolate from neighbors)
    for start, end in zip(starts, ends):
        duration = end - start
        # Only fill if it's a gap BETWEEN activations (not at edges)
        if duration < min_off_duration and start > 0 and end < len(signal):
            # Fill with average of adjacent ON values
            left_val = signal[start - 1] if start > 0 else 0
            right_val = signal[end] if end < len(signal) else 0
            signal[start:end] = (left_val + right_val) / 2
    
    return signal


def apply_nilm_postprocessing(
    power: np.ndarray,
    state_prob: np.ndarray = None,
    median_kernel: int = 5,
    min_on_duration: int = 60,  # 5 min at 5s resolution
    min_off_duration: int = 36,  # 3 min at 5s resolution  
    on_threshold: float = 0.0,
    tau: float = 0.7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete post-processing pipeline for Zero Hallucinations.
    
    Steps:
    1. Hard gating with τ threshold (if state_prob provided)
    2. Median filter to remove spikes
    3. Minimum ON duration filter (HP physics constraint)
    4. Minimum OFF duration filter (merge close activations)
    
    Args:
        power: Predicted power signal [n_samples]
        state_prob: Predicted state probability [n_samples] (optional)
        median_kernel: Median filter kernel size
        min_on_duration: Minimum ON duration in samples
        min_off_duration: Minimum OFF duration in samples
        on_threshold: Power threshold for ON state
        tau: Hard gating threshold for state_prob
        
    Returns:
        (filtered_power, filtered_state) tuple
    """
    # Step 1: Hard gating (if state probabilities available)
    if state_prob is not None:
        on_mask = (state_prob > tau).astype(float)
        power = power * on_mask
    
    # Step 2: Median filter
    power = median_filter(power, kernel_size=median_kernel)
    
    # Step 3: Minimum ON duration (remove short spikes)
    power = minimum_duration_filter(power, min_on_duration, on_threshold)
    
    # Step 4: Minimum OFF duration (merge close activations)
    power = minimum_off_duration_filter(power, min_off_duration, on_threshold)
    
    # Derive state from filtered power
    state = (power > on_threshold).astype(float)
    
    return power, state


def calibrate_threshold(
    state_prob: np.ndarray,
    state_true: np.ndarray,
    tau_range: Tuple[float, float] = (0.4, 0.9),
    n_steps: int = 20,
    min_recall: float = 0.80
) -> Tuple[float, Dict[str, float]]:
    """
    Calibrate hard gating threshold τ for optimal F1 with minimum Recall.
    
    Zero Hallucinations requires high Precision, but we constrain Recall
    to avoid completely missing activations.
    
    Args:
        state_prob: Predicted probabilities [n_samples]
        state_true: Ground truth binary states [n_samples]
        tau_range: (min_tau, max_tau) search range
        n_steps: Number of thresholds to evaluate
        min_recall: Minimum acceptable Recall
        
    Returns:
        (best_tau, metrics_at_best_tau)
    """
    best_tau = 0.5
    best_f1 = 0.0
    best_metrics = {}
    
    for tau in np.linspace(tau_range[0], tau_range[1], n_steps):
        state_pred = (state_prob > tau).astype(int)
        
        # Calculate metrics
        tp = np.sum((state_pred == 1) & (state_true == 1))
        fp = np.sum((state_pred == 1) & (state_true == 0))
        fn = np.sum((state_pred == 0) & (state_true == 1))
        tn = np.sum((state_pred == 0) & (state_true == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Only consider if Recall meets minimum
        if recall >= min_recall and f1 > best_f1:
            best_f1 = f1
            best_tau = tau
            best_metrics = {
                'tau': tau,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'fpr': fpr
            }
    
    return best_tau, best_metrics
