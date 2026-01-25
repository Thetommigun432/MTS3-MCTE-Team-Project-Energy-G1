"""
NILM Utilities
==============
Helper functions for training, evaluation, and metrics.
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
