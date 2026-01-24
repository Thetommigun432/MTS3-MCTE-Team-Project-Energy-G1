"""
NILMFormer Training - ALIGNED TO KDD 2025 PAPER + SPARSITY-AWARE IMPROVEMENTS
==============================================================================
Key differences from previous implementation:
1. Seq2Seq output (full sequence, NOT midpoint only)
2. Instance Normalization global (NOT causal)
3. H200 32GB: Can use large batch + long sequences

IMPROVEMENTS over paper baseline:
- Sparsity-aware loss: higher weight for rare appliances (Dryer 0.1%, RainwaterPump 0.1%)
- Per-appliance ON thresholds for F1 computation
- False Negative penalty for sparse devices
- Adaptive loss based on activity rate

Usage:
    python train_nilmformer_paper.py
    python train_nilmformer_paper.py --epochs 100 --batch_size 256 --d_model 128
    python train_nilmformer_paper.py --loss sparsity  # Use sparsity-aware loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import argparse
from typing import Dict, List, Tuple, Optional
import time

from nilmformer_paper import create_nilmformer_paper, NILMFormerConfig


# =============================================================================
# APPLIANCE STATISTICS (from preprocessing/15min/validation_report.md)
# =============================================================================
APPLIANCE_ACTIVITY_RATES = {
    # Percentage of time appliance is ON (>10W)
    'GarageCabinet': 100.0,      # Always on (standby)
    'HeatPump': 26.7,            # Frequent
    'Dishwasher': 6.2,           # Moderate
    'WashingMachine': 6.2,       # Moderate
    'EVCharger': 7.4,            # Moderate  
    'EVSocket': 6.2,             # Moderate
    'Oven': 2.3,                 # Rare
    'Stove': 1.9,                # Rare
    'RangeHood': 0.6,            # Very rare
    'Dryer': 0.1,                # Almost always OFF
    'RainwaterPump': 0.1,        # Almost always OFF
}

# ON thresholds in normalized units (Watts / P_MAX / 1000)
# P_MAX = 13.5118 kW, so threshold_norm = threshold_W / 13511.8
APPLIANCE_ON_THRESHOLDS = {
    'HeatPump': 100 / 13511.8,        # 100W
    'Dishwasher': 30 / 13511.8,       # 30W
    'WashingMachine': 50 / 13511.8,   # 50W
    'Dryer': 50 / 13511.8,            # 50W
    'Oven': 100 / 13511.8,            # 100W
    'Stove': 50 / 13511.8,            # 50W
    'RangeHood': 20 / 13511.8,        # 20W
    'EVCharger': 100 / 13511.8,       # 100W
    'EVSocket': 100 / 13511.8,        # 100W
    'GarageCabinet': 25 / 13511.8,    # 25W
    'RainwaterPump': 50 / 13511.8,    # 50W
}


# =============================================================================
# CONFIG
# =============================================================================
class TrainingConfig:
    """Training configuration for NILMFormer."""
    
    # Data paths (uses cwd for portability)
    # Default: looks for model_ready in current directory
    data_path: Path = Path.cwd()
    model_ready_subdir: str = 'model_ready'
    save_path: Path = Path.cwd() / 'checkpoints'
    
    # Appliances
    appliances: List[str] = [
        'HeatPump', 'Dishwasher', 'WashingMachine', 'Dryer',
        'Oven', 'Stove', 'RangeHood', 'EVCharger', 'EVSocket',
        'GarageCabinet', 'RainwaterPump'
    ]
    
    # Model (aligned to paper)
    d_model: int = 96           # Paper default
    n_layers: int = 3           # Paper default  
    n_heads: int = 8            # Paper default
    dropout: float = 0.2        # Paper default
    dilations: List[int] = [1, 2, 4, 8]
    
    # Training (H200 32GB can handle large batches!)
    batch_size: int = 128       # Large batch for H200
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 15
    grad_clip: float = 1.0
    # Window
    window_size: int = 1024     # From pretraining
    
    # Loss: 'mse' (paper), 'huber', 'sparsity' (recommended for class imbalance), 'weighted'
    loss_type: str = 'sparsity'
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4        # Can use more workers with H200 system
    
    # Mixed precision (faster on H200)
    use_amp: bool = True
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        self.save_path = Path(self.save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATASET (Seq2Seq)
# =============================================================================
class NILMSeq2SeqDataset(Dataset):
    """
    Dataset for Seq2Seq NILM (predicts full sequence).
    
    Input: (window_size, n_features) - Aggregate + temporal
    Target: (window_size, n_appliances) - All appliances
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        appliance_names: List[str],
    ):
        """
        Args:
            X: Input features (n_samples, window_size, n_features)
            y: Targets (n_samples, window_size, n_appliances)
            appliance_names: List of appliance names
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.appliance_names = appliance_names
        
        print(f"Dataset: {len(self)} samples")
        print(f"  X shape: {self.X.shape}")
        print(f"  y shape: {self.y.shape}")
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = self.X[idx]  # (window_size, n_features)
        
        # Create dict of targets per appliance
        y_dict = {}
        for i, name in enumerate(self.appliance_names):
            y_dict[name] = self.y[idx, :, i:i+1]  # (window_size, 1)
        
        return x, y_dict


def load_pretrained_data(data_path: Path) -> Tuple[Dict, Dict, Dict, Dict]:
    """Load pretrained numpy arrays and metadata."""
    
    print(f"Loading data from {data_path}...")
    
    # Load arrays
    X_train = np.load(data_path / 'X_train.npy')
    y_train = np.load(data_path / 'y_train.npy')
    X_val = np.load(data_path / 'X_val.npy')
    y_val = np.load(data_path / 'y_val.npy')
    X_test = np.load(data_path / 'X_test.npy')
    y_test = np.load(data_path / 'y_test.npy')
    
    # Load metadata
    with open(data_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  X_test: {X_test.shape}")
    
    train_data = {'X': X_train, 'y': y_train}
    val_data = {'X': X_val, 'y': y_val}
    test_data = {'X': X_test, 'y': y_test}
    
    return train_data, val_data, test_data, metadata


def create_dataloaders(
    config: TrainingConfig,
    train_data: Dict,
    val_data: Dict,
    test_data: Dict,
    appliances: List[str],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for Seq2Seq training."""
    
    train_ds = NILMSeq2SeqDataset(train_data['X'], train_data['y'], appliances)
    val_ds = NILMSeq2SeqDataset(val_data['X'], val_data['y'], appliances)
    test_ds = NILMSeq2SeqDataset(test_data['X'], test_data['y'], appliances)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    
    return train_loader, val_loader, test_loader


# =============================================================================
# LOSS (Paper: Simple MSELoss) + SPARSITY-AWARE IMPROVEMENT
# =============================================================================
class HuberLoss(nn.Module):
    """
    Huber Loss for NILM - robust to outliers (Zero Hallucinations strategy).
    
    Huber loss combines MSE for small errors and MAE for large errors:
    - |error| <= δ: 0.5 * error²
    - |error| > δ: δ * (|error| - 0.5δ)
    
    Paper recommends δ=50W for NILM. Since data is normalized by P_MAX (13.5kW),
    δ_norm = 50 / 13511.8 ≈ 0.0037
    """
    
    def __init__(self, delta_watts: float = 50.0, P_MAX: float = 13.5118):
        super().__init__()
        # Convert δ from Watts to normalized units
        self.delta = delta_watts / (P_MAX * 1000)
        print(f"HuberLoss: δ={delta_watts}W (normalized: {self.delta:.6f})")
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        total_loss = 0.0
        n_appliances = 0
        
        for name in predictions.keys():
            if name not in targets:
                continue
            
            pred = predictions[name]
            tgt = targets[name]
            
            # Huber loss
            loss = nn.functional.huber_loss(pred, tgt, delta=self.delta)
            losses[f'{name}_huber'] = loss
            total_loss += loss
            n_appliances += 1
        
        losses['total'] = total_loss / max(n_appliances, 1)
        return losses


class Seq2SeqMSELoss(nn.Module):
    """
    Simple MSE Loss for Seq2Seq NILM (as in paper).
    
    Computes mean MSE across all appliances and all timesteps.
    """
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: {appliance: (B, L, 1)}
            targets: {appliance: (B, L, 1)}
            
        Returns:
            dict with 'total' and per-appliance losses
        """
        losses = {}
        total_loss = 0.0
        n_appliances = 0
        
        for name in predictions.keys():
            if name not in targets:
                continue
            
            pred = predictions[name]
            tgt = targets[name]
            
            loss = self.mse(pred, tgt)
            losses[f'{name}_mse'] = loss
            total_loss += loss
            n_appliances += 1
        
        losses['total'] = total_loss / max(n_appliances, 1)
        
        return losses


class SparsityAwareLoss(nn.Module):
    """
    Sparsity-Aware Loss for NILM with Class Imbalance Handling.
    
    Key innovations:
    1. Per-appliance weights inversely proportional to activity rate
       - Dryer (0.1% activity) → weight ~100x
       - HeatPump (26.7% activity) → weight ~4x
    2. False Negative penalty for rare appliances
    3. Smooth L1 + MSE hybrid for robustness
    
    This addresses the issue where rare appliances (Dryer, RainwaterPump, 
    Stove, RangeHood) get F1=0 because the model learns to predict zeros.
    """
    
    def __init__(
        self,
        activity_rates: Optional[Dict[str, float]] = None,
        on_thresholds: Optional[Dict[str, float]] = None,
        base_fn_weight: float = 10.0,  # False Negative penalty multiplier
        min_weight: float = 1.0,       # Minimum appliance weight
        max_weight: float = 50.0,      # Cap to avoid instability
    ):
        super().__init__()
        self.activity_rates = activity_rates or APPLIANCE_ACTIVITY_RATES
        self.on_thresholds = on_thresholds or APPLIANCE_ON_THRESHOLDS
        self.base_fn_weight = base_fn_weight
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Compute appliance weights from activity rates
        # weight = 1 / sqrt(activity_rate) capped
        self.appliance_weights = {}
        for name, rate in self.activity_rates.items():
            # Inverse square root weighting (less aggressive than inverse)
            if rate > 0:
                weight = min(self.max_weight, max(self.min_weight, 10.0 / np.sqrt(rate)))
            else:
                weight = self.max_weight
            self.appliance_weights[name] = weight
        
        print("SparsityAwareLoss weights:")
        for name, w in sorted(self.appliance_weights.items(), key=lambda x: -x[1]):
            print(f"  {name}: {w:.2f}x (activity {self.activity_rates.get(name, 0):.1f}%)")
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute sparsity-aware loss.
        
        Args:
            predictions: {appliance: (B, L, 1)}
            targets: {appliance: (B, L, 1)}
            
        Returns:
            dict with 'total' and per-appliance losses
        """
        losses = {}
        total_loss = 0.0
        total_weight = 0.0
        
        for name in predictions.keys():
            if name not in targets:
                continue
            
            pred = predictions[name].view(-1)
            tgt = targets[name].view(-1)
            
            # Get appliance-specific parameters
            app_weight = self.appliance_weights.get(name, 1.0)
            on_threshold = self.on_thresholds.get(name, 0.001)
            
            # Determine ON/OFF states
            is_on = (tgt.abs() > on_threshold).float()
            is_off = 1.0 - is_on
            
            # Base loss: Smooth L1 (robust to outliers)
            base_loss = nn.functional.smooth_l1_loss(pred, tgt, reduction='none')
            
            # ON/OFF weighting within sequence
            # ON samples get higher weight (5x), OFF samples get 1x
            sample_weights = is_on * 5.0 + is_off * 1.0
            
            # False Negative penalty: target is ON but prediction is low
            # This is CRITICAL for rare appliances
            fn_mask = is_on * (pred < tgt * 0.5).float()  # Predicting < 50% of true
            fn_penalty = fn_mask * self.base_fn_weight * (tgt - pred).abs()
            
            # Combined loss for this appliance
            app_loss = (base_loss * sample_weights + fn_penalty).mean()
            
            # Weight by appliance rarity
            weighted_app_loss = app_loss * app_weight
            
            losses[f'{name}_mse'] = app_loss
            losses[f'{name}_weight'] = torch.tensor(app_weight)
            
            total_loss += weighted_app_loss
            total_weight += app_weight
        
        # Normalize by total weight
        losses['total'] = total_loss / max(total_weight, 1.0)
        
        return losses


class WeightedNILMLoss(nn.Module):
    """
    Weighted NILM Loss with strong False Negative penalty.
    
    Proven effective in notebook 10 (15min model).
    """
    
    def __init__(
        self,
        on_weight: float = 30.0,
        off_weight: float = 5.0,
        fn_weight: float = 15.0,
        mix: float = 0.5,  # MSE/MAE mix
    ):
        super().__init__()
        self.on_weight = on_weight
        self.off_weight = off_weight
        self.fn_weight = fn_weight
        self.mix = mix
        self.on_thresholds = APPLIANCE_ON_THRESHOLDS
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute weighted loss with FN penalty."""
        losses = {}
        total_loss = 0.0
        n_appliances = 0
        
        for name in predictions.keys():
            if name not in targets:
                continue
            
            pred = predictions[name].view(-1)
            tgt = targets[name].view(-1)
            
            # ON threshold for this appliance
            on_threshold = self.on_thresholds.get(name, 0.001)
            
            # ON/OFF states
            is_on = (tgt.abs() > on_threshold).float()
            is_off = 1.0 - is_on
            
            # Class weights
            weights = is_on * self.on_weight + is_off * self.off_weight
            
            # Hybrid MSE/MAE
            mse = (pred - tgt) ** 2
            mae = torch.abs(pred - tgt)
            base_loss = (1 - self.mix) * mse + self.mix * mae
            
            # Weighted base loss
            weighted_loss = base_loss * weights
            
            # FN penalty: ON but under-predicting
            fn_mask = is_on * (pred < tgt).float()
            fn_penalty = fn_mask * self.fn_weight * mse
            
            # Total for appliance
            app_loss = (weighted_loss + fn_penalty).mean()
            
            losses[f'{name}_mse'] = app_loss
            total_loss += app_loss
            n_appliances += 1
        
        losses['total'] = total_loss / max(n_appliances, 1)
        
        return losses


def create_loss(loss_type: str) -> nn.Module:
    """Create loss function based on type."""
    if loss_type == 'mse':
        print("Using: Simple MSE Loss (paper baseline)")
        return Seq2SeqMSELoss()
    elif loss_type == 'huber':
        print("Using: Huber Loss (δ=50W, robust to outliers)")
        return HuberLoss(delta_watts=50.0)
    elif loss_type == 'sparsity':
        print("Using: Sparsity-Aware Loss (improved)")
        return SparsityAwareLoss()
    elif loss_type == 'weighted':
        print("Using: Weighted NILM Loss with FN penalty")
        return WeightedNILMLoss()
    else:
        print(f"Unknown loss type '{loss_type}', defaulting to huber")
        return HuberLoss(delta_watts=50.0)


# =============================================================================
# METRICS (Seq2Seq) with F1 Score
# =============================================================================
def compute_metrics_seq2seq(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    P_MAX: float = 13.5118,  # kW
) -> Dict[str, Dict[str, float]]:
    """
    Compute NILM metrics for Seq2Seq predictions.
    
    Metrics (from paper + F1):
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - SAE: Signal Aggregate Error = |sum(pred) - sum(true)| / sum(true)
    - F1: F1 Score (using per-appliance ON thresholds)
    """
    metrics = {}
    
    for name in predictions.keys():
        pred = predictions[name].flatten()
        true = targets[name].flatten()
        
        # De-normalize to Watts for interpretable metrics
        pred_watts = pred * P_MAX * 1000
        true_watts = true * P_MAX * 1000
        
        # MAE
        mae = np.mean(np.abs(pred_watts - true_watts))
        
        # RMSE
        rmse = np.sqrt(np.mean((pred_watts - true_watts) ** 2))
        
        # SAE (energy error)
        sum_true = np.sum(np.abs(true_watts)) + 1e-10
        sae = np.abs(np.sum(pred_watts) - np.sum(true_watts)) / sum_true
        
        # F1 Score with per-appliance threshold
        on_threshold_norm = APPLIANCE_ON_THRESHOLDS.get(name, 0.001)
        pred_on = (pred > on_threshold_norm).astype(int)
        true_on = (true > on_threshold_norm).astype(int)
        
        # Confusion matrix elements
        tp = np.sum((pred_on == 1) & (true_on == 1))
        fp = np.sum((pred_on == 1) & (true_on == 0))
        fn = np.sum((pred_on == 0) & (true_on == 1))
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        # Activity rate in this split
        activity_pct = 100 * np.mean(true_on)
        
        metrics[name] = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'SAE': float(sae),
            'F1': float(f1),
            'Precision': float(precision),
            'Recall': float(recall),
            'Activity_%': float(activity_pct),
        }
    
    return metrics


# =============================================================================
# TRAINING LOOP
# =============================================================================
class AverageMeter:
    """Tracks average of values."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True  # Improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # No improvement


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()
    
    pbar = tqdm(loader, desc='Training', leave=False, ncols=100)
    for x, y in pbar:
        x = x.to(device)
        y = {name: val.to(device) for name, val in y.items()}
        
        optimizer.zero_grad()
        
        # Forward (with optional AMP)
        if scaler is not None:
            with autocast(device_type='cuda'):
                pred = model(x)
                losses = criterion(pred, y)
        else:
            pred = model(x)
            losses = criterion(pred, y)
        
        loss = losses['total']
        
        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        loss_meter.update(loss.item(), x.size(0))
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
    
    return {'loss': loss_meter.avg}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Validate model and compute metrics."""
    model.eval()
    loss_meter = AverageMeter()
    
    all_preds = {}
    all_targets = {}
    
    for x, y in tqdm(loader, desc='Validating', leave=False, ncols=100):
        x = x.to(device)
        y_device = {name: val.to(device) for name, val in y.items()}
        
        pred = model(x)
        losses = criterion(pred, y_device)
        
        loss_meter.update(losses['total'].item(), x.size(0))
        
        # Collect predictions
        for name in pred.keys():
            if name not in all_preds:
                all_preds[name] = []
                all_targets[name] = []
            
            all_preds[name].append(pred[name].cpu().numpy())
            all_targets[name].append(y[name].numpy())
    
    # Concatenate
    preds_np = {k: np.concatenate(v) for k, v in all_preds.items()}
    targets_np = {k: np.concatenate(v) for k, v in all_targets.items()}
    
    # Compute metrics
    metrics = compute_metrics_seq2seq(preds_np, targets_np)
    
    return {'loss': loss_meter.avg}, metrics


def train(config: TrainingConfig):
    """Main training function."""
    
    print("=" * 70)
    print("NILMFormer Training (KDD 2025 Paper Aligned)")
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Model: d_model={config.d_model}, layers={config.n_layers}, heads={config.n_heads}")
    print("=" * 70)
    
    # Load data
    model_ready_path = config.data_path / config.model_ready_subdir
    print(f"Loading data from {model_ready_path}...")
    train_data, val_data, test_data, metadata = load_pretrained_data(model_ready_path)
    
    # Get appliances from metadata (all available)
    all_appliances = metadata.get('target_appliances', config.appliances)
    
    # If config.appliances is a subset (single appliance mode), use it
    # Otherwise use all from metadata
    if len(config.appliances) < len(all_appliances):
        appliances = config.appliances
        # Get indices of selected appliances in original data
        appliance_indices = [all_appliances.index(app) for app in appliances if app in all_appliances]
        
        # Filter y data to only include selected appliances
        train_data['y'] = train_data['y'][:, :, appliance_indices]
        val_data['y'] = val_data['y'][:, :, appliance_indices]
        test_data['y'] = test_data['y'][:, :, appliance_indices]
        
        print(f"\n*** SINGLE APPLIANCE MODE: Filtered to {appliances} ***")
        print(f"  y_train new shape: {train_data['y'].shape}")
    else:
        appliances = all_appliances
    
    n_input_features = train_data['X'].shape[2]  # e.g., 7 (1 agg + 6 temporal)
    c_embedding = n_input_features - 1  # Exogenous features (temporal)
    
    print(f"\nAppliances: {appliances}")
    print(f"Input features: {n_input_features} (1 aggregate + {c_embedding} temporal)")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        config, train_data, val_data, test_data, appliances
    )
    
    # Create model
    is_multi = len(appliances) > 1
    model = create_nilmformer_paper(
        appliances=appliances,
        c_embedding=c_embedding,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_head=config.n_heads,
        dropout=config.dropout,
        multi_appliance=is_multi,
    )
    
    model = model.to(config.device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    
    # Loss (based on config)
    criterion = create_loss(config.loss_type)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.lr / 100
    )
    
    # Mixed precision
    scaler = GradScaler() if config.use_amp and config.device == 'cuda' else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.patience)
    
    # Training loop
    best_val_loss = float('inf')
    best_metrics = {}
    
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(config.epochs):
        # Train
        train_results = train_epoch(
            model, train_loader, optimizer, criterion,
            config.device, scaler, config.grad_clip
        )
        
        # Validate
        val_results, val_metrics = validate(
            model, val_loader, criterion, config.device
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Compute avg metrics
        val_avg_mae = np.mean([m['MAE'] for m in val_metrics.values()])
        val_avg_f1 = np.mean([m['F1'] for m in val_metrics.values()])
        
        # Log
        print(f"Epoch {epoch+1:03d}/{config.epochs} | "
              f"Train Loss: {train_results['loss']:.6f} | "
              f"Val Loss: {val_results['loss']:.6f} | "
              f"Val MAE: {val_avg_mae:.1f}W | "
              f"Val F1: {val_avg_f1:.4f} | "
              f"LR: {current_lr:.2e}")
        
        # Check improvement
        improved = early_stopping(val_results['loss'])
        
        if improved:
            best_val_loss = val_results['loss']
            best_metrics = val_metrics
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'metrics': best_metrics,
                'config': config.__dict__,
            }
            torch.save(checkpoint, config.save_path / 'nilmformer_paper_best.pth')
            print(f"  ✓ New best model saved! Val loss: {best_val_loss:.6f}")
        
        if early_stopping.early_stop:
            print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            break
    
    total_time = (time.time() - start_time) / 60
    print(f"\nTraining completed in {total_time:.1f} minutes")
    
    # Final evaluation on test set
    print("\n" + "=" * 70)
    print("Final Evaluation on Test Set")
    print("=" * 70)
    
    # Load best model
    checkpoint = torch.load(config.save_path / 'nilmformer_paper_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_results, test_metrics = validate(model, test_loader, criterion, config.device)
    
    print(f"\nTest Loss: {test_results['loss']:.6f}")
    print("\nPer-Appliance Metrics:")
    print("-" * 70)
    print(f"{'Appliance':<18} {'MAE(W)':<10} {'RMSE(W)':<10} {'SAE':<8} {'F1':<8} {'Act.%':<8}")
    print("-" * 70)
    
    for name, m in sorted(test_metrics.items(), key=lambda x: -x[1]['F1']):
        print(f"{name:<18} {m['MAE']:<10.1f} {m['RMSE']:<10.1f} {m['SAE']:<8.4f} {m['F1']:<8.4f} {m['Activity_%']:<8.1f}")
    
    # Summary stats
    avg_f1 = np.mean([m['F1'] for m in test_metrics.values()])
    avg_mae = np.mean([m['MAE'] for m in test_metrics.values()])
    print("-" * 70)
    print(f"{'AVERAGE':<18} {avg_mae:<10.1f} {'':10} {'':8} {avg_f1:<8.4f}")
    
    # Identify problem appliances (F1 < 0.5)
    problem_apps = [n for n, m in test_metrics.items() if m['F1'] < 0.5]
    if problem_apps:
        print(f"\n⚠ Low F1 (<0.5) appliances: {', '.join(problem_apps)}")
    
    # Save results
    results = {
        'appliances': appliances,
        'test_metrics': test_metrics,
        'best_val_loss': best_val_loss,
        'total_time_minutes': total_time,
        'n_parameters': n_params,
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items()},
    }
    
    with open(config.save_path / 'nilmformer_paper_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {config.save_path / 'nilmformer_paper_results.json'}")
    
    return model, test_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NILMFormer (Paper Aligned)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to model_ready folder with X_train.npy, y_train.npy, etc.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d_model', type=int, default=96)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--loss', type=str, default='sparsity', 
                        choices=['mse', 'huber', 'sparsity', 'weighted'],
                        help='Loss type: mse (paper), huber (δ=50W), sparsity (recommended), weighted')
    parser.add_argument('--appliance', type=str, default=None,
                        help='Single appliance to train on (e.g., HeatPump). If not set, trains on all.')
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision')
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        patience=args.patience,
        loss_type=args.loss,
        use_amp=not args.no_amp,
    )
    
    # Override data path if provided
    if args.data_path:
        config.data_path = Path(args.data_path)
    
    # Single appliance mode
    if args.appliance:
        config.appliances = [args.appliance]
        print(f"\n*** SINGLE APPLIANCE MODE: {args.appliance} ***\n")
    
    train(config)
