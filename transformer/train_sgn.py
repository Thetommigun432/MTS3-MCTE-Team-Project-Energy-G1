"""
Train NILMFormer SGN (Subtask Gated Network)
=============================================
Improved architecture based on literature research:
1. Gated output: y = regression * sigmoid(classification)
2. Focal Loss for sparse class handling
3. BCE weight >> Regression weight (10:1)
4. Smaller model for memory efficiency

Usage:
    python train_sgn.py --data_path data/processed/1sec_new
    python train_sgn.py --epochs 50 --batch_size 32
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
from typing import Dict, List, Tuple
import time

from nilmformer_sgn import create_nilmformer_sgn, SGNLoss


# =============================================================================
# DATASET
# =============================================================================
class NILMDataset(Dataset):
    """Simple NILM dataset."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, appliances: List[str]):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.appliances = appliances
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y_dict = {app: self.y[idx, :, i:i+1] for i, app in enumerate(self.appliances)}
        return x, y_dict


# =============================================================================
# METRICS
# =============================================================================
def compute_metrics(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    on_thresholds: Dict[str, float],
    P_MAX_kW: float = 13.5118,
) -> Dict[str, Dict[str, float]]:
    """Compute MAE, F1, Precision, Recall per appliance."""
    metrics = {}
    
    for app in predictions.keys():
        pred = predictions[app].flatten()
        true = targets[app].flatten()
        
        # Convert to Watts
        pred_W = pred * P_MAX_kW * 1000
        true_W = true * P_MAX_kW * 1000
        
        # MAE
        mae = np.mean(np.abs(pred_W - true_W))
        
        # F1 with threshold
        threshold = on_thresholds.get(app, 0.001)
        pred_on = (pred > threshold).astype(int)
        true_on = (true > threshold).astype(int)
        
        tp = np.sum((pred_on == 1) & (true_on == 1))
        fp = np.sum((pred_on == 1) & (true_on == 0))
        fn = np.sum((pred_on == 0) & (true_on == 1))
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        metrics[app] = {
            'MAE': float(mae),
            'F1': float(f1),
            'Precision': float(precision),
            'Recall': float(recall),
        }
    
    return metrics


# =============================================================================
# TRAINING
# =============================================================================
def train_epoch(model, loader, optimizer, criterion, device, scaler=None, grad_clip=1.0):
    model.train()
    total_loss = 0
    n_batches = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = {k: v.to(device) for k, v in y.items()}
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast(device_type='cuda'):
                outputs = model(x)
                losses = criterion(outputs, y)
        else:
            outputs = model(x)
            losses = criterion(outputs, y)
        
        loss = losses['total']
        
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
        
        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, on_thresholds):
    model.eval()
    total_loss = 0
    n_batches = 0
    
    all_preds = {}
    all_targets = {}
    
    for x, y in tqdm(loader, desc='Validating', leave=False):
        x = x.to(device)
        y_dev = {k: v.to(device) for k, v in y.items()}
        
        outputs = model(x)
        losses = criterion(outputs, y_dev)
        
        total_loss += losses['total'].item()
        n_batches += 1
        
        # Collect gated predictions
        for app in outputs.keys():
            if app not in all_preds:
                all_preds[app] = []
                all_targets[app] = []
            all_preds[app].append(outputs[app]['gated'].cpu().numpy())
            all_targets[app].append(y[app].numpy())
    
    # Compute metrics
    preds_np = {k: np.concatenate(v) for k, v in all_preds.items()}
    targets_np = {k: np.concatenate(v) for k, v in all_targets.items()}
    
    metrics = compute_metrics(preds_np, targets_np, on_thresholds)
    
    return total_loss / max(n_batches, 1), metrics


# =============================================================================
# MAIN
# =============================================================================
def main(args):
    print("=" * 70)
    print("NILMFormer SGN Training (Subtask Gated Network)")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load data
    data_path = Path(args.data_path) / 'model_ready'
    print(f"Loading data from {data_path}...")
    
    X_train = np.load(data_path / 'X_train.npy')
    y_train = np.load(data_path / 'y_train.npy')
    X_val = np.load(data_path / 'X_val.npy')
    y_val = np.load(data_path / 'y_val.npy')
    
    with open(data_path / 'metadata.json') as f:
        metadata = json.load(f)
    
    appliances = metadata['target_appliances']
    P_MAX_kW = metadata.get('scaling', {}).get('P_MAX', 13.5118)
    
    print(f"  X_train: {X_train.shape}")
    print(f"  Appliances: {appliances}")
    
    # ON thresholds (normalized)
    on_thresholds = {
        'HeatPump': 100 / (P_MAX_kW * 1000),
        'Dishwasher': 30 / (P_MAX_kW * 1000),
        'WashingMachine': 50 / (P_MAX_kW * 1000),
        'Dryer': 50 / (P_MAX_kW * 1000),
        'Oven': 100 / (P_MAX_kW * 1000),
        'Stove': 50 / (P_MAX_kW * 1000),
        'RangeHood': 20 / (P_MAX_kW * 1000),
        'EVCharger': 100 / (P_MAX_kW * 1000),
        'EVSocket': 100 / (P_MAX_kW * 1000),
        'GarageCabinet': 25 / (P_MAX_kW * 1000),
        'RainwaterPump': 50 / (P_MAX_kW * 1000),
    }
    
    # Datasets
    train_ds = NILMDataset(X_train, y_train, appliances)
    val_ds = NILMDataset(X_val, y_val, appliances)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # Model (SMALLER!)
    c_embedding = X_train.shape[2] - 1  # temporal features
    model = create_nilmformer_sgn(
        appliances=appliances,
        c_embedding=c_embedding,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    )
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    print(f"  d_model={args.d_model}, n_layers={args.n_layers}, n_heads={args.n_heads}")
    
    # Loss (SGN with Focal)
    criterion = SGNLoss(
        lambda_reg=args.lambda_reg,
        lambda_cls=args.lambda_cls,
        on_thresholds=on_thresholds,
    )
    print(f"\nLoss weights: Reg={args.lambda_reg}, Cls={args.lambda_cls}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 100)
    
    # AMP
    scaler = GradScaler('cuda') if device == 'cuda' and args.use_amp else None
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler, args.grad_clip)
        val_loss, val_metrics = validate(model, val_loader, criterion, device, on_thresholds)
        
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        
        # Average metrics
        avg_mae = np.mean([m['MAE'] for m in val_metrics.values()])
        avg_f1 = np.mean([m['F1'] for m in val_metrics.values()])
        
        print(f"Epoch {epoch+1:03d}/{args.epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"MAE: {avg_mae:.1f}W | F1: {avg_f1:.3f} | LR: {lr:.1e}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'metrics': val_metrics,
            }, save_path / 'sgn_best.pth')
            print(f"  ✓ New best! Saved to {save_path / 'sgn_best.pth'}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⚠ Early stopping at epoch {epoch+1}")
                break
    
    total_time = (time.time() - start_time) / 60
    print(f"\nTraining completed in {total_time:.1f} minutes")
    
    # Final results
    print("\n" + "=" * 70)
    print("Final Validation Metrics")
    print("=" * 70)
    print(f"{'Appliance':<18} {'MAE(W)':<10} {'F1':<8} {'Prec':<8} {'Recall':<8}")
    print("-" * 52)
    
    for app, m in sorted(val_metrics.items(), key=lambda x: -x[1]['F1']):
        print(f"{app:<18} {m['MAE']:<10.1f} {m['F1']:<8.3f} {m['Precision']:<8.3f} {m['Recall']:<8.3f}")
    
    print("-" * 52)
    print(f"{'AVERAGE':<18} {avg_mae:<10.1f} {avg_f1:<8.3f}")
    
    # Save results
    results = {
        'model': 'NILMFormerSGN',
        'n_params': n_params,
        'd_model': args.d_model,
        'n_layers': args.n_layers,
        'best_val_loss': best_val_loss,
        'avg_mae': avg_mae,
        'avg_f1': avg_f1,
        'per_appliance': val_metrics,
        'training_time_min': total_time,
    }
    
    with open(save_path / 'sgn_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {save_path / 'sgn_results.json'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NILMFormer SGN')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='transformer/checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    # Model (smaller!)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Loss weights (BCE >> Reg)
    parser.add_argument('--lambda_reg', type=float, default=1.0)
    parser.add_argument('--lambda_cls', type=float, default=10.0)
    
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--no_amp', action='store_true')
    
    args = parser.parse_args()
    if args.no_amp:
        args.use_amp = False
    
    main(args)
