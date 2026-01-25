"""
NILM Training Pipeline
======================
Complete training script for Hybrid CNN-Transformer NILM.

Usage:
    python train.py                           # Train with defaults
    python train.py --epochs 50               # Custom epochs
    python train.py --appliance heatpump      # Single appliance mode
    
Features:
- Multi-appliance training (one model, multiple outputs)
- Mixed precision training (if CUDA available)
- Cosine annealing learning rate schedule
- Early stopping with patience
- TensorBoard logging
- Checkpoint saving and resumption
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

# Local imports
from config import Config, get_config
from model import HybridCNNTransformer, create_model
from dataset import (
    load_and_prepare_data, create_dataloaders, save_scalers,
    load_pretrained_data, create_pretrained_dataloaders
)
from losses import NILMLoss, create_loss
from utils import (
    EarlyStopping, AverageMeter, Timer,
    calculate_metrics, format_metrics,
    print_model_summary, save_checkpoint, load_checkpoint
)


def train_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion: NILMLoss,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    grad_clip: float = 1.0
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: NILM model
        loader: Training DataLoader
        optimizer: Optimizer
        criterion: Loss function
        device: Device
        scaler: GradScaler for mixed precision
        grad_clip: Gradient clipping value
        
    Returns:
        Dict of loss values
    """
    model.train()
    
    loss_meters = {
        'total': AverageMeter(),
        'mse': AverageMeter(),
        'bce': AverageMeter()
    }
    
    pbar = tqdm(loader, desc='Training', leave=False, ncols=100)
    for batch_idx, (x, y) in enumerate(pbar):
        x = x.to(device)
        y = {name: {k: v.to(device) for k, v in vals.items()} 
             for name, vals in y.items()}
        
        optimizer.zero_grad()
        
        # Forward pass (with optional mixed precision)
        if scaler is not None:
            with autocast(device_type='cuda'):
                pred = model(x)
                losses = criterion(pred, y)
        else:
            pred = model(x)
            losses = criterion(pred, y)
        
        loss = losses['total']
        
        # Backward pass
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
        
        # Update meters
        batch_size = x.size(0)
        loss_meters['total'].update(losses['total'].item(), batch_size)
        loss_meters['mse'].update(losses['mse'].item(), batch_size)
        loss_meters['bce'].update(losses['bce'].item(), batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_meters['total'].avg:.4f}",
            'mse': f"{loss_meters['mse'].avg:.4f}"
        })
    
    pbar.close()
    return {k: v.avg for k, v in loss_meters.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: NILMLoss,
    device: torch.device,
    scalers_y: Optional[Dict] = None
) -> tuple:
    """
    Validate model.
    
    Args:
        model: NILM model
        loader: Validation DataLoader
        criterion: Loss function
        device: Device
        scalers_y: Target scalers for metric calculation
        
    Returns:
        (loss_dict, metrics_dict)
    """
    model.eval()
    
    loss_meters = {
        'total': AverageMeter(),
        'mse': AverageMeter(),
        'bce': AverageMeter()
    }
    
    all_preds = {}
    all_targets = {}
    all_states_pred = {}
    all_states_true = {}
    
    pbar = tqdm(loader, desc='Validating', leave=False, ncols=100)
    for x, y in pbar:
        x = x.to(device)
        y_device = {name: {k: v.to(device) for k, v in vals.items()} 
                   for name, vals in y.items()}
        
        pred = model(x)
        losses = criterion(pred, y_device)
        
        batch_size = x.size(0)
        loss_meters['total'].update(losses['total'].item(), batch_size)
        loss_meters['mse'].update(losses['mse'].item(), batch_size)
        loss_meters['bce'].update(losses['bce'].item(), batch_size)
        
        # Collect predictions for metrics
        for name in pred.keys():
            if name not in all_preds:
                all_preds[name] = []
                all_targets[name] = []
                all_states_pred[name] = []
                all_states_true[name] = []
            
            all_preds[name].append(pred[name]['power'].cpu().numpy())
            all_targets[name].append(y[name]['power'].numpy())
            all_states_pred[name].append(torch.sigmoid(pred[name]['state']).cpu().numpy())
            all_states_true[name].append(y[name]['state'].numpy())
    
    # Concatenate all predictions
    import numpy as np
    preds_np = {k: np.concatenate(v) for k, v in all_preds.items()}
    targets_np = {k: np.concatenate(v) for k, v in all_targets.items()}
    states_pred_np = {k: np.concatenate(v) for k, v in all_states_pred.items()}
    states_true_np = {k: np.concatenate(v) for k, v in all_states_true.items()}
    
    # Calculate metrics
    metrics = calculate_metrics(
        preds_np, targets_np,
        states_pred_np, states_true_np,
        scalers_y
    )
    
    return {k: v.avg for k, v in loss_meters.items()}, metrics


def main(args):
    """Main training function."""
    
    # Configuration
    config = get_config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        loss_mse_weight=args.loss_mse_weight,
        loss_bce_weight=args.loss_bce_weight,
        window_size=args.window_size,
        model_type=args.model_type,
        loss_type=args.loss_type,
        fn_weight=args.fn_weight,
        seq2point=(not args.seq2seq)
    )

    
    # Override data path if provided
    if args.data_path:
        config.data_path = Path(args.data_path)
    
    # Single appliance mode
    if args.appliance:
        config.appliances = [args.appliance]

    # Initialize TensorBoard writer
    log_dir = config.save_path / 'runs' / time.strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")
    
    print("="*70)
    print("HYBRID CNN-TRANSFORMER NILM TRAINING")
    print("="*70)
    print(f"Device: {config.device}")
    print(f"Appliances: {config.appliances}")
    print(f"Window size: {config.window_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print("="*70)
    
    # Load and prepare data
    print("\n[1/5] Loading data...")
    
    # Try pretrained numpy arrays first (from pretraining notebooks)
    # model_ready: NILM-correct scaling (power/P_MAX, temporal untouched)
    model_ready_subdir = getattr(config, 'model_ready_subdir', 'model_ready')
    model_ready_path = config.data_path / model_ready_subdir
    use_pretrained = args.use_pretrained or (model_ready_path / 'X_train.npy').exists()
    
    if use_pretrained and (model_ready_path / 'X_train.npy').exists():
        print(f"Using pretrained numpy arrays from {model_ready_path}")
        
        # Load pretrained data
        train_data, val_data, test_data, metadata = load_pretrained_data(model_ready_path)
        
        # Get appliance names from metadata
        appliance_names = metadata.get('target_appliances', config.appliances)
        config.appliances = appliance_names
        
        # Respect --appliance override for single-appliance training
        if args.appliance and args.appliance in appliance_names:
            config.appliances = [args.appliance]
            appliance_names = [args.appliance]
            print(f"  Single appliance mode: {args.appliance}")
        
        # Override window size from metadata if available
        if 'window_size' in metadata:
            # Only override if explicit arg not provided and we want to match data
            # But here we want to allow cropping (Config=512, Data=1024)
            if args.window_size is None and config.window_size == 1024: # Default was 1024? No, it's 512 now.
                 # If config matches metadata, no op. If config < metadata, we want config (cropping).
                 pass
            
            print(f"  Window size from metadata: {metadata['window_size']} (Config: {config.window_size})")
        
        # Get P_MAX for threshold normalization
        P_MAX_kW = metadata.get('P_MAX', config.P_MAX_kW)
        print(f"  P_MAX: {P_MAX_kW:.4f} kW (for de-scaling predictions)")
        print(f"  Appliances from metadata: {appliance_names}")
        
        # Create DataLoaders (with derivative features)
        print("\n[2/5] Creating dataloaders...")
        add_derivative_features = getattr(config, 'add_derivative_features', True)
        train_loader, val_loader, test_loader = create_pretrained_dataloaders(
            config, train_data, val_data, test_data, appliance_names, 
            P_MAX_kW=P_MAX_kW, 
            add_derivative_features=add_derivative_features
        )
        
        # Store n_features for model creation (7 base + 3 derivative = 10)
        config.n_features = 7 + (3 if add_derivative_features else 0)
        
        # Store P_MAX in config for inference
        config.P_MAX_kW = P_MAX_kW
        
        # No separate scalers needed - data already scaled
        # BUT we need to create mock scalers for metrics calculation (inverse transform)
        # Pretrained data is scaled by P_MAX_kW
        
        class MockScaler:
            def __init__(self, p_max):
                self.p_max = p_max * 1000.0  # Convert kW to W
                
            def inverse_transform(self, x):
                return x * self.p_max
                
        # Create a mock scaler for each appliance
        scalers_y = {app: MockScaler(P_MAX_kW) for app in appliance_names}
        print(f"Created mock scalers using P_MAX={P_MAX_kW:.4f} kW")
        
    else:
        # Fall back to raw parquet loading (legacy mode)
        print("Using raw parquet data (legacy mode)")
        
        # Try to find data file
        data_file = None
        candidates = [
            config.data_path / 'nilm_ready_1sec_new.parquet',
            config.data_path / 'nilm_ready_1sec.parquet',
            config.data_path / 'nilm_ready_dataset.parquet',
        ]
        
        for candidate in candidates:
            if candidate.exists():
                data_file = candidate
                break
        
        if data_file is None:
            print(f"Error: No data file found. Tried: {candidates}")
            print("Please run pretraining notebook first or provide --data_path")
            return
        
        print(f"Using data: {data_file}")
        
        train_data, val_data, test_data, scaler_agg, scalers_y = load_and_prepare_data(
            data_file,
            config.appliances,
            config.temporal_features
        )
        
        # Filter to appliances that exist in data
        available_appliances = list(train_data['targets'].keys())
        config.appliances = available_appliances
        print(f"Available appliances in data: {available_appliances}")
        
        if not available_appliances:
            print("Error: No target appliances found in data!")
            return
        
        # Create DataLoaders
        print("\n[2/5] Creating dataloaders...")
        train_loader, val_loader, test_loader = create_dataloaders(
            config, train_data, val_data, test_data
        )
        
        # Save scalers
        save_scalers(scaler_agg, scalers_y, config.save_path)
    
    # Create model
    print("\n[3/5] Creating model...")
    model = create_model(config)
    print_model_summary(model, config)
    
    # Loss, optimizer, scheduler
    criterion = create_loss(config)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.lr / 100
    )
    
    # Mixed precision scaler
    scaler = GradScaler('cuda') if config.device.type == 'cuda' else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.patience)
    
    # Resume from checkpoint if exists
    start_epoch = 0
    best_val_loss = float('inf')
    checkpoint_path = config.save_path / 'hybrid_nilm_best.pth'
    
    if args.resume and checkpoint_path.exists():
        print(f"Resuming from {checkpoint_path}")
        start_epoch, _ = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
        start_epoch += 1
    
    # Training loop
    print("\n[4/5] Training...")
    print("-"*70)
    
    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    start_time = time.time()
    
    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        
        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, criterion,
            config.device, scaler, config.grad_clip
        )
        
        # Validate
        val_losses, val_metrics = validate(
            model, val_loader, criterion, config.device, scalers_y
        )
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_losses['total'])
        history['val_loss'].append(val_losses['total'])
        history['val_metrics'].append(val_metrics)
        
        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            save_checkpoint(
                model, optimizer, scheduler,
                epoch, val_losses['total'], val_metrics,
                checkpoint_path, config
            )
        
        # Logging
        epoch_time = time.time() - epoch_start
        lr = scheduler.get_last_lr()[0]
        
        # Logging - every epoch
        print(f"Epoch {epoch+1:3d}/{config.epochs} | "
              f"Train: {train_losses['total']:.4f} | "
              f"Val: {val_losses['total']:.4f} | "
              f"LR: {lr:.2e} | "
              f"Time: {epoch_time:.1f}s")
        
        # TensorBoard Logging
        writer.add_scalar('Loss/Train', train_losses['total'], epoch)
        writer.add_scalar('Loss/Val', val_losses['total'], epoch)
        writer.add_scalar('LR', lr, epoch)
        
        # Log aggregated validation metrics
        agg_metrics = calculate_aggregate_metrics(val_metrics)
        for metric_name, val in agg_metrics.items():
             writer.add_scalar(f'Metrics/Val_Avg_{metric_name.replace("avg_", "")}', val, epoch)

        # Log per-appliance metrics
        for app, metrics in val_metrics.items():
            for metric_name, val in metrics.items():
                writer.add_scalar(f'Metrics/Val_{app}_{metric_name}', val, epoch)
        
        # Print metrics for the trained appliance
        if args.appliance and args.appliance in val_metrics:
            m = val_metrics[args.appliance]
            print(f"      >> {args.appliance} | MAE: {m.get('MAE', 0.0):.1f}W | F1: {m.get('F1', 0.0):.3f}")
        
        # Early stopping
        if early_stopping(val_losses['total']):
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    
    # Evaluate on test set
    print("\n[5/5] Evaluating on test set...")
    
    # Load best model
    load_checkpoint(checkpoint_path, model)
    
    test_losses, test_metrics = validate(
        model, test_loader, criterion, config.device, scalers_y
    )
    
    print(format_metrics(test_metrics, "TEST RESULTS"))
    
    # Save final results
    results = {
        'appliances': config.appliances,
        'test_metrics': test_metrics,
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch + 1,
        'total_time_minutes': total_time / 60,
        'n_parameters': model.count_parameters(),
        'config': {k: str(v) if hasattr(v, '__fspath__') else v 
                   for k, v in vars(config).items() if not k.startswith('_')}
    }
    
    results_path = config.save_path / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Best model saved to: {checkpoint_path}")
    print(f"✅ Results saved to: {results_path}")
    print(f"✅ Scalers saved to: {config.save_path / 'scalers.pkl'}")
    
    writer.close()
    return model, history, test_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NILM Hybrid CNN-Transformer Training')
    
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--loss_mse_weight', type=float, default=1.0,
                        help='MSE loss weight')
    parser.add_argument('--loss_bce_weight', type=float, default=2.0,
                        help='BCE loss weight')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to data directory')
    parser.add_argument('--appliance', type=str, default=None,
                        help='Train single appliance only')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--use_pretrained', action='store_true',
                        help='Force use of pretrained numpy arrays from model_ready/')
    
    # Model architecture arguments
    parser.add_argument('--d_model', type=int, default=256,
                        help='Transformer embedding dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, default=1024,
                        help='Feed-forward dimension')
    parser.add_argument('--window_size', type=int, default=1024,
                        help='Input window size')
    
    # New options (Notebook features)
    parser.add_argument('--model_type', type=str, default='simple', choices=['hybrid', 'simple', 'nilmformer'],
                        help='Model architecture type (default: simple)')
    parser.add_argument('--loss_type', type=str, default='weighted_nilm', choices=['focal', 'mse', 'weighted_nilm'],
                        help='Loss function type (default: weighted_nilm)')
    parser.add_argument('--fn_weight', type=float, default=15.0,
                        help='False Negative penalty weight (for weighted_nilm)')
    parser.add_argument('--seq2seq', action='store_true',
                        help='Use Sequence-to-Sequence (predict full window) instead of Seq2Point (midpoint)')
    
    args = parser.parse_args()
    main(args)
