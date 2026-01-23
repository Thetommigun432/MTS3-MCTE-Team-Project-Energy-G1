"""
Train NILMFormer on 15-min resolution HeatPump data.
Simplified script for single-appliance training.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path
import argparse
import json

from model import create_model
from types import SimpleNamespace


def load_15min_data(data_path):
    """Load pre-windowed 15min numpy arrays."""
    X_train = np.load(data_path / 'X_train.npy')
    y_train = np.load(data_path / 'y_train.npy')
    X_val = np.load(data_path / 'X_val.npy')
    y_val = np.load(data_path / 'y_val.npy')
    X_test = np.load(data_path / 'X_test.npy')
    y_test = np.load(data_path / 'y_test.npy')
    
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"Test:  X={X_test.shape}, y={y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


class Simple15minDataset(Dataset):
    """Simple dataset for pre-windowed 15min data."""
    
    def __init__(self, X, y, seq2point=True):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq2point = seq2point
        self.mid = X.shape[1] // 2
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]  # (seq_len, features)
        
        if self.seq2point:
            # Midpoint prediction
            y_power = self.y[idx, self.mid, 0]
            y_state = 1.0 if y_power > 0.01 else 0.0
        else:
            y_power = self.y[idx, :, 0]
            y_state = (y_power > 0.01).float()
        
        return x, {
            'HeatPump': {
                'power': y_power.unsqueeze(0) if self.seq2point else y_power,
                'state': torch.tensor([y_state]) if self.seq2point else y_state
            }
        }


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for x, y_dict in loader:
        x = x.to(device)
        
        optimizer.zero_grad()
        outputs = model(x)
        
        # Get target
        y_power = y_dict['HeatPump']['power'].to(device)
        y_state = y_dict['HeatPump']['state'].to(device)
        
        # Get predictions
        pred_power = outputs['HeatPump']['power'].squeeze()
        pred_state = outputs['HeatPump']['state'].squeeze()
        
        # Loss
        loss_mse = nn.MSELoss()(pred_power, y_power.squeeze())
        loss_bce = nn.BCEWithLogitsLoss()(pred_state, y_state.squeeze())
        loss = loss_mse + 0.5 * loss_bce
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y_dict in loader:
            x = x.to(device)
            outputs = model(x)
            
            y_power = y_dict['HeatPump']['power'].to(device)
            pred_power = outputs['HeatPump']['power'].squeeze()
            
            loss = nn.MSELoss()(pred_power, y_power.squeeze())
            total_loss += loss.item()
            

            all_preds.append(pred_power.cpu().numpy().flatten())
            all_targets.append(y_power.squeeze().cpu().numpy().flatten())
    
    # Compute MAE
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    mae = np.mean(np.abs(all_preds - all_targets))
    
    return total_loss / len(loader), mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    data_path = Path('data/processed/15min/model_ready/heatpump_controller')
    X_train, y_train, X_val, y_val, X_test, y_test = load_15min_data(data_path)
    
    # Create datasets
    train_ds = Simple15minDataset(X_train, y_train, seq2point=True)
    val_ds = Simple15minDataset(X_val, y_val, seq2point=True)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    
    # Model config for 15min data
    cfg = SimpleNamespace(
        model_type='nilmformer',
        input_features=7,
        window_size=96,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=256,
        dropout=0.1,
        dilations=[1, 2, 4, 8],
        kernel_size_embed=3,
        kernel_size_head=3,
        pffn_ratio=4,
        appliances=['HeatPump'],
        seq2point=True,
        use_rope=True,
        use_flash_attention=False,
        use_stationarization=False,
        use_pooling_for_state=True,
        device=device
    )
    
    model = create_model(cfg)
    model = model.to(device)
    
    # Count params
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_loss = float('inf')
    save_path = Path('transformer/checkpoints/nilmformer_15min_best.pth')
    
    print(f"\n{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Val MAE':<10}")
    print("-" * 40)
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, nn.MSELoss(), device)
        val_loss, val_mae = validate(model, val_loader, device)
        scheduler.step()
        
        print(f"{epoch+1:<6} {train_loss:<12.6f} {val_loss:<12.6f} {val_mae:<10.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': vars(cfg),
                'epoch': epoch,
                'val_loss': val_loss
            }, save_path)
            print(f"  -> Saved best model")
    
    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
