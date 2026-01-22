"""
Script per DISTRIBUTED TRAINING su più laptop (PyTorch DDP).
Allena UN SINGOLO modello distribuendo i dati su più machine.

Setup:
1. Assicurati che tutti i laptop siano sulla stessa rete
2. Inizia da un laptop "master" (con IP noto):
   python train_distributed.py --model unet --appliance heatpump --master-addr 192.168.1.100 --rank 0 --world-size 3
3. Dagli altri laptop (worker):
   python train_distributed.py --model unet --appliance heatpump --master-addr 192.168.1.100 --rank 1 --world-size 3
   python train_distributed.py --model unet --appliance heatpump --master-addr 192.168.1.100 --rank 2 --world-size 3

--rank: numero del processo (0=master, 1,2,...=workers)
--world-size: numero totale di processi (=numero di laptop)
--master-addr: IP del laptop master
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import argparse
import os
import pickle
import time
from pathlib import Path
import json
from datetime import datetime

# ==================== CONFIG ====================
APPLIANCES = [
    'chargingstation_socket', 'dishwasher', 'dryer', 'garagecabinet',
    'heatpump', 'heatpump_controller', 'oven', 'rainwaterpump',
    'rangehood', 'smappeecharger', 'stove', 'washingmachine'
]

# ==================== DATASET ====================
class EnergyDataset(Dataset):
    def __init__(self, X, y, augment=False, noise_std=0.01):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.augment = augment
        self.noise_std = noise_std
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = torch.clamp(x + noise, 0.0, 1.0)
        return x, self.y[idx]

# ==================== MODELLI ====================
class CNNSeq2Seq(nn.Module):
    """CNN Encoder-Decoder per NILM"""
    def __init__(self, input_channels=7, hidden_channels=48, num_layers=3):
        super(CNNSeq2Seq, self).__init__()
        
        encoder_layers = []
        in_ch = input_channels
        for i in range(num_layers):
            out_ch = hidden_channels * (2 ** i)
            encoder_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.25)
            ])
            in_ch = out_ch
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.bottleneck_ch = hidden_channels * (2 ** (num_layers - 1))
        
        decoder_layers = []
        in_ch = self.bottleneck_ch
        for i in range(num_layers - 1, -1, -1):
            out_ch = hidden_channels * (2 ** i) if i > 0 else hidden_channels
            decoder_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            in_ch = out_ch
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.output_layer = nn.Sequential(
            nn.Conv1d(hidden_channels, 1, kernel_size=1),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = x.transpose(1, 2)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        output = self.output_layer(decoded)
        return output.transpose(1, 2)

class UNet1D(nn.Module):
    """U-Net 1D per NILM con skip connections"""
    def __init__(self, input_channels=7, base_channels=24):
        super(UNet1D, self).__init__()
        
        self.enc1 = self._conv_block(input_channels, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._conv_block(base_channels * 4, base_channels * 8)
        
        self.bottleneck = self._conv_block(base_channels * 8, base_channels * 16)
        
        self.dec4 = self._conv_block(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.dec3 = self._conv_block(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = self._conv_block(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = self._conv_block(base_channels * 2 + base_channels, base_channels)
        
        self.output = nn.Sequential(
            nn.Conv1d(base_channels, 1, kernel_size=1),
            nn.ReLU()
        )
        
        self.pool = nn.MaxPool1d(2)
        
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.1),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25)
        )
    
    def _upsample_and_concat(self, x, skip):
        x = nn.functional.interpolate(x, size=skip.shape[2], mode='linear', align_corners=True)
        return torch.cat([x, skip], dim=1)
    
    def forward(self, x):
        original_len = x.shape[1]
        x = x.transpose(1, 2)
        
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        b = self.bottleneck(self.pool(e4))
        
        d4 = self.dec4(self._upsample_and_concat(b, e4))
        d3 = self.dec3(self._upsample_and_concat(d4, e3))
        d2 = self.dec2(self._upsample_and_concat(d3, e2))
        d1 = self.dec1(self._upsample_and_concat(d2, e1))
        
        out = self.output(d1)
        
        if out.shape[2] != original_len:
            out = nn.functional.interpolate(out, size=original_len, mode='linear', align_corners=True)
        
        return out.transpose(1, 2)

class NILMLoss(nn.Module):
    """Loss personalizzata per NILM con focus sui picchi"""
    def __init__(self, alpha=0.5):
        super(NILMLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss(reduction='none')
        
    def forward(self, pred, target):
        mse = self.mse(pred, target)
        weight = 1.0 + 99.0 * (target / (target.max() + 1e-8))
        weighted_loss = (mse * weight).mean()
        under_pred = torch.clamp(target - pred, min=0)
        under_penalty = (under_pred ** 2 * weight * 2).mean()
        mae = torch.abs(pred - target).mean()
        return weighted_loss + under_penalty + self.alpha * mae

# ==================== TRAINING ====================
def load_data(appliance, data_base_path="data/processed/15min/model_ready"):
    """Carica i dati .npy per un'appliance"""
    appliance_path = os.path.join(data_base_path, appliance)
    
    X_train = np.load(os.path.join(appliance_path, "X_train.npy"))
    X_val = np.load(os.path.join(appliance_path, "X_val.npy"))
    X_test = np.load(os.path.join(appliance_path, "X_test.npy"))
    y_train = np.load(os.path.join(appliance_path, "y_train.npy"))
    y_val = np.load(os.path.join(appliance_path, "y_val.npy"))
    y_test = np.load(os.path.join(appliance_path, "y_test.npy"))
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def setup_distributed(rank, world_size, master_addr, master_port=29500):
    """Setup comunicazione distribuita tra processi"""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    
    # Per Windows, usa 'gloo' se disponibile, altrimenti 'nccl'
    # Se world_size=1, non serve distributed training
    if world_size == 1:
        return  # Skip per single process
    
    # Inizializza il processo group
    try:
        dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    except Exception as e:
        print(f"⚠️  Errore init gloo: {e}, skipping distributed training")
        return

def cleanup_distributed():
    """Pulizia dopo training distribuito"""
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except:
        pass

def is_main_process(rank):
    """Check se questo è il processo master (rank 0)"""
    return rank == 0

def train_epoch(model, train_loader, criterion, optimizer, device, rank, world_size):
    """Allena per un'epoca (versione distribuita)"""
    model.train()
    epoch_loss = 0
    num_batches = len(train_loader)
    
    for batch_idx, (inp, tgt) in enumerate(train_loader):
        inp, tgt = inp.to(device), tgt.to(device)
        
        pred = model(inp)
        loss = criterion(pred, tgt)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Progress ogni 100 batch
        if rank == 0 and (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx+1}/{num_batches} - Loss: {loss.item():.4f}")
    
    # Sincronizza loss tra i processi (solo se distribuito)
    if world_size > 1 and dist.is_initialized():
        loss_tensor = torch.tensor([epoch_loss], device=device)
        dist.all_reduce(loss_tensor)
        loss_tensor /= len(train_loader)
        return float(loss_tensor)
    
    return epoch_loss / len(train_loader)

def validate(model, val_loader, criterion, device, rank, world_size):
    """Valuta il modello (versione distribuita)"""
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for inp, tgt in val_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            pred = model(inp)
            val_loss += criterion(pred, tgt).item()
    
    # Sincronizza loss tra i processi (solo se distribuito)
    if world_size > 1 and dist.is_initialized():
        loss_tensor = torch.tensor([val_loss], device=device)
        dist.all_reduce(loss_tensor)
        loss_tensor /= len(val_loader)
        return float(loss_tensor)
    
    return val_loss / len(val_loader)

def train_model(args):
    """Funzione principale di training distribuito"""
    
    # Se world_size = 1, disabilita distributed training
    use_distributed = args.world_size > 1
    
    if use_distributed:
        setup_distributed(args.rank, args.world_size, args.master_addr, args.master_port)
    
    is_main = is_main_process(args.rank)
    
    if is_main:
        print(f"\n{'='*60}")
        print(f"🚀 DISTRIBUTED TRAINING {args.model.upper()} su {args.appliance}")
        print(f"{'='*60}")
        print(f"Processi: {args.world_size} (questo è rank {args.rank})")
        print(f"Master: {args.master_addr}:{args.master_port}\n")
    
    # Device (forza CPU se richiesto)
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if is_main:
        print(f"Device: {device}")
    
    # Carica dati
    if is_main:
        print(f"📂 Caricamento dati {args.appliance}...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(args.appliance)
    if is_main:
        print(f"   Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    
    # Dataset
    train_dataset = EnergyDataset(X_train, y_train, augment=True, noise_std=0.015)
    val_dataset = EnergyDataset(X_val, y_val, augment=False)
    
    # DistributedSampler divide i dati tra i processi
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=args.world_size, 
        rank=args.rank, 
        shuffle=True
    ) if use_distributed else None
    val_sampler = DistributedSampler(
        val_dataset, 
        num_replicas=args.world_size, 
        rank=args.rank, 
        shuffle=False
    ) if use_distributed else None
    
    # DataLoader senza shuffle (già fatto dal sampler)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler, 
        shuffle=train_sampler is None,
        num_workers=2, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        sampler=val_sampler,
        shuffle=False,
        num_workers=2, 
        pin_memory=True
    )
    
    # Modello
    num_input_features = X_train.shape[2]
    if args.model == 'cnn':
        model = CNNSeq2Seq(input_channels=num_input_features, hidden_channels=48, num_layers=5)
    elif args.model == 'unet':
        model = UNet1D(input_channels=num_input_features, base_channels=24)
    else:
        raise ValueError(f"Modello non supportato: {args.model}")
    
    model.to(device)
    
    # Wrap il modello con DistributedDataParallel (solo se world_size > 1)
    if args.world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[0] if torch.cuda.is_available() else None)
    
    if is_main:
        num_params = sum(p.numel() for p in (model.module if args.world_size > 1 else model).parameters())
        print(f"\n📊 Modello: {num_params:,} parametri (distribuito su {args.world_size} processi)")
    
    # Loss, optimizer, scheduler
    criterion = NILMLoss(alpha=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)
    
    # Training
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    if is_main:
        print(f"\n🔥 Training per {args.epochs} epoche (patience={args.patience})...")
        print(f"   Dati distribuiti tra {args.world_size} processi\n")
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Sincronizza i sampler tra i processi
        if use_distributed:
            train_sampler.set_epoch(epoch)
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, args.rank, args.world_size)
        val_loss = validate(model, val_loader, criterion, device, args.rank, args.world_size)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Early stopping (solo sul rank 0)
        if is_main:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Salva lo state dict (con o senza .module a seconda di DDP)
                if use_distributed:
                    best_model_state = model.module.state_dict().copy()
                else:
                    best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
                elapsed = time.time() - epoch_start
                print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                      f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                      f"Patience: {patience_counter}/{args.patience} | {elapsed:.1f}s")
            
            if patience_counter >= args.patience:
                print(f"\n⚠️  Early stopping all'epoca {epoch+1}")
                break
        
        # Sincronizza early stopping tra i processi
        if args.world_size > 1 and dist.is_initialized():
            should_stop = torch.tensor([patience_counter >= args.patience], dtype=torch.bool, device=device)
            dist.broadcast(should_stop, src=0)
            if should_stop.item():
                break
    
    # Carica il miglior modello (solo su rank 0)
    if is_main:
        if args.world_size > 1:
            model.module.load_state_dict(best_model_state)
        else:
            model.load_state_dict(best_model_state)
    
    # Sincronizza il modello finale su tutti i processi
    if args.world_size > 1 and dist.is_initialized():
        dist.barrier()
    
    # Salva il modello (solo dal rank 0)
    if is_main:
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / f"{args.model}_{args.appliance}_distributed_model.pth"
        torch.save({
            'model_state_dict': (model.module if args.world_size > 1 else model).state_dict(),
            'hyperparameters': {
                'model': args.model,
                'appliance': args.appliance,
                'num_input_features': num_input_features,
                'epochs_trained': epoch + 1,
                'world_size': args.world_size
            },
            'history': history,
            'best_val_loss': best_val_loss
        }, model_path)
        
        # Salva il log
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"{args.model}_{args.appliance}_distributed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(log_path, 'w') as f:
            json.dump({
                'model': args.model,
                'appliance': args.appliance,
                'world_size': args.world_size,
                'epochs': epoch + 1,
                'best_val_loss': float(best_val_loss),
                'num_params': num_params,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'training_time_minutes': (time.time() - start_time) / 60,
                'speedup': f"~{args.world_size}x (teoricamente)",
                'history': {
                    'train_loss': [float(x) for x in history['train_loss']],
                    'val_loss': [float(x) for x in history['val_loss']]
                }
            }, f, indent=2)
        
        print(f"\n✅ Training distribuito completato!")
        print(f"   Modello salvato: {model_path}")
        print(f"   Log salvato: {log_path}")
        print(f"   Tempo totale: {(time.time() - start_time)/60:.1f} min")
        print(f"   Speedup: ~{args.world_size}x con {args.world_size} processi")
        print(f"   Best val loss: {best_val_loss:.6f}\n")
    
    cleanup_distributed()

# ==================== CLI ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed training NILM su più laptop")
    parser.add_argument('--model', type=str, choices=['cnn', 'unet'], default='cnn',
                       help='Tipo di modello (default: cnn)')
    parser.add_argument('--appliance', type=str, choices=APPLIANCES, default='heatpump',
                       help='Appliance da allenare (default: heatpump)')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Numero di epoche (default: 150)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size PER PROCESSO (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.0008,
                       help='Learning rate (default: 0.0008)')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience (default: 5)')
    
    # Parametri per distributed training
    parser.add_argument('--rank', type=int, required=True,
                       help='Rank del processo (0=master, 1,2,...=workers)')
    parser.add_argument('--world-size', type=int, required=True,
                       help='Numero totale di processi (laptop)')
    parser.add_argument('--master-addr', type=str, required=True,
                       help='IP del laptop master (es. 192.168.1.100)')
    parser.add_argument('--master-port', type=int, default=29500,
                       help='Porta del master (default: 29500)')
    parser.add_argument('--cpu', action='store_true',
                       help='Forza uso della CPU invece di CUDA')
    
    args = parser.parse_args()
    train_model(args)
