
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import time
import json
import argparse
import math

# ==============================================================================
# CONFIGURATION
# ==============================================================================
class Config:
    # Look for the data file in common locations
    _possible_paths = [
        Path(r'.tmp/10sec/10sec/nilm_10sec_mar_may.parquet'),
        Path(r'data/processed/10sec/nilm_10sec_mar_may.parquet'),
        Path(r'c:\Users\gamek\School\TeamProject\MTS3-MCTE-Team-Project-Energy-G1\data\processed\10sec\nilm_10sec_mar_may.parquet')
    ]
    DATA_PATH = next((p for p in _possible_paths if p.exists()), _possible_paths[0])
    
    # Path where this script is located
    try:
        SAVE_PATH = Path(__file__).parent
    except NameError:
        SAVE_PATH = Path.cwd() / "model_highfreq"
    
    WINDOW_SIZE = 512  # ~85 minutes context (10s resolution)
    BATCH_SIZE = 128   # Increased for GPU speed
    EPOCHS = 5         # Reduced for quick feedback
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10
    
    # Augmentation
    NOISE_STD = 0.01
    MAGNITUDE_SCALE = 0.1
    MASK_PROB = 0.1
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Transformer Config
    D_MODEL = 128
    NHEAD = 4
    NUM_LAYERS = 2
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.1

# ==============================================================================
# DATASET
# ==============================================================================
class NILMDataset10Sec(Dataset):
    def __init__(self, X, y, window_size=512, augment=False, on_threshold=0.02, config=None):
        self.X = X
        self.y = y
        self.window_size = window_size
        self.midpoint = window_size // 2
        self.n_samples = len(X) - window_size + 1
        self.augment = augment
        self.config = config
        self.on_threshold = on_threshold
        
        # Pre-calc indices for stratification
        self.on_indices = np.where(y[window_size//2 : -window_size//2 + 1] > on_threshold)[0]
        self.off_indices = np.where(y[window_size//2 : -window_size//2 + 1] <= on_threshold)[0]

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        x = self.X[idx:idx + self.window_size].copy()
        y = self.y[idx + self.midpoint]
        
        if self.augment and self.config:
            # Noise
            if np.random.random() < 0.5:
                noise = np.random.normal(0, self.config.NOISE_STD, x.shape)
                x += noise
            # Mag Scaling
            if np.random.random() < 0.3:
                scale = np.random.uniform(1 - self.config.MAGNITUDE_SCALE, 1 + self.config.MAGNITUDE_SCALE)
                x *= scale
                
        return torch.tensor(x, dtype=torch.float32), torch.tensor([y], dtype=torch.float32)

    def get_stratified_sampler(self, on_ratio=0.4):
        n_on = len(self.on_indices)
        n_off = len(self.off_indices)
        if n_on == 0: return None
        
        w_on = on_ratio * n_off / ((1 - on_ratio) * n_on + 1e-10)
        weights = np.zeros(self.n_samples)
        weights[self.on_indices] = w_on
        weights[self.off_indices] = 1.0
        
        return WeightedRandomSampler(torch.DoubleTensor(weights), self.n_samples, replacement=True)

# ==============================================================================
# TRANSFORMER MODEL
# ==============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]

class NILMTransformer(nn.Module):
    def __init__(self, n_features=10, window_size=512):
        super().__init__()
        cfg = Config
        
        self.embedding = nn.Linear(n_features, cfg.D_MODEL)
        self.pos_encoder = PositionalEncoding(cfg.D_MODEL, max_len=window_size)
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.D_MODEL, 
            nhead=cfg.NHEAD, 
            dim_feedforward=cfg.DIM_FEEDFORWARD, 
            dropout=cfg.DROPOUT,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=cfg.NUM_LAYERS)
        
        self.fc_out = nn.Sequential(
            nn.Linear(cfg.D_MODEL, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # x: [Batch, Seq, Feat]
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # Take the midpoint token for prediction (Seq2Point)
        mid_token = x[:, x.shape[1]//2, :]
        return self.fc_out(mid_token)

# ==============================================================================
# UTILS
# ==============================================================================
class WeightedFocalMSE(nn.Module):
    def __init__(self, gamma=2.0, on_weight=10.0, threshold=0.02):
        super().__init__()
        self.gamma, self.on_weight, self.threshold = gamma, on_weight, threshold
    def forward(self, pred, target):
        mse = (pred - target) ** 2
        with torch.no_grad():
            focal = 1 + self.gamma * torch.tanh(mse / (target.abs() + 0.01))
            class_w = torch.where(target.abs() > self.threshold, 
                                  torch.tensor(self.on_weight, device=target.device), 
                                  torch.tensor(1.0, device=target.device))
        return (focal * class_w * mse).mean()

def load_data(path, appliance):
    df = pd.read_parquet(path)
    
    # 10s columns: Time, Aggregate, Stove, Dishwasher, etc.
    # Feature Engineering
    agg = df['Aggregate'].values
    
    # Derivative features?
    # For Transformer, raw sequence is often enough, but derivative helps
    dP_dt = np.zeros_like(agg)
    dP_dt[1:] = agg[1:] - agg[:-1]
    
    # Simple Temporal (resampled 10s doesn't map perfectly to hour unless properly indexed)
    dt = pd.to_datetime(df['Time'])
    hour = dt.dt.hour.values
    dow = dt.dt.dayofweek.values
    
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    features = np.column_stack([
        agg, dP_dt, hour_sin, hour_cos
    ]) # 4 features
    
    target = df[appliance].values
    return features, target

# ==============================================================================
# MAIN
# ==============================================================================
def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss = 0
    
    for i, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
        
        if i % 100 == 0:
            print(f"   Batch {i}/{len(loader)} | Loss: {loss.item():.4f}")
        
    return total_loss / len(loader)

def evaluate(model, loader, scaler_y, device, threshold):
    model.eval()
    mse_total = 0
    preds = []
    actuals = []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            p = model(X)
            
            p_inv = scaler_y.inverse_transform(p.cpu().numpy()).flatten()
            y_inv = scaler_y.inverse_transform(y.cpu().numpy()).flatten()
            
            preds.extend(p_inv)
            actuals.extend(y_inv)
            
    preds = np.array(preds)
    actuals = np.array(actuals)
    preds = np.maximum(preds, 0)
    
    mae = np.mean(np.abs(preds - actuals))
    
    p_bin = preds > threshold
    t_bin = actuals > threshold
    
    from sklearn.metrics import f1_score
    f1 = f1_score(t_bin, p_bin, zero_division=0)
    
    return {'mae': mae, 'f1': f1, 'preds': preds, 'actuals': actuals}

def main(appliance):
    print(f"🚀 High-Freq Transformer Training: {appliance}")
    print(f"🔥 DEVICE: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    cfg = Config()
    
    # Load Data
    X, y = load_data(cfg.DATA_PATH, appliance)
    
    # Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Split (Time-based for NILM)
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    thresh = 0.02 # 20W detection threshold
    
    train_ds = NILMDataset10Sec(X_scaled[:train_end], y_scaled[:train_end], cfg.WINDOW_SIZE, augment=True, config=cfg)
    val_ds = NILMDataset10Sec(X_scaled[train_end:val_end], y_scaled[train_end:val_end], cfg.WINDOW_SIZE)
    test_ds = NILMDataset10Sec(X_scaled[val_end:], y_scaled[val_end:], cfg.WINDOW_SIZE)
    
    sampler = train_ds.get_stratified_sampler(on_ratio=0.4)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    # Model
    model = NILMTransformer(n_features=X.shape[1], window_size=cfg.WINDOW_SIZE).to(cfg.DEVICE)
    opt = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    loss_fn = WeightedFocalMSE(on_weight=20.0, threshold=thresh)
    
    # Train
    best_score = float('inf')
    
    for ep in range(cfg.EPOCHS):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, opt, loss_fn, cfg.DEVICE)
        res = evaluate(model, val_loader, scaler_y, cfg.DEVICE, thresh)
        
        # Hybrid Score: MAE - 200*F1
        score = res['mae'] - 200.0 * res['f1']
        
        if score < best_score:
            best_score = score
            torch.save({
                'model': model.state_dict(),
                'scaler_X': scaler_X,
                'scaler_y': scaler_y
            }, cfg.SAVE_PATH / f'transformer_10sec_{appliance.lower()}_best.pth')
            
        print(f"Ep {ep+1} | Loss: {loss:.4f} | MAE: {res['mae']*1000:.1f}W | F1: {res['f1']:.3f} | {time.time()-t0:.1f}s")
        
    print("Training Complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--appliance', type=str, required=True)
    args = parser.parse_args()
    main(args.appliance)
