"""
Training Script for TCN_SA - SOTA Fast Training
================================================

Script ottimizzato per training veloce su H200 MIG con risorse limitate (16GB RAM).

Ottimizzazioni implementate:
1. FastNILMDataset: numpy array slicing (no DataFrame.iloc overhead)
2. AMP fp16: torch.autocast + GradScaler per mixed precision
3. Flash Attention: is_causal=True per O(T) memory invece di O(T²)
4. num_workers=0: evita memory duplication su sistemi 16GB

Regularization SOTA:
- Data Augmentation: noise ±2% + time-shift su Aggregate (--augment)
- Label Smoothing: 0.0 default (focal loss assumes sharp labels, 0.05 conflicts!)
- Weight Decay: 0.02 con AdamW
- Spatial Dropout: 0.25 channel-wise nel modello
- Gradient Clipping: max_norm=1.0
- OneCycleLR: pct_start=0.3 per warm-up

Loss Design (SOTA for dual-head NILM):
- Gate: Focal Loss (gamma=2.0, alpha=0.75 ON) - handles class imbalance
- Power: Huber conditioned on gate.detach() - reduces gate/power conflict
- Early Stop: composite score = F1 - 0.0003*MAE_ON (not pure F1)

Configurazione default:
- Window: 4096, Stride: 512, n_blocks: 12 (RF=4101)
- Batch: 256, LR: 0.002, Epochs: 60 con early stopping

Usage:
    python -m training.train_tcn_sa --appliance HeatPump --epochs 60 --augment
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.tcn_sa import TCN_SA, count_parameters
from nilm.causal.train_ab_test import create_block_split


class FastNILMDataset(Dataset):
    """
    Fast NILM Dataset - Ottimizzato per 16GB RAM
    
    Carica dati come numpy array e usa slicing efficiente.
    Evita DataFrame.iloc che è 100x più lento.
    
    Args:
        X_data: (N, 7) numpy array - features [Aggregate + 6 temporal]
        y_data: (N,) numpy array - target power normalized [0, 1]
        window_size: 4096 default (68 min @ 1Hz)
        stride: 512 per training, 1 per inference
        augment: True per aggiungere noise ±2% su Aggregate
        lookahead: secondi di futuro visibili (0 = causale puro)
        
    Con lookahead > 0:
        - Input: finestra [t-window+lookahead, t+lookahead] (include futuro)
        - Target: y[t] (il valore al "presente", NON al futuro)
        - Questo evita data leakage: il target è sempre nel "passato" rispetto al lookahead
    """
    
    def __init__(self, X_data, y_data, window_size=4096, stride=512, augment=False, target_smooth=30, lookahead=0):
        self.X = X_data  # (N, 7) numpy array - shared memory
        self.y = y_data  # (N,) numpy array
        self.window_size = window_size
        self.augment = augment
        self.target_smooth = target_smooth  # 30s default, 1 for EVCharger
        self.lookahead = lookahead  # secondi di futuro visibili
        
        # Pre-compute valid window starts
        # Con lookahead: serve spazio extra alla fine per vedere il futuro
        n_samples = len(self.X)
        max_end = n_samples - lookahead  # Non possiamo andare oltre
        self.window_starts = np.arange(0, max_end - window_size + 1, stride)
        print(f"    Windows: {len(self.window_starts):,} (lookahead={lookahead})")
    
    def __len__(self):
        return len(self.window_starts)
    
    def __getitem__(self, idx):
        start = self.window_starts[idx]
        end = start + self.window_size
        
        # === CON LOOKAHEAD: evita data leakage ===
        # Input: finestra [start, end+lookahead) - vede il futuro
        # Target: y[end-1] - il "presente" (fine della finestra SENZA lookahead)
        # Il modello vede lookahead secondi nel futuro ma predice il "presente"
        if self.lookahead > 0:
            X = self.X[start:end + self.lookahead]  # Include futuro
            target_idx = end - 1  # Target è al "presente"
        else:
            X = self.X[start:end]  # Causale puro
            target_idx = end - 1
        
        # Target smoothing causale: configurable per appliance
        # EVCharger: NO smoothing (transizioni istantanee 0→7kW)
        # HeatPump/altri: 30s smoothing (riduce rumore sensore)
        smooth_window = min(self.target_smooth, self.window_size)
        if smooth_window > 1:
            y = self.y[target_idx - smooth_window + 1 : target_idx + 1].mean()
        else:
            y = self.y[target_idx]  # Ultimo istante (no smoothing)
        
        # Convert to tensor
        X_t = torch.from_numpy(X.copy() if self.augment else X)
        seq_len = X_t.shape[0]  # window_size + lookahead se lookahead > 0
        
        # Augmentation AGGRESSIVA per class imbalance (soprattutto samples ON)
        # Per appliance con <10% ON, augmentation aiuta a generalizzare
        if self.augment:
            is_on = y > 0.01  # Sample è ON?
            
            # 1. Noise injection - PIÙ AGGRESSIVO su ON
            noise_prob = 0.5 if is_on else 0.3
            noise_mag = 0.04 if is_on else 0.02  # ±4% su ON, ±2% su OFF
            if torch.rand(1).item() < noise_prob:
                X_t[:, 0] += torch.randn(seq_len) * noise_mag
            
            # 2. Time-shift - PIÙ AGGRESSIVO su ON
            shift_prob = 0.4 if is_on else 0.2
            shift_range = 128 if is_on else 64
            if torch.rand(1).item() < shift_prob:
                shift = torch.randint(-shift_range, shift_range + 1, (1,)).item()
                if shift != 0:
                    X_t[:, 0] = torch.roll(X_t[:, 0], shifts=shift)
            
            # 3. Magnitude scaling SOLO su ON (±10%) - simula variazione carica EV
            if is_on and torch.rand(1).item() < 0.3:
                scale = 0.9 + torch.rand(1).item() * 0.2  # [0.9, 1.1]
                X_t[:, 0] *= scale
            
            # 4. Dropout temporale su ON (5% prob) - simula micro-interruzioni
            if is_on and torch.rand(1).item() < 0.05:
                dropout_len = torch.randint(10, 60, (1,)).item()  # 10-60 sec
                dropout_start = torch.randint(0, max(1, seq_len - dropout_len), (1,)).item()
                X_t[dropout_start:dropout_start+dropout_len, 0] *= 0.5
        
        return X_t, torch.tensor(y, dtype=torch.float32)


def train_epoch(model, dataloader, optimizer, scheduler, device, on_weight, on_ratio, label_smoothing=0.05, scaler=None, lookahead=0):
    """
    Training epoch con ottimizzazioni SOTA
    
    Ottimizzazioni:
    - AMP fp16: autocast per forward, fp32 per loss (BCE non safe con fp16)
    - Label smoothing: 0.05 per BCE loss (reduce overconfidence)
    - Weighted BCE: on_weight per class imbalance
    - ON-Weighted Huber: peso dinamico basato su on_ratio per MAE_ON
    - Gradient clipping: max_norm=1.0
    - OneCycleLR: step per batch
    
    Loss = BCE(gate) + 3.0 × Huber(power)
    """
    model.train()
    total_loss = 0
    use_amp = scaler is not None
    
    # Target timestep: con lookahead, il target è -(lookahead+1) dall'ultimo
    target_timestep = -(lookahead + 1) if lookahead > 0 else -1
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for X_batch, y_batch in pbar:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        
        # AMP autocast for fp16 (forward pass only)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            gate, power = model(X_batch, target_timestep=target_timestep)
            gate = gate.squeeze(-1)
            power = power.squeeze(-1)
        
        # Loss computation in fp32 (BCE not safe with fp16)
        gate = gate.float()
        power = power.float()
        
        # Labels (soglia 0.01 = 1% P_MAX, paper-consistent)
        # 0.02 era troppo alto → uccideva recall su cicli deboli
        y_on = (y_batch > 0.01).float()
        y_smooth = y_on * (1 - label_smoothing) + (1 - y_on) * label_smoothing
        
        # === FOCAL LOSS (SOTA per class imbalance) ===
        # Lin et al. 2017 - RetinaNet: down-weight easy examples, focus on hard ones
        # gamma=1.5 (non 2.0): NILM ha noise strutturale, gamma=2 enfatizza troppo outlier
        # alpha=0.70/0.30: empiricamente ottimale (0.85 troppo aggressivo → modello conservativo)
        gamma = 1.5
        bce = F.binary_cross_entropy(gate, y_smooth, reduction='none')
        pt = torch.where(y_on > 0.5, gate, 1 - gate)  # prob of correct class
        focal_weight = (1 - pt) ** gamma  # esempi facili (pt~1) → peso~0
        alpha = torch.where(y_on > 0.5, 0.70, 0.30)  # fisso, testato empiricamente
        gate_loss = (alpha * focal_weight * bce).mean()
        
        # Power loss pesata su TARGET (non gate predetto)
        # Quando target è ON: peso alto (1.0) → modello impara potenza corretta
        # Quando target è OFF: peso basso (0.1) → non importa se sbaglia
        # Questo migliora MAE_ON senza danneggiare F1
        power_weights = torch.where(y_on > 0.5, 1.0, 0.1)
        power_loss = (F.huber_loss(power, y_batch, delta=0.1, reduction='none') * power_weights).mean()
        loss = gate_loss + 2.0 * power_loss
        
        # Backward with scaler
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, device, P_MAX, on_weight=0.5, on_ratio=0.1, lookahead=0):
    """
    Validazione con metriche NILM standard + validation loss
    
    Metriche calcolate:
    - val_loss: stessa loss del training per confronto
    - F1 Score: armonica di precision e recall per classificazione ON/OFF
    - MAE: errore medio assoluto in Watt (denormalizzato)
    - MAE_ON: errore medio solo quando appliance è ON (metrica critica)
    - Ghost Power: energia predetta quando appliance è OFF (false positive)
    
    Threshold ON/OFF: target > 0.01 (1% di P_MAX)
    """
    model.eval()
    all_gates, all_powers, all_targets = [], [], []
    total_loss = 0
    n_batches = 0
    
    # Target timestep: con lookahead, il target è -(lookahead+1) dall'ultimo
    target_timestep = -(lookahead + 1) if lookahead > 0 else -1
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch_dev = y_batch.to(device, non_blocking=True)
        gate, power = model(X_batch, target_timestep=target_timestep)
        
        # Compute validation loss (SAME as training - Focal + conditioned power)
        gate_sq = gate.squeeze(-1)
        power_sq = power.squeeze(-1)
        y_on = (y_batch_dev > 0.01).float()  # Soglia 0.01 coerente con training
        
        # Focal Loss (same as training)
        gamma = 1.5
        bce = F.binary_cross_entropy(gate_sq, y_on, reduction='none')
        pt = torch.where(y_on > 0.5, gate_sq, 1 - gate_sq)
        focal_weight = (1 - pt) ** gamma
        alpha = torch.where(y_on > 0.5, 0.70, 0.30)  # fisso, coerente con training
        gate_loss = (alpha * focal_weight * bce).mean()
        
        # Power loss condizionata (same as training)
        power_weights = gate_sq.detach() + 0.1
        power_loss = (F.huber_loss(power_sq, y_batch_dev, delta=0.1, reduction='none') * power_weights).mean()
        loss = gate_loss + 2.0 * power_loss
        total_loss += loss.item()
        n_batches += 1
        
        all_gates.append(gate_sq.cpu())
        all_powers.append(power_sq.cpu())
        all_targets.append(y_batch)
    
    gate = torch.cat(all_gates).numpy()
    power = torch.cat(all_powers).numpy()
    target = torch.cat(all_targets).numpy()
    
    # Denormalize to Watts
    power_w = power * P_MAX * 1000
    target_w = target * P_MAX * 1000
    
    # Metrics (soglia 0.01 coerente con training labels)
    pred_on = gate > 0.5
    true_on = target > 0.01
    
    TP = np.sum(pred_on & true_on)
    FP = np.sum(pred_on & ~true_on)
    FN = np.sum(~pred_on & true_on)
    
    prec = TP / (TP + FP + 1e-8)    
    rec = TP / (TP + FN + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    
    mae = np.mean(np.abs(power_w - target_w))
    mae_on = np.mean(np.abs(power_w[true_on] - target_w[true_on])) if true_on.sum() > 0 else 0
    ghost = np.mean(power_w[~true_on]) if (~true_on).sum() > 0 else 0
    val_loss = total_loss / n_batches
    
    return {'val_loss': val_loss, 'f1': f1, 'precision': prec, 'recall': rec, 
            'mae': mae, 'mae_on': mae_on, 'ghost': ghost}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--appliance', default='HeatPump')
    parser.add_argument('--data', default='/home/jovyan/mts3-mcte-team-project-g1-model-v1-datavol-1/data/nilm_ready_1sec_new.parquet')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--window', type=int, default=4096)
    parser.add_argument('--stride', type=int, default=512)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--n_blocks', type=int, default=12)  # RF=4101 for window 4096
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--no_inception', action='store_true')
    parser.add_argument('--no_attention', action='store_true')
    parser.add_argument('--augment', action='store_true', help='Enable SOTA data augmentation')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing for BCE (0.0 recommended with focal loss)')
    parser.add_argument('--weight_decay', type=float, default=0.02, help='AdamW weight decay')
    parser.add_argument('--lookahead', type=int, default=0, help='Lookahead in seconds for bidirectional mode (0=causal)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("WaveNILM v6 - TCN + Causal Attention (SOTA)")
    print("=" * 60)
    print(f"Appliance: {args.appliance}")
    print(f"Window: {args.window} | Stride: {args.stride} | n_blocks: {args.n_blocks}")
    print(f"SOTA: augment={args.augment} | label_smooth={args.label_smoothing} | wd={args.weight_decay}")
    if args.lookahead > 0:
        print(f"BIDIRECTIONAL MODE: lookahead={args.lookahead}s")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Count samples
    pf = pq.ParquetFile(args.data)
    n_samples = pf.metadata.num_rows
    print(f"Total samples: {n_samples:,}")
    
    # Detect resolution: if ~658K samples → 1min, if ~37M → 1sec
    if n_samples < 1_000_000:
        samples_per_day = 1440  # 1min resolution
        print(f"Detected: 1-minute resolution ({samples_per_day} samples/day)")
    else:
        samples_per_day = 86400  # 1sec resolution
        print(f"Detected: 1-second resolution ({samples_per_day} samples/day)")
    
    # Create splits
    train_idx, val_idx, _ = create_block_split(n_samples, samples_per_day=samples_per_day)
    print(f"Train: {len(train_idx):,} | Val: {len(val_idx):,}")
    
    # P_MAX: usa valori FISSI per appliance note (evita problemi di normalizzazione)
    P_MAX_FIXED = {
        'EVCharger': 7.5,      # 7.5 kW (monofase max) o 11 kW trifase
        'HeatPump': 5.0,       # 5 kW tipico
        'Dishwasher': 2.5,     # 2.5 kW
        'WashingMachine': 2.5, # 2.5 kW
        'Dryer': 3.0,          # 3 kW
    }
    
    if args.appliance in P_MAX_FIXED:
        P_MAX = P_MAX_FIXED[args.appliance]
        print(f"P_MAX: {P_MAX:.1f} kW = {P_MAX*1000:.0f} W (FIXED for {args.appliance})")
    else:
        # Fallback: calcola da dati
        table = pq.read_table(args.data, columns=[args.appliance])
        y_all = table.to_pandas()[args.appliance].values
        y_train_only = y_all[train_idx]
        P_MAX = float(np.percentile(y_train_only[y_train_only > 0.01], 99)) if (y_train_only > 0.01).any() else 1.0
        print(f"P_MAX: {P_MAX:.4f} kW = {P_MAX*1000:.1f} W (from train 99th percentile)")
        del table, y_all, y_train_only
    
    # Class balance - ADAPTIVE WEIGHTING for extreme unbalance
    # Use random sample of train indices for unbiased estimate
    table = pq.read_table(args.data, columns=[args.appliance])
    y_all_temp = table.to_pandas()[args.appliance].values
    np.random.seed(42)
    sample_idx = np.random.choice(train_idx, size=min(2_000_000, len(train_idx)), replace=False)
    y_train = y_all_temp[sample_idx]
    on_ratio = (y_train > 0.01).mean()
    del table, y_all_temp, y_train
    
    # Adaptive on_weight: MODERATO per evitare all-ON predictions
    # Formula più conservativa per bilanciare precision/recall
    if on_ratio < 0.01:  # Extreme: <1% ON (e.g. Dryer 0.08%)
        on_weight = 0.90  # 9:1 ratio (era 0.98, troppo aggressivo)
        print(f"⚠️ EXTREME UNBALANCE: ON={on_ratio:.2%} → using on_weight=0.90")
    elif on_ratio < 0.10:  # Heavy: <10% ON (e.g. EVCharger 5.5%)
        on_weight = min(0.85, 1 - on_ratio + 0.3)  # ~0.85 for 5% ON
        print(f"⚠️ HEAVY UNBALANCE: ON={on_ratio:.1%} → on_weight={on_weight:.2f}")
    else:  # Normal: >10% ON
        on_weight = 1 - on_ratio
        print(f"ON ratio: {on_ratio:.1%} | on_weight: {on_weight:.2f}")
    
    # === LOAD DATA TO RAM (FAST!) ===
    print("\nLoading data to RAM...")
    feature_cols = ['Aggregate', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
    
    table = pq.read_table(args.data, columns=feature_cols + [args.appliance])
    df = table.to_pandas()
    
    # Pre-process features
    X_all = df[feature_cols].values.astype(np.float32)
    
    # === DELTA P (derivata) - calcola PRIMA della normalizzazione ===
    # Importante: la derivata su raw aggregate preserva la scala assoluta
    # ΔP per EVCharger: ±7kW (step netto), per HeatPump: ±0.5kW (rampa)
    # Se calcolato dopo normalizzazione, questa informazione si perde
    raw_agg = X_all[:, 0].copy()  # kW originali
    delta_P_raw = np.diff(raw_agg, prepend=raw_agg[0])
    # Normalizza ΔP separatamente (clip a ±5kW, poi scala a [-1,1])
    delta_P = np.clip(delta_P_raw / 5.0, -1, 1).astype(np.float32)
    
    # IMPORTANTE: normalizza Aggregate per p95 DELL'AGGREGATE, non per P_MAX appliance
    # Se Aggregate >> appliance (es. casa 8kW, HP 2kW), il segnale appliance diventa
    # una perturbazione piccola → il modello non lo vede
    # Paper WaveNILM: "aggregate normalized by percentiles"
    agg_p95 = np.percentile(X_all[:, 0], 95)
    X_all[:, 0] = np.clip(X_all[:, 0] / agg_p95, 0, 2) * 2 - 1  # Clip to [0,2] then scale to [-1,3]
    print(f"Aggregate normalization: p95={agg_p95:.2f} kW")
    
    # Aggiungi ΔP come feature (calcolato su raw, normalizzato separatamente)
    X_all = np.column_stack([X_all, delta_P])  # Ora 8 features
    print(f"Added ΔP feature (from raw): shape {X_all.shape}")
    
    y_all = (df[args.appliance].values / P_MAX).astype(np.float32)
    del table, df
    
    # Split
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    del X_all, y_all
    
    print(f"Train data: {X_train.nbytes / 1e9:.2f} GB | Val data: {X_val.nbytes / 1e9:.2f} GB")
    
    # Target smoothing: appliance-specific
    # NO smoothing: transizioni istantanee, eventi brevi
    # SI smoothing: rampe graduali, cicli termici, rumore sensore
    INSTANT_APPLIANCES = {
        'EVCharger',      # 0→7kW istantaneo
        'EVSocket',       # 0→3.7kW istantaneo  
        'Dishwasher',     # Fasi nette ON/OFF
        'Stove',          # Piastre resistive ON/OFF
        'RangeHood',      # Ventola ON/OFF, eventi brevi
        'RainwaterPump',  # Pompa ON/OFF, eventi MOLTO brevi (30-120s)
        'Dryer',          # Micro-episodi (mediana 12s!)
    }
    if args.appliance in INSTANT_APPLIANCES:
        target_smooth = 1  # No smoothing
        print(f"⚡ Target smoothing: DISABLED for {args.appliance} (instant ON/OFF)")
    else:
        target_smooth = 30  # 30 sec
        print(f"🔄 Target smoothing: 30s for {args.appliance} (gradual transitions)")
    
    # Create datasets (FAST numpy slicing)
    train_dataset = FastNILMDataset(X_train, y_train, args.window, args.stride, 
                                    augment=args.augment, target_smooth=target_smooth,
                                    lookahead=args.lookahead)
    val_dataset = FastNILMDataset(X_val, y_val, args.window, args.stride, 
                                  augment=False, target_smooth=target_smooth,
                                  lookahead=args.lookahead)
    
    # === BALANCED BATCH SAMPLER for unbalanced datasets ===
    # IMPORTANTE: calcola is_on con STESSA LOGICA del dataset
    smooth_window = target_smooth
    if smooth_window > 1:
        train_targets_smoothed = np.array([
            y_train[start + args.window - smooth_window : start + args.window].mean()
            for start in train_dataset.window_starts
        ])
    else:
        # No smoothing: ultimo istante
        train_targets_smoothed = np.array([
            y_train[start + args.window - 1]
            for start in train_dataset.window_starts
        ])
    is_on = (train_targets_smoothed > 0.01).astype(np.float32)  # Soglia 0.01 coerente
    
    # Peso inversamente proporzionale alla frequenza
    n_on = is_on.sum()
    n_off = len(is_on) - n_on
    
    # SAMPLER solo per casi ESTREMI (<1% ON, es. Dryer 0.08%)
    # Per 5-15% ON: Focal Loss è sufficiente, sampler introduce bias
    if on_ratio < 0.01:  # Solo Dryer e simili
        weight_on = 1.0 / (n_on + 1)
        weight_off = 1.0 / (n_off + 1)
        sample_weights = np.where(is_on > 0.5, weight_on, weight_off)
        sample_weights = torch.from_numpy(sample_weights).double()
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        use_sampler = True
        print(f"\n🎯 BALANCED SAMPLING (extreme case): {n_on:,} ON ({n_on/len(is_on)*100:.2f}%) vs {n_off:,} OFF")
    else:
        sampler = None
        use_sampler = False
        print(f"\n📊 Natural distribution: {n_on:,} ON ({n_on/len(is_on)*100:.1f}%) - Focal Loss handles imbalance")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=(not use_sampler), sampler=sampler if use_sampler else None,
                              num_workers=2, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size*2,
                            shuffle=False, num_workers=2, pin_memory=True,
                            persistent_workers=True)
    
    print(f"Train batches: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")
    
    # Model (8 features: Aggregate + 6 temporal + ΔP)
    model = WaveNILMv6STAFN(
        n_features=8,
        n_appliances=1,
        hidden_channels=args.hidden,
        n_blocks=args.n_blocks,
        use_psa=not args.no_attention,
        use_inception=not args.no_inception,
        lookahead=args.lookahead
    ).to(device)
    
    print(f"\nParameters: {count_parameters(model):,}")
    
    # AMP GradScaler for mixed precision (fp16)
    scaler = torch.amp.GradScaler('cuda')
    print("AMP enabled (fp16 mixed precision)")
    
    # Optimizer (SOTA: higher weight decay = 0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.3
    )
    
    # Training loop
    # NOTA: early stop su score composito, non F1 puro
    # F1 puro può fermarsi troppo presto con focal loss
    best_score = -float('inf')
    best_f1 = 0
    patience_counter = 0
    checkpoint_dir = Path(__file__).parent.parent.parent / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Timestamp per evitare sovrascrittura tra run
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_path = checkpoint_dir / f'wavenilm_v6_{args.appliance}_{run_id}_best.pt'
    print(f"Checkpoint path: {best_path.name}")
    
    print("\nStarting training...")
    
    # Show power weight for this appliance
    power_on_weight = min(20.0, max(3.0, 0.5 / (on_ratio + 0.01)))
    print(f"Power ON weight: {power_on_weight:.1f}x (based on {on_ratio:.1%} ON ratio)")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, on_weight, on_ratio, args.label_smoothing, scaler, lookahead=args.lookahead)
        
        metrics = validate(model, val_loader, device, P_MAX, on_weight, on_ratio, lookahead=args.lookahead)
        print(f"Loss: train={train_loss:.4f} val={metrics['val_loss']:.4f} | F1={metrics['f1']:.4f} | MAE={metrics['mae']:.1f}W | MAE_ON={metrics['mae_on']:.1f}W | Ghost={metrics['ghost']:.1f}W")
        
        # Score composito: F1 alto + MAE_ON basso (paper NILM standard)
        # Evita early stop su F1 che sale ma MAE_ON esplode
        score = metrics['f1'] - 0.0003 * metrics['mae_on']
        
        if score > best_score:
            best_score = score
            best_f1 = metrics['f1']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
                'P_MAX': P_MAX
            }, best_path)
            print(f"✅ Saved best (F1={best_f1:.4f}, score={score:.4f})")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print("⚠️ Early stopping")
                break
    
    print(f"\nBest F1: {best_f1:.4f}")


if __name__ == '__main__':
    main()
