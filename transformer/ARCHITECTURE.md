# Hybrid CNN-Transformer NILM Architecture

## Overview

Architettura **SOTA 2025/2026** per Non-Intrusive Load Monitoring (NILM) a 5-second resolution.

| Spec | Valore |
|------|--------|
| **Parametri** | 7,322,136 (~7.3M) |
| **d_model** | 256 |
| **n_heads** | 8 (32 dim/head) |
| **n_layers** | 6 |
| **d_ff** | 1024 |
| **CNN** | [64, 128, 256] |
| **Input Features** | 10 (7 base + 3 derivative) |
| **Window** | 1024 samples = 85 min @ 5sec |

---

## Input Features (10 total)

| # | Feature | Range | Description |
|---|---------|-------|-------------|
| 0 | Aggregate | [0, 1] | Total power / P_MAX (13.51 kW) |
| 1 | hour_sin | [-1, 1] | sin(2π × hour/24) |
| 2 | hour_cos | [-1, 1] | cos(2π × hour/24) |
| 3 | dow_sin | [-1, 1] | sin(2π × day_of_week/7) |
| 4 | dow_cos | [-1, 1] | cos(2π × day_of_week/7) |
| 5 | month_sin | [-1, 1] | sin(2π × month/12) |
| 6 | month_cos | [-1, 1] | cos(2π × month/12) |
| 7 | **dP/dt** | dynamic | Power derivative (appliance signatures) |
| 8 | **rolling_mean** | [0, 1] | 40-second rolling mean |
| 9 | **rolling_std** | dynamic | 40-second rolling std |

### Derivative Features (Computed On-The-Fly)

```python
def _compute_derivative_features(agg):
    # dP/dt - captures sudden changes (appliance ON/OFF events)
    dP_dt = np.zeros(n); dP_dt[1:] = agg[1:] - agg[:-1]
    
    # Rolling mean (8 samples @ 5sec = 40 seconds)
    rolling_mean = np.convolve(agg, np.ones(8)/8, mode='same')
    
    # Rolling std - captures variability
    rolling_std = [np.std(agg[i-8:i]) for i in range(8, n)]
    
    return [dP_dt, rolling_mean, rolling_std]
```

**Why dP/dt matters:**
- Fridge compressor: gradual dP/dt (soft-start)
- Washing machine: sharp dP/dt (direct motor)
- Heat pump: distinctive dP/dt signature

---

## Flusso Dati

```
Input [batch, 1024, 10]
       │
       │  Aggregate + Temporal sin/cos + dP/dt + rolling stats
       ▼
┌──────────────────────────┐
│ Causal Stationarization  │  ← Welford vectorizzato O(n)
│ (solo dati passati)      │     Normalizza usando solo info causale
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│ CNN Feature Extractor    │  ← ResNet 64→128→256, kernel [7,5,3]
│ (transitori locali)      │     Cattura spike, ramp-up, patterns
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│ Linear Projection        │  ← 256 → d_model
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│ Transformer Encoder ×6   │  ← 8 heads, RoPE positional encoding
│ (dipendenze long-range)  │     Cattura durata cicli appliance
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│ Multi-Task Output Heads  │  ← 11 appliance-specific heads
│ ├─ Regression: ±5 attn   │     Neighborhood attention pooling
│ └─ Classification: pool  │     Global avg+max pooling
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│ DeStationarization       │  ← Identity-init, learned
└──────────────────────────┘
       │
       ▼
Output: {appliance: {power: [batch,1], state: [batch,1]}}
```

---

## NILM-Correct Scaling

**Critical for energy conservation!**

All power columns scaled by the **same P_MAX**:

```python
P_MAX = 13.5118 kW  # max(Aggregate_train)

# Scaling (preprocessing)
aggregate_scaled = aggregate_kW / P_MAX
appliance_scaled = appliance_kW / P_MAX  # SAME P_MAX!

# De-scaling (inference)
power_W = prediction * P_MAX * 1000
```

**Why single P_MAX?**
- Energy conservation: `sum(appliances) ≈ aggregate`
- If each appliance had its own scaler, the sum wouldn't match
- NILM literature standard (BERT4NILM, NILMFormer)

---

## Componenti Chiave

### 1. Causal Stationarization

**Problema:** Dati smart meter non-stazionari (drift stagionale).

**Soluzione:** Normalizzazione usando solo dati passati (no leakage).

```python
# Vectorizzato O(n) con cumsum
causal_mean = cumsum(x) / [1, 2, 3, ..., T]
x_norm = (x - causal_mean) / causal_std
```

### 2. CNN + Transformer Ibrido

| Componente | Ruolo |
|------------|-------|
| **CNN** | Cattura transitori locali (accensioni, spegnimenti) |
| **Transformer** | Modella dipendenze long-range (cicli, pattern) |
| **RoPE** | Positional encoding relativo, migliore per sequenze lunghe |

### 3. Multi-Task Head (SOTA)

| Task | Metodo | Motivazione |
|------|--------|-------------|
| **Regression** | Neighborhood ±5 + attention | Robusto per streaming |
| **Classification** | Global avg+max pooling | Cattura contesto globale |

```python
# Regression: attention pooling su ±5 timestep
x_neighborhood = x[:, mid-5:mid+5, :]
attn_weights = softmax(Q·K^T / √d)
x_pooled = attn_weights @ x_neighborhood
power = regression(x_pooled)
```

### 4. DeStationarization

Ri-scala output alla scala originale usando statistiche causali.

```python
# Identity init (parte da no-op, impara se necessario)
output = x * (1 + scale) + bias  # scale=0, bias=0 inizialmente
```

---

## Target Appliances (11)

| Appliance | ON Threshold | Noise Threshold | Typical Power |
|-----------|-------------|-----------------|---------------|
| HeatPump | 100W | 8W | 500-3000W |
| Dishwasher | 30W | 5W | 1500-2000W |
| WashingMachine | 50W | 50W | 500-2000W |
| Dryer | 50W | 5W | 2000-3000W |
| Oven | 100W | 100W | 2000-4000W |
| Stove | 50W | 50W | 1000-3000W |
| RangeHood | 20W | 5W | 100-300W |
| EVCharger | 100W | 5W | 3000-7000W |
| EVSocket | 100W | 5W | 3000-7000W |
| GarageCabinet | 25W | 25W | 50-200W |
| RainwaterPump | 50W | 10W | 500-1000W |

---

## Loss Function

### Multi-Objective (3 componenti)

| Loss | Peso | Scopo |
|------|------|-------|
| **MSE** | 1.0 | Accuratezza power regression |
| **BCE** | 0.5 | Transizioni ON/OFF classification |
| **Soft Dice** | 0.1 | Class imbalance |

```python
L_total = L_mse + 0.5 * L_bce + 0.1 * L_soft_dice
```

---

## Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Window size | 1024 | 85 minutes @ 5sec resolution |
| Batch size | 32 | Limited by 8GB VRAM |
| Learning rate | 1e-4 | With cosine annealing |
| LR min | 1e-6 | Cosine annealing minimum |
| Epochs | 100 | Early stopping patience=15 |
| Dropout | 0.1 | Regularization |
| Gradient clip | 1.0 | Prevent exploding gradients |
| Weight decay | 1e-5 | AdamW regularization |

### Learning Rate Schedule

```
Cosine Annealing: lr(t) = lr_min + 0.5(lr_max - lr_min)(1 + cos(πt/T))
```

---

## Model Variants

### Full Model (Current)
```bash
python train.py --epochs 100 --batch_size 32 --use_pretrained
# 7.3M params, fits in 8GB VRAM
```

### Light Version (for experiments)
```bash
python train.py --epochs 20 --batch_size 64 \
    --d_model 64 --n_layers 2 --n_heads 4 --d_ff 128 --use_pretrained
# ~1M params, faster training
```

---

## Files

```
transformer/
├── config.py          # Hyperparameters and configuration
├── model.py           # Model architecture (7.3M params)
├── dataset.py         # Data loading + derivative features
├── losses.py          # Loss functions (MSE, BCE, Dice)
├── train.py           # Training pipeline + cosine annealing
├── utils.py           # Metrics, checkpointing, logging
├── visualize_*.py     # Visualization scripts
├── checkpoints/       # Saved models
│   ├── hybrid_nilm_best.pth
│   └── training_results.json
└── plots/             # Generated visualizations
```

---

## References

1. **NILMFormer** (Petralia et al., KDD 2025) - Causal Stationarization
2. **BERT4NILM** (Yue et al., 2020) - Transformer per NILM
3. **RoPE** (Su et al., 2021) - Rotary Positional Embedding
4. **Seq2Point** (Zhang et al., 2018) - CNN baseline
