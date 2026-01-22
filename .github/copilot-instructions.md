# NILM Energy Disaggregation - Copilot Instructions

## Project Overview
Non-Intrusive Load Monitoring (NILM) project for energy consumption disaggregation. Predicts individual appliance power from aggregate building consumption using deep learning (PyTorch).

## Architecture & Data Flow

### Data Pipeline
```
Raw InfluxDB Export → preprocessing/{resolution}/ → data/processed/{resolution}/model_ready/
```
- **Resolutions**: `10sec`, `15min`, `1sec`
- **Model-ready format**: `X_train.npy`, `y_train.npy` per appliance (shared X across appliances)
- **12 appliances**: dishwasher, heatpump_controller, washingmachine, dryer, oven, stove, etc.

### Model Paradigm: Seq2Point
All models use **Seq2Point** architecture (NOT Seq2Seq):
- Input: Window of aggregate consumption `[batch, seq_len, features]`
- Output: Single appliance power at window midpoint `[batch, 1]`
- Target extraction: `y_batch[:, seq_len // 2]` (always use integer division)

## Key Conventions

### PyTorch Model Structure
```python
# Standard forward pass pattern
x = x.permute(0, 2, 1)  # [B, F, S] for Conv1d
x = self.conv_embedding(x)
x = x.permute(0, 2, 1)  # [B, S, D] for Transformer
mid_token = x[:, x.shape[1] // 2, :]  # Seq2Point extraction
```

### Loss Functions
Use `WeightedNILMLoss` for class imbalance (appliances mostly OFF):
- `on_weight=15-30`: Higher penalty for ON-state errors
- `fn_weight=10-15`: Extra penalty for False Negatives (predict OFF when ON)
- `threshold=0.01-0.1`: ON/OFF detection threshold (scaled space)

### Temporal Features (Cyclical Encoding)
Always encode time features as sin/cos pairs for continuity:
```python
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
```
Standard features: `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos`

## Critical Patterns

### Multi-Task Training (Regression + Classification)
Models output dual heads: power regression + ON/OFF classification.
```python
loss = loss_reg + (CLF_WEIGHT * loss_clf)  # CLF_WEIGHT ~0.5
```
Classification acts as a "gate" during inference to zero-out power when OFF.

### Data Loading (Shared X, Per-Appliance Y)
```python
X = {"train": np.load(".../dishwasher/X_train.npy"), ...}  # Load once
appliance_data[appliance] = {"y_train": np.load(f".../{appliance}/y_train.npy"), ...}
```

### Model Saving Convention
```python
torch.save(model.state_dict(), f"{model_type}_{appliance}_best.pth")
# Examples: transformer_heatpump_best.pth, cnn_seq2seq_heatpump_model.pth
```

## Notebooks Organization
- `model_exploration/1_Data_Processing.ipynb` - Initial raw data splitting
- `model_exploration/[2-10]_Model_*.ipynb` - Architecture experiments (CNN, LSTM, UNet, Transformer, LGBM, VAR)
- `preprocessing/15min/data_preparation_15min.ipynb` - Main preprocessing pipeline with energy balance corrections

## NILM-Specific Metrics
- **SAE** (Signal Aggregate Error): Total energy accuracy `|Σ_true - Σ_pred| / Σ_true`
- **F1 Score**: ON/OFF detection balance
- Standard: MAE, RMSE, R²

## Code Style
- Minimize boilerplate; no excessive docstrings
- Explain approach BEFORE coding
- Prefer working solutions over theoretical discussions
