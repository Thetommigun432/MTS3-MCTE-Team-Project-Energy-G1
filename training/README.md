# Training - ML Model Development

This directory contains all machine learning training code, notebooks, and experiments.

> ⚠️ **Not for Runtime**: This code is for model training only. Runtime inference uses `apps/backend/`.

## Contents

```
training/
├── train_model.py        # Main training script
├── requirements.txt      # Training dependencies (TensorFlow, PyTorch, etc.)
├── notebooks/            # Jupyter notebooks for experimentation
└── logs/                 # Training logs
```

## Model Exploration

The following directories contain historical training experiments (kept for reference):
- `model_exploration/` - Jupyter notebooks with various architectures
- `model_highfreq/` - High-frequency model experiments
- `preprocessing/` - Data preprocessing scripts

## Usage

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install training dependencies
pip install -r training/requirements.txt

# Train a model
python training/train_model.py --appliance heatpump --model cnn
```

## Output

Trained model files (*.pth, *.safetensors) are gitignored. Export to `apps/backend/models/` for runtime use.

## PT model metrics and plots

`pt_metrics_and_plots.py` extracts MAE and F1 per TCN_SA `.pt` model and can produce comparison plots (predicted vs ground truth).

```bash
# Extract metrics from checkpoints (MAE, F1); writes training/pt_metrics.csv
python -m training.pt_metrics_and_plots --checkpoint-dir apps/backend/models/tcn_sa --out-csv training/pt_metrics.csv

# With pred-vs-GT plots (needs parquet path)
python -m training.pt_metrics_and_plots --plots --data path/to/nilm_ready_1sec_new.parquet --out-dir training/plots_pt
```

- **Metrics**: Reads `metrics` from each `.pt` if present. Backend export `.pt` files (in `apps/backend/models/tcn_sa/`) contain only `state_dict` and have no metrics; run `train_tcn_sa` to get checkpoints with MAE/F1.
- **Bar plot**: When MAE/F1 exist, saves `metrics_bar.png` in `--out-dir`.
- **Per-appliance plots**: For each appliance, runs inference on the last 10% of data, selects periods with activity, saves `plot_<Appliance>_period_1.png`, etc.
- **All-appliances plot**: Saves `plot_all_appliances.png` (pred vs GT overlaid).
