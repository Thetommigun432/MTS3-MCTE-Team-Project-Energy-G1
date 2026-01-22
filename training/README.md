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
