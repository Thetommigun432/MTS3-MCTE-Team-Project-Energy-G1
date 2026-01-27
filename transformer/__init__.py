"""
Hybrid CNN-Transformer NILM Architecture
==========================================
State-of-the-art Non-Intrusive Load Monitoring model for 1Hz data.

Architecture based on:
- CNN Feature Extraction (local transient detection)
- Transformer Encoder with RoPE (long-range dependencies)
- Multi-Task Output Heads (simultaneous multi-appliance prediction)

Components:
- model.py: HybridCNNTransformer architecture
- dataset.py: Data loading and windowing
- losses.py: Multi-objective loss functions
- train.py: Training pipeline
- config.py: Configuration
- utils.py: Utilities and metrics
"""

from .model import HybridCNNTransformer
from .dataset import NILMDataset, create_dataloaders
from .losses import NILMLoss
from .config import Config
from .utils import calculate_metrics, EarlyStopping

__version__ = "1.0.0"
__all__ = [
    "HybridCNNTransformer",
    "NILMDataset", 
    "create_dataloaders",
    "NILMLoss",
    "Config",
    "calculate_metrics",
    "EarlyStopping"
]
