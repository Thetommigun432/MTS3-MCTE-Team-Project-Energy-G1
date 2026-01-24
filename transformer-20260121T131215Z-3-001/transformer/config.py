"""
Configuration for Hybrid CNN-Transformer NILM
==============================================
Hyperparameters for 5-second resampled data with NILM-correct scaling.

Data Scaling (NILM-correct):
- Power columns: scaled by P_MAX = max(Aggregate_train) = 13.51 kW
- Temporal sin/cos: NOT scaled (already in [-1, 1])
- All appliances use SAME P_MAX (energy conservation)
- For inference: power_watts = prediction * P_MAX * 1000
"""
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    """
    NILM Training Configuration.
    
    Optimized for 5-second resampled data with 1024-sample window (85 min context).
    """
    
    # -------------------------------------------------------------------------
    # Data Paths
    # -------------------------------------------------------------------------
    # model_ready: NILM-correct scaling (power/P_MAX, temporal untouched)
    # Data Paths
    # -------------------------------------------------------------------------
    # model_ready: NILM-correct scaling (power/P_MAX, temporal untouched)
    # Use cwd() for portability (works on both Local and Colab/Drive)
    data_path: Path = Path.cwd() / 'data' / 'processed' / '1sec_new'
    model_ready_subdir: str = 'model_ready'  # NILM-correct scaling (power/P_MAX)
    save_path: Path = Path.cwd() / 'transformer' / 'checkpoints'
    
    # Scaling parameters (from pretraining - DO NOT CHANGE)
    P_MAX_kW: float = 13.5118  # max(Aggregate_train) in kW
    
    # -------------------------------------------------------------------------
    # Target Appliances (multi-output) - aligned with 1sec_new data
    # -------------------------------------------------------------------------
    appliances: List[str] = field(default_factory=lambda: [
        'HeatPump',
        'Dishwasher',
        'WashingMachine',
        'Dryer',
        'Oven',
        'Stove',
        'RangeHood',
        'EVCharger',
        'EVSocket',
        'GarageCabinet',
        'RainwaterPump'
    ])
    
    # Per-appliance ON thresholds (Watts) for classification head
    on_thresholds: dict = field(default_factory=lambda: {
        'HeatPump': 100,
        'Dishwasher': 30,
        'WashingMachine': 50,
        'Dryer': 50,
        'Oven': 100,
        'Stove': 50,
        'RangeHood': 20,
        'EVCharger': 100,
        'EVSocket': 100,
        'GarageCabinet': 25,
        'RainwaterPump': 50,
    })
    
    # Per-appliance noise thresholds (Watts)
    noise_thresholds: dict = field(default_factory=lambda: {
        'HeatPump': 8,
        'Dishwasher': 5,
        'WashingMachine': 50,
        'Dryer': 5,
        'Oven': 100,
        'Stove': 50,
        'RangeHood': 5,
        'EVCharger': 5,
        'EVSocket': 5,
        'GarageCabinet': 25,
        'RainwaterPump': 10,
    })
    
    # -------------------------------------------------------------------------
    # Sequence Parameters (aligned with pretraining)
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Sequence Parameters (aligned with pretraining)
    # -------------------------------------------------------------------------
    resolution_sec: int = 1          # Data resolution (from pretraining resample)
    window_size: int = 512           # Reduced to 512 (8.5 min) to fix OOM and focus on local patterns
    stride: int = 256                # 50% overlap
    stride_inference: int = 512      # Non-overlapping inference
    
    # -------------------------------------------------------------------------
    # CNN Feature Extractor
    # -------------------------------------------------------------------------
    cnn_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [7, 5, 3])
    cnn_use_residual: bool = True
    
    # -------------------------------------------------------------------------
    # Transformer Architecture (SOTA scale: ~3-5M params)
    # -------------------------------------------------------------------------
    model_type: str = 'hybrid'       # 'hybrid', 'simple', or 'nilmformer'
    
    # NILMFormer specific
    dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    kernel_size_embed: int = 3
    kernel_size_head: int = 3
    pffn_ratio: int = 4
    
    d_model: int = 256               # Embedding dimension (was 128)
    n_heads: int = 8                 # Multi-head attention heads (32 dim/head)
    n_layers: int = 6                # Transformer encoder layers (was 4)
    d_ff: int = 1024                 # Feed-forward dimension (was 512)
    dropout: float = 0.1
    use_rope: bool = True            # Rotary Positional Embedding
    use_flash_attention: bool = False  # Enable for CUDA if available
    
    # -------------------------------------------------------------------------
    # Multi-Task Output
    # -------------------------------------------------------------------------
    seq2point: bool = True           # True = Midpoint only (Seq2Point, paper optimal)
    output_length: int = 1            # Single midpoint prediction
    
    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 15               # Early stopping patience
    grad_clip: float = 1.0           # Gradient clipping
    
    # Loss weights
    loss_mse_weight: float = 1.0
    loss_bce_weight: float = 2.0     # High classification weight for class imbalance
    loss_sdl_weight: float = 0.1     # Soft Dice Loss weight
    energy_loss_weight: float = 0.1  # Energy consistency loss weight
    
    # New Loss Parameters (for WeightedNILMLoss)
    loss_type: str = 'focal'         # 'focal', 'mse', or 'weighted_nilm'
    fn_weight: float = 15.0          # False Negative penalty weight (for weighted_nilm)

    energy_warmup_steps: int = 1000  # Warmup steps before full energy loss
    
    # -------------------------------------------------------------------------
    # SOTA Features (NILMFormer-style)
    # -------------------------------------------------------------------------
    use_stationarization: bool = True   # Causal stationarization (Welford)
    use_pooling_for_state: bool = True  # Temporal pooling for classification
    
    # -------------------------------------------------------------------------
    # Data Augmentation (V3 Robustness)
    # -------------------------------------------------------------------------
    augment: bool = True
    noise_std: float = 0.02          # Gaussian noise std (normalized)
    mag_scale: float = 0.2           # Magnitude warping scale (+/- 20%)
    mask_prob: float = 0.2           # Probability of masking a chunk
    mask_size: int = 10              # Size of mask (in timesteps)
    
    # -------------------------------------------------------------------------
    # Derivative Features (dP/dt, rolling_mean, rolling_std)
    # -------------------------------------------------------------------------
    add_derivative_features: bool = False  # DISABLED for NILMFormer strict alignment
    
    
    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------
    device: torch.device = field(default_factory=lambda: 
        torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    num_workers: int = 0             # DataLoader workers (0 for Windows compatibility)
    
    # -------------------------------------------------------------------------
    # Temporal Features
    # -------------------------------------------------------------------------
    temporal_features: List[str] = field(default_factory=lambda: [
        'hour_sin', 'hour_cos',
        'dow_sin', 'dow_cos', 
        'month_sin', 'month_cos'
    ])
    
    # Override n_features (set dynamically after data loading)
    n_features: Optional[int] = None
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.save_path.mkdir(parents=True, exist_ok=True)
        
    @property
    def n_appliances(self) -> int:
        return len(self.appliances)
    
    @property
    def input_features(self) -> int:
        """Number of input features: aggregate + temporal + optional derivatives."""
        if self.n_features is not None:
            return self.n_features
        base = 1 + len(self.temporal_features)  # 7 = 1 aggregate + 6 temporal
        if self.add_derivative_features:
            return base + 3  # +3 for dP/dt, rolling_mean, rolling_std
        return base


def get_config(**overrides) -> Config:
    """Create config with optional overrides."""
    return Config(**overrides)
