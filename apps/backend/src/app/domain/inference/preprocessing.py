
import numpy as np
import pandas as pd
from typing import List, Tuple
from datetime import datetime

def compute_cyclical_features(dt: datetime) -> List[float]:
    """
    Compute cyclical time features for a given datetime.
    Returns [hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos]
    """
    # Hour of day (0-23)
    hour = dt.hour + dt.minute / 60.0
    hour_rad = 2 * np.pi * hour / 24.0
    
    # Day of week (0-6)
    dow = dt.weekday()
    dow_rad = 2 * np.pi * dow / 7.0
    
    # Month (1-12) -> 0-11 for consistency 
    # Or typically 1-12. Let's assume standard month encoding
    month = dt.month - 1
    month_rad = 2 * np.pi * month / 12.0
    
    return [
        np.sin(hour_rad), np.cos(hour_rad),
        np.sin(dow_rad), np.cos(dow_rad),
        np.sin(month_rad), np.cos(month_rad)
    ]

def build_feature_window(
    samples: List[Tuple[datetime, float]], 
    p_max_kw: float,
    window_size: int = 1536
) -> np.ndarray:
    """
    Construct input feature window (T, 7) for inference.
    
    Args:
        samples: List of (timestamp, power_value_amperes/watts?)
                 Note: Checkpoints imply input is Watts normalized by 15.0kW usually.
        p_max_kw: Max power for normalization (kW).
        window_size: Length of window.
        
    Returns:
        np.ndarray: Shape (1, 7, window_size) as float32
        
    Raises:
        ValueError if insufficient samples provided.
    """
    if len(samples) < window_size:
        raise ValueError(f"Insufficient samples: {len(samples)} < {window_size}")
        
    # Take last window_size samples
    window_samples = samples[-window_size:]
    
    # Pre-allocate (window, 7)
    features = np.zeros((window_size, 7), dtype=np.float32)
    
    for i, (ts, val) in enumerate(window_samples):
        # 1. Normalize Power
        # Assuming val is in Watts if p_max_kw is in kW?
        # Or val is kW?
        # Standard NILM datasets usually watts. 
        # If p_max_kw is 15.0 (15000 W?) usually scaling is power_kw / p_max_kw
        # Let's assume input val is raw Watts.
        
        # If val > 20000, clip?
        norm_val = (val / 1000.0) / p_max_kw
        features[i, 0] = np.clip(norm_val, 0.0, 1.0)
        
        # 2. Time Features
        time_feats = compute_cyclical_features(ts)
        features[i, 1:] = time_feats
        
    # Transpose to (7, window) and add batch dim -> (1, 7, window)
    # Model WaveNILM_v3 expects (B, C, T)
    return features.transpose(1, 0)[np.newaxis, ...]
