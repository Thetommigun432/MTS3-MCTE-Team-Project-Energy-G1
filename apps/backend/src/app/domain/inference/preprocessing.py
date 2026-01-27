
import numpy as np
from datetime import datetime, timezone
from typing import List, Tuple, Optional

class DataPreprocessor:
    """
    Preprocesses raw sensor data into model-ready features.
    
    Supports 7 or 8 features depending on model:
    
    7 features (HeatPump, older models):
        0: Aggregate (normalized by p95, scaled to [-1, 3])
        1: hour_sin   = sin(2π × hour/24)
        2: hour_cos   = cos(2π × hour/24)
        3: dow_sin    = sin(2π × day_of_week/7)
        4: dow_cos    = cos(2π × day_of_week/7)
        5: month_sin  = sin(2π × month/12)
        6: month_cos  = cos(2π × month/12)
    
    8 features (newer models with delta_P):
        0-6: Same as above
        7: ΔP = delta power (clipped to ±5kW, scaled to [-1, 1])
    
    NOTE: Temporal features are in [-1, 1] (NO SCALING needed).
    NOTE: agg_p95 is in WATTS (from training metadata)!
    """
    
    def __init__(self, agg_p95: float = 8000.0, n_features: int = 7, P_MAX: float = None):
        """
        Args:
            agg_p95: 95th percentile of Aggregate in WATTS (from training metadata)
                     Default ~8kW, but should be loaded from metadata.pkl
            n_features: Number of features (7 or 8). 8 includes delta_P.
            P_MAX: Alias for agg_p95 (for backwards compatibility)
        """
        # Support both names for backwards compatibility
        if P_MAX is not None:
            self.agg_p95 = P_MAX
        else:
            self.agg_p95 = agg_p95
        self.n_features = n_features
        self.prev_power = None  # For ΔP calculation
        
    def process_sample(self, timestamp: float, power_watts: float) -> np.ndarray:
        """
        Process a single raw sample into feature vector.
        
        Args:
            timestamp: Unix timestamp
            power_watts: Total power in WATTS
        
        Returns:
            np.ndarray: Feature vector of shape (n_features,) dtype=float32
        """
        # 1. Normalize Aggregate power using p95 (SAME as train_v6_simple.py)
        #    Clip to [0, 2] then scale to [-1, 3]
        aggregate_norm = np.clip(power_watts / self.agg_p95, 0, 2) * 2 - 1
        
        # 2. Extract time components
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        day_of_week = dt.weekday()  # 0=Monday, 6=Sunday
        month = dt.month - 1 + dt.day / 31.0  # 0-11 continuous
        
        # 3. Cyclical encoding (sin/cos)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin = np.sin(2 * np.pi * day_of_week / 7)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        # Base features (7 total)
        features = [
            aggregate_norm,  # 0: Aggregate
            hour_sin,        # 1: hour_sin
            hour_cos,        # 2: hour_cos
            dow_sin,         # 3: dow_sin
            dow_cos,         # 4: dow_cos
            month_sin,       # 5: month_sin
            month_cos,       # 6: month_cos
        ]
        
        # 4. Optional: ΔP (delta power) for 8-feature models
        if self.n_features == 8:
            if self.prev_power is None:
                delta_p = 0.0
            else:
                # ΔP in watts, clipped to ±5kW, scaled to [-1, 1]
                delta_p = np.clip((power_watts - self.prev_power) / 5000.0, -1, 1)
            self.prev_power = power_watts
            features.append(delta_p)  # 7: ΔP
        
        return np.array(features, dtype=np.float32)
    
    def reset(self):
        """Reset preprocessor state."""
        self.prev_power = None
