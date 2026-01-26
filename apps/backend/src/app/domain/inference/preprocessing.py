
import numpy as np
from datetime import datetime, timezone
from typing import List, Tuple, Optional

class DataPreprocessor:
    """
    Preprocesses raw sensor data into model-ready features.
    
    Features (7 total) - MUST MATCH TRAINING DATA:
        0: Aggregate (power_normalized 0-1, scaled by P_MAX)
        1: hour_sin   = sin(2π × hour/24)
        2: hour_cos   = cos(2π × hour/24)
        3: dow_sin    = sin(2π × day_of_week/7)
        4: dow_cos    = cos(2π × day_of_week/7)
        5: month_sin  = sin(2π × month/12)
        6: month_cos  = cos(2π × month/12)
    
    NOTE: Temporal features are in [-1, 1] (NO SCALING needed).
    NOTE: P_MAX is in WATTS (same as metadata from training)!
    """
    
    def __init__(self, P_MAX: float = 15000.0):
        """
        Args:
            P_MAX: Maximum power in WATTS for normalization (from training metadata)
                   Default ~15kW, but should be loaded from metadata.pkl
        """
        self.P_MAX = P_MAX  # WATTS (NOT kW!)
        
    def process_sample(self, timestamp: float, power_watts: float) -> np.ndarray:
        """
        Process a single raw sample into feature vector.
        
        Args:
            timestamp: Unix timestamp
            power_watts: Total power in WATTS
        
        Returns:
            np.ndarray: Feature vector of shape (7,) dtype=float32
        """
        # 1. Normalize Aggregate power: Aggregate_scaled = Aggregate / P_MAX
        aggregate_norm = np.clip(power_watts / self.P_MAX, 0, 1)
        
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
        
        # Return in EXACT order as training data
        return np.array([
            aggregate_norm,  # 0: Aggregate
            hour_sin,        # 1: hour_sin
            hour_cos,        # 2: hour_cos
            dow_sin,         # 3: dow_sin
            dow_cos,         # 4: dow_cos
            month_sin,       # 5: month_sin
            month_cos        # 6: month_cos
        ], dtype=np.float32)
    
    def reset(self):
        """Reset preprocessor state (stateless, but kept for API compatibility)."""
        pass
