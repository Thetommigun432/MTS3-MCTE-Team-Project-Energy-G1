"""
Prediction Smoothing Module.

Applies rolling window smoothing to predictions for each appliance.
This reduces noise and provides smoother output for visualization.
"""

from collections import deque
from typing import Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class PredictionSmoother:
    """
    Maintains rolling windows of predictions per (building, appliance).
    
    Default window size is 30 samples (30 seconds at 1Hz).
    Returns the mean of the window for each prediction.
    """
    
    window_size: int = 30
    # {(building_id, appliance_key): deque of (power_kw, confidence) tuples}
    _buffers: Dict[Tuple[str, str], deque] = field(default_factory=dict)
    
    def smooth(
        self,
        building_id: str,
        predictions: Dict[str, Tuple[float, float]],
    ) -> Dict[str, Tuple[float, float]]:
        """
        Apply smoothing to predictions.
        
        Args:
            building_id: Building identifier
            predictions: Dict of {appliance_key: (power_kw, confidence)}
            
        Returns:
            Smoothed predictions with same structure
        """
        smoothed: Dict[str, Tuple[float, float]] = {}
        
        for appliance_key, (power_kw, confidence) in predictions.items():
            buffer_key = (building_id, appliance_key)
            
            # Get or create buffer
            if buffer_key not in self._buffers:
                self._buffers[buffer_key] = deque(maxlen=self.window_size)
            
            buffer = self._buffers[buffer_key]
            
            # Add new prediction
            buffer.append((power_kw, confidence))
            
            # Compute smoothed values (mean)
            if len(buffer) > 0:
                sum_power = sum(p for p, _ in buffer)
                sum_conf = sum(c for _, c in buffer)
                n = len(buffer)
                smoothed_power = sum_power / n
                smoothed_conf = sum_conf / n
            else:
                smoothed_power = power_kw
                smoothed_conf = confidence
            
            smoothed[appliance_key] = (smoothed_power, smoothed_conf)
        
        return smoothed
    
    def clear(self, building_id: str = None) -> None:
        """Clear buffers for a building or all buffers."""
        if building_id is None:
            self._buffers.clear()
        else:
            keys_to_remove = [k for k in self._buffers if k[0] == building_id]
            for k in keys_to_remove:
                del self._buffers[k]
    
    def get_buffer_sizes(self, building_id: str) -> Dict[str, int]:
        """Get current buffer sizes for debugging."""
        return {
            k[1]: len(v) 
            for k, v in self._buffers.items() 
            if k[0] == building_id
        }


# Global singleton
_smoother: PredictionSmoother = None


def get_prediction_smoother(window_size: int = 30) -> PredictionSmoother:
    """Get or create the global prediction smoother."""
    global _smoother
    if _smoother is None:
        _smoother = PredictionSmoother(window_size=window_size)
    return _smoother
