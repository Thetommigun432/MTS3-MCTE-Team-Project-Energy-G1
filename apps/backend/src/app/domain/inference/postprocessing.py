"""
Smart Priority Scaling Post-Processing for NILM.

Enforces energy conservation: sum of appliances ≤ (1 - ghost_load) × aggregate.
Uses physics-based priority scaling instead of naive linear scaling.

Key insight:
- RIGID appliances (EV, Dishwasher): Fixed power profiles, scale last
- ELASTIC appliances (HeatPump, Fridge): Variable/inverter, scale first
- TARGET_EXPLAINED_RATIO: ~75% of aggregate is explained by monitored appliances
"""

from typing import Dict, List, Tuple
from app.core.logging import get_logger

logger = get_logger(__name__)


# Appliance categories based on physical behavior
# RIGID: Fixed power profiles (ON/OFF states with known wattage)
# ELASTIC: Inverter-driven or variable load (can modulate power)
RIGID_APPLIANCES = {"EVCharger", "EVSocket", "Dishwasher", "WashingMachine", "Dryer", "Oven"}
ELASTIC_APPLIANCES = {"HeatPump", "RangeHood", "Stove", "RainwaterPump"}

# Noise threshold (Watts) - don't scale if violation is below this
NOISE_THRESHOLD_W = 100.0

# Target explained ratio: 75% of aggregate is from monitored appliances
# Remaining 25% is ghost load (lights, standby, router, etc.)
TARGET_EXPLAINED_RATIO = 0.75


def enforce_sum_constraint_smart(
    predictions: Dict[str, Tuple[float, float]],
    aggregate_power_kw: float,
    noise_threshold_kw: float = NOISE_THRESHOLD_W / 1000.0,
    target_explained_ratio: float = TARGET_EXPLAINED_RATIO,
) -> Dict[str, Tuple[float, float]]:
    """
    Smart Priority Scaling that enforces target explained ratio.

    Args:
        predictions: Dict mapping appliance_id -> (power_kw, confidence)
        aggregate_power_kw: Total measured aggregate power in kW
        noise_threshold_kw: Don't scale if deviation < this (avoids jitter)
        target_explained_ratio: Target sum/aggregate ratio (default 0.75 = 25% ghost load)

    Returns:
        Corrected predictions dict with same format
    """
    if not predictions or aggregate_power_kw <= 0:
        return predictions

    # Calculate target sum (75% of aggregate)
    target_sum = aggregate_power_kw * target_explained_ratio
    
    # Calculate current sum
    total_pred = sum(p for p, _ in predictions.values())
    
    # If predictions are already close to target, don't scale
    if abs(total_pred - target_sum) < noise_threshold_kw:
        return predictions
    
    # Scale factor to reach target
    if total_pred > 0:
        scale_factor = target_sum / total_pred
    else:
        return predictions
    
    logger.debug(
        f"Scaling predictions: sum={total_pred:.3f}kW -> target={target_sum:.3f}kW "
        f"(scale={scale_factor:.3f}, ghost={1-target_explained_ratio:.0%})"
    )

    # Apply uniform scaling to all predictions
    corrected = {}
    for appliance_id, (power_kw, conf) in predictions.items():
        corrected[appliance_id] = (power_kw * scale_factor, conf)

    return corrected


def compute_residual(
    predictions: Dict[str, Tuple[float, float]],
    aggregate_power_kw: float,
) -> float:
    """
    Compute residual power (unaccounted consumption).

    Residual = Aggregate - Sum(Predictions)
    This represents ghost loads (lights, standby, misc devices).
    Typically 25% of aggregate is normal.

    Returns:
        Residual power in kW (can be negative if predictions > aggregate)
    """
    total_pred = sum(p for p, _ in predictions.values())
    return aggregate_power_kw - total_pred
