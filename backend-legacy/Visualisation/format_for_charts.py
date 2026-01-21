import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional


def format_for_charts(predictions: pd.DataFrame, 
                      timestamps: Optional[List[datetime]] = None,
                      include_confidence: bool = True) -> Dict[str, Any]:
    """
    Format predictions for frontend chart visualization.
    
    Args:
        predictions: DataFrame with columns 'predicted_power', 'on_off_probability', 'is_on'
        timestamps: Optional list of timestamps for predictions
        include_confidence: Whether to include confidence probabilities
    
    Returns:
        Dictionary formatted for JavaScript/React charts
    """
    result = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_predictions": len(predictions),
            "unit": "W"
        },
        "data": []
    }
    
    for i, row in predictions.iterrows():
        point = {
            "index": int(i),
            "power": float(row['predicted_power']),
            "is_on": bool(row['is_on'])
        }
        
        if timestamps and i < len(timestamps):
            point["timestamp"] = timestamps[i].isoformat()
        
        if include_confidence:
            point["confidence"] = float(row['on_off_probability'])
        
        result["data"].append(point)
    
    # Calculate aggregated statistics
    result["statistics"] = calculate_statistics(predictions)
    
    return result


def calculate_statistics(predictions: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate aggregated statistics from predictions.
    
    Args:
        predictions: DataFrame with predictions
    
    Returns:
        Dictionary with statistics
    """
    power_values = predictions['predicted_power'].values
    on_states = predictions['is_on'].values
    
    # Filter values when appliance is on
    power_when_on = power_values[on_states.astype(bool)]
    
    stats = {
        "total_energy_wh": float(np.sum(power_values) / 4),  # Assuming 15min intervals
        "average_power_w": float(np.mean(power_values)),
        "max_power_w": float(np.max(power_values)),
        "min_power_w": float(np.min(power_values)),
        "on_time_percentage": float(np.mean(on_states) * 100),
        "on_periods_count": int(count_on_periods(on_states)),
    }
    
    if len(power_when_on) > 0:
        stats["average_power_when_on_w"] = float(np.mean(power_when_on))
    else:
        stats["average_power_when_on_w"] = 0.0
    
    return stats


def count_on_periods(on_states: np.ndarray) -> int:
    """Count the number of consecutive ON periods."""
    if len(on_states) == 0:
        return 0
    
    # Find state changes
    changes = np.diff(on_states.astype(int))
    # Count OFF to ON transitions
    return int(np.sum(changes == 1)) + (1 if on_states[0] else 0)


def format_timeseries_for_grafana(predictions: pd.DataFrame,
                                   timestamps: List[datetime],
                                   metric_name: str = "predicted_power") -> List[Dict]:
    """
    Format predictions for Grafana/InfluxDB.
    
    Args:
        predictions: DataFrame with predictions
        timestamps: List of timestamps
        metric_name: Name of the metric
    
    Returns:
        List of points in Grafana format
    """
    points = []
    
    for i, (ts, row) in enumerate(zip(timestamps, predictions.itertuples())):
        point = {
            "time": ts.isoformat(),
            "measurement": metric_name,
            "fields": {
                "power": float(row.predicted_power),
                "probability": float(row.on_off_probability)
            },
            "tags": {
                "appliance": "heatpump",
                "is_on": str(row.is_on)
            }
        }
        points.append(point)
    
    return points


def format_24h_summary(predictions: pd.DataFrame,
                       timestamps: List[datetime]) -> Dict[str, Any]:
    """
    Create a summary of predictions for the next 24 hours.
    
    Args:
        predictions: DataFrame with predictions
        timestamps: List of timestamps
    
    Returns:
        Dictionary with 24h summary
    """
    # Group by hour
    hourly_data = {}
    
    for ts, row in zip(timestamps, predictions.itertuples()):
        hour = ts.hour
        if hour not in hourly_data:
            hourly_data[hour] = {
                "power_sum": 0,
                "count": 0,
                "on_count": 0
            }
        
        hourly_data[hour]["power_sum"] += row.predicted_power
        hourly_data[hour]["count"] += 1
        hourly_data[hour]["on_count"] += int(row.is_on)
    
    # Calculate hourly averages
    hourly_summary = []
    for hour in sorted(hourly_data.keys()):
        data = hourly_data[hour]
        hourly_summary.append({
            "hour": hour,
            "average_power": data["power_sum"] / data["count"] if data["count"] > 0 else 0,
            "on_percentage": data["on_count"] / data["count"] * 100 if data["count"] > 0 else 0
        })
    
    return {
        "hourly_breakdown": hourly_summary,
        "peak_hour": max(hourly_summary, key=lambda x: x["average_power"])["hour"],
        "total_predicted_energy_kwh": sum(h["average_power"] for h in hourly_summary) / 1000
    }
