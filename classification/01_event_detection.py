
"""
Real-time NILM Stage 1: Event Detection (Z-Score)
-------------------------------------------------
Implementation of the Z-Score adaptive thresholding method for 1 Hz data.

Reference: "Non-Intrusive Load Monitoring (NILM) Event-Driven Architectures for Real-Time Edge Computing"
Section 2.2: Event Detection: Adaptive Thresholding

Algorithm:
1. Calculate rolling mean (mu) and std (sigma) over a window (e.g., 10 seconds).
2. Compute Z-score: z_t = (P_t - mu) / sigma
3. Trigger event if |z_t| > Threshold (e.g., 3.0)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class ZScoreDetector:
    def __init__(self, window_size=10, threshold=3.5, min_power_diff=20):
        """
        Args:
            window_size (int): Rolling window size in samples (seconds).
            threshold (float): Z-score threshold for detection.
            min_power_diff (float): Minimum absolute power change (Watts) to consider as a valid event.
                                    Helps filter out noise even if Z-score is high during quiet periods.
        """
        self.window_size = window_size
        self.threshold = threshold
        self.min_power_diff = min_power_diff
        
    def detect_events(self, power_series):
        """
        Detect events in a power series.
        
        Args:
            power_series (pd.Series): 1-second resolution power data (Watts).
            
        Returns:
            pd.DataFrame: DataFrame of detected events with columns:
                          ['timestamp', 'power_before', 'power_after', 'delta_power', 'z_score']
        """
        # Calculate rolling stats
        # Shift detection by 1 to compare current value against *past* statistics (causal)
        # However, Z-score usually compares point t to window around it or preceeding it.
        # For real-time, we compare t to [t-window, t-1].
        
        rolling_mean = power_series.rolling(window=self.window_size).mean().shift(1)
        rolling_std = power_series.rolling(window=self.window_size).std().shift(1)
        
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, 1.0)
        
        # Calculate Z-score
        z_scores = (power_series - rolling_mean) / rolling_std
        
        # DEBUG stats
        print(f"Power stats: Mean={power_series.mean():.2f}, Std={power_series.std():.2f}")
        try:
            print(f"Z-score stats: Max={z_scores.max():.2f}, Min={z_scores.min():.2f}")
            print(f"Diff stats: Max={power_series.diff().abs().max():.2f}")
        except:
            print("Could not calc stats (empty?)")
        
        # Detect candidate events
        candidates = np.abs(z_scores) > self.threshold
        
        # Filter candidates
        # Let's use a simpler differential approach combined with z-score for Step 1
        # P_t - P_{t-1}
        diff = power_series.diff()
        
        # Mask: Z-score high AND Abs Diff > min_power
        mask = (candidates) & (diff.abs() > self.min_power_diff)
        
        detected_indices = power_series.index[mask]
        print(f"Candidates (Z>{self.threshold}): {candidates.sum()}")
        print(f"Final Events (Diff>{self.min_power_diff}): {len(detected_indices)}") 
                
        event_list = []
        for ts in detected_indices:
            # We need the loc index to get previous value easily
            # Assuming discrete 1s steps. 
            
            p_curr = power_series.loc[ts]
            # Approximate 'before' as current - diff
            p_before = p_curr - diff.loc[ts]
            
            event_list.append({
                'timestamp': ts,
                'power_before': p_before,
                'power_after': p_curr,
                'delta_power': diff.loc[ts],
                'z_score': z_scores.loc[ts]
            })
            
        return pd.DataFrame(event_list)

def main():
    # Load a chunk of data for testing
    print("Loading data...")
    # Absolute path to be safe
    BASE_DIR = Path('c:/Users/gamek/School/TeamProject/MTS3-MCTE-Team-Project-Energy-G1')
    DATA_PATH = BASE_DIR / "data/processed/1sec_new/nilm_ready_1sec_new.parquet"
    
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.")
        return

    # Load full dataset (15 months / ~39M rows)
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df)} rows for detection.")
    
    # Select Aggregate power
    # Note: Column is 'Aggregate' based on previous context
    power_data = df.set_index('Time')['Aggregate']
    
    # 1. Event Detection
    print("Running Z-Score Event Detection...")
    # Data appears to be in kW. Max diff was ~9.08.
    # Threshold: Z=2.0 (relative), MinDiff=0.02 kW (20 Watts)
    detector = ZScoreDetector(window_size=10, threshold=2.0, min_power_diff=0.020) 
    events_df = detector.detect_events(power_data)
    
    print(f"Detected {len(events_df)} events.")
    print(events_df.head(10))
    
    # Save for next stage
    events_df.to_csv("detected_events.csv", index=False)
    print("Events saved to detected_events.csv")
    
    # Optional: Plotting specific segment
    if len(events_df) > 0:
        # Plot around the first large event 
        # Pick top 5 events by delta
        top_events = events_df.reindex(events_df.delta_power.abs().sort_values(ascending=False).index).head(5)
        
        for i, (idx, row) in enumerate(top_events.iterrows()):
            ts = row['timestamp']
            window_sec = 60
            start_plot = ts - pd.Timedelta(seconds=window_sec)
            end_plot = ts + pd.Timedelta(seconds=window_sec)
            
            subset = power_data.loc[start_plot:end_plot]
            
            plt.figure(figsize=(10, 4))
            plt.plot(subset.index, subset.values, label='Aggregate Power')
            plt.axvline(ts, color='red', linestyle='--', label='Detected Event')
            plt.title(f"Event at {ts} | Delta: {row['delta_power']:.2f}W | Z: {row['z_score']:.2f}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"event_plot_{i}.png")
            plt.close()
        print("Generated sample event plots.")

if __name__ == "__main__":
    main()
