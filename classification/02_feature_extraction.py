
"""
Real-time NILM Stage 2: Feature Extraction
------------------------------------------
Extracts transient and steady-state features for each detected event.

Reference: "NILM Event-Driven Architectures for Real-Time Edge Computing"
Section 3.2: Low-Frequency Feature Extraction (1 Hz)

Features to Extract:
1. Transient:
   - Delta Power (ΔP) - Already from detection
   - Rise Time (Slope) - Not easily available at 1Hz 1-step, but can check 2-3 step ramp.
2. Steady-State (Window post-event, e.g., 5-10 seconds):
   - Mean Power
   - Variance (Std Dev)
   - Min/Max Range
3. Context:
   - Hour of Day (Cyclical)
   - Day of Week (Cyclical)
4. Historical/Database (Simulated):
   - Proximity to known appliance signatures (for KNN fallback later)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def extract_features(events_df, power_series, steady_state_window=10):
    """
    Extract features using vectorized numpy operations for speed.
    """
    features_list = []
    
    # Ensure sorted by time
    events_df = events_df.sort_values('timestamp')
    
    # Convert series to numpy
    # Ensure time is monotonic
    if not power_series.index.is_monotonic_increasing:
        power_series = power_series.sort_index()
        
    times = power_series.index.values
    values = power_series.values
    
    # Config for waveform
    window_len = 50
    pre_event = 10
    post_event = 40
    
    print(f"Extracting features for {len(events_df)} events (Optimized)...")
    
    # Map event timestamps to integer indices in power_series
    # searchsorted is O(logN) per event
    event_times = events_df['timestamp'].values
    event_indices = np.searchsorted(times, event_times)
    
    # Iterate indices
    for i, idx in enumerate(tqdm(event_indices)):
        ts = event_times[i]
        
        # Check bounds
        if idx < pre_event or idx + post_event > len(values):
            continue
            
        # Get row data
        row = events_df.iloc[i]
        
        # 0. Waveform (Fast Numpy Slice)
        waveform = values[idx - pre_event : idx + post_event]
        
        # Check length
        if len(waveform) != window_len:
            continue
            
        # Normalize
        baseline = np.mean(waveform[:pre_event]) if pre_event > 0 else waveform[0]
        norm_waveform = waveform - baseline
        
        # 1. Steady State (Slice: idx+1 to idx+window)
        # Note: steady_state_window in seconds. Assuming 1Hz data -> window samples
        ss_window_len = steady_state_window
        ss_slice = values[idx + 1 : idx + 1 + ss_window_len]
        
        if len(ss_slice) == 0: continue
            
        ss_mean = np.mean(ss_slice)
        ss_std = np.std(ss_slice)
        ss_min = np.min(ss_slice)
        ss_max = np.max(ss_slice)
        ss_range = ss_max - ss_min
        ss_p25 = np.percentile(ss_slice, 25)
        ss_p50 = np.median(ss_slice)
        ss_p75 = np.percentile(ss_slice, 75)
        
        # 2. Ramp (idx + 2)
        if idx + 2 < len(values):
            ramp_2s = values[idx + 2] - values[idx]
        else:
            ramp_2s = 0.0
            
        # Context
        # Need to convert numpy datetime64 [ns] to hour/day
        # ts is numpy datetime64
        # Converting each is slow. Use vectorized beforehand?
        # Or faster: pd.Timestamp(ts).hour
        p_ts = pd.Timestamp(ts)
        hour = p_ts.hour
        dow = p_ts.dayofweek
        
        feat = {
            'event_id': p_ts,
            'timestamp': p_ts,
            'delta_power': row['delta_power'],
            'waveform': norm_waveform.tolist(),
            'ss_mean': ss_mean,
            'ss_std': ss_std,
            'ss_range': ss_range,
            'ss_p25': ss_p25,
            'ss_p50': ss_p50,
            'ss_p75': ss_p75,
            'ramp_2s': ramp_2s,
            'hour': hour,
            'day_of_week': dow,
            # SOTA 2025 Spectral Features
            'dominant_freq': 0.0,
            'spectral_power': 0.0,
            'spectral_entropy': 0.0
        }
        
        # FFT Calculation
        # Window is likely available as 'window'
        # If not, use 'norm_waveform' (centered/scaled is fine for freq)
        sig_fft = norm_waveform
        n = len(sig_fft)
        if n > 10:
            freqs = np.fft.rfftfreq(n, d=1.0)
            fft_vals = np.abs(np.fft.rfft(sig_fft))
            
            # Dominant Freq (skip DC)
            if len(fft_vals) > 1:
                dom_idx = np.argmax(fft_vals[1:]) + 1
                feat['dominant_freq'] = freqs[dom_idx]
                feat['spectral_power'] = fft_vals[dom_idx]
                
                # Spectral Entropy
                psd = fft_vals[1:]**2
                psd_sum = np.sum(psd)
                if psd_sum > 0:
                    psd_norm = psd / psd_sum
                    feat['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

        features_list.append(feat)
            
    return pd.DataFrame(features_list)

def main():
    # Load events
    input_path = Path("detected_events.csv")
    if not input_path.exists():
        print("detected_events.csv not found. Run Stage 1 first.")
        return
        
    events_df = pd.read_csv(input_path)
    events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
    
    # Load raw data again (needed for feature extraction context)
    BASE_DIR = Path('c:/Users/gamek/School/TeamProject/MTS3-MCTE-Team-Project-Energy-G1')
    DATA_PATH = BASE_DIR / "data/processed/1sec_new/nilm_ready_1sec_new.parquet"
    df = pd.read_parquet(DATA_PATH)
    power_data = df.set_index('Time')['Aggregate']
    
    # Extract
    features_df = extract_features(events_df, power_data)
    
    # Save
    features_df.to_csv("event_features.csv", index=False)
    print(f"Features saved to event_features.csv. Shape: {features_df.shape}")

if __name__ == "__main__":
    main()
