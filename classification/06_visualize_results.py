
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
import json
from pathlib import Path

def main():
    # User Request: "Giornate Intense" (Intense Days)
    # We pick one Winter day (Heavy Heating) and one Summer day (Cooling)
    INTENSE_DAYS = ['2024-01-15', '2024-07-02'] 
    APPLIANCE = 'HeatPump' 
    
    # 1. Load Data (Once)
    print("Loading Data for Visualization...")
    BASE_DIR = Path('c:/Users/gamek/School/TeamProject/MTS3-MCTE-Team-Project-Energy-G1')
    DATA_PATH = BASE_DIR / "data/processed/1sec_new/nilm_ready_1sec_new.parquet"
    
    df_full = pd.read_parquet(DATA_PATH)
    if 'Time' in df_full.columns: df_full = df_full.set_index('Time')
    if df_full.index.tz is not None: df_full.index = df_full.index.tz_localize(None)
    
    # Load Model (Once)
    model = xgb.XGBClassifier()
    model.load_model("xgb_nilm_model.json")
    
    with open("model_metadata.json", "r") as f:
        meta = json.load(f)
    classes = meta['classes']
    features = meta['features']
    
    # Load Features (Once)
    print("Loading extracted features...")
    feat_path = Path("event_features.csv")
    feat_df_full = pd.read_csv(feat_path)
    feat_df_full['timestamp'] = pd.to_datetime(feat_df_full['timestamp'])
    if feat_df_full['timestamp'].dt.tz is not None:
        feat_df_full['timestamp'] = feat_df_full['timestamp'].dt.tz_localize(None)

    for day_str in INTENSE_DAYS:
        START_DATE = f"{day_str} 00:00:00"
        END_DATE = f"{day_str} 23:59:59"
        print(f"\n--- Visualizing {day_str} (Full Day) ---")
        
        # 2. Slice Data
        mask = (df_full.index >= START_DATE) & (df_full.index < END_DATE)
        df_slice = df_full[mask]
        
        if len(df_slice) == 0:
            print(f"No data for {day_str}. Skipping.")
            continue
            
        real_signal = df_slice[APPLIANCE]
        
        # Slice Features
        feat_slice = feat_df_full[(feat_df_full['timestamp'] >= START_DATE) & 
                                  (feat_df_full['timestamp'] < END_DATE)].copy()
        
        if len(feat_slice) == 0:
            print(f"No events found in {day_str}.")
            # Continue to plot empty prediction vs real? Maybe real has data but no events detected?
            # We should plot even if no events, to show flat line.
            # But we need feat_slice for logic.
            # Create empty logic handled below?
            pass

        if len(feat_slice) > 0:
            print(f"Predicting {len(feat_slice)} events...")
            X = feat_slice[features]
    # Predict
    y_pred_idx = model.predict(X)
    feat_slice['pred_class'] = [classes[i] for i in y_pred_idx]
    
    # 5. Reconstruct Logic within plot_day
    plot_day(df_slice, real_signal, feat_slice, day_str, APPLIANCE)

def plot_day(df_slice, real_signal, feat_slice, day_str, APPLIANCE):
    print(f"  > Reconstructing signal for {day_str}...")
    
    # Sort events
    feat_slice = feat_slice.sort_values('timestamp')
    
    # Track state
    current_state = 0.0
    synth_events = []
    
    # We need fast access to raw signal for backtracking
    raw_series = df_slice['Aggregate'] if 'Aggregate' in df_slice.columns else real_signal # Best guess for total load
    
    feature_events = feat_slice.to_dict('records')
    
    # ... (Backtracking Logic) ...
    for i, evt in enumerate(feature_events):
        if evt['pred_class'] == APPLIANCE:
            delta = evt['delta_power']
            
            # If turning OFF (large negative) and we think it's OFF (state ~0)
            if delta < -200 and current_state < 200:
                # print(f"  > Detected Hidden Ramp ending at {evt['timestamp']}...")
                
                # Search backwards up to 3 hours
                end_time = evt['timestamp']
                start_search = end_time - pd.Timedelta(hours=3)
                
                try:
                    # Robust Loop Approach (Pure Numpy)
                    window = raw_series.loc[start_search:end_time]
                    
                    found_time = None
                    if len(window) > 0:
                        # Convert to numpy immediately to bypass pandas indexing issues
                        # Force 1D array for values
                        vals = window.values.flatten()
                        times = window.index.values
                        
                        # Iterate backwards
                        n_points = len(vals)
                        for j in range(n_points - 1, -1, -1):
                            if vals[j] < 200:
                                found_time = times[j]
                                break
                                
                        if found_time is not None:
                            # Create Synthetic ON Event
                            synth_on = evt.copy()
                            synth_on['timestamp'] = found_time
                            synth_on['delta_power'] = abs(delta) 
                            synth_on['pred_class'] = APPLIANCE
                            synth_on['type'] = 'Synthetic Ramp'
                            synth_events.append(synth_on)
                except Exception as e:
                    pass

            # Update tracking state
            current_state += delta
            if current_state < 0: current_state = 0

    # Merge synthetic events
    if synth_events:
        all_events = pd.DataFrame(feature_events + synth_events)
        all_events = all_events.sort_values('timestamp')
    else:
        all_events = feat_slice
        
    # Reconstruct 
    # Use dense array as before
    power_values = np.zeros(len(df_slice.index))
    # Ensure event_times are in the window
    # Filter all_events to be within df_slice index range (just in case)
    # Actually, feat_slice was already filtered.
    
    event_times = all_events['timestamp'].values
    event_indices = df_slice.index.get_indexer(event_times, method='nearest')
    
    curr = 0.0
    curr_idx = 0
    
    # Vectorized loop for reconstruction? 
    # The loop approach is fine for 86400 points.
    
    # Pre-sort indices to be safe
    # sorted_pairs = sorted(zip(event_indices, all_events['delta_power'], all_events['pred_class']))
    
    # But all_events is sorted by timestamp, so indices should be roughly sorted.
    
    for i, evt_idx in enumerate(event_indices):
        if evt_idx < 0: continue
        power_values[curr_idx:evt_idx] = curr
        
        row = all_events.iloc[i]
        if row['pred_class'] == APPLIANCE:
            curr += row['delta_power']
            if curr < 0: curr = 0.0
            
        curr_idx = evt_idx
    
    power_values[curr_idx:] = curr
    reconstructed = pd.Series(power_values, index=df_slice.index)
    
    # 6. Plot
    plt.figure(figsize=(15, 6))
    plt.plot(real_signal.index, real_signal.values, label=f'Real {APPLIANCE}', color='black', alpha=0.6, linewidth=1.5)
    plt.plot(reconstructed.index, reconstructed.values, label=f'Predicted (Full Day)', color='red', linestyle='-', alpha=0.8, linewidth=1.5)
    
    # Plot matched events as dots
    events = all_events[all_events['pred_class'] == APPLIANCE]
    
    # Mark Real vs Synthetic
    real_evts = events[events.get('type', 'Real') != 'Synthetic Ramp']
    
    plt.scatter(real_evts['timestamp'], [real_signal.max()] * len(real_evts), color='red', marker='v', s=20, label='Detected Event')
    
    plt.title(f"{APPLIANCE}: Real vs Predicted (24H Intense Day) - {day_str}")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_file = f"classification/plot_{APPLIANCE}_{day_str}.png"
    plt.savefig(out_file)
    print(f"Saved plot to {out_file}")

if __name__ == "__main__":
    main()
