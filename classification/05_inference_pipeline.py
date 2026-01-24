
"""
Real-time NILM Stage 4: Business Logic & Inference Engine
---------------------------------------------------------
The "Brain" that connects:
1. Data Stream (Smart Meter)
2. Detector (Z-Score)
3. Classifier (XGBoost)
4. State Machine (Business Logic)

Features:
- Real-time processing simulation
- State tracking (ON/OFF pairing)
- Cost calculation (EUR)
- Anomaly alerts
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import json
import time
from collections import deque

class NILMInferenceEngine:
    def __init__(self, model_path, metadata_path, cost_per_kwh=0.25):
        # Load Model
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        
        # Load Metadata
        with open(metadata_path, 'r') as f:
            self.meta = json.load(f)
            self.features = self.meta['features']
            self.classes = self.meta['classes']
            
        self.cost_per_kwh = cost_per_kwh
        
        # Runtime State
        self.buffer = deque(maxlen=20) # 20s buffer for Z-score/Features
        self.appliance_states = {} # {app_name: {'start_time': ts, 'avg_power': W}}
        self.total_cost = 0.0
        
        # Z-Score State
        self.window_size = 10
        self.threshold = 2.0
        self.min_power_diff = 0.020 # kW
        
    def process_stream(self, stream_data):
        """
        Simulate real-time processing.
        stream_data: DataFrame with index=Time, col='Aggregate' (kW)
        """
        print("Starting Real-Time Inference Simulation...")
        print("-" * 50)
        
        # Rolling stats for Z-score
        rolling_window = deque(maxlen=self.window_size)
        
        last_val = 0
        
        for ts, power in stream_data.items():
            current_val = power
            
            # 1. Detection (Online Z-Score)
            event_detected = False
            delta = 0
            
            if len(rolling_window) == self.window_size:
                mu = np.mean(rolling_window)
                sigma = np.std(rolling_window)
                if sigma < 1e-6: sigma = 1e-6
                
                z = (current_val - mu) / sigma
                diff = current_val - last_val
                
                if abs(z) > self.threshold and abs(diff) > self.min_power_diff:
                    event_detected = True
                    delta = diff
            
            # Update rolling stats
            rolling_window.append(current_val)
            self.buffer.append(current_val)
            last_val = current_val
            
            if event_detected:
                self._handle_event(ts, delta, current_val)
                
    def _handle_event(self, ts, delta, current_power):
        # 2. Feature Extraction (On current buffer)
        # We need steady state... in real-time we'd wait N seconds.
        # Here we cheat by peaking ahead? No, 'buffer' is past. Not enough for steady state.
        # REAL-TIME LOGIC: Trigger "Candidate Event", wait 5s, then classify.
        
        # Simplified: We assume we have the future in 'buffer' 
        # (Impossible in true 1-step stream without lag).
        # We will simulate "Response Latency" of 5 seconds.
        # For this demo, we skip the waiting logic and pretend we extracted it.
        
        # We just need to conform to the model's feature shape.
        # Features: ['delta_power', 'ss_mean', 'ss_std', 'ss_range', 'ss_p25', 'ss_p50', 'ss_p75', 'ramp_2s', 'hour', 'day_of_week']
        
        # Mock features based on single point (Limitation of simple loop)
        # In prod, use a 5s buffer delay.
        
        feat_vector = pd.DataFrame([{
            'delta_power': delta,
            'ss_mean': current_power, # Approx
            'ss_std': 0.01, # Mock
            'ss_range': 0.01,
            'ss_p25': current_power,
            'ss_p50': current_power,
            'ss_p75': current_power,
            'ramp_2s': delta, # Approx
            'hour': ts.hour,
            'day_of_week': ts.dayofweek
        }])
        
        # 3. Classification
        pred_idx = self.model.predict(feat_vector[self.features])[0]
        label = self.classes[pred_idx]
        
        # 4. State Machine & Business Logic
        self._update_state(label, delta, ts)
        
        print(f"[{ts}] EVENT: {label:<15} | Delta: {delta:+.3f} kW | Cost so far: €{self.total_cost:.2f}")

    def _update_state(self, label, delta, ts):
        """
        Pairing Logic:
        If Delta > 0: ON
        If Delta < 0: OFF
        """
        is_on = delta > 0
        
        if is_on:
            if label not in self.appliance_states:
                self.appliance_states[label] = {'start': ts, 'power': abs(delta)}
                print(f"   >>> {label} TURNED ON")
            else:
                # Already on? Maybe state change
                # Update power
                self.appliance_states[label]['power'] = abs(delta)
        else:
            if label in self.appliance_states:
                start_time = self.appliance_states[label]['start']
                avg_power = self.appliance_states[label]['power']
                
                duration_h = (ts - start_time).total_seconds() / 3600
                kwh = avg_power * duration_h
                cost = kwh * self.cost_per_kwh
                
                self.total_cost += cost
                
                print(f"   <<< {label} TURNED OFF (Duration: {duration_h*60:.1f} min, Cost: €{cost:.4f})")
                del self.appliance_states[label]
                
                # Anomaly Check
                if duration_h > 5: # e.g. Oven on for 5 hours
                     print(f"   !!! ALERT: {label} ran for {duration_h:.1f} hours!")

def main():
    # Load test data (last chunk of what we loaded)
    # Using the same parquet reading for simplicity
    from pathlib import Path
    BASE_DIR = Path('c:/Users/gamek/School/TeamProject/MTS3-MCTE-Team-Project-Energy-G1')
    DATA_PATH = BASE_DIR / "data/processed/1sec_new/nilm_ready_1sec_new.parquet"
    
    # Take a slice from the middle to ensure activity
    df = pd.read_parquet(DATA_PATH).iloc[100000:110000] # 10k samples ~ 2.7 hours
    stream = df.set_index('Time')['Aggregate']
    
    engine = NILMInferenceEngine(
        model_path="xgb_nilm_model.json",
        metadata_path="model_metadata.json"
    )
    
    engine.process_stream(stream)

if __name__ == "__main__":
    main()
