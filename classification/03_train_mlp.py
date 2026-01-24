
"""
Real-time NILM Stage 3: Hybrid Deep Learning (Waveform + Scalars)
-----------------------------------------------------------------
Fuses raw waveform data (for shape) with engineered scalars (for magnitude)
into a single Deep MLP.

Results: 
- Waveform alone: Good for HeatPump (Shape)
- Scalars alone: Good for EVCharger (Magnitude)
- Hybrid: Best of both worlds (SOTA)
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import json
import ast
import joblib

def match_labels(features_df, appliance_data, tolerance_sec=5, min_on_power=0.020):
    print("Matching events to ground truth labels (Vectorized)...")
    APPLIANCES = [
        'HeatPump', 'Dishwasher', 'WashingMachine', 'Dryer',
        'Oven', 'Stove', 'RangeHood', 'EVCharger', 'EVSocket',
        'GarageCabinet', 'RainwaterPump'
    ]
    app_diffs = appliance_data[APPLIANCES].diff().fillna(0)
    matches = []
    feature_events = features_df.sort_values('timestamp')
    
    for app in APPLIANCES:
        changes = app_diffs[app]
        mask = changes.abs() > min_on_power
        if not mask.any(): continue
        sig = changes[mask].reset_index()
        sig.columns = ['Time', 'power_change']
        sig['appliance_candidate'] = app
        merged = pd.merge_asof(
            feature_events[['timestamp', 'event_id', 'delta_power']],
            sig, left_on='timestamp', right_on='Time',
            direction='nearest', tolerance=pd.Timedelta(seconds=tolerance_sec)
        )
        merged = merged.dropna(subset=['appliance_candidate'])
        merged['error'] = (merged['delta_power'] - merged['power_change']).abs()
        valid = merged['error'] < np.maximum(0.050, merged['delta_power'].abs() * 0.5)
        matches.append(merged[valid])
        
    if not matches:
        features_df['label'] = 'Unknown'
        return features_df
    all_matches = pd.concat(matches)
    best = all_matches.sort_values('error').drop_duplicates(subset=['event_id'], keep='first')
    label_map = best.set_index('event_id')['appliance_candidate']
    features_df['label'] = features_df['event_id'].map(label_map).fillna('Unknown')
    return features_df

def main():
    feat_path = Path("event_features.csv")
    if not feat_path.exists():
        print("event_features.csv not found.")
        return
    
    print("Loading Features...")
    df = pd.read_csv(feat_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is not None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    
    print("Parsing Waveforms...")
    df['waveform'] = df['waveform'].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x)
    
    print("Loading Labels...")
    BASE_DIR = Path('c:/Users/gamek/School/TeamProject/MTS3-MCTE-Team-Project-Energy-G1')
    DATA_PATH = BASE_DIR / "data/processed/1sec_new/nilm_ready_1sec_new.parquet"
    ground_truth = pd.read_parquet(DATA_PATH)
    
    if isinstance(ground_truth.index, pd.DatetimeIndex):
        if ground_truth.index.tz is not None:
            ground_truth.index = ground_truth.index.tz_localize(None)
    if 'Time' in ground_truth.columns:
        if not pd.api.types.is_datetime64_any_dtype(ground_truth['Time']):
            ground_truth['Time'] = pd.to_datetime(ground_truth['Time'])
        if ground_truth['Time'].dt.tz is not None:
            ground_truth['Time'] = ground_truth['Time'].dt.tz_localize(None)
        outer = ground_truth.set_index('Time')
    else:
        outer = ground_truth
        
    df = match_labels(df, outer)
    df = df[df['label'] != 'Unknown']
    print(f"Labeled: {len(df)}")
    
    if len(df) < 50: return
    
    # --- Hybrid Feature Construction ---
    # 1. Waveforms (N, 50)
    X_wave = np.stack(df['waveform'].values)
    
    # 2. Scalars (N, K)
    # Ensure these cols exist in 'event_features.csv'
    scalar_cols = ['delta_power', 'ss_mean', 'ss_std', 'ss_range', 'hour']
    X_scalars = df[scalar_cols].values
    
    # Scaling
    print("Scaling Features...")
    scaler_w = StandardScaler()
    X_wave_scaled = scaler_w.fit_transform(X_wave)
    
    scaler_s = StandardScaler()
    X_scalars_scaled = scaler_s.fit_transform(X_scalars)
    
    # Concatenate
    X_combined = np.hstack([X_wave_scaled, X_scalars_scaled])
    
    y = df['label'].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    
    print(f"Training Hybrid MLP (Features: {X_combined.shape[1]})...")
    
    # Robust Architecture
    clf = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42,
        verbose=True
    )
    
    clf.fit(X_train, y_train)
    
    print("\nEvaluation:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    joblib.dump(clf, "nilm_hybrid_model.pkl")
    meta = {
        'classes': le.classes_.tolist(),
        'input_dim': X_combined.shape[1],
        'features': scalar_cols # needed to reconstruct input during inference
    }
    with open("hybrid_metadata.json", "w") as f:
        json.dump(meta, f)
    print("Saved nilm_hybrid_model.pkl")

if __name__ == "__main__":
    main()
