
"""
Stage 3: Event Classification (XGBoost)
---------------------------------------
Trains a gradient boosting classifier on the extracted features.
Includes robust Timezone handling for ground truth matching.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
from pathlib import Path
import ast

def match_labels(features_df, appliance_data, tolerance_sec=5, min_on_power=0.020):
    """
    Match detected events to ground truth appliance starts using available data.
    Algorithm:
    1. Identify all appliance 'start' events in ground truth (diff > threshold)
    2. Use merge_asof to find nearest ground truth start for each detected event
    3. Filter matches by time tolerance and power magnitude similarity
    """
    print("Matching events to ground truth labels (Vectorized)...")
    
    APPLIANCES = [
        'HeatPump', 'Dishwasher', 'WashingMachine', 'Dryer',
        'Oven', 'Stove', 'RangeHood', 'EVCharger', 'EVSocket',
        'GarageCabinet', 'RainwaterPump'
    ]
    
    # Calculate difference to find edges in ground truth
    # We need to handle this carefully.
    # If appliance_data index is Time, great.
    
    # Optimized: Calculate diffs once
    app_diffs = appliance_data[APPLIANCES].diff().fillna(0)
    
    matches = []
    
    # Ensure features sorted
    feature_events = features_df.sort_values('timestamp')
    
    for app in APPLIANCES:
        # Find starts > 20W (adjustable)
        changes = app_diffs[app]
        significant_mask = changes.abs() > min_on_power
        
        if not significant_mask.any():
            continue
            
        sig_changes = changes[significant_mask].reset_index()
        # Rename 'index' (or whatever name) to 'Time' if needed
        if 'Time' not in sig_changes.columns and sig_changes.columns[0] == 'index':
             sig_changes.rename(columns={'index': 'Time'}, inplace=True)
        elif 'Time' not in sig_changes.columns:
             # Assume first column is time
             sig_changes.rename(columns={sig_changes.columns[0]: 'Time'}, inplace=True)

        sig_changes.columns = ['Time', 'power_change']
        sig_changes['appliance_candidate'] = app
        
        # Merge
        merged = pd.merge_asof(
            feature_events[['timestamp', 'event_id', 'delta_power']],
            sig_changes,
            left_on='timestamp',
            right_on='Time',
            direction='nearest',
            tolerance=pd.Timedelta(seconds=tolerance_sec)
        )
        
        # Filter valid matches
        merged = merged.dropna(subset=['appliance_candidate'])
        
        # Power Check: The change in appliance should match the delta_power
        # Allow some error (e.g. 50W + 50%)
        # Note: delta_power might be aggregate, power_change is individual app.
        merged['error'] = (merged['delta_power'] - merged['power_change']).abs()
        
        # Dynamic threshold: 50W absolute or 50% relative
        valid_mask = merged['error'] < np.maximum(0.050, merged['delta_power'].abs() * 0.5)
        
        matches.append(merged[valid_mask])
        
    if not matches:
        print("No matches found!")
        features_df['label'] = 'Unknown'
        return features_df
        
    all_matches = pd.concat(matches)
    
    # Resolve duplicates: If an event matches multiple appliances, pick lowest error
    best_matches = all_matches.sort_values('error').drop_duplicates(subset=['event_id'], keep='first')
    
    # Map back to features
    label_map = best_matches.set_index('event_id')['appliance_candidate']
    features_df['label'] = features_df['event_id'].map(label_map).fillna('Unknown')
    
    print(f"Matched {len(best_matches)} events out of {len(features_df)}.")
    return features_df

def train_model(X, y):
    # 5. Train with Class Weights
    print("Training XGBoost with Class Balancing...")
    
    # Calculate weights: Total / (n_classes * count)
    from sklearn.utils.class_weight import compute_sample_weight
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    sample_weights = compute_sample_weight(
        class_weight='balanced',
        y=y_enc # Use full y encoded
    )
    
    # Split weights to match train set
    X_train_w, X_test_w, y_train_w, y_test_w, sw_train, sw_test = train_test_split(
        X, y_enc, sample_weights, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    # ULTIMATE MODEL (Max Performance, No Edge Constraints)
    # User Request: "IL FOTTUTO MODELLO" (The Best Model)
    print("Training ULTIMATE XGBoost (Server-Grade Config)...")
    model = xgb.XGBClassifier(
        n_estimators=1500,       # Massive ensemble
        max_depth=8,             # Deep interaction capture
        learning_rate=0.02,      # Very slow, precise learning
        objective='multi:softprob',
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method='hist',
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(
        X_train_w, y_train_w,
        sample_weight=sw_train,
        verbose=True
    )
    
    print("Evaluating...")
    y_pred = model.predict(X_test_w)
    
    print(classification_report(y_test_w, y_pred, target_names=le.classes_))
    
    return model, le

def main():
    # 1. Load context
    print("Loading data for labeling...")
    BASE_DIR = Path('c:/Users/gamek/School/TeamProject/MTS3-MCTE-Team-Project-Energy-G1')
    DATA_PATH = BASE_DIR / "data/processed/1sec_new/nilm_ready_1sec_new.parquet"
    
    df = pd.read_parquet(DATA_PATH)
    
    # --- TIMEZONE FIX ---
    # Ensure TZ naive for ground truth
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
    if 'Time' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['Time']):
            df['Time'] = pd.to_datetime(df['Time'])
        if df['Time'].dt.tz is not None:
            df['Time'] = df['Time'].dt.tz_localize(None)
        df = df.set_index('Time')
    
    # 2. Load Features
    feat_path = Path("event_features.csv")
    if not feat_path.exists():
        print("event_features.csv not found.")
        return
        
    features_df = pd.read_csv(feat_path)
    features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
    
    # --- TIMEZONE FIX for Features ---
    if features_df['timestamp'].dt.tz is not None:
        features_df['timestamp'] = features_df['timestamp'].dt.tz_localize(None)
        
    # 3. Match
    labeled_features = match_labels(features_df, df)
    
    # Match
    labeled_features = match_labels(features_df, df)
    
    # Filter unknowns
    labeled_features = labeled_features[labeled_features['label'] != 'Unknown']
    
    # Reverting Filter: User wants ALL appliances, even if < 80%.
    # We will use the Ultimate Model to maximize their score.
    
    # 4. Prepare X, y
    feature_cols = [
        'delta_power', 'ss_mean', 'ss_std', 'ss_range',
        'ss_p25', 'ss_p50', 'ss_p75', 'ramp_2s',
        'hour', 'day_of_week',
        # SOTA 2025 Spectral Features
        'dominant_freq', 'spectral_power', 'spectral_entropy'
    ]
    # Check if cols exist
    existing_cols = [c for c in feature_cols if c in labeled_features.columns]
    
    X = labeled_features[existing_cols]
    y = labeled_features['label']
    
    if len(y) < 50:
        print("Not enough labeled events to train.")
        return

    # --- SOTA 2025: LOF Ghost Filtering ---
    # Section 1.3: Filter "Unknown" or "Ghost" devices before classification
    print("Applying LOF for Ghost Device Filtering...")
    from sklearn.neighbors import LocalOutlierFactor
    
    # Use LOF to detect density outliers
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1) # Assume 10% ghosts
    is_inlier = lof.fit_predict(X) 
    # 1 is inlier, -1 is outlier
    
    n_outliers = np.sum(is_inlier == -1)
    print(f"LOF identified {n_outliers} Ghost Events (Outliers). Removing them.")
    
    # Filter
    X = X[is_inlier == 1]
    y = y[is_inlier == 1]
        
    # 5. Train
    model, le = train_model(X, y)
    
    # Save
    model.save_model("xgb_nilm_model.json")
    
    meta = {
        'classes': le.classes_.tolist(),
        'features': existing_cols
    }
    with open("model_metadata.json", "w") as f:
        json.dump(meta, f)
        
    print("Model saved to xgb_nilm_model.json")

if __name__ == "__main__":
    main()
