
"""
Real-time NILM Stage 3: Deep Learning Classifier (1D-CNN)
---------------------------------------------------------
Trains a 1D Convolutional Neural Network on raw waveform windows 
to distinguish appliance transients with high precision.

Approach:
- Input: 50-point waveform window (10s before, 40s after event)
- Architecture: Conv1D -> MaxPooling -> Flatten -> Dense
- Advantages: Learns shape patterns (motor start surge vs resistive step) automatically.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import json
import ast

def match_labels(features_df, appliance_data, tolerance_sec=5, min_on_power=0.020):
    """
    Generate labels using vectorized merge_asof.
    """
    print("Matching events to ground truth labels (Vectorized)...")
    
    APPLIANCES = [
        'HeatPump', 'Dishwasher', 'WashingMachine', 'Dryer',
        'Oven', 'Stove', 'RangeHood', 'EVCharger', 'EVSocket',
        'GarageCabinet', 'RainwaterPump'
    ]
    
    # Calculate appliance changes
    app_diffs = appliance_data[APPLIANCES].diff().fillna(0)
    
    # Store candidate matches
    matches = []
    
    feature_events = features_df.sort_values('timestamp')
    
    # Iterate over each appliance to find significant changes
    for app in APPLIANCES:
        changes = app_diffs[app]
        significant_mask = changes.abs() > min_on_power
        
        if not significant_mask.any():
            continue
            
        sig_changes = changes[significant_mask].reset_index() 
        sig_changes.columns = ['Time', 'power_change']
        sig_changes['appliance_candidate'] = app
        
        # Match
        merged = pd.merge_asof(
            feature_events[['timestamp', 'event_id', 'delta_power']],
            sig_changes,
            left_on='timestamp',
            right_on='Time',
            direction='nearest',
            tolerance=pd.Timedelta(seconds=tolerance_sec)
        )
        
        merged = merged.dropna(subset=['appliance_candidate'])
        merged['error'] = (merged['delta_power'] - merged['power_change']).abs()
        valid_mask = merged['error'] < np.maximum(0.050, merged['delta_power'].abs() * 0.5)
        matches.append(merged[valid_mask])
        
    if not matches:
        features_df['label'] = 'Unknown'
        return features_df
        
    all_matches = pd.concat(matches)
    best_matches = all_matches.sort_values('error').drop_duplicates(subset=['event_id'], keep='first')
    
    # Map
    label_map = best_matches.set_index('event_id')['appliance_candidate']
    features_df['label'] = features_df['event_id'].map(label_map).fillna('Unknown')
    
    return features_df

def create_cnn_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        
        # Conv Block 1
        tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.2),
        
        # Conv Block 2
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.2),
        
        # Dense Head
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    feat_path = Path("event_features.csv")
    if not feat_path.exists():
        print("event_features.csv not found.")
        return
        
    print("Loading features...")
    features_df = pd.read_csv(feat_path)
    features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
    # Ensure naive
    if features_df['timestamp'].dt.tz is not None:
        features_df['timestamp'] = features_df['timestamp'].dt.tz_localize(None)
    
    # Parse waveform column (stored as string representation of list in CSV)
    print("Parsing waveforms...")
    # Safe eval for lists
    features_df['waveform'] = features_df['waveform'].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x)
    
    print("Loading ground truth for labeling...")
    BASE_DIR = Path('c:/Users/gamek/School/TeamProject/MTS3-MCTE-Team-Project-Energy-G1')
    DATA_PATH = BASE_DIR / "data/processed/1sec_new/nilm_ready_1sec_new.parquet"
    ground_truth = pd.read_parquet(DATA_PATH).head(5000000) # Subset match
    
    # Handle Index / Time column
    if 'Time' in ground_truth.columns:
        # Move Time to Column if it's index? No, set_index it if needed, or keep as column.
        # But ground_truth needs to be DataFrame where columns are appliances? No, match_labels expects DF.
        # But app_diffs = appliance_data[APPS].diff().
        # So appliance_data must have APPS columns.
        pass
        
    # Ensure TZ naive for ground truth index or Time column
    # If index is datetime
    if isinstance(ground_truth.index, pd.DatetimeIndex):
        if ground_truth.index.tz is not None:
            ground_truth.index = ground_truth.index.tz_localize(None)
    # If 'Time' column exists
    if 'Time' in ground_truth.columns:
        if not pd.api.types.is_datetime64_any_dtype(ground_truth['Time']):
            ground_truth['Time'] = pd.to_datetime(ground_truth['Time'])
        if ground_truth['Time'].dt.tz is not None:
            ground_truth['Time'] = ground_truth['Time'].dt.tz_localize(None)
        
        outer = ground_truth.set_index('Time')
    else:
        outer = ground_truth
    
    # Label
    df = match_labels(features_df, outer)
    df = df[df['label'] != 'Unknown']
    
    print(f"Labeled events: {len(df)}")
    if len(df) < 50:
        print("Not enough events.")
        return
        
    # Prepare X, y
    # X needs to be (N, 50, 1)
    X_list = np.stack(df['waveform'].values)
    # Ensure shape
    if X_list.ndim == 2:
        X_list = np.expand_dims(X_list, axis=-1)
        
    y_raw = df['label'].values
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_list, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    
    # Train
    model = create_cnn_model((X_train.shape[1], X_train.shape[2]), len(le.classes_))
    model.summary()
    
    print("Training CNN...")
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating...")
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save
    model.save("nilm_cnn_model.keras")
    
    # Save meta
    meta = {
        'classes': le.classes_.tolist(),
        'input_len': L
    }
    with open("cnn_metadata.json", "w") as f:
        json.dump(meta, f)
        
    print("Saved nilm_cnn_model.keras")

if __name__ == "__main__":
    main()
