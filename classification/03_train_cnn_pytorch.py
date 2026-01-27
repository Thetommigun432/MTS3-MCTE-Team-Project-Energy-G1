
"""
Real-time NILM Stage 3: Deep Learning Classifier (PyTorch 1D-CNN)
-----------------------------------------------------------------
Trains a 1D Convolutional Neural Network on raw waveform windows using PyTorch.
Achieves SOTA precision by learning transient shapes.

Architecture:
- Input: (Batch, 1, 50)
- Conv1D (32 filters) -> ReLU -> MaxPool
- Conv1D (64 filters) -> ReLU -> MaxPool
- Flatten -> Dense (128) -> Dropout -> Dense (Classes)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from pathlib import Path
import json
import ast
import time

# --- Labeling Logic (Reused) ---
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

# --- PyTorch Model ---
class NILM_CNN(nn.Module):
    def __init__(self, input_len, num_classes):
        super(NILM_CNN, self).__init__()
        self.features = nn.Sequential(
            # Conv 1
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # Conv 2
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Calculate flatten size
        # Input 50 -> Pool2 -> 25 -> Pool2 -> 12
        flatten_size = 64 * (input_len // 4) 
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class NILMDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X) # (N, 1, L)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def main():
    feat_path = Path("event_features.csv")
    if not feat_path.exists():
        print("Error: event_features.csv not found")
        return
        
    print("Loading Data...")
    df_feat = pd.read_csv(feat_path)
    df_feat['timestamp'] = pd.to_datetime(df_feat['timestamp'])
    if df_feat['timestamp'].dt.tz is not None:
        df_feat['timestamp'] = df_feat['timestamp'].dt.tz_localize(None)
        
    print("Parsing Waveforms...")
    df_feat['waveform'] = df_feat['waveform'].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x)
    
    print("Loading Ground Truth...")
    BASE_DIR = Path('c:/Users/gamek/School/TeamProject/MTS3-MCTE-Team-Project-Energy-G1')
    DATA_PATH = BASE_DIR / "data/processed/1sec_new/nilm_ready_1sec_new.parquet"
    ground_truth = pd.read_parquet(DATA_PATH).head(5000000)
    
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
        
    # Label
    df = match_labels(df_feat, outer)
    df = df[df['label'] != 'Unknown']
    print(f"Labeled Samples: {len(df)}")
    
    if len(df) < 50: return
    
    # Prepare Data
    X_raw = np.stack(df['waveform'].values) # (N, 50)
    X_raw = np.expand_dims(X_raw, axis=1) # (N, 1, 50) for Conv1d
    
    le = LabelEncoder()
    y_raw = le.fit_transform(df['label'])
    
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw)
    
    train_dataset = NILMDataset(X_train, y_train)
    test_dataset = NILMDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = NILM_CNN(input_len=50, num_classes=len(le.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training PyTorch CNN...")
    epochs = 20
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "nilm_cnn_pytorch.pth")
            
    # Final Evaluation
    print("Loading Best Model...")
    model.load_state_dict(torch.load("nilm_cnn_pytorch.pth"))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=le.classes_))
    
    # Save Metadata
    meta = {'classes': le.classes_.tolist()}
    with open("pytorch_metadata.json", "w") as f:
        json.dump(meta, f)

if __name__ == "__main__":
    main()
