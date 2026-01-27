
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json

def analyze_errors():
    # Load data
    df = pd.read_csv("labeled_events.csv")
    
    # Feature Engineering (re-create training set)
    feature_cols = ['delta_power', 'ss_mean', 'ss_std', 'ss_range', 'hour', 'day_of_week']
    X = df[feature_cols]
    y = df['label']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Load Model
    model = xgb.XGBClassifier()
    model.load_model("xgb_nilm_model.json")
    
    # Convert labels
    le = LabelEncoder()
    # Fit on ALL labels to ensure mapping is correct
    le.fit(y) 
    y_test_enc = le.transform(y_test)
    
    # Predict
    y_pred_enc = model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=le.classes_)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
    plt.title("Confusion Matrix (Event Classification)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("classification_confusion_matrix.png")
    print("Saved classification_confusion_matrix.png")
    
    # 2. Analyze "Dishwasher" Confusion
    # Find what Dishwasher (True) was predicted as
    dw_mask = (y_test == "Dishwasher")
    dw_pred = y_pred[dw_mask]
    
    print("\nDishwasher Predictions Distribution:")
    print(pd.Series(dw_pred).value_counts())
    
    # 3. Analyze Feature Overlap
    # Plot 'delta_power' density for Dishwasher vs its main confusers
    confusers = pd.Series(dw_pred).value_counts().index[:3] # Top 3
    
    plt.figure(figsize=(10, 6))
    for app in confusers:
        subset = df[df['label'] == app]
        sns.kdeplot(subset['delta_power'], label=f"{app} (n={len(subset)})", fill=True, alpha=0.3)
    
    plt.title("Delta Power Distribution of Dishwasher vs Confusers")
    plt.xlabel("Delta Power (kW)")
    plt.xlim(0, 3.0) # Focus on heating range
    plt.legend()
    plt.tight_layout()
    plt.savefig("feature_overlap_delta.png")
    print("Saved feature_overlap_delta.png")

if __name__ == "__main__":
    analyze_errors()
