import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import pickle
import os


class PreprocessingEngine:
    """
    Preprocessing engine for NILM (Non-Intrusive Load Monitoring) data.
    Transforms raw data into sequences ready for model inference.
    """
    
    # Default configuration
    DEFAULT_WINDOW_SIZE = 99  # Corresponds to model's SEQ_LEN
    DEFAULT_FEATURES = ['power', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']

    def __init__(self, scaler_path: Optional[str] = None, window_size: int = None):
        """
        Initialize the preprocessing engine.
        
        Args:
            scaler_path: Path to the pre-trained scaler pickle file
            window_size: Window size for sequences
        """
        self.window_size = window_size or self.DEFAULT_WINDOW_SIZE
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        if scaler_path and os.path.exists(scaler_path):
            self._load_scaler(scaler_path)

    def _load_scaler(self, path: str):
        """Load a pre-trained scaler from pickle file."""
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        self.is_fitted = True
        print(f"Scaler loaded from {path}")

    def save_scaler(self, path: str):
        """Save the current scaler to a pickle file."""
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to {path}")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw data into clean data.
        
        Steps:
            1. Timestamp handling
            2. Duplicate removal
            3. Missing value interpolation
            4. Normalization
            5. Temporal feature engineering
        
        Args:
            df: DataFrame with raw data
        
        Returns:
            Cleaned and normalized DataFrame
        """
        data = df.copy()
        
        # 1. Timestamp handling
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.sort_values(by='timestamp')
            data.set_index('timestamp', inplace=True)
        elif not isinstance(data.index, pd.DatetimeIndex):
            # Try to convert the index
            try:
                data.index = pd.to_datetime(data.index)
            except Exception:
                pass

        # 2. Cleanup: Remove duplicates
        data = data[~data.index.duplicated(keep='first')]

        # 3. Handle missing values
        data = data.ffill().bfill()

        # 4. Find power column (search for common variants)
        power_col = self._find_power_column(data)
        if power_col is None:
            raise ValueError("No 'power' column found in DataFrame")
        
        # 5. Temporal Feature Engineering
        data = self._add_temporal_features(data)

        # 6. Normalize power column
        power_values = data[[power_col]].values
        
        if not self.is_fitted:
            self.scaler.fit(power_values)
            self.is_fitted = True
        
        data['power_normalized'] = self.scaler.transform(power_values)

        return data

    def _find_power_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the column containing power data."""
        power_variants = ['power', 'Power', 'POWER', 'active_power', 'consumption', 
                          'energy_consumption', 'watt', 'watts', 'W']
        
        for col in power_variants:
            if col in df.columns:
                return col
        
        # If not found, take the first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return numeric_cols[0]
        
        return None

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclic temporal features (sin/cos encoding)."""
        if isinstance(df.index, pd.DatetimeIndex):
            # Hours of the day (24h cycle)
            hours = df.index.hour + df.index.minute / 60
            df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
            df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
            
            # Day of the week (7 day cycle)
            day_of_week = df.index.dayofweek
            df['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
            df['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)
            
            # Month of the year (12 month cycle)
            month = df.index.month
            df['month_sin'] = np.sin(2 * np.pi * month / 12)
            df['month_cos'] = np.cos(2 * np.pi * month / 12)
        
        return df

    def prepare_sequences(self, data: pd.DataFrame, 
                         feature_cols: list = None) -> np.ndarray:
        """
        Prepare sequences for the Seq2Point model.
        
        Args:
            data: Preprocessed DataFrame
            feature_cols: List of columns to use as features
        
        Returns:
            Numpy array of shape [N, window_size, n_features]
        """
        if feature_cols is None:
            # Use default features if available
            available_features = []
            for feat in ['power_normalized', 'hour_sin', 'hour_cos', 
                        'day_sin', 'day_cos', 'month_sin', 'month_cos']:
                if feat in data.columns:
                    available_features.append(feat)
            feature_cols = available_features if available_features else ['power_normalized']
        
        # Extract values
        values = data[feature_cols].values
        
        # Create sliding window sequences
        sequences = []
        for i in range(len(values) - self.window_size + 1):
            seq = values[i:i + self.window_size]
            sequences.append(seq)
        
        if len(sequences) == 0:
            raise ValueError(
                f"Insufficient data: {len(values)} points, "
                f"at least {self.window_size} required"
            )
        
        return np.array(sequences, dtype=np.float32)

    def inverse_transform_power(self, normalized_values: np.ndarray) -> np.ndarray:
        """
        Convert normalized values back to original scale.
        
        Args:
            normalized_values: Normalized values
        
        Returns:
            Values in original scale
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Run clean_data first.")
        
        values = normalized_values.reshape(-1, 1)
        return self.scaler.inverse_transform(values).flatten()

    def process_pipeline(self, raw_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Complete preprocessing pipeline.
        
        Args:
            raw_df: DataFrame with raw data
        
        Returns:
            Tuple of (sequences array, cleaned dataframe)
        """
        cleaned = self.clean_data(raw_df)
        sequences = self.prepare_sequences(cleaned)
        return sequences, cleaned