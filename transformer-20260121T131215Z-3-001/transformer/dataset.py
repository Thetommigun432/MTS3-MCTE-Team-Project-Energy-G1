"""
NILM Dataset and DataLoader
===========================
Sliding window dataset for NILM with temporal features.
Supports multi-appliance output and data augmentation.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import pickle


class NILMDataset(Dataset):
    """
    Sliding window dataset for NILM training.
    
    Features:
    - Aggregate power as main input
    - Temporal features (hour, day-of-week, month as sin/cos)
    - Multi-appliance targets
    - Optional data augmentation (Gaussian noise)
    """
    
    def __init__(
        self,
        aggregate: np.ndarray,
        targets: Dict[str, np.ndarray],
        temporal: np.ndarray,
        window_size: int,
        stride: int = 1,
        augment: bool = False,
        noise_std: float = 0.02,
        on_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            aggregate: Normalized aggregate power [n_samples]
            targets: Dict of normalized appliance powers {name: [n_samples]}
            temporal: Temporal features [n_samples, n_temporal_features]
            window_size: Input sequence length
            stride: Sliding window stride
            augment: Whether to apply data augmentation
            noise_std: Gaussian noise std for augmentation
            on_thresholds: Dict of ON thresholds per appliance (in normalized units)
        """
        self.aggregate = aggregate.astype(np.float32)
        self.targets = {k: v.astype(np.float32) for k, v in targets.items()}
        self.temporal = temporal.astype(np.float32)
        self.window_size = window_size
        self.stride = stride
        self.augment = augment
        self.noise_std = noise_std
        self.on_thresholds = on_thresholds or {}
        
        # Calculate number of sequences
        self.n_samples = (len(aggregate) - window_size) // stride + 1
        
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
        start = idx * self.stride
        end = start + self.window_size
        
        # Input: aggregate + temporal features
        agg = self.aggregate[start:end].reshape(-1, 1)
        temp = self.temporal[start:end]
        x = np.concatenate([agg, temp], axis=1)
        
        # Apply augmentation
        if self.augment:
            noise = np.random.randn(*x.shape).astype(np.float32) * self.noise_std
            x = x + noise
        
        # Targets: midpoint for seq2point
        mid = self.window_size // 2
        y = {}
        for name, values in self.targets.items():
            power = values[start + mid]
            # ON/OFF state based on threshold
            threshold = self.on_thresholds.get(name, 0.01)
            state = 1.0 if power > threshold else 0.0
            y[name] = {
                'power': np.array([power], dtype=np.float32),
                'state': np.array([state], dtype=np.float32)
            }
        
        return torch.from_numpy(x), {
            name: {k: torch.from_numpy(v) for k, v in vals.items()}
            for name, vals in y.items()
        }


class NILMSeq2SeqDataset(Dataset):
    """
    Dataset for Seq2Seq NILM (output full sequence).
    """
    
    def __init__(
        self,
        aggregate: np.ndarray,
        targets: Dict[str, np.ndarray],
        temporal: np.ndarray,
        window_size: int,
        stride: int = 1,
        augment: bool = False,
        noise_std: float = 0.02
    ):
        self.aggregate = aggregate.astype(np.float32)
        self.targets = {k: v.astype(np.float32) for k, v in targets.items()}
        self.temporal = temporal.astype(np.float32)
        self.window_size = window_size
        self.stride = stride
        self.augment = augment
        self.noise_std = noise_std
        
        self.n_samples = (len(aggregate) - window_size) // stride + 1
        
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        start = idx * self.stride
        end = start + self.window_size
        
        # Input
        agg = self.aggregate[start:end].reshape(-1, 1)
        temp = self.temporal[start:end]
        x = np.concatenate([agg, temp], axis=1)
        
        if self.augment:
            noise = np.random.randn(*x.shape).astype(np.float32) * self.noise_std
            x = x + noise
        
        # Targets: full sequence
        y = {name: values[start:end].reshape(-1, 1) for name, values in self.targets.items()}
        
        return torch.from_numpy(x), {name: torch.from_numpy(v) for name, v in y.items()}


def load_and_prepare_data(
    data_path: Path,
    appliances: List[str],
    temporal_cols: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[Dict, Dict, Dict, StandardScaler, Dict[str, StandardScaler]]:
    """
    Load data and prepare train/val/test splits.
    
    Args:
        data_path: Path to parquet/csv file
        appliances: List of target appliance column names
        temporal_cols: List of temporal feature column names
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        
    Returns:
        train_data, val_data, test_data dicts
        scaler_agg: StandardScaler for aggregate
        scalers_y: Dict of StandardScalers for each appliance
    """
    # Load data
    if data_path.suffix == '.parquet':
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df):,} samples from {data_path.name}")
    
    # Extract arrays
    aggregate = df['Aggregate'].values if 'Aggregate' in df.columns else df['aggregate'].values
    
    targets = {}
    for app in appliances:
        # Try different column name formats
        col = None
        for candidate in [app, app.lower(), app.replace('_', '')]:
            if candidate in df.columns:
                col = candidate
                break
        if col:
            targets[app] = df[col].values
        else:
            print(f"Warning: Appliance '{app}' not found in data")
    
    temporal = df[temporal_cols].values if all(c in df.columns for c in temporal_cols) else None
    
    # Create temporal features if not present
    if temporal is None:
        print("Creating temporal features from timestamp...")
        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'])
        else:
            ts = pd.to_datetime(df.index)
        
        # Ensure ts is a DatetimeIndex for attribute access
        if not isinstance(ts, pd.DatetimeIndex):
            ts = pd.DatetimeIndex(ts)
        
        temporal = np.column_stack([
            np.sin(2 * np.pi * ts.hour / 24),
            np.cos(2 * np.pi * ts.hour / 24),
            np.sin(2 * np.pi * ts.minute / 60),  # Added minute features
            np.cos(2 * np.pi * ts.minute / 60),
            np.sin(2 * np.pi * ts.dayofweek / 7),
            np.cos(2 * np.pi * ts.dayofweek / 7),
            np.sin(2 * np.pi * ts.month / 12),
            np.cos(2 * np.pi * ts.month / 12)
        ])
    
    # Normalize
    scaler_agg = StandardScaler()
    aggregate_arr = np.asarray(aggregate)  # Convert to numpy array if needed
    aggregate_scaled = scaler_agg.fit_transform(aggregate_arr.reshape(-1, 1)).flatten()
    
    scalers_y = {}
    targets_scaled = {}
    for name, values in targets.items():
        scalers_y[name] = StandardScaler()
        targets_scaled[name] = scalers_y[name].fit_transform(values.reshape(-1, 1)).flatten()
    
    # Time-based split (no data leakage)
    n = len(aggregate_scaled)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = {
        'aggregate': aggregate_scaled[:train_end],
        'targets': {k: v[:train_end] for k, v in targets_scaled.items()},
        'temporal': temporal[:train_end]
    }
    
    val_data = {
        'aggregate': aggregate_scaled[train_end:val_end],
        'targets': {k: v[train_end:val_end] for k, v in targets_scaled.items()},
        'temporal': temporal[train_end:val_end]
    }
    
    test_data = {
        'aggregate': aggregate_scaled[val_end:],
        'targets': {k: v[val_end:] for k, v in targets_scaled.items()},
        'temporal': temporal[val_end:]
    }
    
    print(f"Train: {len(train_data['aggregate']):,} | Val: {len(val_data['aggregate']):,} | Test: {len(test_data['aggregate']):,}")
    
    return train_data, val_data, test_data, scaler_agg, scalers_y


def create_dataloaders(
    config,
    train_data: Dict,
    val_data: Dict,
    test_data: Dict
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders from prepared data.
    
    Args:
        config: Config object
        train_data, val_data, test_data: Data dicts from load_and_prepare_data
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Compute normalized ON thresholds
    norm_thresholds = {}
    for app in config.appliances:
        if app in config.on_thresholds:
            # Approximate: threshold / typical_max_power (rough normalization)
            norm_thresholds[app] = 0.01  # Will be recalculated based on scaler
    
    # Create datasets
    train_ds = NILMDataset(
        aggregate=train_data['aggregate'],
        targets=train_data['targets'],
        temporal=train_data['temporal'],
        window_size=config.window_size,
        stride=config.stride,
        augment=config.augment,
        noise_std=config.noise_std,
        on_thresholds=norm_thresholds
    )
    
    val_ds = NILMDataset(
        aggregate=val_data['aggregate'],
        targets=val_data['targets'],
        temporal=val_data['temporal'],
        window_size=config.window_size,
        stride=config.window_size // 2,  # Less overlap for validation
        augment=False
    )
    
    test_ds = NILMDataset(
        aggregate=test_data['aggregate'],
        targets=test_data['targets'],
        temporal=test_data['temporal'],
        window_size=config.window_size,
        stride=config.stride_inference,
        augment=False
    )
    
    print(f"Train sequences: {len(train_ds):,}")
    print(f"Val sequences: {len(val_ds):,}")
    print(f"Test sequences: {len(test_ds):,}")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    return train_loader, val_loader, test_loader


def save_scalers(scaler_agg: StandardScaler, scalers_y: Dict[str, StandardScaler], path: Path):
    """Save scalers for inference."""
    with open(path / 'scalers.pkl', 'wb') as f:
        pickle.dump({'aggregate': scaler_agg, 'targets': scalers_y}, f)
    print(f"Scalers saved to {path / 'scalers.pkl'}")


def load_scalers(path: Path) -> Tuple[StandardScaler, Dict[str, StandardScaler]]:
    """Load scalers for inference."""
    with open(path / 'scalers.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['aggregate'], data['targets']


# =============================================================================
# PRETRAINED NUMPY DATASET (from pretraining notebooks)
# =============================================================================

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ... (Previous code)

def apply_robust_augmentations(x: np.ndarray, noise_std=0.02, mag_scale=0.0, mask_prob=0.0, mask_size=0):
    """
    Apply robust augmentations (V3): Scale, Mask, Noise.
    x: [window_size, features] (Feature 0 is Aggregate)
    """
    # 1. Magnitude Warping (Scale Aggregate)
    if mag_scale > 0 and np.random.random() < 0.5:
        scale = 1.0 + (np.random.random() * 2 - 1) * mag_scale
        # Assume feature 0 is aggregate, and if derivatives exist (1,2,3), scale them too
        # This function is called AFTER derivative computation in __getitem__
        x[:, 0] *= scale # Aggregate
        if x.shape[1] >= 4: # Assuming 1-3 are derivatives
            x[:, 1] *= scale # dP/dt
            x[:, 2] *= scale # Rolling Mean
            x[:, 3] *= scale # Rolling Std
            
    # 2. Time Masking (Zero out a random chunk of Aggregate)
    if mask_prob > 0 and mask_size > 0 and np.random.random() < mask_prob:
        start = np.random.randint(0, len(x) - mask_size)
        x[start:start+mask_size, 0] = 0  # Mask aggregate
        # We don't mask temporal features or derivatives usually, but let's be consistent
        # If we mask aggregate, derivatives should technically change, but masking them is a good approx
        if x.shape[1] >= 4:
            x[start:start+mask_size, 1:4] = 0
            
    # 3. Noise Injection
    if noise_std > 0:
        noise = np.random.randn(*x.shape).astype(np.float32) * noise_std
        x += noise
        
    return x


class PretrainedNILMDataset(Dataset):
    """
    Dataset for pre-windowed numpy arrays from pretraining notebooks.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        appliance_names: List[str],
        seq2point: bool = True,
        augment: bool = False,
        noise_std: float = 0.02,
        mag_scale: float = 0.0,
        mask_prob: float = 0.0,
        mask_size: int = 0,
        on_thresholds: Optional[Dict[str, float]] = None,
        add_derivative_features: bool = True,
        crop_window: Optional[int] = None
    ):
        # Avoid copying if already float32
        self.X = X if X.dtype == np.float32 else X.astype(np.float32)
        self.y = y if y.dtype == np.float32 else y.astype(np.float32)
        self.appliance_names = appliance_names
        self.seq2point = seq2point
        self.augment = augment
        
        self.on_thresholds = on_thresholds or {}
        self.add_derivative_features = add_derivative_features
        
        # Augmentation params (Restored)
        self.noise_std = noise_std
        self.mag_scale = mag_scale
        self.mask_prob = mask_prob
        self.mask_size = mask_size
        
        # Window cropping
        self.crop_window = crop_window
        self.n_samples, self.original_window_size, self.n_features = X.shape # Rename to original
        
        if self.crop_window and self.crop_window < self.original_window_size:
            self.crop_start = (self.original_window_size - self.crop_window) // 2
            self.window_size = self.crop_window
            print(f"Dataset: Auto-cropping inputs from {self.original_window_size} to {self.window_size}")
        else:
            self.crop_start = 0
            self.window_size = self.original_window_size
        self.n_appliances = y.shape[2]
        
        # If adding derivative features, n_features will increase by 3
        if self.add_derivative_features:
            self.output_n_features = self.n_features + 3
        else:
            self.output_n_features = self.n_features
            
        # Pre-calculate ON indices for Stratified Sampling (Primary appliance is index 0 usually)
        # We'll calculate 'is_on' for the first appliance in the list (usually the target one for single-appliance models)
        self.midpoint = self.original_window_size // 2 # Use ORIGINAL midpoint for correct target extraction
        
        # We assume single appliance or first appliance is the target for balancing
        target_idx = 0 
        target_name = self.appliance_names[target_idx]
        thresh = self.on_thresholds.get(target_name, 0.01)
        
        # Check ON states at midpoint
        if self.seq2point:
             self.is_on = self.y[:, self.midpoint, target_idx] > thresh
        else:
             self.is_on = np.mean(self.y[:, :, target_idx] > thresh, axis=1) > 0.1 # >10% ON
             
        # --- ROBUST FILTERING (V4) ---
        # Filter out samples where Target > Aggregate (Physics Violation)
        # Both are normalized by same P_MAX, so we can compare directly.
        # Allow small tolerance (e.g. 100W normalized = 100/13500 = 0.007)
        tolerance = 0.01 
        
        # Check midpoint validity for Seq2Point
        if self.seq2point:
            # Aggregate at midpoint (feature 0) vs Target at midpoint
            agg_mid = self.X[:, self.midpoint, 0]
            target_mid = self.y[:, self.midpoint, target_idx]
            
            # Valid if Target <= Agg + tolerance
            self.is_valid = target_mid <= (agg_mid + tolerance)
            
            # Also filter out extremely negative aggregates (sensor errors)
            self.is_valid &= (agg_mid > -0.01)
        else:
            # Valid if mean violation is low
            self.is_valid = np.ones(self.n_samples, dtype=bool)

        valid_count = np.sum(self.is_valid)
        total_count = self.n_samples
        print(f"Data Filtering: Kept {valid_count}/{total_count} ({valid_count/total_count:.1%}) samples based on physics (Target <= Agg)")
        
        # Combine ON/OFF with Validity
        self.on_indices = np.where(self.is_on & self.is_valid)[0]
        self.off_indices = np.where(~self.is_on & self.is_valid)[0]
        
    def __len__(self) -> int:
        return self.n_samples
    
    def _compute_derivative_features(self, agg: np.ndarray) -> np.ndarray:
        # Same as before
        n = len(agg)
        dP_dt = np.zeros(n, dtype=np.float32)
        dP_dt[1:] = agg[1:] - agg[:-1]
        
        window = 8
        rolling_mean = np.convolve(agg, np.ones(window)/window, mode='same').astype(np.float32)
        
        rolling_std = np.zeros(n, dtype=np.float32)
        for i in range(window, n):
            rolling_std[i] = np.std(agg[i-window:i])
        rolling_std[:window] = rolling_std[window] if window < n else 0
        
        return np.stack([dP_dt, rolling_mean, rolling_std], axis=1)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
        x = self.X[idx].copy()  # (original_window, n_features)
        
        # Apply cropping if needed
        if self.crop_window and self.crop_window < self.original_window_size:
            x = x[self.crop_start : self.crop_start + self.crop_window]
        
        # Add derivative features if enabled
        if self.add_derivative_features:
            agg = x[:, 0]  # First feature is aggregate power
            deriv_features = self._compute_derivative_features(agg)
            x = np.concatenate([x, deriv_features], axis=1)
        
        # Augmentation (V3)
        if self.augment:
            x = apply_robust_augmentations(
                x, 
                noise_std=self.noise_std,
                mag_scale=self.mag_scale,
                mask_prob=self.mask_prob,
                mask_size=self.mask_size
            )
        
        # Targets
        if self.seq2point:
            # Midpoint prediction (Original midpoint is valid for centered crop)
            # The crop is centered around original center, so original center index works
            mid = self.original_window_size // 2
            y_dict = {}
            for i, name in enumerate(self.appliance_names):
                power = self.y[idx, mid, i]
                threshold = self.on_thresholds.get(name, 0.01)
                state = 1.0 if power > threshold else 0.0
                y_dict[name] = {
                    'power': np.array([power], dtype=np.float32),
                    'state': np.array([state], dtype=np.float32)
                }
        else:
            # Full sequence (seq2seq)
            # MUST CROP Y TO MATCH X
            y_full = self.y[idx] # [OriginalWindow, Appliances]
            if self.crop_window and self.crop_window < self.original_window_size:
                y_full = y_full[self.crop_start : self.crop_start + self.crop_window]
            
            y_dict = {}
            for i, name in enumerate(self.appliance_names):
                y_app = y_full[:, i]
                y_dict[name] = {
                    'power': y_app.reshape(-1, 1),
                    'state': (y_app > self.on_thresholds.get(name, 0.01)).astype(np.float32).reshape(-1, 1)
                }
        
        return torch.from_numpy(x), {
            name: {k: torch.from_numpy(v) for k, v in vals.items()}
            for name, vals in y_dict.items()
        }

    def get_stratified_sampler(self, on_ratio=0.3):
        """Create a WeightedRandomSampler to balance ON/OFF samples."""
        n_on = len(self.on_indices)
        n_off = len(self.off_indices)
        if n_on == 0: 
            return None
        
        # Calculate weights to force on_ratio
        # w_on * n_on / (w_on * n_on + w_off * n_off) = on_ratio
        # Assuming w_off = 1.0
        w_on = on_ratio * n_off / ((1 - on_ratio) * n_on + 1e-10)
        
        weights = np.zeros(self.n_samples)
        weights[self.on_indices] = w_on
        weights[self.off_indices] = 1.0
        
        return WeightedRandomSampler(torch.DoubleTensor(weights), self.n_samples, replacement=True)


def load_pretrained_data(
    model_ready_path: Path
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Load pretrained numpy arrays and metadata.
    
    Args:
        model_ready_path: Path to model_ready folder with .npy files
        
    Returns:
        train_data, val_data, test_data, metadata
    """
    import json
    
    print(f"Loading pretrained data from {model_ready_path}")
    
    # Load metadata
    metadata_path = model_ready_path / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Load numpy arrays
    X_train = np.load(model_ready_path / 'X_train.npy')
    y_train = np.load(model_ready_path / 'y_train.npy')
    X_val = np.load(model_ready_path / 'X_val.npy')
    y_val = np.load(model_ready_path / 'y_val.npy')
    X_test = np.load(model_ready_path / 'X_test.npy')
    y_test = np.load(model_ready_path / 'y_test.npy')
    
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    
    train_data = {'X': X_train, 'y': y_train}
    val_data = {'X': X_val, 'y': y_val}
    test_data = {'X': X_test, 'y': y_test}
    
    # Get P_MAX from metadata (NILM-correct scaling)
    if 'scaling' in metadata and 'P_MAX' in metadata['scaling']:
        metadata['P_MAX'] = metadata['scaling']['P_MAX']
        print(f"  P_MAX: {metadata['P_MAX']:.4f} kW (from metadata)")
    else:
        # Fallback: load from P_MAX.pkl file
        pmax_path = model_ready_path / 'P_MAX.pkl'
        if pmax_path.exists():
            with open(pmax_path, 'rb') as f:
                metadata['P_MAX'] = pickle.load(f)
            print(f"  P_MAX: {metadata['P_MAX']:.4f} kW (from pkl)")
    
    # Load scaling_params.pkl if exists
    scaling_params_path = model_ready_path / 'scaling_params.pkl'
    if scaling_params_path.exists():
        with open(scaling_params_path, 'rb') as f:
            metadata['scaling_params'] = pickle.load(f)
    
    # Legacy: also check for old scaler.pkl (for backward compatibility)
    scaler_path = model_ready_path / 'scaler.pkl'
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            metadata['scaler'] = pickle.load(f)
    
    return train_data, val_data, test_data, metadata


def create_pretrained_dataloaders(
    config,
    train_data: Dict,
    val_data: Dict,
    test_data: Dict,
    appliance_names: List[str],
    P_MAX_kW: float = 13.5118,
    add_derivative_features: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders from pretrained numpy arrays.
    
    Args:
        config: Config object
        train_data, val_data, test_data: Dicts with 'X' and 'y' numpy arrays
        appliance_names: List of appliance names
        P_MAX_kW: Max aggregate power in kW (for threshold normalization)
        add_derivative_features: Add dP/dt, rolling_mean, rolling_std
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Normalized ON thresholds: threshold_W / (P_MAX_kW * 1000)
    # ON thresholds are in Watts, P_MAX is in kW
    norm_thresholds = {}
    for name in appliance_names:
        threshold_W = config.on_thresholds.get(name, 50)  # Default 50W
        norm_thresholds[name] = threshold_W / (P_MAX_kW * 1000)  # Convert to scaled units
    
    train_ds = PretrainedNILMDataset(
        X=train_data['X'],
        y=train_data['y'],
        appliance_names=appliance_names,
        seq2point=config.seq2point,
        augment=config.augment,
        noise_std=config.noise_std,
        mag_scale=getattr(config, 'mag_scale', 0.0),
        mask_prob=getattr(config, 'mask_prob', 0.0),
        mask_size=getattr(config, 'mask_size', 0),
        on_thresholds=norm_thresholds,
        add_derivative_features=add_derivative_features,
        crop_window=config.window_size # Pass config window size for cropping
    )
    
    val_ds = PretrainedNILMDataset(
        X=val_data['X'],
        y=val_data['y'],
        appliance_names=appliance_names,
        seq2point=config.seq2point,
        augment=False,
        on_thresholds=norm_thresholds,
        add_derivative_features=add_derivative_features,
        crop_window=config.window_size
    )
    
    test_ds = PretrainedNILMDataset(
        X=test_data['X'],
        y=test_data['y'],
        appliance_names=appliance_names,
        seq2point=config.seq2point,
        augment=False,
        on_thresholds=norm_thresholds,
        add_derivative_features=add_derivative_features,
        crop_window=config.window_size
    )
    
    # Report actual feature count
    actual_features = train_ds.output_n_features
    print(f"Input features: {actual_features} (base={train_ds.n_features}, +derivatives={add_derivative_features})")
    print(f"Train sequences: {len(train_ds):,}")
    print(f"Val sequences:   {len(val_ds):,}")
    print(f"Test sequences:  {len(test_ds):,}")
    
    # Stratified Sampling (Improved V3)
    # If using robust training (augment=True), we also use stratified sampling to fix recall
    sampler = None
    if config.augment:
        print("Using Stratified Sampling (Target 30% ON ratio)")
        sampler = train_ds.get_stratified_sampler(on_ratio=0.3)
        if sampler is None:
            print("Warning: Could not create stratified sampler (no ON events?)")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True if config.device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    return train_loader, val_loader, test_loader
