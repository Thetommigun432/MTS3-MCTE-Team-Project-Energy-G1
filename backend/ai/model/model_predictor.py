import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import os


class ImprovedPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequence modeling."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]


class MultiTaskNILMTransformer(nn.Module):
    """
    Multi-task Transformer for Non-Intrusive Load Monitoring (NILM).
    
    Architecture:
        1. Conv1D embedding for local feature extraction
        2. Positional encoding for temporal awareness
        3. Transformer encoder for global context
        4. Dual output heads: regression (power) + classification (ON/OFF)
    """
    def __init__(self, input_dim=7, d_model=256, n_heads=32, n_layers=8, 
                 seq_len=99, ff_dim=512, dropout=0.25, gate_threshold=0.5):
        super().__init__()
        self.gate_threshold = gate_threshold
        
        # 1. FEATURE EXTRACTION (Conv1d for local patterns)
        self.conv_embedding = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. POSITIONAL ENCODING
        self.pos_encoding = ImprovedPositionalEncoding(d_model, max_len=seq_len)
        
        # 3. TRANSFORMER ENCODER
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim, 
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 4. OUTPUT HEADS
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor [Batch, Seq_Len, Features]
        Returns:
            Training: (raw_power, on_off_prob)
            Inference: (gated_power, on_off_prob)
        """
        x = x.permute(0, 2, 1)  # [B, F, S] for Conv1d
        x = self.conv_embedding(x)
        x = x.permute(0, 2, 1)  # [B, S, D] for Transformer
        
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)
        
        # Seq2Point: Focus on midpoint token
        mid_token = x[:, x.shape[1] // 2, :]
        
        raw_power = self.regression_head(mid_token)
        on_off_prob = self.classification_head(mid_token)
        
        if not self.training:
            is_on = (on_off_prob > self.gate_threshold).float()
            final_power = raw_power * is_on
            return final_power, on_off_prob
        
        return raw_power, on_off_prob


class ModelPredictor:
    """
    Class to load and use the Transformer model for NILM predictions.
    """
    
    # Default model configuration (must match training configuration)
    DEFAULT_CONFIG = {
        'input_dim': 7,
        'd_model': 256,
        'n_heads': 32,
        'n_layers': 8,
        'seq_len': 99,
        'ff_dim': 512,
        'dropout': 0.25,
        'gate_threshold': 0.5
    }

    def __init__(self, model_path: str, config: dict = None, device: str = None):
        """
        Initialize the predictor by loading the model.
        
        Args:
            model_path: Path to the .pth model file
            config: Model configuration (uses DEFAULT_CONFIG if None)
            device: Execution device ('cuda', 'cpu', or None for auto-detect)
        """
        self.config = config or self.DEFAULT_CONFIG
        self.device = torch.device(device) if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model = self._load_model(model_path)
        print(f"ModelPredictor initialized on {self.device}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load the model from .pth file."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Create the model architecture
        model = MultiTaskNILMTransformer(**self.config)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume checkpoint is directly the state_dict
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        print(f"Model loaded successfully from {model_path}")
        return model

    def get_prediction(self, input_data: np.ndarray) -> pd.DataFrame:
        """
        Run predictions using the loaded model.
        
        Args:
            input_data: Numpy array of shape [N, seq_len, features] or [N, seq_len]
        
        Returns:
            DataFrame with predictions (power and on_off_probability)
        """
        # Convert to tensor
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
        else:
            input_tensor = input_data
        
        # Add batch dimension if needed
        if input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Add features dimension if needed (for 1D data)
        if input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(-1)
        
        input_tensor = input_tensor.to(self.device)
        
        # Run prediction
        with torch.no_grad():
            power_pred, on_off_prob = self.model(input_tensor)
        
        # Convert to numpy
        power_np = power_pred.cpu().numpy().flatten()
        prob_np = on_off_prob.cpu().numpy().flatten()
        
        # Create result DataFrame
        result = pd.DataFrame({
            'predicted_power': power_np,
            'on_off_probability': prob_np,
            'is_on': (prob_np > self.config['gate_threshold']).astype(int)
        })
        
        return result

    def predict_batch(self, sequences: np.ndarray, batch_size: int = 64) -> pd.DataFrame:
        """
        Batch prediction for large amounts of data.
        
        Args:
            sequences: Array of sequences [N, seq_len, features]
            batch_size: Batch size
        
        Returns:
            DataFrame with all predictions
        """
        all_results = []
        n_samples = len(sequences)
        
        for i in range(0, n_samples, batch_size):
            batch = sequences[i:i + batch_size]
            batch_result = self.get_prediction(batch)
            all_results.append(batch_result)
        
        return pd.concat(all_results, ignore_index=True)