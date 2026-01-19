import torch
import torch.nn as nn
import math
from backend.ai.model.model_architecture.improved_positional_encoding import ImprovedPositionalEncoding

class MultiTaskNILMTransformer(nn.Module):
    """
    Multi-task Transformer for Non-Intrusive Load Monitoring (NILM).
    
    Architecture:
        1. Conv1D embedding for local feature extraction
        2. Positional encoding for temporal awareness
        3. Transformer encoder for global context
        4. Dual output heads: regression (power) + classification (ON/OFF)
    
    The classification head acts as a "gate" during inference to set 
    power to zero when the appliance is predicted to be OFF.
    """
    def __init__(self, input_dim, d_model, n_heads, n_layers, seq_len, ff_dim, dropout=0.1, gate_threshold=0.5):
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
        # Regression head (Power estimation)
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Classification head (ON/OFF state)
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
            Training: (raw_power, on_off_prob) for separate loss computation
            Inference: (gated_power, on_off_prob) with automatic zero-gating
        """
        # A. Feature Extraction & Transformer
        x = x.permute(0, 2, 1)  # [B, F, S] for Conv1d
        x = self.conv_embedding(x)
        x = x.permute(0, 2, 1)  # [B, S, D] for Transformer
        
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)
        
        # B. Seq2Point: Focus on midpoint token
        mid_token = x[:, x.shape[1] // 2, :]
        
        # C. Compute outputs
        raw_power = self.regression_head(mid_token)
        on_off_prob = self.classification_head(mid_token)
        
        # D. Apply gating during inference
        if not self.training:
            is_on = (on_off_prob > self.gate_threshold).float()
            final_power = raw_power * is_on  # Zero power when OFF
            return final_power, on_off_prob
        
        return raw_power, on_off_prob