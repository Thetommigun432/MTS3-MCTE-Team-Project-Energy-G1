"""
NILMFormer SGN - Subtask Gated Network Architecture
=====================================================
Based on literature research:
1. Classification × Regression gating to eliminate hallucinations
2. Focal Loss for sparse class imbalance
3. Smaller model for GPU memory efficiency

Key improvements over base NILMFormer:
- Gated output: y = regression * sigmoid(classification)
- Focal Loss: down-weights easy negatives (OFF states)
- Reduced model: d_model=64, n_layers=2 for memory efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field


# =============================================================================
# FOCAL LOSS (for sparse classification)
# =============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for handling extreme class imbalance.
    
    FL(p) = -alpha * (1-p)^gamma * log(p)  for positive class
    FL(p) = -(1-alpha) * p^gamma * log(1-p)  for negative class
    
    gamma=2.0 reduces loss for well-classified examples (OFF states)
    alpha=0.75 gives more weight to rare ON states
    """
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_logits: (B, L, 1) raw logits (pre-sigmoid)
            target: (B, L, 1) binary ON/OFF labels
        """
        pred_prob = torch.sigmoid(pred_logits)
        
        # Compute focal weight
        pt = torch.where(target > 0.5, pred_prob, 1 - pred_prob)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weighting (higher for positive class)
        alpha_t = torch.where(target > 0.5, self.alpha, 1 - self.alpha)
        
        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        
        # Apply focal weight and alpha
        loss = alpha_t * focal_weight * bce
        
        return loss.mean()


# =============================================================================
# SGN LOSS (Combined Regression + Gated Classification)
# =============================================================================
class SGNLoss(nn.Module):
    """
    Subtask Gated Network Loss.
    
    Components:
    1. Huber Loss for regression (robust to outliers)
    2. Focal Loss for classification (handles class imbalance)
    3. BCE weight >> Regression weight (prevent hallucinations)
    
    Architecture: output = regression * gate(classification)
    """
    
    def __init__(
        self,
        lambda_reg: float = 1.0,
        lambda_cls: float = 10.0,  # 10x weight on classification
        huber_delta: float = 0.05,  # ~50W in normalized space
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        on_thresholds: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_cls = lambda_cls
        self.huber_delta = huber_delta
        
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        
        # Default ON thresholds (normalized by P_MAX = 13.5 kW)
        self.on_thresholds = on_thresholds or {
            'HeatPump': 100 / 13511.8,
            'Dishwasher': 30 / 13511.8,
            'WashingMachine': 50 / 13511.8,
            'Dryer': 50 / 13511.8,
            'Oven': 100 / 13511.8,
            'Stove': 50 / 13511.8,
            'RangeHood': 20 / 13511.8,
            'EVCharger': 100 / 13511.8,
            'EVSocket': 100 / 13511.8,
            'GarageCabinet': 25 / 13511.8,
            'RainwaterPump': 50 / 13511.8,
        }
    
    def forward(
        self,
        predictions: Dict[str, Dict[str, torch.Tensor]],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: {app: {'power': (B,L,1), 'state_logits': (B,L,1), 'gated': (B,L,1)}}
            targets: {app: (B, L, 1) power values}
        """
        losses = {}
        total_reg = 0.0
        total_cls = 0.0
        n_apps = 0
        
        for app_name in predictions.keys():
            if app_name not in targets:
                continue
            
            pred = predictions[app_name]
            target = targets[app_name]
            
            # Get gated output and classification logits
            gated_power = pred['gated']  # (B, L, 1)
            state_logits = pred['state_logits']  # (B, L, 1)
            
            # Create binary ON/OFF labels from target
            threshold = self.on_thresholds.get(app_name, 0.001)
            target_state = (target.abs() > threshold).float()  # (B, L, 1)
            
            # === Regression Loss (Huber on gated output) ===
            reg_loss = F.huber_loss(gated_power, target, delta=self.huber_delta)
            
            # === Classification Loss (Focal) ===
            cls_loss = self.focal(state_logits, target_state)
            
            losses[f'{app_name}_reg'] = reg_loss
            losses[f'{app_name}_cls'] = cls_loss
            
            total_reg += reg_loss
            total_cls += cls_loss
            n_apps += 1
        
        # Combine with weighting (BCE >> Reg)
        avg_reg = total_reg / max(n_apps, 1)
        avg_cls = total_cls / max(n_apps, 1)
        
        losses['reg_total'] = avg_reg
        losses['cls_total'] = avg_cls
        losses['total'] = self.lambda_reg * avg_reg + self.lambda_cls * avg_cls
        
        return losses


# =============================================================================
# SIMPLE BUILDING BLOCKS
# =============================================================================
class ConvBlock(nn.Module):
    """Simple Conv1D block with residual."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding='same', dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x))) + self.skip(x)


class TransformerLayer(nn.Module):
    """Lightweight transformer layer."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# SGN MODEL: Subtask Gated Network
# =============================================================================
class NILMFormerSGN(nn.Module):
    """
    Subtask Gated Network for NILM.
    
    Architecture:
    - Shared CNN backbone (dilated convolutions)
    - Shared Transformer encoder
    - Per-appliance heads:
      - Regression head: predicts power
      - Classification head: predicts ON/OFF probability
      - Gated output: power * sigmoid(classification)
    
    Smaller than base NILMFormer:
    - d_model=64 (was 96)
    - n_layers=2 (was 3)
    - Lighter FFN (2x instead of 4x)
    """
    
    def __init__(
        self,
        appliances: List[str],
        c_embedding: int = 6,  # temporal features
        d_model: int = 64,     # REDUCED from 96
        n_layers: int = 2,     # REDUCED from 3
        n_heads: int = 4,      # REDUCED from 8
        dropout: float = 0.1,
        dilations: List[int] = [1, 2, 4, 8],
    ):
        super().__init__()
        
        self.appliances = appliances
        self.d_model = d_model
        
        # === Shared CNN Backbone ===
        self.embed_conv = nn.Sequential(
            ConvBlock(1, d_model // 2, kernel_size=7, dilation=1),
            ConvBlock(d_model // 2, d_model // 2, kernel_size=5, dilation=2),
            ConvBlock(d_model // 2, d_model // 2, kernel_size=3, dilation=4),
        )
        
        # Temporal feature projection
        self.temporal_proj = nn.Conv1d(c_embedding, d_model // 2, kernel_size=1)
        
        # === Shared Transformer ===
        self.transformer = nn.Sequential(*[
            TransformerLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        
        # === Per-Appliance SGN Heads ===
        self.regression_heads = nn.ModuleDict()
        self.classification_heads = nn.ModuleDict()
        
        for app in appliances:
            # Regression: predicts power value
            self.regression_heads[app] = nn.Sequential(
                nn.Conv1d(d_model, d_model // 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(d_model // 2, 1, kernel_size=1),
            )
            
            # Classification: predicts ON/OFF logits
            self.classification_heads[app] = nn.Sequential(
                nn.Conv1d(d_model, d_model // 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(d_model // 2, 1, kernel_size=1),
            )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # CRITICAL: Initialize classification head bias to +2.0
        # This makes sigmoid(2) ≈ 0.88, so gate starts OPEN
        # Allows regression to learn first, then classification refines
        for app in self.appliances:
            # Get the last conv layer of classification head
            cls_head = self.classification_heads[app]
            last_conv = cls_head[-1]  # nn.Conv1d(d_model // 2, 1, 1)
            if hasattr(last_conv, 'bias') and last_conv.bias is not None:
                nn.init.constant_(last_conv.bias, 2.0)  # sigmoid(2) ≈ 0.88
    
    def forward(self, x: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Args:
            x: (B, L, C) where C = 1 + c_embedding (aggregate + temporal)
        
        Returns:
            Dict[app_name, {'power': (B,L,1), 'state_logits': (B,L,1), 'gated': (B,L,1)}]
        """
        # (B, L, C) -> (B, C, L) for Conv1d
        x = x.permute(0, 2, 1)
        
        # Split aggregate and temporal
        aggregate = x[:, :1, :]  # (B, 1, L)
        temporal = x[:, 1:, :]   # (B, c_emb, L)
        
        # Instance normalization on aggregate
        mean = aggregate.mean(dim=-1, keepdim=True)
        std = (aggregate.var(dim=-1, keepdim=True) + 1e-6).sqrt()
        aggregate_norm = (aggregate - mean) / std
        
        # CNN embedding
        cnn_out = self.embed_conv(aggregate_norm)  # (B, d_model//2, L)
        temporal_out = self.temporal_proj(temporal)  # (B, d_model//2, L)
        
        # Concatenate
        x = torch.cat([cnn_out, temporal_out], dim=1)  # (B, d_model, L)
        
        # Transformer: (B, d_model, L) -> (B, L, d_model)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = self.final_norm(x)
        
        # Back to (B, d_model, L) for conv heads
        x = x.permute(0, 2, 1)
        
        # Per-appliance outputs
        outputs = {}
        for app in self.appliances:
            # Regression: raw power prediction
            power = self.regression_heads[app](x)  # (B, 1, L)
            power = F.relu(power)  # Non-negative power
            
            # Classification: ON/OFF logits
            state_logits = self.classification_heads[app](x)  # (B, 1, L)
            
            # Gated output: power * sigmoid(classification)
            # This is the "killer delle allucinazioni"
            gate = torch.sigmoid(state_logits)
            gated = power * gate
            
            # Reshape to (B, L, 1)
            outputs[app] = {
                'power': power.permute(0, 2, 1),
                'state_logits': state_logits.permute(0, 2, 1),
                'gated': gated.permute(0, 2, 1),
            }
        
        return outputs
    
    def get_gated_predictions(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convenience method: returns only gated outputs for inference."""
        outputs = self.forward(x)
        return {app: out['gated'] for app, out in outputs.items()}


# =============================================================================
# FACTORY
# =============================================================================
def create_nilmformer_sgn(
    appliances: List[str],
    c_embedding: int = 6,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 4,
    dropout: float = 0.1,
) -> NILMFormerSGN:
    """
    Create smaller SGN model optimized for GPU memory.
    
    ~200K parameters (was ~500K+)
    """
    return NILMFormerSGN(
        appliances=appliances,
        c_embedding=c_embedding,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
    )


if __name__ == "__main__":
    # Quick test
    print("Testing NILMFormer SGN...")
    
    appliances = ['HeatPump', 'Dishwasher', 'WashingMachine']
    model = create_nilmformer_sgn(appliances)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Test forward
    x = torch.randn(4, 1024, 7)  # (B, L, C)
    outputs = model(x)
    
    for app, out in outputs.items():
        print(f"{app}:")
        print(f"  power: {out['power'].shape}")
        print(f"  state_logits: {out['state_logits'].shape}")
        print(f"  gated: {out['gated'].shape}")
    
    # Test loss
    targets = {app: torch.rand(4, 1024, 1) * 0.1 for app in appliances}
    loss_fn = SGNLoss()
    losses = loss_fn(outputs, targets)
    print(f"\nLoss: {losses['total'].item():.4f}")
    print(f"  Reg: {losses['reg_total'].item():.4f}")
    print(f"  Cls: {losses['cls_total'].item():.4f}")
    
    print("\n✅ SGN model ready!")
