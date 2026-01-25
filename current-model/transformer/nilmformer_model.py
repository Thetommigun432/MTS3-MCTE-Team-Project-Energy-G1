"""
NILMFormer Model Architecture (KDD 2025)
========================================
Adapted from: https://github.com/adrienpetralia/NILMFormer

Key Features:
- Instance Normalization (Global Stationarization)
- Dilated Convolutions (DilatedBlock)
- TokenStats (Encoding statistical properties)
- ProjStats (De-stationarization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Any, Optional

# =============================================================================
# LAYERS: Embedding & CNN
# =============================================================================

class ResUnit(nn.Module):
    """Residual Unit with Dilated Convolution."""
    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int = 8,
        dilation: int = 1,
        stride: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=c_in,
                out_channels=c_out,
                kernel_size=k,
                dilation=dilation,
                stride=stride,
                bias=bias,
                padding="same",
            ),
            nn.GELU(),
            nn.BatchNorm1d(c_out),
        )
        if c_in > 1 and c_in != c_out:
            self.match_residual = True
            self.conv = nn.Conv1d(in_channels=c_in, out_channels=c_out, kernel_size=1)
        else:
            self.match_residual = False

    def forward(self, x) -> torch.Tensor:
        if self.match_residual:
            x_bottleneck = self.conv(x)
            x = self.layers(x)
            return torch.add(x_bottleneck, x)
        else:
            return torch.add(x, self.layers(x))


class DilatedBlock(nn.Module):
    """
    Dilated Convolutional Block.
    Stack of ResUnits with increasing dilation factors.
    """
    def __init__(
        self,
        c_in: int = 1,
        c_out: int = 32,
        kernel_size: int = 3,
        dilation_list: list = [1, 2, 4, 8],
        bias: bool = True,
    ):
        super().__init__()

        layers = []
        for i, dilation in enumerate(dilation_list):
            if i == 0:
                layers.append(
                    ResUnit(c_in, c_out, k=kernel_size, dilation=dilation, bias=bias)
                )
            else:
                layers.append(
                    ResUnit(c_out, c_out, k=kernel_size, dilation=dilation, bias=bias)
                )
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        return self.network(x)


# =============================================================================
# LAYERS: Transformer
# =============================================================================

class DiagonalMaskFromSeqlen:
    def __init__(self, B, L, device="cpu"):
        with torch.no_grad():
            self._mask = torch.diag(
                torch.ones(L, dtype=torch.bool, device=device)
            ).repeat(B, 1, 1, 1)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask


class DiagonnalyMaskedSelfAttention(nn.Module):
    """
    Masked Self Attention to prevent leakage if needed, 
    or just standard SA with specific dropout/scaling.
    """
    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        dropout: float,
        use_efficient_attention: bool = False,
    ):
        super().__init__()
        self.use_efficient_attention = use_efficient_attention
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.scale = head_dim**-0.5

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * head_dim, bias=False)

        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(batch, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(batch, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(batch, seqlen, self.n_heads, self.head_dim)

        # Diagonal mask usage (optional in standard transformers, but used in NILMFormer)
        # Here we implement standard attention for simplicity unless strict diagonal masking is required
        # NILMFormer uses diagonal masking for specific sequence handling
        diag_mask = DiagonalMaskFromSeqlen(batch, seqlen, device=xq.device)

        # Standard attention implementation
        scale = 1.0 / xq.shape[-1] ** 0.5
        # (B, L, H, E) -> (B, H, L, E)
        xq = xq.permute(0, 2, 1, 3)
        xk = xk.permute(0, 2, 1, 3)
        xv = xv.permute(0, 2, 1, 3)

        scores = torch.matmul(xq, xk.transpose(-2, -1)) # (B, H, L, L)
        
        # Apply mask
        scores = scores.masked_fill(diag_mask.mask, -float('inf'))
        
        attn = self.attn_dropout(torch.softmax(scores * scale, dim=-1))
        output = torch.matmul(attn, xv) # (B, H, L, E)
        
        output = output.permute(0, 2, 1, 3).contiguous().reshape(batch, seqlen, -1)
        return self.out_dropout(self.wo(output))


class PositionWiseFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dp_rate: float = 0.0,
        activation: Any = F.gelu,
        bias1: bool = True,
        bias2: bool = True,
    ):
        super().__init__()
        self.layer1 = nn.Linear(dim, hidden_dim, bias=bias1)
        self.layer2 = nn.Linear(hidden_dim, dim, bias=bias2)
        self.dropout = nn.Dropout(dp_rate)
        self.activation = activation

    def forward(self, x) -> torch.Tensor:
        x = self.layer2(self.dropout(self.activation(self.layer1(x))))
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dp_rate, pffn_ratio=4, norm_eps=1e-5):
        super().__init__()
        
        assert d_model % n_head == 0, f"d_model {d_model} not divisible by n_head {n_head}"

        self.attention_layer = DiagonnalyMaskedSelfAttention(
            dim=d_model,
            n_heads=n_head,
            head_dim=d_model // n_head,
            dropout=dp_rate,
        )

        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)

        self.dropout = nn.Dropout(dp_rate)

        self.pffn = PositionWiseFeedForward(
            dim=d_model,
            hidden_dim=d_model * pffn_ratio,
            dp_rate=dp_rate,
        )

    def forward(self, x) -> torch.Tensor:
        # Attention Block
        x_norm = self.norm1(x)
        new_x = self.attention_layer(x_norm)
        x = torch.add(x, new_x)

        # PFFN Block
        x_norm = self.norm2(x)
        new_x = self.pffn(x_norm)
        x = torch.add(x, self.dropout(new_x))

        return x


# =============================================================================
# MAIN MODEL: NILMFormer
# =============================================================================

class NILMFormer(nn.Module):
    """
    NILMFormer architecture for Single-Appliance NILM.
    """
    def __init__(
        self,
        c_in: int = 1,              # Input channels (load curve)
        c_embedding: int = 8,       # Exogenous channels (if any, typically 0 or small)
        c_out: int = 1,             # Output channels (1 for power)
        kernel_size: int = 3,
        kernel_size_head: int = 3,
        dilations: List[int] = [1, 2, 4, 8],
        conv_bias: bool = True,
        n_encoder_layers: int = 3,
        d_model: int = 96,
        n_head: int = 8,
        dropout: float = 0.1,
        pffn_ratio: int = 4
    ):
        super().__init__()

        self.c_in = c_in
        self.d_model = d_model

        # ============ Embedding ============#
        # If no exogenous variables, d_model_ is typically d_model
        # But NILMFormer splits space for exogenous. 
        # Here we adapt: if c_embedding > 0 (exogenous exists), we split.
        # Otherwise we use full d_model for main embedding.
        
        if c_embedding > 0:
            d_model_main = 3 * d_model // 4
            d_model_exo = d_model // 4
        else:
            d_model_main = d_model
            d_model_exo = 0

        self.EmbedBlock = DilatedBlock(
            c_in=c_in,
            c_out=d_model_main,
            kernel_size=kernel_size,
            dilation_list=dilations,
            bias=conv_bias,
        )

        if c_embedding > 0:
            self.ProjEmbedding = nn.Conv1d(
                in_channels=c_embedding, out_channels=d_model_exo, kernel_size=1
            )
        else:
            self.ProjEmbedding = None

        # TokenStats: Projections for Mean and Std
        self.ProjStats1 = nn.Linear(2, d_model)
        self.ProjStats2 = nn.Linear(d_model, 2)

        # ============ Encoder ============#
        layers = []
        for _ in range(n_encoder_layers):
            layers.append(EncoderLayer(
                d_model=d_model,
                n_head=n_head,
                dp_rate=dropout,
                pffn_ratio=pffn_ratio
            ))
        layers.append(nn.LayerNorm(d_model))
        self.EncoderBlock = nn.Sequential(*layers)

        # ============ Downstream Task Head ============#
        # Classification Head (State) - Linear for midpoint features
        self.ClassificationHead = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Regression Head - Linear for midpoint features  
        self.DownstreamTaskHead = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, c_out)
        )

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize nn.Linear and nn.LayerNorm weights."""
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x) -> dict:
        """
        Forward pass for NILMFormer.
        Input shape: (B, L, C) where C = 1 (load) + exogenous
        We expect input in standard format (Batch, Seq, Feat)
        """
        # Transpose to (B, C, L) for CNNs if needed, or handle splitting
        # x: [Batch, Seq, Features]
        
        # Split channels
        # x[:, :, :1] => load curve
        # x[:, :, 1:] => exogenous input(s)
        
        load = x[:, :, :1].permute(0, 2, 1) # (B, 1, L)
        
        if x.shape[2] > 1:
            encoding = x[:, :, 1:].permute(0, 2, 1) # (B, e, L)
        else:
            encoding = None

        # === Instance Normalization (Stationarization) === #
        inst_mean = torch.mean(load, dim=-1, keepdim=True).detach() # (B, 1, 1)
        inst_std = torch.sqrt(
            torch.var(load, dim=-1, keepdim=True, unbiased=False) + 1e-6
        ).detach() # (B, 1, 1)

        load_norm = (load - inst_mean) / inst_std  # shape (B, 1, L)

        # === Embedding === #
        # 1) Dilated Conv block on normalized load
        x_emb = self.EmbedBlock(load_norm)  # (B, d_model_main, L)

        # 2) Project exogenous features if they exist
        if encoding is not None and self.ProjEmbedding is not None:
            enc_emb = self.ProjEmbedding(encoding)  # (B, d_model_exo, L)
            x_emb = torch.cat([x_emb, enc_emb], dim=1) # (B, d_model, L)
        
        # Prepare for Transformer: (B, L, d_model)
        x_emb = x_emb.permute(0, 2, 1)

        # === TokenStats === #
        # Concatenate Mean and Std and project to d_model
        # inst_mean/std are (B, 1, 1), need to be compatible
        stats = torch.cat([inst_mean, inst_std], dim=1).permute(0, 2, 1) # (B, 1, 2)
        stats_token = self.ProjStats1(stats)  # (B, 1, d_model)
        
        # Append stats token to sequence
        x_tf = torch.cat([x_emb, stats_token], dim=1)  # (B, L + 1, d_model)

        # === Transformer Encoder === #
        x_tf = self.EncoderBlock(x_tf)  # (B, L + 1, d_model)
        
        # Remove stats token
        x_tf = x_tf[:, :-1, :]  # (B, L, d_model)

        # === Heads (Seq2Point: Extract Midpoint) === #
        # Get midpoint position
        L = x_tf.shape[1]
        mid = L // 2
        
        # Extract midpoint features only
        midpoint_feat = x_tf[:, mid, :]  # (B, d_model)
        
        # === Dual-Head Output === #
        # 1. Classification Head (ON/OFF probability)
        state_logits = self.ClassificationHead(midpoint_feat)  # (B, 1)
        state_prob = torch.sigmoid(state_logits)  # (B, 1) ∈ [0,1]
        
        # 2. Regression Head (Power)
        power_raw = self.DownstreamTaskHead(midpoint_feat)  # (B, c_out)
        
        # === Reverse Instance Normalization (De-Stationarization) === #
        # FIX: Use ORIGINAL inst_mean/inst_std, not learned projection!
        # The model predicts in normalized space, we denormalize to original scale
        
        # Squeeze to match power_raw shape: inst_mean/std are (B, 1, 1) -> (B, 1)
        orig_mean = inst_mean.squeeze(-1)  # (B, 1)
        orig_std = inst_std.squeeze(-1)    # (B, 1)
        
        # Denormalize: power_raw is in normalized space, scale back
        power_denorm = power_raw * orig_std + orig_mean  # (B, 1)
        
        # Ensure non-negative power output
        power_denorm = torch.relu(power_denorm)
        
        # === ON/OFF Conditioning === #
        # If classification < 0.5, force power toward 0
        # Soft gating: power_final = power × state_prob
        power_final = power_denorm * state_prob  # (B, 1)
        
        return {
            'power': power_final,         # Conditioned power (always >= 0)
            'state': state_logits,        # Raw logits for BCE loss
            'power_uncond': power_denorm  # Unconditioned for debugging
        }

