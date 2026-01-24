"""
NILMFormer Model - ALIGNED TO KDD 2025 PAPER
=============================================
Reference: https://github.com/adrienpetralia/NILMFormer

Key Design Choices (from paper):
1. Seq2Seq output (NOT Seq2Point)
2. Instance Normalization (global mean/std, NOT causal)
3. TokenStats: mean+std as learnable token
4. ProjStats: learned de-normalization
5. DilatedBlock: [1,2,4,8] dilations with residual
6. DiagonallyMaskedSelfAttention: masks diagonal in attention
7. Loss: Simple MSELoss (NOT weighted/focal)

H200 32GB: Can handle large batch + long sequences + Seq2Seq
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field


# =============================================================================
# CONFIG (aligned to paper defaults)
# =============================================================================
@dataclass
class NILMFormerConfig:
    """NILMFormer configuration - paper defaults."""
    c_in: int = 1                    # Input channels (aggregate power only)
    c_embedding: int = 8             # Exogenous channels (temporal sin/cos)
    c_out: int = 1                   # Output channels (single appliance)
    
    kernel_size: int = 3             # Dilated conv kernel
    kernel_size_head: int = 3        # Output head kernel
    dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    conv_bias: bool = True
    
    n_encoder_layers: int = 3        # Transformer layers
    d_model: int = 96                # Model dimension (must be divisible by 4)
    n_head: int = 8                  # Attention heads
    dp_rate: float = 0.2             # Dropout
    pffn_ratio: int = 4              # FFN expansion ratio
    norm_eps: float = 1e-5


# =============================================================================
# LAYERS: Dilated Convolution Embedding
# =============================================================================
class ResUnit(nn.Module):
    """Residual Unit with Dilated Convolution (from paper)."""
    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int = 3,
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
        
        # Residual projection if channels don't match
        if c_in > 1 and c_in != c_out:
            self.match_residual = True
            self.conv = nn.Conv1d(in_channels=c_in, out_channels=c_out, kernel_size=1)
        else:
            self.match_residual = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.match_residual:
            x_bottleneck = self.conv(x)
            x = self.layers(x)
            return torch.add(x_bottleneck, x)
        else:
            return torch.add(x, self.layers(x))


class DilatedBlock(nn.Module):
    """
    Dilated Convolutional Block (from paper).
    Stack of ResUnits with increasing dilation factors [1, 2, 4, 8].
    """
    def __init__(
        self,
        c_in: int = 1,
        c_out: int = 72,  # 3/4 of d_model if d_model=96
        kernel_size: int = 3,
        dilation_list: List[int] = [1, 2, 4, 8],
        bias: bool = True,
    ):
        super().__init__()
        
        layers = []
        for i, dilation in enumerate(dilation_list):
            if i == 0:
                layers.append(ResUnit(c_in, c_out, k=kernel_size, dilation=dilation, bias=bias))
            else:
                layers.append(ResUnit(c_out, c_out, k=kernel_size, dilation=dilation, bias=bias))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# =============================================================================
# LAYERS: Transformer with Diagonal Masking
# =============================================================================
class DiagonalMaskFromSeqlen:
    """Create diagonal mask to prevent self-attention on same position."""
    def __init__(self, B: int, L: int, device: str = "cpu"):
        with torch.no_grad():
            # Diagonal mask: True on diagonal (will be masked to -inf)
            self._mask = torch.diag(
                torch.ones(L, dtype=torch.bool, device=device)
            ).unsqueeze(0).expand(B, -1, -1)  # (B, L, L)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask


class DiagonallyMaskedSelfAttention(nn.Module):
    """
    Self-Attention with diagonal masking (from paper).
    
    The diagonal mask prevents each position from attending to itself,
    forcing the model to rely on context from other positions.
    """
    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        dropout: float,
    ):
        super().__init__()
        
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seqlen, _ = x.shape
        
        # Project Q, K, V
        xq = self.wq(x).view(batch, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(batch, seqlen, self.n_heads, self.head_dim)
        xv = self.wv(x).view(batch, seqlen, self.n_heads, self.head_dim)
        
        # Rearrange to (B, H, L, D)
        xq = xq.permute(0, 2, 1, 3)
        xk = xk.permute(0, 2, 1, 3)
        xv = xv.permute(0, 2, 1, 3)
        
        # Attention scores
        scores = torch.matmul(xq, xk.transpose(-2, -1)) * self.scale  # (B, H, L, L)
        
        # Apply diagonal mask (mask self-attention on same position)
        diag_mask = DiagonalMaskFromSeqlen(batch, seqlen, device=x.device)
        scores = scores.masked_fill(diag_mask.mask.unsqueeze(1), float('-inf'))
        
        # Softmax and dropout
        attn = self.attn_dropout(F.softmax(scores, dim=-1))
        
        # Apply attention to values
        output = torch.matmul(attn, xv)  # (B, H, L, D)
        
        # Rearrange back to (B, L, H*D)
        output = output.permute(0, 2, 1, 3).contiguous().reshape(batch, seqlen, -1)
        
        return self.out_dropout(self.wo(output))


class PositionWiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dp_rate: float = 0.0,
    ):
        super().__init__()
        self.layer1 = nn.Linear(dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dp_rate)
        self.activation = F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.dropout(self.activation(self.layer1(x))))


class EncoderLayer(nn.Module):
    """Transformer Encoder Layer (Pre-LN style)."""
    def __init__(self, config: NILMFormerConfig):
        super().__init__()
        
        assert config.d_model % config.n_head == 0, \
            f"d_model ({config.d_model}) must be divisible by n_head ({config.n_head})"
        
        self.attention = DiagonallyMaskedSelfAttention(
            dim=config.d_model,
            n_heads=config.n_head,
            head_dim=config.d_model // config.n_head,
            dropout=config.dp_rate,
        )
        
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.dropout = nn.Dropout(config.dp_rate)
        
        self.pffn = PositionWiseFeedForward(
            dim=config.d_model,
            hidden_dim=config.d_model * config.pffn_ratio,
            dp_rate=config.dp_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN: Norm -> Attention -> Residual
        x_norm = self.norm1(x)
        x = x + self.attention(x_norm)
        
        # Pre-LN: Norm -> FFN -> Residual
        x_norm = self.norm2(x)
        x = x + self.dropout(self.pffn(x_norm))
        
        return x


# =============================================================================
# MAIN MODEL: NILMFormer (Seq2Seq, aligned to paper)
# =============================================================================
class NILMFormerPaper(nn.Module):
    """
    NILMFormer architecture (KDD 2025 paper).
    
    Key features:
    - Seq2Seq output (predicts full sequence, NOT just midpoint)
    - Instance Normalization (subtract mean/std globally)
    - TokenStats: project mean/std to d_model and append as token
    - ProjStats: learn to de-normalize output
    - DilatedBlock: captures local patterns with increasing receptive field
    - Diagonal-masked attention: forces contextual learning
    
    Input: (B, C, L) where C = 1 (aggregate) + c_embedding (exogenous)
    Output: (B, 1, L) power prediction for target appliance
    """
    
    def __init__(self, config: NILMFormerConfig):
        super().__init__()
        
        assert config.d_model % 4 == 0, "d_model must be divisible by 4"
        
        self.config = config
        
        # ============ Embedding ============
        # Main embedding: 3/4 of d_model for load curve
        d_model_main = 3 * config.d_model // 4  # e.g., 72 if d_model=96
        d_model_exo = config.d_model // 4       # e.g., 24 if d_model=96
        
        self.EmbedBlock = DilatedBlock(
            c_in=config.c_in,
            c_out=d_model_main,
            kernel_size=config.kernel_size,
            dilation_list=config.dilations,
            bias=config.conv_bias,
        )
        
        # Exogenous projection (temporal features)
        self.ProjEmbedding = nn.Conv1d(
            in_channels=config.c_embedding,
            out_channels=d_model_exo,
            kernel_size=1
        )
        
        # TokenStats: project (mean, std) -> d_model
        self.ProjStats1 = nn.Linear(2, config.d_model)
        # ProjStats: project d_model -> (mean, std) for de-normalization
        self.ProjStats2 = nn.Linear(config.d_model, 2)
        
        # ============ Encoder ============
        layers = []
        for _ in range(config.n_encoder_layers):
            layers.append(EncoderLayer(config))
        layers.append(nn.LayerNorm(config.d_model))
        self.EncoderBlock = nn.Sequential(*layers)
        
        # ============ Downstream Task Head ============
        # Conv1d head for Seq2Seq output
        self.DownstreamTaskHead = nn.Conv1d(
            in_channels=config.d_model,
            out_channels=config.c_out,
            kernel_size=config.kernel_size_head,
            padding=config.kernel_size_head // 2,
            padding_mode="replicate",
        )
        
        # Initialize weights
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for NILMFormer.
        
        Input: (B, C, L) where C = 1 + c_embedding
            - x[:, :1, :] => load curve (aggregate power)
            - x[:, 1:, :] => exogenous inputs (temporal sin/cos)
        
        Output: (B, 1, L) predicted appliance power
        """
        # Split channels
        encoding = x[:, 1:, :]  # (B, c_embedding, L) - exogenous
        x = x[:, :1, :]         # (B, 1, L) - load curve
        
        # === Instance Normalization (Global) ===
        # Compute mean and std over the sequence dimension
        inst_mean = torch.mean(x, dim=-1, keepdim=True).detach()  # (B, 1, 1)
        inst_std = torch.sqrt(
            torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-6
        ).detach()  # (B, 1, 1)
        
        # Normalize
        x = (x - inst_mean) / inst_std  # (B, 1, L)
        
        # === Embedding ===
        # 1) Dilated Conv block on normalized load
        x = self.EmbedBlock(x)  # (B, d_model_main, L)
        
        # 2) Project exogenous features
        encoding = self.ProjEmbedding(encoding)  # (B, d_model_exo, L)
        
        # 3) Concatenate and transpose for Transformer
        x = torch.cat([x, encoding], dim=1)  # (B, d_model, L)
        x = x.permute(0, 2, 1)  # (B, L, d_model)
        
        # === TokenStats ===
        # Project (mean, std) to d_model and append as special token
        stats = torch.cat([inst_mean, inst_std], dim=1)  # (B, 2, 1)
        stats = stats.permute(0, 2, 1)  # (B, 1, 2)
        stats_token = self.ProjStats1(stats)  # (B, 1, d_model)
        
        # Append stats token to sequence
        x = torch.cat([x, stats_token], dim=1)  # (B, L+1, d_model)
        
        # === Transformer Encoder ===
        x = self.EncoderBlock(x)  # (B, L+1, d_model)
        
        # Remove stats token (keep sequence)
        stats_token_out = x[:, -1:, :]  # (B, 1, d_model) - save for de-norm
        x = x[:, :-1, :]  # (B, L, d_model)
        
        # === Conv Head (Seq2Seq) ===
        x = x.permute(0, 2, 1)  # (B, d_model, L)
        x = self.DownstreamTaskHead(x)  # (B, c_out, L)
        
        # === Reverse Instance Normalization (ProjStats) ===
        # Learn to de-normalize based on transformed stats token
        stats_out = self.ProjStats2(stats_token_out)  # (B, 1, 2)
        out_mean = stats_out[:, :, 0:1]  # (B, 1, 1)
        out_std = stats_out[:, :, 1:2]   # (B, 1, 1)
        
        # De-normalize output
        x = x * out_std + out_mean  # (B, c_out, L)
        
        return x


# =============================================================================
# MULTI-APPLIANCE WRAPPER (for your 11-appliance setup)
# =============================================================================
class MultiApplianceNILMFormer(nn.Module):
    """
    Multi-appliance NILMFormer wrapper.
    
    Creates separate NILMFormer heads for each appliance.
    Shares the embedding and encoder, but has separate output heads.
    
    This is memory-efficient for H200 32GB.
    """
    
    def __init__(
        self,
        appliances: List[str],
        c_embedding: int = 6,  # 6 temporal features (sin/cos)
        d_model: int = 96,
        n_encoder_layers: int = 3,
        n_head: int = 8,
        dp_rate: float = 0.2,
        kernel_size: int = 3,
        dilations: List[int] = [1, 2, 4, 8],
    ):
        super().__init__()
        
        self.appliances = appliances
        
        config = NILMFormerConfig(
            c_in=1,
            c_embedding=c_embedding,
            c_out=1,
            kernel_size=kernel_size,
            dilations=dilations,
            n_encoder_layers=n_encoder_layers,
            d_model=d_model,
            n_head=n_head,
            dp_rate=dp_rate,
        )
        
        # Shared components
        d_model_main = 3 * d_model // 4
        d_model_exo = d_model // 4
        
        self.EmbedBlock = DilatedBlock(
            c_in=1,
            c_out=d_model_main,
            kernel_size=kernel_size,
            dilation_list=dilations,
        )
        
        self.ProjEmbedding = nn.Conv1d(c_embedding, d_model_exo, kernel_size=1)
        self.ProjStats1 = nn.Linear(2, d_model)
        
        # Shared encoder
        layers = []
        for _ in range(n_encoder_layers):
            layers.append(EncoderLayer(config))
        layers.append(nn.LayerNorm(d_model))
        self.EncoderBlock = nn.Sequential(*layers)
        
        # Per-appliance output heads
        self.heads = nn.ModuleDict()
        self.proj_stats_out = nn.ModuleDict()
        
        for app in appliances:
            self.heads[app] = nn.Conv1d(
                d_model, 1, kernel_size=3, padding=1, padding_mode="replicate"
            )
            self.proj_stats_out[app] = nn.Linear(d_model, 2)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass.
        
        Input: (B, L, C) where C = 1 + c_embedding
        Output: dict of {appliance: (B, L, 1)}
        """
        # Transpose to (B, C, L) for Conv1d
        x = x.permute(0, 2, 1)
        
        # Split
        encoding = x[:, 1:, :]  # (B, c_emb, L)
        load = x[:, :1, :]      # (B, 1, L)
        
        # Instance Norm
        inst_mean = load.mean(dim=-1, keepdim=True).detach()
        inst_std = (load.var(dim=-1, keepdim=True, unbiased=False) + 1e-6).sqrt().detach()
        load_norm = (load - inst_mean) / inst_std
        
        # Embedding
        x_emb = self.EmbedBlock(load_norm)
        enc_emb = self.ProjEmbedding(encoding)
        x_emb = torch.cat([x_emb, enc_emb], dim=1).permute(0, 2, 1)  # (B, L, d_model)
        
        # TokenStats
        stats = torch.cat([inst_mean, inst_std], dim=1).permute(0, 2, 1)  # (B, 1, 2)
        stats_token = self.ProjStats1(stats)  # (B, 1, d_model)
        x_full = torch.cat([x_emb, stats_token], dim=1)  # (B, L+1, d_model)
        
        # Encoder
        x_out = self.EncoderBlock(x_full)
        stats_token_out = x_out[:, -1:, :]
        x_seq = x_out[:, :-1, :].permute(0, 2, 1)  # (B, d_model, L)
        
        # Per-appliance heads
        outputs = {}
        for app in self.appliances:
            # Conv head
            y = self.heads[app](x_seq)  # (B, 1, L)
            
            # De-normalize
            stats_proj = self.proj_stats_out[app](stats_token_out)  # (B, 1, 2)
            out_mean = stats_proj[:, :, 0:1]
            out_std = stats_proj[:, :, 1:2]
            y = y * out_std + out_mean
            
            # Ensure non-negative
            y = F.relu(y)
            
            outputs[app] = y.permute(0, 2, 1)  # (B, L, 1)
        
        return outputs


# =============================================================================
# FACTORY FUNCTION
# =============================================================================
def create_nilmformer_paper(
    appliances: List[str],
    c_embedding: int = 6,
    d_model: int = 96,
    n_layers: int = 3,
    n_head: int = 8,
    dropout: float = 0.2,
    multi_appliance: bool = True,
) -> nn.Module:
    """
    Create NILMFormer model aligned to KDD 2025 paper.
    
    Args:
        appliances: List of appliance names
        c_embedding: Number of exogenous features (temporal sin/cos)
        d_model: Model dimension (must be divisible by 4)
        n_layers: Number of transformer encoder layers
        n_head: Number of attention heads
        dropout: Dropout rate
        multi_appliance: If True, create shared-encoder multi-head model
        
    Returns:
        NILMFormer model
    """
    if multi_appliance and len(appliances) > 1:
        return MultiApplianceNILMFormer(
            appliances=appliances,
            c_embedding=c_embedding,
            d_model=d_model,
            n_encoder_layers=n_layers,
            n_head=n_head,
            dp_rate=dropout,
        )
    else:
        # Single appliance
        config = NILMFormerConfig(
            c_in=1,
            c_embedding=c_embedding,
            c_out=1,
            d_model=d_model,
            n_encoder_layers=n_layers,
            n_head=n_head,
            dp_rate=dropout,
        )
        return NILMFormerPaper(config)


if __name__ == "__main__":
    # Quick test
    print("Testing NILMFormer Paper Implementation...")
    
    # Config
    appliances = ['HeatPump', 'Dishwasher', 'WashingMachine']
    batch_size = 8
    seq_len = 1024
    c_in = 1 + 6  # aggregate + 6 temporal features
    
    # Create model
    model = create_nilmformer_paper(
        appliances=appliances,
        c_embedding=6,
        d_model=96,
        n_layers=3,
        multi_appliance=True,
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, c_in)
    
    with torch.no_grad():
        outputs = model(x)
    
    print(f"Input shape: {x.shape}")
    for name, out in outputs.items():
        print(f"  {name}: {out.shape}")
    
    print("✅ All tests passed!")
