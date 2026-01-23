"""
Hybrid CNN-Transformer Model for NILM.

State-of-the-art architecture combining:
1. Causal Stationarization (NILMFormer-style, Welford algorithm)
2. CNN Feature Extractor (captures local transients at 1Hz)
3. Transformer Encoder with RoPE (long-range temporal dependencies)
4. Multi-Task Output Heads (simultaneous multi-appliance prediction)

Adapted from transformer/model.py for backend inference.
"""
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CAUSAL STATIONARIZATION (NILMFormer-style - VECTORIZED)
# =============================================================================
class CausalStationarization(nn.Module):
    """
    Causal stationarization using vectorized cumulative statistics.

    OPTIMIZED: Uses cumsum for O(n) instead of O(n^2) loop.
    Normalizes input using only past information (no data leakage).
    """

    def __init__(self, eps: float = 1e-6, min_samples: int = 10):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Stationarize input causally using vectorized cumsum.

        Args:
            x: [batch, seq_len, features]

        Returns:
            x_norm: Normalized input
            causal_mean: Running mean at each timestep
            causal_std: Running std at each timestep
        """
        batch, seq_len, features = x.shape
        device = x.device

        # Counts for each position: [1, 2, 3, ..., seq_len]
        counts = torch.arange(1, seq_len + 1, device=device, dtype=x.dtype)
        counts = counts.view(1, -1, 1)  # [1, seq_len, 1]

        # Cumulative mean: cumsum(x) / counts
        cumsum = x.cumsum(dim=1)  # [batch, seq_len, features]
        causal_mean = cumsum / counts

        # Cumulative variance approximation using E[X^2] - E[X]^2
        x_sq = x**2
        cumsum_sq = x_sq.cumsum(dim=1)
        mean_sq = cumsum_sq / counts  # E[X^2]
        sq_mean = causal_mean**2  # E[X]^2

        variance = mean_sq - sq_mean
        # Clamp to avoid negative variance from numerical issues
        variance = torch.clamp(variance, min=0.0)
        causal_std = torch.sqrt(variance + self.eps)

        # For first min_samples timesteps, use std=1 to avoid instability
        mask = counts < self.min_samples
        causal_std = torch.where(mask, torch.ones_like(causal_std), causal_std)

        # Normalize
        x_norm = (x - causal_mean) / causal_std

        return x_norm, causal_mean, causal_std


class DeStationarization(nn.Module):
    """
    De-stationarization using ORIGINAL statistics (not learned).

    Uses the causal mean/std from stationarization directly.
    """

    def __init__(self, n_features: int = 1):
        super().__init__()
        # Keep for backward compatibility but not used
        self.proj = nn.Linear(2 * n_features, 2 * n_features)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self, x: torch.Tensor, causal_mean: torch.Tensor, causal_std: torch.Tensor
    ) -> torch.Tensor:
        """
        De-stationarize predictions using ORIGINAL statistics.

        Args:
            x: Model output [batch, 1] or [batch, seq_len, 1]
            causal_mean: Stats from stationarization [batch, seq_len, features]
            causal_std: Stats from stationarization [batch, seq_len, features]

        Returns:
            De-stationarized output (always >= 0)
        """
        # Use midpoint stats for consistency
        mid = causal_mean.size(1) // 2
        mean_mid = causal_mean[:, mid, :1]  # [batch, 1]
        std_mid = causal_std[:, mid, :1]  # [batch, 1]

        # Use original stats directly, not learned projection
        output = x * std_mid + mean_mid

        # Ensure non-negative power output
        output = torch.relu(output)

        return output


# =============================================================================
# ROTARY POSITIONAL EMBEDDING (RoPE)
# =============================================================================
class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding for better long-sequence handling.

    RoPE encodes position information directly into attention scores,
    providing better extrapolation to longer sequences than absolute PE.
    """

    def __init__(self, dim: int, max_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base

        # Precompute rotation frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin for positions
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int) -> None:
        """Build cos/sin cache for given sequence length."""
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [seq_len, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]

        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin embeddings for the sequence."""
        if seq_len > self.cos_cached.size(2):
            self._build_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================================================================
# SINUSOIDAL POSITIONAL ENCODING (fallback)
# =============================================================================
class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for transformers."""

    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# =============================================================================
# CNN FEATURE EXTRACTOR
# =============================================================================
class ResidualConvBlock(nn.Module):
    """1D Residual Convolutional Block for local feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Residual connection
        self.residual = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv1d(in_channels, out_channels, 1, stride=stride)
        )

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.act(out)

        return out


class CNNFeatureExtractor(nn.Module):
    """
    Multi-scale CNN for extracting local features from power signals.

    Captures transient patterns (power spikes/dips) at different scales.
    """

    def __init__(
        self,
        in_features: int,
        channels: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        use_residual: bool = True,
    ):
        super().__init__()
        channels = channels or [32, 64, 128]
        kernel_sizes = kernel_sizes or [7, 5, 3]

        self.input_proj = nn.Conv1d(in_features, channels[0], 1)

        layers: list[nn.Module] = []
        for i in range(len(channels) - 1):
            if use_residual:
                layers.append(
                    ResidualConvBlock(channels[i], channels[i + 1], kernel_sizes[i])
                )
            else:
                layers.append(
                    nn.Conv1d(
                        channels[i],
                        channels[i + 1],
                        kernel_sizes[i],
                        padding=kernel_sizes[i] // 2,
                    )
                )
                layers.append(nn.BatchNorm1d(channels[i + 1]))
                layers.append(nn.GELU())

        self.layers = nn.Sequential(*layers)
        self.output_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, features]
        Returns:
            [batch, seq_len, output_dim]
        """
        # Conv1d expects [batch, channels, seq_len]
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.layers(x)
        # Back to [batch, seq_len, channels]
        return x.transpose(1, 2)


# =============================================================================
# TRANSFORMER ENCODER WITH OPTIONAL RoPE
# =============================================================================
class TransformerEncoderLayer(nn.Module):
    """
    Custom Transformer Encoder Layer with optional RoPE and efficient attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_rope: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope

        # Multi-head attention components
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # RoPE if enabled
        if use_rope:
            self.rope = RotaryPositionalEmbedding(self.head_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Self-attention with pre-norm
        residual = x
        x = self.norm1(x)

        # Project to Q, K, V
        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Apply RoPE
        if self.use_rope:
            cos, sin = self.rope(x, seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )
        attn_output = self.out_proj(attn_output)

        x = residual + self.dropout(attn_output)

        # Feed-forward with pre-norm
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)

        return x


# =============================================================================
# MULTI-TASK OUTPUT HEAD
# =============================================================================
class ApplianceHead(nn.Module):
    """
    Output head for a single appliance.

    Design choices (SOTA-aligned):
    - Regression: Uses neighborhood around midpoint with attention
    - Classification: Uses temporal pooling (captures context for ON/OFF)
    """

    def __init__(
        self,
        d_model: int,
        seq2point: bool = True,
        dropout: float = 0.1,
        use_pooling_for_state: bool = True,
        neighborhood_size: int = 5,
    ):
        super().__init__()
        self.seq2point = seq2point
        self.use_pooling_for_state = use_pooling_for_state
        self.neighborhood_size = neighborhood_size

        # Attention pooling
        self.attn_query = nn.Linear(d_model, d_model // 4)
        self.attn_key = nn.Linear(d_model, d_model // 4)

        # Regression head
        self.regression = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Classification head
        state_input_dim = d_model * 3 if use_pooling_for_state else d_model
        self.classification = nn.Sequential(
            nn.Linear(state_input_dim, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            dict with 'power' and 'state' predictions
        """
        batch_size, seq_len, d_model = x.size()

        if self.seq2point:
            if x.dim() == 3:
                mid = seq_len // 2

                # Simple midpoint if neighborhood is 0
                if self.neighborhood_size < 1:
                    x_pooled = x[:, mid, :]
                else:
                    # REGRESSION: Neighborhood with attention pooling
                    start = max(0, mid - self.neighborhood_size)
                    end = min(seq_len, mid + self.neighborhood_size + 1)
                    x_neighborhood = x[:, start:end, :]  # [batch, ~11, d_model]

                    x_mid = x[:, mid : mid + 1, :]  # [batch, 1, d_model]

                    q = self.attn_query(x_mid)
                    k = self.attn_key(x_neighborhood)

                    attn_scores = torch.matmul(q, k.transpose(-2, -1))
                    attn_scores = attn_scores / (d_model // 4) ** 0.5
                    attn_weights = F.softmax(attn_scores, dim=-1)

                    x_pooled = torch.matmul(attn_weights, x_neighborhood)
                    x_pooled = x_pooled.squeeze(1)

                power = self.regression(x_pooled)

                # CLASSIFICATION for Seq2Point (Global Pooling for centered state)
                if self.use_pooling_for_state:
                    x_t = x.transpose(1, 2)
                    avg_pool = F.adaptive_avg_pool1d(x_t, 1).squeeze(-1)
                    max_pool = F.adaptive_max_pool1d(x_t, 1).squeeze(-1)
                    x_mid_flat = x[:, mid, :]
                    state_input = torch.cat([x_mid_flat, avg_pool, max_pool], dim=-1)
                else:
                    state_input = x[:, mid, :]

                state = self.classification(state_input)
            else:
                power = self.regression(x)
                state = self.classification(x)
        else:
            # SEQ2SEQ: Predict for every timestep
            power = self.regression(x)
            state = self.classification(x)

        return {"power": power, "state": state}


# =============================================================================
# HYBRID CNN-TRANSFORMER NILM MODEL
# =============================================================================
class HybridCNNTransformer(nn.Module):
    """
    Hybrid CNN-Transformer for multi-appliance NILM.
    """

    def __init__(
        self,
        n_features: int = 7,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        appliances: list[str] | None = None,
        cnn_channels: list[int] | None = None,
        cnn_kernel_sizes: list[int] | None = None,
        use_rope: bool = True,
        seq2point: bool = True,
        max_len: int = 2048,
        use_stationarization: bool = True,
        use_pooling_for_state: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.seq2point = seq2point
        self.use_stationarization = use_stationarization
        self.appliances = appliances or ["heatpump", "washingmachine", "dishwasher"]

        cnn_channels = cnn_channels or [32, 64, 128]
        cnn_kernel_sizes = cnn_kernel_sizes or [7, 5, 3]

        # 0. Causal Stationarization
        if use_stationarization:
            self.stationarize = CausalStationarization()
            self.destationarize = nn.ModuleDict(
                {name: DeStationarization(n_features=1) for name in self.appliances}
            )

        # 1. CNN Feature Extractor
        self.cnn = CNNFeatureExtractor(
            in_features=n_features,
            channels=cnn_channels,
            kernel_sizes=cnn_kernel_sizes,
            use_residual=True,
        )

        # 2. Projection to d_model
        self.proj = nn.Linear(cnn_channels[-1], d_model)

        # 3. Positional Encoding
        self.use_rope = use_rope
        if not use_rope:
            self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        # 4. Transformer Encoder Layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, use_rope)
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)

        # 5. Multi-Task Output Heads
        self.heads = nn.ModuleDict(
            {
                name: ApplianceHead(d_model, seq2point, dropout, use_pooling_for_state)
                for name in self.appliances
            }
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if self.use_stationarization:
            for destat in self.destationarize.values():
                nn.init.zeros_(destat.proj.weight)
                nn.init.zeros_(destat.proj.bias)

    def forward(self, x: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        causal_mean, causal_std = None, None
        if self.use_stationarization:
            x, causal_mean, causal_std = self.stationarize(x)

        x = self.cnn(x)
        x = self.proj(x)

        if not self.use_rope:
            x = self.pos_encoder(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        outputs = {}
        for name, head in self.heads.items():
            out = head(x)
            if self.use_stationarization and causal_mean is not None:
                out["power"] = self.destationarize[name](
                    out["power"], causal_mean, causal_std
                )
            outputs[name] = out

        return outputs

    def predict_power(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.forward(x)
        return {name: out["power"] for name, out in outputs.items()}


# =============================================================================
# ADAPTER FOR BACKEND INFERENCE
# =============================================================================
class HybridCNNTransformerAdapter(nn.Module):
    """
    Adapter wrapping HybridCNNTransformer for backend inference.

    Handles:
    1. Input shape conversion: (B, L, 1) -> (B, L, 7) with temporal features
    2. Output extraction: dict of appliance predictions -> (B, n_appliances) tensor
    3. Output scaling: normalized -> kW (multiply by P_MAX_kW)
    """

    APPLIANCES = [
        "HeatPump",
        "Dishwasher",
        "WashingMachine",
        "Dryer",
        "Oven",
        "Stove",
        "RangeHood",
        "EVCharger",
        "EVSocket",
        "GarageCabinet",
        "RainwaterPump",
    ]

    def __init__(
        self,
        n_features: int = 7,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        cnn_channels: list[int] | None = None,
        cnn_kernel_sizes: list[int] | None = None,
        use_rope: bool = True,
        seq2point: bool = True,
        use_stationarization: bool = True,
        use_pooling_for_state: bool = True,
        p_max_kw: float = 13.5118,
        **kwargs: Any,
    ):
        super().__init__()
        self.p_max_kw = p_max_kw
        self.n_features = n_features

        cnn_channels = cnn_channels or [64, 128, 256]
        cnn_kernel_sizes = cnn_kernel_sizes or [7, 5, 3]

        # Instantiate the underlying model
        self.model = HybridCNNTransformer(
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            appliances=self.APPLIANCES,
            cnn_channels=cnn_channels,
            cnn_kernel_sizes=cnn_kernel_sizes,
            use_rope=use_rope,
            seq2point=seq2point,
            use_stationarization=use_stationarization,
            use_pooling_for_state=use_pooling_for_state,
        )

    def _build_temporal_features(
        self, x: torch.Tensor, timestamp_hours: float = 12.0
    ) -> torch.Tensor:
        """
        Build 7-feature tensor from aggregate input.

        Args:
            x: (B, L, 1) - aggregate power in kW
            timestamp_hours: hour of day for temporal features (default noon)

        Returns:
            (B, L, 7) tensor with [aggregate, hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos]
        """
        B, L, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Normalize aggregate by P_MAX
        x_norm = x / self.p_max_kw

        # Generate temporal features (constant across window for inference)
        hour = timestamp_hours
        dow = 2.0  # Wednesday
        month = 6.0  # June

        hour_sin = torch.full(
            (B, L, 1), math.sin(2 * math.pi * hour / 24), device=device, dtype=dtype
        )
        hour_cos = torch.full(
            (B, L, 1), math.cos(2 * math.pi * hour / 24), device=device, dtype=dtype
        )
        dow_sin = torch.full(
            (B, L, 1), math.sin(2 * math.pi * dow / 7), device=device, dtype=dtype
        )
        dow_cos = torch.full(
            (B, L, 1), math.cos(2 * math.pi * dow / 7), device=device, dtype=dtype
        )
        month_sin = torch.full(
            (B, L, 1), math.sin(2 * math.pi * month / 12), device=device, dtype=dtype
        )
        month_cos = torch.full(
            (B, L, 1), math.cos(2 * math.pi * month / 12), device=device, dtype=dtype
        )

        return torch.cat(
            [x_norm, hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos], dim=-1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for backend inference.

        Args:
            x: (B, L, 1) - aggregate power in kW (already preprocessed by engine)

        Returns:
            (B, 11) - predicted power per appliance in kW
        """
        # Build 7-feature input
        x_full = self._build_temporal_features(x)

        # Run model
        outputs = self.model(x_full)  # Dict[str, Dict[str, Tensor]]

        # Extract power predictions and stack
        powers = []
        for appliance in self.APPLIANCES:
            power = outputs[appliance]["power"]  # (B, 1)
            powers.append(power)

        result = torch.cat(powers, dim=-1)  # (B, 11)

        # Scale to kW (model outputs are in normalized units)
        # Note: The model's destationarization already handles some denormalization
        # but output is still relative to the normalized space
        result = result * self.p_max_kw

        return result
