"""
Inference engine with model loading and execution.
Supports CNNTransformer, CNNSeq2Seq, and UNet1D architectures.
Uses safetensors-only loading with thread offloading for inference.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from datetime import datetime

from app.core.errors import ErrorCode, ModelError, ValidationError
from app.core.logging import get_logger
from app.core.telemetry import INFERENCE_COUNT, INFERENCE_LATENCY, MODEL_CACHE_SIZE
from app.domain.inference.registry import ModelEntry, PreprocessingConfig, get_model_registry
from app.domain.inference.architectures import HybridCNNTransformerAdapter, TCN_Gated, TCN_SA

logger = get_logger(__name__)


# =============================================================================
# Model Architectures
# =============================================================================


class CNNTransformer(nn.Module):
    """
    CNN + Transformer hybrid architecture for NILM.

    Architecture:
    - CNN feature extractor (1D convolutions)
    - Transformer encoder
    - Pooling and output head
    """

    def __init__(
        self,
        input_channels: int = 1,
        cnn_channels: list[int] | None = None,
        cnn_kernel_sizes: list[int] | None = None,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        cnn_channels = cnn_channels or [32, 64]
        cnn_kernel_sizes = cnn_kernel_sizes or [5, 5]

        # CNN feature extractor
        cnn_layers: list[nn.Module] = []
        in_ch = input_channels
        for out_ch, kernel in zip(cnn_channels, cnn_kernel_sizes):
            cnn_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # Project to d_model
        self.project = nn.Conv1d(in_ch, d_model, 1)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, features)

        Returns:
            Output tensor of shape (batch, 1)
        """
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)

        # CNN feature extraction
        x = self.cnn(x)

        # Project to d_model
        x = self.project(x)

        # Transformer expects (batch, seq_len, d_model)
        x = x.transpose(1, 2)
        x = self.transformer(x)

        # Pool and output
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.pool(x).squeeze(-1)  # (batch, d_model)
        x = self.output(x)  # (batch, 1)

        return x


class CNNSeq2Seq(nn.Module):
    """CNN Encoder-Decoder for NILM power disaggregation (matches existing)."""

    def __init__(
        self,
        input_channels: int = 7,
        hidden_channels: int = 48,
        num_layers: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Encoder
        encoder_layers: list[nn.Module] = []
        in_ch = input_channels
        for i in range(num_layers):
            out_ch = hidden_channels * (2 ** i)
            encoder_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.25),
            ])
            in_ch = out_ch
        self.encoder = nn.Sequential(*encoder_layers)

        self.bottleneck_ch = hidden_channels * (2 ** (num_layers - 1))

        # Decoder
        decoder_layers: list[nn.Module] = []
        in_ch = self.bottleneck_ch
        for i in range(num_layers - 1, -1, -1):
            out_ch = hidden_channels * (2 ** i) if i > 0 else hidden_channels
            decoder_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2),
            ])
            in_ch = out_ch
        self.decoder = nn.Sequential(*decoder_layers)

        self.output_layer = nn.Sequential(
            nn.Conv1d(hidden_channels, 1, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        output = self.output_layer(decoded)
        return output.transpose(1, 2)  # (batch, seq_len, 1)


class UNet1D(nn.Module):
    """U-Net 1D for NILM with skip connections (matches existing)."""

    def __init__(self, input_channels: int = 7, base_channels: int = 24, **kwargs: Any) -> None:
        super().__init__()

        self.enc1 = self._conv_block(input_channels, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._conv_block(base_channels * 4, base_channels * 8)

        self.bottleneck = self._conv_block(base_channels * 8, base_channels * 16)

        self.dec4 = self._conv_block(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.dec3 = self._conv_block(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = self._conv_block(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = self._conv_block(base_channels * 2 + base_channels, base_channels)

        self.output = nn.Sequential(
            nn.Conv1d(base_channels, 1, kernel_size=1),
            nn.ReLU(),
        )

        self.pool = nn.MaxPool1d(2)

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.1),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),
        )

    def _upsample_and_concat(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2], mode='linear', align_corners=True)
        return torch.cat([x, skip], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_len = x.shape[1]
        x = x.transpose(1, 2)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(self._upsample_and_concat(b, e4))
        d3 = self.dec3(self._upsample_and_concat(d4, e3))
        d2 = self.dec2(self._upsample_and_concat(d3, e2))
        d1 = self.dec1(self._upsample_and_concat(d2, e1))

        out = self.output(d1)

        if out.shape[2] != original_len:
            out = F.interpolate(out, size=original_len, mode='linear', align_corners=True)

        return out.transpose(1, 2)


# Model factory
MODEL_CLASSES: dict[str, type[nn.Module]] = {
    "cnntransformer": CNNTransformer,
    "cnn_transformer": CNNTransformer,
    "cnnseq2seq": CNNSeq2Seq,
    "cnn": CNNSeq2Seq,
    "unet": UNet1D,
    "unet1d": UNet1D,
    # HybridCNNTransformer architecture (NILMFormer-style)
    "hybrid_cnn_transformer": HybridCNNTransformerAdapter,
    "hybridcnntransformer": HybridCNNTransformerAdapter,
    "hybrid": HybridCNNTransformerAdapter,
    # TCN architectures
    "tcn_gated": TCN_Gated,
    "tcn-gated": TCN_Gated,
    "tcn_sa": TCN_SA,
    "tcn-sa": TCN_SA,
}


def create_model(architecture: str, params: dict[str, Any]) -> nn.Module:
    """
    Create a model instance from architecture name and parameters.

    Raises:
        ModelError: If architecture is unknown
    """
    key = architecture.lower().replace("-", "_").replace(" ", "")
    if key not in MODEL_CLASSES:
        raise ModelError(
            code=ErrorCode.MODEL_FACTORY_ERROR,
            message=f"Unknown architecture: {architecture}",
            details={"available": list(MODEL_CLASSES.keys())},
        )

    model_class = MODEL_CLASSES[key]
    try:
        return model_class(**params)
    except Exception as e:
        raise ModelError(
            code=ErrorCode.MODEL_FACTORY_ERROR,
            message=f"Failed to instantiate {architecture}: {e}",
        )


# =============================================================================
# Preprocessing
# =============================================================================


def apply_preprocessing(
    data: list[float],
    config: PreprocessingConfig,
) -> np.ndarray:
    """Apply preprocessing to input data."""
    arr = np.array(data, dtype=np.float32)

    if config.type == "standard":
        mean = np.array(config.mean or 0.0, dtype=np.float32)
        std = np.array(config.std or 1.0, dtype=np.float32)
        if std.ndim > 0:
            mean = mean[0] if len(mean) > 0 else 0.0
            std = std[0] if len(std) > 0 else 1.0
        arr = (arr - mean) / (std + 1e-8)

    elif config.type == "minmax":
        min_val = config.min_val or 0.0
        max_val = config.max_val or 1.0
        arr = (arr - min_val) / (max_val - min_val + 1e-8)

    # identity: no transformation

    return arr


def apply_inverse_preprocessing(
    value: float,
    config: PreprocessingConfig,
) -> float:
    """
    Apply inverse preprocessing to model output.
    
    For TCN_SA models: output is normalized [0,1] by P_MAX.
    To get kW: predicted_kw = output * p_max_kw
    """
    # TCN_SA models: output normalized by P_MAX
    # De-normalize: multiply by p_max_kw
    if hasattr(config, 'p_max_kw') and config.p_max_kw:
        return value * config.p_max_kw
    
    # Legacy: standard normalization with mean/std
    if config.type == "standard":
        mean = config.mean
        std = config.std
        if isinstance(mean, list):
            mean = mean[0] if mean else 0.0
        if isinstance(std, list):
            std = std[0] if std else 1.0
        mean = mean or 0.0
        std = std or 1.0
        value = value * std + mean

    elif config.type == "minmax":
        min_val = config.min_val or 0.0
        max_val = config.max_val or 1.0
        value = value * (max_val - min_val) + min_val

    return value


# =============================================================================
# Inference Engine
# =============================================================================


class InferenceEngine:
    """Engine for running model inference."""

    def __init__(self) -> None:
        self._model_cache: dict[tuple[str, str], nn.Module] = {}

    def _load_model(self, entry: ModelEntry) -> nn.Module:
        """Load a model from safetensors or .pt checkpoint."""
        cache_key = (entry.model_id, entry.model_version)

        # Check cache
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        # Create model architecture
        model = create_model(entry.architecture, entry.architecture_params)

        # Ensure artifact is present (auto-download if missing)
        from app.domain.inference.artifacts import ensure_artifact_present

        try:
            artifact_path = ensure_artifact_present(entry)
        except Exception as e:
            if isinstance(e, ModelError):
                raise
            raise ModelError(
                code=ErrorCode.MODEL_LOAD_ERROR,
                message=f"Failed to ensure artifact availability: {e}",
            )

        # Support both .safetensors and .pt formats
        if artifact_path.suffix == ".safetensors":
            try:
                state_dict = load_safetensors(str(artifact_path))
                model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                raise ModelError(
                    code=ErrorCode.MODEL_LOAD_ERROR,
                    message=f"Failed to load safetensors: {e}",
                )
        elif artifact_path.suffix in (".pt", ".pth"):
            try:
                checkpoint = torch.load(str(artifact_path), map_location="cpu", weights_only=False)
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if "model_state_dict" in checkpoint:
                        state_dict = checkpoint["model_state_dict"]
                    elif "model" in checkpoint:
                        state_dict = checkpoint["model"]
                    elif "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                raise ModelError(
                    code=ErrorCode.MODEL_LOAD_ERROR,
                    message=f"Failed to load .pt checkpoint: {e}",
                )
        else:
            raise ModelError(
                code=ErrorCode.MODEL_ARTIFACT_INVALID,
                message=f"Unsupported artifact format: {artifact_path.suffix}. Use .safetensors or .pt",
            )

        model.eval()

        # Cache model
        self._model_cache[cache_key] = model
        MODEL_CACHE_SIZE.set(len(self._model_cache))

        logger.info(
            "Model loaded",
            extra={"model_id": entry.model_id, "version": entry.model_version},
        )

        return model

    def get_model(self, model_id: str | None, appliance_id: str) -> tuple[nn.Module, ModelEntry]:
        """
        Get a model for inference.

        If model_id is provided, use that specific model.
        Otherwise, get the active model for the appliance.
        """
        registry = get_model_registry()

        if model_id:
            entry = registry.get(model_id)
            if not entry:
                raise ModelError(
                    code=ErrorCode.MODEL_NOT_FOUND,
                    message=f"Model not found: {model_id}",
                )
        else:
            entry = registry.get_active_for_appliance(appliance_id)
            if not entry:
                raise ModelError(
                    code=ErrorCode.MODEL_NOT_FOUND,
                    message=f"No active model for appliance: {appliance_id}",
                )

        model = self._load_model(entry)
        return model, entry

    def run_inference(
        self,
        model: nn.Module,
        entry: ModelEntry,
        window: list[float],
    ) -> tuple[float, float]:
        """
        Run inference on a window of data.

        Returns:
            Tuple of (predicted_kw, confidence)
        """
        start_time = time.time()
        status = "success"

        try:
            # Validate window length
            if len(window) != entry.input_window_size:
                raise ValidationError(
                    code=ErrorCode.VALIDATION_WINDOW_LENGTH,
                    message=f"Expected window of {entry.input_window_size} values, got {len(window)}",
                )

            # Preprocess
            preprocessed = apply_preprocessing(window, entry.preprocessing)

            # Convert to tensor: (1, seq_len, 1)
            x = torch.tensor(preprocessed, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

            # Run inference
            with torch.no_grad():
                output = model(x)

            # Extract output value
            if output.dim() == 3:
                # Sequence output -> take mean or last
                output_val = output.mean().item()
            else:
                output_val = output.squeeze().item()

            # Apply inverse preprocessing
            predicted_kw = apply_inverse_preprocessing(output_val, entry.preprocessing)

            # Clamp to non-negative
            predicted_kw = max(0.0, predicted_kw)

            # Dynamic confidence based on prediction certainty
            # Use max power from preprocessing config
            p_max_kw = (entry.preprocessing.max_val or 15000.0) / 1000.0  # Convert W to kW
            if p_max_kw < 0.1:
                p_max_kw *= 1000  # Was already in kW
            
            # Normalize by rated power
            norm_power = min(predicted_kw / max(p_max_kw, 0.1), 1.0)
            
            # Confidence logic: high for clear ON/OFF, lower for uncertain middle
            import math
            if norm_power > 0.4:
                confidence = 0.75 + 0.20 * min(norm_power, 1.0)
            elif norm_power < 0.05:
                confidence = 0.80 + 0.15 * (1 - norm_power / 0.05)
            else:
                midpoint = 0.225
                distance_from_mid = abs(norm_power - midpoint) / midpoint
                confidence = 0.45 + 0.30 * math.tanh(distance_from_mid * 2)

            return predicted_kw, confidence

        except Exception as e:
            status = "error"
            if isinstance(e, (ModelError, ValidationError)):
                raise
            raise ModelError(
                code=ErrorCode.INFERENCE_FAILED,
                message=f"Inference failed: {e}",
            )
        finally:
            duration = time.time() - start_time
            INFERENCE_LATENCY.labels(model_id=entry.model_id).observe(duration)
            INFERENCE_COUNT.labels(model_id=entry.model_id, status=status).inc()

    def run_inference_multi_head(
        self,
        model: nn.Module,
        entry: ModelEntry,
        window: list[float],
        samples: list[tuple[str, float]] | None = None,
    ) -> dict[str, tuple[float, float]]:
        """
        Run multi-head inference on a window of data.

        For multi-head models, the model output has N values (one per head).
        For single-head models, this wraps the output in a dict.

        Returns:
            Dict mapping field_key to (predicted_kw, confidence)
        """
        start_time = time.time()
        status = "success"

        try:
            # Validate window length
            if len(window) != entry.input_window_size:
                raise ValidationError(
                    code=ErrorCode.VALIDATION_WINDOW_LENGTH,
                    message=f"Expected window of {entry.input_window_size} values, got {len(window)}",
                )

            # SOTA Architectures require 7/8 temporal features
            is_sota = entry.architecture.lower() in ("tcn_gated", "tcn-gated", "tcn_sa", "tcn-sa", "hybrid_cnn_transformer", "cnnseq2seq", "unet1d")

            if is_sota and samples:
                # Use SOTA preprocessor for 7 or 8 features (based on model config)
                from app.domain.inference.preprocessing import DataPreprocessor
                import numpy as _np
                # Get P_MAX from architecture_params (default 15 kW = 15000 W)
                # DataPreprocessor expects P_MAX in WATTS
                p_max_watts = entry.architecture_params.get("p_max_kw", 15.0) * 1000.0
                # Get n_features from architecture_params (7 for old models, 8 for new with delta_P)
                n_features = entry.architecture_params.get("n_features", 7)
                preprocessor = DataPreprocessor(P_MAX=p_max_watts, n_features=n_features)

                features_list = []
                for ts_iso, val_kw in samples:
                    try:
                        # Handle ISO string with potential Z or +00:00
                        ts_clean = ts_iso.replace("Z", "+00:00")
                        dt = datetime.fromisoformat(ts_clean)
                        # CRITICAL: Rolling window stores values in kW, preprocessor expects WATTS
                        # Convert kW -> W before passing to preprocessor
                        val_watts = val_kw * 1000.0
                        # process_sample returns (n_features,) array
                        features = preprocessor.process_sample(dt.timestamp(), val_watts)
                        features_list.append(features)
                    except Exception as e:
                        logger.warning(f"Failed to preprocess sample: {e}")
                        # Fallback to zeros for this sample if date parsing fails
                        features_list.append(_np.zeros(n_features, dtype=_np.float32))
                
                # features_list is (seq_len, n_features)
                # TCN_SA expects input (B, T, F) where T=seq_len, F=n_features
                # It does internal transpose: x.transpose(1, 2) -> (B, F, T) for Conv1d
                x = torch.tensor(_np.array(features_list), dtype=torch.float32).unsqueeze(0)  # (1, T, F)
            else:
                # Standard preprocessing
                preprocessed = apply_preprocessing(window, entry.preprocessing)
                # Convert to tensor: (1, 1, seq_len)
                x = torch.tensor(preprocessed, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
                
            # Run inference
            with torch.no_grad():
                output = model(x)

            # Handle Multi-Task Learning (MTL) output - typically (power, probability)
            if isinstance(output, (list, tuple)):
                # We primarily want the first output (power)
                output = output[0]

            # Get number of output heads
            num_heads = len(entry.heads)
            field_keys = entry.head_field_keys

            # Parse output based on shape
            if output.dim() == 3:
                # Sequence output (batch, seq_len, heads) -> take mean over sequence
                output_vals = output.mean(dim=1).squeeze(0)  # (heads,) or scalar
            elif output.dim() == 2:
                # (batch, heads) or (batch, 1)
                output_vals = output.squeeze(0)  # (heads,) or scalar
            else:
                # Scalar
                output_vals = output

            # Convert to list of floats
            if output_vals.numel() == 1:
                # Single output - distribute to all heads (legacy single-head models)
                raw_val = output_vals.item()
                output_list = [raw_val] * num_heads
            else:
                # Multi-output
                output_list = output_vals.tolist()
                if len(output_list) < num_heads:
                    # Pad with zeros if fewer outputs than heads
                    output_list.extend([0.0] * (num_heads - len(output_list)))
                elif len(output_list) > num_heads:
                    # Truncate if more outputs than heads
                    output_list = output_list[:num_heads]

            # Build result dict
            results: dict[str, tuple[float, float]] = {}
            for i, field_key in enumerate(field_keys):
                raw_val = output_list[i]

                # Apply inverse preprocessing
                predicted_kw = apply_inverse_preprocessing(raw_val, entry.preprocessing)

                # Clamp to non-negative
                predicted_kw = max(0.0, predicted_kw)

                # Dynamic confidence based on prediction certainty
                # Use max power for this appliance from preprocessing config
                p_max_kw = (entry.preprocessing.max_val or 15000.0) / 1000.0  # Convert W to kW
                if p_max_kw < 0.1:
                    p_max_kw *= 1000  # Was already in kW
                
                # Normalize by rated power
                norm_power = min(predicted_kw / max(p_max_kw, 0.1), 1.0)
                
                # Confidence logic:
                # - High confidence (0.75-0.95) for clear ON states (>40% of rated power)
                # - High confidence (0.80-0.95) for clear OFF states (<5% of rated power)  
                # - Lower confidence (0.45-0.75) for uncertain middle region
                import math
                if norm_power > 0.4:
                    # Clearly ON - confidence increases with power
                    confidence = 0.75 + 0.20 * min(norm_power, 1.0)
                elif norm_power < 0.05:
                    # Clearly OFF - high confidence for very low values
                    confidence = 0.80 + 0.15 * (1 - norm_power / 0.05)
                else:
                    # Uncertain region - use sigmoid-like curve centered at ~22%
                    midpoint = 0.225
                    distance_from_mid = abs(norm_power - midpoint) / midpoint
                    confidence = 0.45 + 0.30 * math.tanh(distance_from_mid * 2)

                results[field_key] = (predicted_kw, confidence)

            return results

        except Exception as e:
            status = "error"
            if isinstance(e, (ModelError, ValidationError)):
                raise
            raise ModelError(
                code=ErrorCode.INFERENCE_FAILED,
                message=f"Multi-head inference failed: {e}",
            )
        finally:
            duration = time.time() - start_time
            INFERENCE_LATENCY.labels(model_id=entry.model_id).observe(duration)
            INFERENCE_COUNT.labels(model_id=entry.model_id, status=status).inc()

    def get_loaded_models(self) -> list[str]:
        """Get list of loaded model IDs."""
        return [f"{mid}:{ver}" for mid, ver in self._model_cache.keys()]

    def clear_cache(self) -> int:
        """Clear model cache. Returns number of models cleared."""
        count = len(self._model_cache)
        self._model_cache.clear()
        MODEL_CACHE_SIZE.set(0)
        return count


# Global engine instance
_engine: InferenceEngine | None = None


def get_inference_engine() -> InferenceEngine:
    """Get the global inference engine instance."""
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine
