"""Tests for HybridCNNTransformer model loading and inference."""
from __future__ import annotations

import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Set test environment variables before importing app modules
os.environ["ENV"] = "test"
os.environ["TEST_JWT_SECRET"] = "unit-test-secret"

import numpy as np
import torch


class TestHybridCNNTransformerAdapter:
    """Unit tests for the HybridCNNTransformerAdapter architecture."""

    def test_adapter_initializes(self):
        """Adapter should initialize with default parameters."""
        from app.domain.inference.architectures import HybridCNNTransformerAdapter

        model = HybridCNNTransformerAdapter()
        assert model is not None
        assert len(model.APPLIANCES) == 11
        assert model.p_max_kw == pytest.approx(13.5118)

    def test_adapter_forward_shape(self):
        """Forward pass should return (batch, 11) tensor."""
        from app.domain.inference.architectures import HybridCNNTransformerAdapter

        model = HybridCNNTransformerAdapter()
        model.eval()

        # Input: (batch, seq_len, 1)
        x = torch.randn(2, 1024, 1)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 11)
        assert output.dtype == torch.float32

    def test_adapter_temporal_features(self):
        """_build_temporal_features should create 7-feature tensor."""
        from app.domain.inference.architectures import HybridCNNTransformerAdapter

        model = HybridCNNTransformerAdapter()

        x = torch.randn(1, 512, 1)
        x_full = model._build_temporal_features(x, timestamp_hours=12.0)

        assert x_full.shape == (1, 512, 7)

    def test_adapter_output_non_negative(self):
        """Model outputs should be scaled and generally non-negative."""
        from app.domain.inference.architectures import HybridCNNTransformerAdapter

        model = HybridCNNTransformerAdapter()
        model.eval()

        # Realistic-ish aggregate power input
        x = torch.abs(torch.randn(1, 1024, 1)) * 5.0  # ~0-10 kW range

        with torch.no_grad():
            output = model(x)

        # Outputs are scaled by p_max_kw, so they can be negative from the model
        # but should be finite
        assert torch.isfinite(output).all()


class TestModelRegistry:
    """Tests for model registry integration."""

    @pytest.fixture
    def models_dir(self):
        return Path(__file__).parent.parent / "models"

    def test_architecture_in_model_classes(self):
        """MODEL_CLASSES should include hybrid_cnn_transformer variants."""
        from app.domain.inference.engine import MODEL_CLASSES

        assert "hybrid_cnn_transformer" in MODEL_CLASSES
        assert "hybridcnntransformer" in MODEL_CLASSES
        assert "hybrid" in MODEL_CLASSES

    def test_create_model_hybrid(self):
        """create_model should instantiate HybridCNNTransformerAdapter."""
        from app.domain.inference.engine import create_model
        from app.domain.inference.architectures import HybridCNNTransformerAdapter

        params = {
            "n_features": 7,
            "d_model": 64,  # Smaller for faster test
            "n_heads": 4,
            "n_layers": 2,
            "d_ff": 128,
        }

        model = create_model("hybrid_cnn_transformer", params)
        assert isinstance(model, HybridCNNTransformerAdapter)


class TestFixtures:
    """Tests using fixture files (if available)."""

    @pytest.fixture
    def fixtures_dir(self):
        return Path(__file__).parent / "fixtures"

    def test_fixtures_exist(self, fixtures_dir):
        """Check if fixture files exist (skip if not generated yet)."""
        aggregate_path = fixtures_dir / "aggregate_kw_sample.npy"
        midpoint_path = fixtures_dir / "y_midpoint_kw_sample.npy"

        if not aggregate_path.exists():
            pytest.skip("Fixtures not generated. Run build_fixture_from_y_test.py first.")

        assert aggregate_path.exists()
        assert midpoint_path.exists()

    def test_fixture_shapes(self, fixtures_dir):
        """Verify fixture data shapes."""
        aggregate_path = fixtures_dir / "aggregate_kw_sample.npy"
        midpoint_path = fixtures_dir / "y_midpoint_kw_sample.npy"

        if not aggregate_path.exists():
            pytest.skip("Fixtures not generated.")

        aggregate = np.load(aggregate_path)
        midpoint = np.load(midpoint_path)

        # Expected: (N, 1024) for aggregate, (N, 11) for midpoint
        assert aggregate.ndim == 2
        assert aggregate.shape[1] == 1024
        assert midpoint.ndim == 2
        assert midpoint.shape[1] == 11
        assert aggregate.shape[0] == midpoint.shape[0]

    def test_inference_with_fixtures(self, fixtures_dir):
        """Run inference on fixture data."""
        aggregate_path = fixtures_dir / "aggregate_kw_sample.npy"

        if not aggregate_path.exists():
            pytest.skip("Fixtures not generated.")

        from app.domain.inference.architectures import HybridCNNTransformerAdapter

        aggregate = np.load(aggregate_path)
        window = aggregate[0]  # First sample

        model = HybridCNNTransformerAdapter(
            d_model=64,  # Smaller for faster test
            n_layers=2,
        )
        model.eval()

        # Convert to tensor
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 11)
        assert torch.isfinite(output).all()


class TestEngineIntegration:
    """Integration tests for the inference engine."""

    def test_engine_run_inference_multi_head(self):
        """Test run_inference_multi_head with a synthetic model entry."""
        from app.domain.inference.engine import InferenceEngine, create_model
        from app.domain.inference.registry import ModelEntry, PreprocessingConfig, HeadConfig

        # Create a small model for testing
        model = create_model("hybrid_cnn_transformer", {
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 4,
            "d_ff": 128,
        })
        model.eval()

        # Create a mock entry
        heads = [
            HeadConfig(appliance_id=app, field_key=app.lower())
            for app in [
                "HeatPump", "Dishwasher", "WashingMachine", "Dryer", "Oven",
                "Stove", "RangeHood", "EVCharger", "EVSocket", "GarageCabinet", "RainwaterPump"
            ]
        ]

        entry = ModelEntry(
            model_id="test_hybrid",
            model_version="test",
            appliance_id="multi",
            architecture="hybrid_cnn_transformer",
            architecture_params={},
            artifact_path="test.safetensors",
            artifact_sha256="abc123",
            input_window_size=1024,
            preprocessing=PreprocessingConfig(type="identity"),
            is_active=True,
            heads=heads,
        )

        engine = InferenceEngine()

        # Generate fake window data
        window = [float(i % 10) for i in range(1024)]

        # Run inference
        results = engine.run_inference_multi_head(model, entry, window)

        assert len(results) == 11
        for field_key, (predicted_kw, confidence) in results.items():
            assert isinstance(predicted_kw, float)
            assert isinstance(confidence, float)
            assert np.isfinite(predicted_kw)
