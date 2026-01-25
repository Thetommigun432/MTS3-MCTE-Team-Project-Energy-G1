"""
Integration tests for model loading with safetensors.
"""

import pytest
import torch
import numpy as np
from pathlib import Path


class TestModelLoading:
    """Tests for loading models from safetensors files."""
    
    @pytest.fixture
    def models_dir(self):
        """Get the models directory."""
        return Path(__file__).parent.parent.parent / "apps" / "backend" / "models"
    
    @pytest.fixture
    def registry_path(self, models_dir):
        """Get the registry.json path."""
        return models_dir / "registry.json"
    
    def test_registry_exists(self, registry_path):
        """Verify registry.json exists."""
        assert registry_path.exists(), f"Registry not found at {registry_path}"
    
    def test_registry_structure(self, registry_path):
        """Verify registry has expected structure."""
        import json
        with open(registry_path) as f:
            data = json.load(f)
        
        assert "models" in data
        # Expect dict or list
        models = data["models"]
        assert len(models) > 0, "Registry should have at least one model"
        
        # Check first model entry
        if isinstance(models, dict):
            entry = list(models.values())[0]
        else:
            entry = models[0]
            
        required_keys = ["model_id", "architecture", "appliance_id"]
        for key in required_keys:
            assert key in entry, f"Missing key: {key}"
    
    def test_safetensors_files_exist(self, models_dir, registry_path):
        """Verify all referenced safetensors files exist for ACTIVE models."""
        import json
        with open(registry_path) as f:
            data = json.load(f)
        
        models = data["models"]
        entries = models.values() if isinstance(models, dict) else models
        
        missing = []
        for entry in entries:
            # Only check active models
            if not entry.get("is_active", False):
                continue
            artifact_path = entry.get("artifact_path")
            if artifact_path:
                full_path = models_dir / artifact_path
                if not full_path.exists():
                    missing.append(str(full_path))
                    
        assert not missing, f"Missing artifacts: {missing}"
    
    def test_load_wavenilm_v3(self, models_dir, registry_path):
        """Test loading a WaveNILM v3 model from safetensors."""
        import json
        from safetensors.torch import load_file as load_safetensors
        
        with open(registry_path) as f:
            data = json.load(f)
        
        models = data["models"]
        entries = list(models.values()) if isinstance(models, dict) else models
        
        # Find a wavenilm_v3 entry that is active
        wavenilm_entry = None
        for entry in entries:
            if entry.get("architecture") == "wavenilm_v3" and entry.get("is_active"):
                wavenilm_entry = entry
                break
        
        if not wavenilm_entry:
            pytest.skip("No active wavenilm_v3 model in registry")
        
        # Load weights
        artifact_path = models_dir / wavenilm_entry["artifact_path"]
        if not artifact_path.exists():
            pytest.skip(f"Artifact not found: {artifact_path}")
            
        state_dict = load_safetensors(str(artifact_path))
        
        assert len(state_dict) > 0, "State dict should not be empty"
        
        # Check for expected layer keys (blocks is the main pattern)
        assert any("blocks" in k for k in state_dict.keys()), f"Missing blocks layer. Keys: {list(state_dict.keys())[:5]}"
    
    def test_model_inference(self, models_dir, registry_path):
        """Test running inference with a loaded model."""
        import json
        import sys
        
        # Add backend src to path
        backend_src = Path(__file__).parent.parent.parent / "apps" / "backend" / "src"
        sys.path.insert(0, str(backend_src))
        
        from safetensors.torch import load_file as load_safetensors
        from app.domain.inference.architectures.wavenilm_v3 import WaveNILM_v3
        
        with open(registry_path) as f:
            data = json.load(f)
        
        models = data["models"]
        entries = list(models.values()) if isinstance(models, dict) else models
        
        wavenilm_entry = None
        for entry in entries:
            if entry.get("architecture") == "wavenilm_v3":
                wavenilm_entry = entry
                break
        
        if not wavenilm_entry:
            pytest.skip("No wavenilm_v3 model in registry")
        
        # Get architecture params
        params = wavenilm_entry.get("architecture_params", {})
        n_blocks = params.get("n_blocks", 9)
        hidden_channels = params.get("hidden_channels", 64)
        window_size = wavenilm_entry.get("input_window_size", 1536)
        
        # Create model
        model = WaveNILM_v3(
            n_input_features=7,
            hidden_channels=hidden_channels,
            n_blocks=n_blocks,
        )
        
        # Load weights
        artifact_path = models_dir / wavenilm_entry["artifact_path"]
        state_dict = load_safetensors(str(artifact_path))
        
        # Try to load - may fail if architecture mismatch
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            pytest.fail(f"Failed to load state dict: {e}")
        
        model.eval()
        
        # Run inference with dummy data
        dummy_input = torch.randn(1, 7, window_size)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        # WaveNILM returns tuple (power, prob)
        assert isinstance(output, tuple), "Expected tuple output"
        power, prob = output
        
        assert power.shape == (1, window_size, 1), f"Unexpected power shape: {power.shape}"
        assert prob.shape == (1, window_size, 1), f"Unexpected prob shape: {prob.shape}"
        
        # Check values are reasonable
        assert not torch.isnan(power).any(), "Power contains NaN"
        assert not torch.isnan(prob).any(), "Prob contains NaN"
        assert (prob >= 0).all() and (prob <= 1).all(), "Prob should be in [0, 1]"
