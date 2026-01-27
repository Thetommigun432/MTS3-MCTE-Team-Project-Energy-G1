"""
Unit tests for the model registry.

Tests registry parsing, validation, and multi-head model support.
"""

import hashlib
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.domain.inference.registry import (
    ModelRegistry,
    ModelEntry,
    HeadConfig,
    PreprocessingConfig,
)
from app.core.errors import ModelError


@pytest.fixture
def temp_registry_dir(tmp_path):
    """Create a temporary directory for registry tests."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return tmp_path, models_dir


@pytest.fixture
def sample_artifact(temp_registry_dir):
    """Create a sample model artifact file."""
    _, models_dir = temp_registry_dir
    artifact_path = models_dir / "model.safetensors"
    artifact_content = b"fake safetensors content for testing"
    artifact_path.write_bytes(artifact_content)

    # Compute actual SHA256
    sha256 = hashlib.sha256(artifact_content).hexdigest()
    return artifact_path, sha256


@pytest.fixture
def mock_settings(temp_registry_dir):
    """Mock settings to use temp directory."""
    tmp_path, models_dir = temp_registry_dir

    mock_settings_obj = MagicMock()
    mock_settings_obj.model_registry_path = str(tmp_path / "registry.json")
    mock_settings_obj.models_dir = str(models_dir)

    with patch("app.domain.inference.registry.get_settings", return_value=mock_settings_obj):
        yield mock_settings_obj


class TestModelRegistryParsing:
    """Tests for registry JSON parsing."""

    @pytest.mark.unit
    def test_load_empty_registry(self, temp_registry_dir, mock_settings):
        """Empty registry file loads successfully with no models."""
        tmp_path, _ = temp_registry_dir
        registry_file = tmp_path / "registry.json"
        registry_file.write_text('{"models": {}}')

        registry = ModelRegistry()
        registry.load(str(registry_file), mock_settings.models_dir)

        assert registry.is_loaded
        assert len(registry.list_all()) == 0

    @pytest.mark.unit
    def test_load_nonexistent_registry(self, temp_registry_dir, mock_settings):
        """Missing registry file loads as empty with warning."""
        registry = ModelRegistry()
        registry.load(str(temp_registry_dir[0] / "nonexistent.json"), mock_settings.models_dir)

        assert registry.is_loaded
        assert len(registry.list_all()) == 0

    @pytest.mark.unit
    def test_load_single_model_dict_format(self, temp_registry_dir, mock_settings, sample_artifact):
        """Registry with dictionary format parses correctly."""
        tmp_path, models_dir = temp_registry_dir
        artifact_path, sha256 = sample_artifact

        registry_data = {
            "models": {
                "heatpump_v1": {
                    "model_version": "1.0.0",
                    "appliance_id": "heatpump",
                    "architecture": "CNNTransformer",
                    "architecture_params": {"d_model": 128},
                    "artifact_path": "model.safetensors",
                    "artifact_sha256": sha256,
                    "input_window_size": 1000,
                    "is_active": True,
                }
            }
        }

        registry_file = tmp_path / "registry.json"
        registry_file.write_text(json.dumps(registry_data))

        registry = ModelRegistry()
        registry.load(str(registry_file), str(models_dir))

        assert registry.is_loaded
        assert len(registry.list_all()) == 1

        entry = registry.get("heatpump_v1")
        assert entry is not None
        assert entry.model_id == "heatpump_v1"
        assert entry.model_version == "1.0.0"
        assert entry.appliance_id == "heatpump"
        assert entry.architecture == "CNNTransformer"
        assert entry.is_active is True

    @pytest.mark.unit
    def test_load_array_format(self, temp_registry_dir, mock_settings, sample_artifact):
        """Registry with array format parses correctly."""
        tmp_path, models_dir = temp_registry_dir
        artifact_path, sha256 = sample_artifact

        registry_data = {
            "models": [
                {
                    "model_id": "model_array_1",
                    "model_version": "1.0.0",
                    "appliance_id": "dishwasher",
                    "architecture": "UNet1D",
                    "artifact_path": "model.safetensors",
                    "artifact_sha256": sha256,
                    "input_window_size": 500,
                }
            ]
        }

        registry_file = tmp_path / "registry.json"
        registry_file.write_text(json.dumps(registry_data))

        registry = ModelRegistry()
        registry.load(str(registry_file), str(models_dir))

        assert len(registry.list_all()) == 1
        entry = registry.get("model_array_1")
        assert entry is not None
        assert entry.appliance_id == "dishwasher"

    @pytest.mark.unit
    def test_missing_required_field_raises_error(self, temp_registry_dir, mock_settings):
        """Missing required field raises ModelError."""
        tmp_path, models_dir = temp_registry_dir

        registry_data = {
            "models": {
                "bad_model": {
                    "model_version": "1.0.0",
                    # Missing: appliance_id, architecture, artifact_path, artifact_sha256, input_window_size
                }
            }
        }

        registry_file = tmp_path / "registry.json"
        registry_file.write_text(json.dumps(registry_data))

        registry = ModelRegistry()
        with pytest.raises(ModelError) as exc_info:
            registry.load(str(registry_file), str(models_dir))

        assert "missing required fields" in str(exc_info.value.message).lower()

    @pytest.mark.unit
    def test_invalid_json_raises_error(self, temp_registry_dir, mock_settings):
        """Invalid JSON raises ModelError."""
        tmp_path, models_dir = temp_registry_dir

        registry_file = tmp_path / "registry.json"
        registry_file.write_text('{"models": {invalid json}')

        registry = ModelRegistry()
        with pytest.raises(ModelError) as exc_info:
            registry.load(str(registry_file), str(models_dir))

        assert "invalid json" in str(exc_info.value.message).lower()


class TestMultiHeadModels:
    """Tests for multi-head model configurations."""

    @pytest.mark.unit
    def test_multi_head_parsing(self, temp_registry_dir, mock_settings, sample_artifact):
        """Multi-head model with heads list parses correctly."""
        tmp_path, models_dir = temp_registry_dir
        _, sha256 = sample_artifact

        registry_data = {
            "models": {
                "multi_v1": {
                    "model_version": "2.0.0",
                    "appliance_id": "multi",
                    "architecture": "tcn_sa",
                    "artifact_path": "model.safetensors",
                    "artifact_sha256": sha256,
                    "input_window_size": 1000,
                    "is_active": True,
                    "heads": [
                        {"appliance_id": "HeatPump", "field_key": "HeatPump"},
                        {"appliance_id": "Dishwasher", "field_key": "Dishwasher"},
                        {"appliance_id": "WashingMachine", "field_key": "WashingMachine"},
                    ]
                }
            }
        }

        registry_file = tmp_path / "registry.json"
        registry_file.write_text(json.dumps(registry_data))

        registry = ModelRegistry()
        registry.load(str(registry_file), str(models_dir))

        entry = registry.get("multi_v1")
        assert entry is not None
        assert len(entry.heads) == 3
        assert entry.heads[0].appliance_id == "HeatPump"
        assert entry.heads[1].field_key == "Dishwasher"

    @pytest.mark.unit
    def test_is_multi_head_property(self):
        """is_multi_head returns True when heads > 1."""
        entry = ModelEntry(
            model_id="test",
            model_version="1.0",
            appliance_id="multi",
            architecture="test",
            architecture_params={},
            artifact_path="test.safetensors",
            artifact_sha256="abc123",
            input_window_size=100,
            preprocessing=PreprocessingConfig(type="identity"),
            heads=[
                HeadConfig(appliance_id="a", field_key="a"),
                HeadConfig(appliance_id="b", field_key="b"),
            ]
        )
        entry._resolved_path = Path("/fake/path")

        assert entry.is_multi_head is True
        assert entry.head_appliances == ["a", "b"]
        assert entry.head_field_keys == ["a", "b"]

    @pytest.mark.unit
    def test_single_head_is_not_multi_head(self):
        """Single-head model is_multi_head returns False."""
        entry = ModelEntry(
            model_id="test",
            model_version="1.0",
            appliance_id="single",
            architecture="test",
            architecture_params={},
            artifact_path="test.safetensors",
            artifact_sha256="abc123",
            input_window_size=100,
            preprocessing=PreprocessingConfig(type="identity"),
            heads=[HeadConfig(appliance_id="single", field_key="single")]
        )
        entry._resolved_path = Path("/fake/path")

        assert entry.is_multi_head is False

    @pytest.mark.unit
    def test_single_head_fallback(self, temp_registry_dir, mock_settings, sample_artifact):
        """Single-head model without heads list creates default head from appliance_id."""
        tmp_path, models_dir = temp_registry_dir
        _, sha256 = sample_artifact

        registry_data = {
            "models": {
                "single_v1": {
                    "model_version": "1.0.0",
                    "appliance_id": "fridge",
                    "architecture": "CNNSeq2Seq",
                    "artifact_path": "model.safetensors",
                    "artifact_sha256": sha256,
                    "input_window_size": 500,
                    # No "heads" field - should use appliance_id as single head
                }
            }
        }

        registry_file = tmp_path / "registry.json"
        registry_file.write_text(json.dumps(registry_data))

        registry = ModelRegistry()
        registry.load(str(registry_file), str(models_dir))

        entry = registry.get("single_v1")
        assert entry is not None
        assert len(entry.heads) == 1
        assert entry.heads[0].appliance_id == "fridge"
        assert entry.heads[0].field_key == "fridge"
        assert entry.is_multi_head is False

    @pytest.mark.unit
    def test_head_config_defaults_field_key(self):
        """HeadConfig defaults field_key to appliance_id if not provided."""
        head = HeadConfig.from_dict({"appliance_id": "test_appliance"})

        assert head.appliance_id == "test_appliance"
        assert head.field_key == "test_appliance"


class TestRegistryValidation:
    """Tests for registry validation."""

    @pytest.mark.unit
    def test_missing_artifact_reports_error(self, temp_registry_dir, mock_settings):
        """Validation reports missing artifact files."""
        tmp_path, models_dir = temp_registry_dir

        registry_data = {
            "models": {
                "missing_v1": {
                    "model_version": "1.0.0",
                    "appliance_id": "missing",
                    "architecture": "CNNTransformer",
                    "artifact_path": "nonexistent.safetensors",
                    "artifact_sha256": "abc123",
                    "input_window_size": 100,
                }
            }
        }

        registry_file = tmp_path / "registry.json"
        registry_file.write_text(json.dumps(registry_data))

        registry = ModelRegistry()
        registry.load(str(registry_file), str(models_dir))

        errors = registry.validate()

        assert len(errors) == 1
        assert "artifact not found" in errors[0].lower()

    @pytest.mark.unit
    def test_sha256_mismatch_reports_error(self, temp_registry_dir, mock_settings, sample_artifact):
        """Validation reports SHA256 mismatch."""
        tmp_path, models_dir = temp_registry_dir

        registry_data = {
            "models": {
                "wrong_hash_v1": {
                    "model_version": "1.0.0",
                    "appliance_id": "test",
                    "architecture": "CNNTransformer",
                    "artifact_path": "model.safetensors",
                    "artifact_sha256": "wrong_sha256_hash_intentionally_incorrect",
                    "input_window_size": 100,
                }
            }
        }

        registry_file = tmp_path / "registry.json"
        registry_file.write_text(json.dumps(registry_data))

        registry = ModelRegistry()
        registry.load(str(registry_file), str(models_dir))

        errors = registry.validate()

        assert len(errors) == 1
        assert "sha256 mismatch" in errors[0].lower()

    @pytest.mark.unit
    def test_valid_artifact_no_errors(self, temp_registry_dir, mock_settings, sample_artifact):
        """Valid artifact with correct SHA256 passes validation."""
        tmp_path, models_dir = temp_registry_dir
        _, sha256 = sample_artifact

        registry_data = {
            "models": {
                "valid_v1": {
                    "model_version": "1.0.0",
                    "appliance_id": "test",
                    "architecture": "CNNTransformer",
                    "artifact_path": "model.safetensors",
                    "artifact_sha256": sha256,
                    "input_window_size": 100,
                    "is_active": True,
                }
            }
        }

        registry_file = tmp_path / "registry.json"
        registry_file.write_text(json.dumps(registry_data))

        registry = ModelRegistry()
        registry.load(str(registry_file), str(models_dir))

        errors = registry.validate()

        assert len(errors) == 0


class TestRegistryQueries:
    """Tests for registry query methods."""

    @pytest.mark.unit
    def test_get_active_for_appliance(self, temp_registry_dir, mock_settings, sample_artifact):
        """get_active_for_appliance returns the active model."""
        tmp_path, models_dir = temp_registry_dir
        _, sha256 = sample_artifact

        registry_data = {
            "models": {
                "v1": {
                    "model_version": "1.0.0",
                    "appliance_id": "fridge",
                    "architecture": "A",
                    "artifact_path": "model.safetensors",
                    "artifact_sha256": sha256,
                    "input_window_size": 100,
                    "is_active": False,
                },
                "v2": {
                    "model_version": "2.0.0",
                    "appliance_id": "fridge",
                    "architecture": "B",
                    "artifact_path": "model.safetensors",
                    "artifact_sha256": sha256,
                    "input_window_size": 100,
                    "is_active": True,
                },
            }
        }

        registry_file = tmp_path / "registry.json"
        registry_file.write_text(json.dumps(registry_data))

        registry = ModelRegistry()
        registry.load(str(registry_file), str(models_dir))

        active = registry.get_active_for_appliance("fridge")
        assert active is not None
        assert active.model_id == "v2"

    @pytest.mark.unit
    def test_get_active_returns_none_if_none_active(self, temp_registry_dir, mock_settings, sample_artifact):
        """get_active_for_appliance returns None if no active model."""
        tmp_path, models_dir = temp_registry_dir
        _, sha256 = sample_artifact

        registry_data = {
            "models": {
                "v1": {
                    "model_version": "1.0.0",
                    "appliance_id": "test",
                    "architecture": "A",
                    "artifact_path": "model.safetensors",
                    "artifact_sha256": sha256,
                    "input_window_size": 100,
                    "is_active": False,
                },
            }
        }

        registry_file = tmp_path / "registry.json"
        registry_file.write_text(json.dumps(registry_data))

        registry = ModelRegistry()
        registry.load(str(registry_file), str(models_dir))

        active = registry.get_active_for_appliance("test")
        assert active is None

    @pytest.mark.unit
    def test_get_models_for_appliance(self, temp_registry_dir, mock_settings, sample_artifact):
        """get_models_for_appliance returns all models for appliance."""
        tmp_path, models_dir = temp_registry_dir
        _, sha256 = sample_artifact

        registry_data = {
            "models": {
                "v1": {
                    "model_version": "1.0.0",
                    "appliance_id": "fridge",
                    "architecture": "A",
                    "artifact_path": "model.safetensors",
                    "artifact_sha256": sha256,
                    "input_window_size": 100,
                },
                "v2": {
                    "model_version": "2.0.0",
                    "appliance_id": "fridge",
                    "architecture": "B",
                    "artifact_path": "model.safetensors",
                    "artifact_sha256": sha256,
                    "input_window_size": 100,
                },
                "other": {
                    "model_version": "1.0.0",
                    "appliance_id": "dishwasher",
                    "architecture": "C",
                    "artifact_path": "model.safetensors",
                    "artifact_sha256": sha256,
                    "input_window_size": 100,
                },
            }
        }

        registry_file = tmp_path / "registry.json"
        registry_file.write_text(json.dumps(registry_data))

        registry = ModelRegistry()
        registry.load(str(registry_file), str(models_dir))

        fridge_models = registry.get_models_for_appliance("fridge")
        assert len(fridge_models) == 2

        dishwasher_models = registry.get_models_for_appliance("dishwasher")
        assert len(dishwasher_models) == 1


class TestPreprocessingConfig:
    """Tests for preprocessing configuration parsing."""

    @pytest.mark.unit
    def test_preprocessing_identity_default(self):
        """Default preprocessing is identity."""
        config = PreprocessingConfig.from_dict({})
        assert config.type == "identity"

    @pytest.mark.unit
    def test_preprocessing_standard(self):
        """Standard preprocessing parses mean and std."""
        config = PreprocessingConfig.from_dict({
            "type": "standard",
            "mean": 0.5,
            "std": 0.2,
        })

        assert config.type == "standard"
        assert config.mean == 0.5
        assert config.std == 0.2

    @pytest.mark.unit
    def test_preprocessing_minmax(self):
        """Minmax preprocessing parses min and max."""
        config = PreprocessingConfig.from_dict({
            "type": "minmax",
            "min": 0.0,
            "max": 100.0,
        })

        assert config.type == "minmax"
        assert config.min_val == 0.0
        assert config.max_val == 100.0

    @pytest.mark.unit
    def test_preprocessing_list_values(self):
        """Preprocessing supports list values for multi-channel data."""
        config = PreprocessingConfig.from_dict({
            "type": "standard",
            "mean": [0.5, 0.6, 0.7],
            "std": [0.1, 0.2, 0.3],
        })

        assert config.mean == [0.5, 0.6, 0.7]
        assert config.std == [0.1, 0.2, 0.3]
