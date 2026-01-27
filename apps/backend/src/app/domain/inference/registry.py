"""
Model registry for managing ML model metadata and artifacts.
Supports safetensors-only loading with SHA256 verification.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.errors import ErrorCode, ModelError
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration for a model."""

    type: str  # "standard", "minmax", "identity"
    mean: float | list[float] | None = None
    std: float | list[float] | None = None
    min_val: float | None = None
    max_val: float | None = None
    p_max_kw: float | None = None  # P_MAX for de-normalizing TCN_SA output
    agg_p95: float | None = None   # 95th percentile of aggregate (W) for input normalization

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreprocessingConfig":
        """Create from dictionary."""
        return cls(
            type=data.get("type", "identity"),
            mean=data.get("mean"),
            std=data.get("std"),
            min_val=data.get("min"),
            max_val=data.get("max"),
            p_max_kw=data.get("p_max_kw"),
            agg_p95=data.get("agg_p95"),
        )


@dataclass
class HeadConfig:
    """
    Configuration for a single output head in multi-head models.
    
    Attributes:
        appliance_id: Appliance identifier (e.g., "fridge")
        field_key: Safe identifier for InfluxDB fields (e.g., "fridge")
    """
    appliance_id: str
    field_key: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HeadConfig":
        """Create from dictionary."""
        return cls(
            appliance_id=data["appliance_id"],
            field_key=data.get("field_key", data["appliance_id"]),
        )


@dataclass
class ModelEntry:
    """
    A model entry in the registry.
    
    Supports both single-head (legacy) and multi-head models.
    Multi-head models have a `heads` list with output configurations.
    """

    model_id: str
    model_version: str
    appliance_id: str  # Primary appliance for single-head, or first head for multi-head
    architecture: str
    architecture_params: dict[str, Any]
    artifact_path: str
    input_window_size: int
    preprocessing: PreprocessingConfig
    artifact_sha256: str | None = None  # Optional - skip verification if not present
    is_active: bool = False
    heads: list[HeadConfig] = field(default_factory=list)  # Multi-head config

    # Resolved absolute path (set during validation)
    _resolved_path: Path | None = field(default=None, repr=False)

    @property
    def resolved_path(self) -> Path:
        """Get the resolved absolute path to the artifact."""
        if self._resolved_path is None:
            raise ValueError("Model entry not validated - call validate() first")
        return self._resolved_path

    @property
    def is_multi_head(self) -> bool:
        """Check if this is a multi-head model."""
        return len(self.heads) > 1

    @property
    def head_appliances(self) -> list[str]:
        """Get list of appliance IDs for all heads."""
        if not self.heads:
            return [self.appliance_id]
        return [h.appliance_id for h in self.heads]

    @property
    def head_field_keys(self) -> list[str]:
        """Get list of field keys for all heads."""
        if not self.heads:
            return [self.appliance_id]
        return [h.field_key for h in self.heads]


class ModelRegistry:
    """Registry for managing ML models."""

    def __init__(self) -> None:
        self._entries: dict[str, ModelEntry] = {}
        self._by_appliance: dict[str, list[str]] = {}
        self._active_by_appliance: dict[str, str] = {}
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if registry is loaded."""
        return self._loaded

    def load(self, registry_path: str | None = None, models_dir: str | None = None) -> None:
        """
        Load model registry from JSON file.

        Args:
            registry_path: Path to registry.json (default from config)
            models_dir: Base directory for model artifacts (default from config)
        """
        settings = get_settings()
        registry_path = registry_path or settings.model_registry_path
        models_dir = models_dir or settings.models_dir

        path = Path(registry_path)
        if not path.exists():
            logger.warning("Model registry not found", extra={"path": str(path)})
            self._loaded = True
            return

        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ModelError(
                code=ErrorCode.REGISTRY_ERROR,
                message=f"Invalid JSON in model registry: {e}",
            )

        # Parse models section
        models = data.get("models", {})
        if isinstance(models, list):
            # Array format
            for entry_data in models:
                self._parse_entry(entry_data, models_dir)
        else:
            # Dictionary format (model_id -> entry)
            for model_id, entry_data in models.items():
                entry_data["model_id"] = entry_data.get("model_id", model_id)
                self._parse_entry(entry_data, models_dir)

        self._loaded = True
        logger.info(
            "Model registry loaded",
            extra={"models_count": len(self._entries)},
        )

    def _parse_entry(self, data: dict[str, Any], models_dir: str) -> None:
        """Parse a single registry entry."""
        required_fields = [
            "model_id", "model_version", "appliance_id", "architecture",
            "artifact_path", "input_window_size",
        ]

        # Check required fields
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ModelError(
                code=ErrorCode.REGISTRY_ERROR,
                message=f"Model entry missing required fields: {missing}",
                details={"entry": data.get("model_id", "unknown")},
            )

        # Parse preprocessing
        preprocessing_data = data.get("preprocessing", {"type": "identity"})
        preprocessing = PreprocessingConfig.from_dict(preprocessing_data)

        entry = ModelEntry(
            model_id=data["model_id"],
            model_version=data["model_version"],
            appliance_id=data["appliance_id"],
            architecture=data["architecture"],
            architecture_params=data.get("architecture_params", {}),
            artifact_path=data["artifact_path"],
            input_window_size=data["input_window_size"],
            artifact_sha256=data.get("artifact_sha256"),
            preprocessing=preprocessing,
            is_active=data.get("is_active", False),
            heads=[
                HeadConfig.from_dict(h) for h in data.get("heads", [])
            ],
        )

        # Backward compat: if no heads, create single head from appliance_id
        if not entry.heads:
            entry.heads = [HeadConfig(appliance_id=entry.appliance_id, field_key=entry.appliance_id)]

        # Resolve artifact path
        artifact_path = Path(data["artifact_path"])
        if not artifact_path.is_absolute():
            artifact_path = Path(models_dir) / artifact_path
        entry._resolved_path = artifact_path

        # Store entry
        self._entries[entry.model_id] = entry

        # Index by appliance
        if entry.appliance_id not in self._by_appliance:
            self._by_appliance[entry.appliance_id] = []
        self._by_appliance[entry.appliance_id].append(entry.model_id)

        # Track active model
        if entry.is_active:
            if entry.appliance_id in self._active_by_appliance:
                logger.warning(
                    "Multiple active models for appliance",
                    extra={"appliance_id": entry.appliance_id},
                )
            self._active_by_appliance[entry.appliance_id] = entry.model_id

    def validate(self) -> list[str]:
        """
        Validate all registry entries.

        Returns:
            List of validation error messages
        """
        errors: list[str] = []

        for model_id, entry in self._entries.items():
            # Check artifact exists
            if not entry.resolved_path.exists():
                errors.append(f"Model {model_id}: artifact not found at {entry.resolved_path}")
                continue

            # Check SHA256 (skip if not provided)
            if entry.artifact_sha256:
                actual_sha256 = self._compute_sha256(entry.resolved_path)
                if actual_sha256 != entry.artifact_sha256:
                    errors.append(
                        f"Model {model_id}: SHA256 mismatch "
                        f"(expected {entry.artifact_sha256[:16]}..., got {actual_sha256[:16]}...)"
                    )

        # Check for appliances without active model
        for appliance_id, model_ids in self._by_appliance.items():
            if appliance_id not in self._active_by_appliance:
                logger.warning(
                    "No active model for appliance",
                    extra={"appliance_id": appliance_id},
                )

        return errors

    def _compute_sha256(self, path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def get(self, model_id: str) -> ModelEntry | None:
        """Get a model entry by ID."""
        return self._entries.get(model_id)

    def get_active_for_appliance(self, appliance_id: str) -> ModelEntry | None:
        """Get the active model for an appliance."""
        model_id = self._active_by_appliance.get(appliance_id)
        if model_id:
            return self._entries.get(model_id)
        return None

    def get_models_for_appliance(self, appliance_id: str) -> list[ModelEntry]:
        """Get all models for an appliance."""
        model_ids = self._by_appliance.get(appliance_id, [])
        return [self._entries[mid] for mid in model_ids if mid in self._entries]

    def list_all(self) -> list[ModelEntry]:
        """List all model entries."""
        return list(self._entries.values())

    def reload(self) -> None:
        """Reload the registry from disk."""
        self._entries.clear()
        self._by_appliance.clear()
        self._active_by_appliance.clear()
        self._loaded = False
        self.load()


# Global registry instance
_registry: ModelRegistry | None = None


def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def init_model_registry() -> ModelRegistry:
    """Initialize and load the global model registry."""
    registry = get_model_registry()
    if not registry.is_loaded:
        registry.load()
    return registry
