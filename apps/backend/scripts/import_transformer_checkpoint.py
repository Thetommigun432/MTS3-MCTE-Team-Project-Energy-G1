#!/usr/bin/env python3
"""
Convert HybridCNNTransformer .pth checkpoint to .safetensors format.

Usage:
    python scripts/import_transformer_checkpoint.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from safetensors.torch import save_file

from scripts._asset_finder import find_transformer_zip
from app.domain.inference.architectures import HybridCNNTransformerAdapter

# Appliance configuration (must match model training order)
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

FIELD_KEYS = [
    "heatpump",
    "dishwasher",
    "washingmachine",
    "dryer",
    "oven",
    "stove",
    "rangehood",
    "evcharger",
    "evsocket",
    "garagecabinet",
    "rainwaterpump",
]


def main() -> None:
    print("=" * 60)
    print("Transformer Checkpoint Import")
    print("=" * 60)

    # 1. Find and extract checkpoint
    zip_path = find_transformer_zip()
    print(f"Found: {zip_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Extract the checkpoint file from the zip
        with zipfile.ZipFile(zip_path, "r") as zf:
            # List contents to find checkpoint path
            checkpoint_files = [
                n for n in zf.namelist() if n.endswith(".pth") and "checkpoint" in n.lower()
            ]
            if not checkpoint_files:
                # Try alternate paths
                checkpoint_files = [n for n in zf.namelist() if n.endswith(".pth")]

            if not checkpoint_files:
                raise FileNotFoundError("No .pth checkpoint found in zip")

            # Prefer the "best" checkpoint
            pth_name = next(
                (f for f in checkpoint_files if "best" in f.lower()),
                checkpoint_files[0],
            )

            print(f"Extracting: {pth_name}")
            zf.extract(pth_name, tmpdir_path)

        pth_path = tmpdir_path / pth_name
        print(f"Extracted: {pth_path}")

        # 2. Instantiate model with training config
        model = HybridCNNTransformerAdapter(
            n_features=7,
            d_model=256,
            n_heads=8,
            n_layers=6,
            d_ff=1024,
            dropout=0.1,
            cnn_channels=[64, 128, 256],
            cnn_kernel_sizes=[7, 5, 3],
            use_rope=True,
            seq2point=True,
            use_stationarization=True,
            use_pooling_for_state=True,
            p_max_kw=13.5118,
        )

        # 3. Load checkpoint safely
        print("Loading checkpoint...")
        checkpoint = torch.load(pth_path, map_location="cpu", weights_only=True)

        # Handle common checkpoint structures
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Adapt keys: the checkpoint may or may not have "model." prefix
        # Our adapter expects keys under "model." since it wraps HybridCNNTransformer
        cleaned: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                cleaned[k] = v
            else:
                cleaned[f"model.{k}"] = v

        # Load into model (strict=False to handle any minor mismatches)
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"Loaded checkpoint: {len(cleaned)} tensors")
        if missing:
            print(f"  Missing keys: {len(missing)} (first 5: {missing[:5]})")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")

        # 4. Save as safetensors
        models_dir = Path(__file__).parent.parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        out_path = models_dir / "transformer_hybrid_v1.safetensors"

        print(f"Saving safetensors to: {out_path}")
        save_file(model.state_dict(), str(out_path))

        # 5. Compute SHA256
        with open(out_path, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()
        print(f"SHA256: {sha256}")

        # 6. Update registry.json
        registry_path = models_dir / "registry.json"
        with open(registry_path) as f:
            registry = json.load(f)

        # Build heads config
        heads = [
            {"appliance_id": app, "field_key": fk}
            for app, fk in zip(APPLIANCES, FIELD_KEYS)
        ]

        new_entry = {
            "model_id": "transformer_hybrid_v1",
            "model_version": "2026-01-21",
            "appliance_id": "multi",  # Multi-head model
            "architecture": "hybrid_cnn_transformer",
            "architecture_params": {
                "n_features": 7,
                "d_model": 256,
                "n_heads": 8,
                "n_layers": 6,
                "d_ff": 1024,
                "dropout": 0.1,
                "cnn_channels": [64, 128, 256],
                "cnn_kernel_sizes": [7, 5, 3],
                "use_rope": True,
                "seq2point": True,
                "use_stationarization": True,
                "use_pooling_for_state": True,
                "p_max_kw": 13.5118,
            },
            "artifact_path": "transformer_hybrid_v1.safetensors",
            "artifact_sha256": sha256,
            "input_window_size": 1024,
            "preprocessing": {"type": "identity"},
            "is_active": True,
            "heads": heads,
        }

        # Check if entry already exists
        existing_idx = next(
            (i for i, m in enumerate(registry["models"]) if m["model_id"] == "transformer_hybrid_v1"),
            None,
        )

        if existing_idx is not None:
            # Update existing entry
            registry["models"][existing_idx] = new_entry
            print("Updated existing registry entry")
        else:
            # Set other models to inactive (new model becomes primary)
            for m in registry["models"]:
                m["is_active"] = False
            registry["models"].append(new_entry)
            print("Added new registry entry")

        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=4)

        print(f"Updated: {registry_path}")
        print("=" * 60)
        print("SUCCESS")
        print("=" * 60)


if __name__ == "__main__":
    main()
