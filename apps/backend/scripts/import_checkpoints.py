
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import warnings
import numpy as np
import torch
from safetensors.torch import save_file

# Import registry types (assuming available in path)
import sys
# Add apps/backend/src to path so 'app' module is resolvable
backend_src = Path.cwd() / "apps" / "backend" / "src"
sys.path.append(str(backend_src))

from app.domain.inference.architectures.tcn_gated import TCN_Gated

def compute_sha256(path: Path) -> str:
    hash_sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def strict_load(path: Path):
    """
    Load checkpoint with weights_only=False locally to extract metadata,
    since we trust the provided artifacts for this operation.
    """
    # Note: In a fully untrusted environment, we would use a restricted Unpickler.
    # Here we assume the user provided valid model files.
    return torch.load(path, map_location='cpu', weights_only=False)

def convert_checkpoint(ckpt_path: Path, output_dir: Path, registry_data: dict):
    print(f"Processing {ckpt_path.name}...")
    
    try:
        data = strict_load(ckpt_path)
    except Exception as e:
        print(f"❌ Failed to load {ckpt_path}: {e}")
        return

    # Extract metadata
    meta = {}
    state_dict = None
    
    if 'model' in data:
        state_dict = data['model']
    elif 'state_dict' in data:
        state_dict = data['state_dict']
    else:
        state_dict = data
        
    # Infer or get architecture params
    n_blocks = data.get('n_blocks', 8)
    hidden_channels = data.get('hidden_channels', 128)
    window = data.get('window', 1536)
    optimal_threshold = float(data.get('optimal_threshold', 0.5))
    
    metadata = data.get('scaling', {})
    p_max = float(metadata.get('P_MAX', 15.0)) if 'P_MAX' in metadata else 15.0
    
    # Infer appliance from filename (e.g. TCN_SA_HeatPump_best.pt)
    appliance = "unknown"
    parts = ckpt_path.stem.split('_')
    for p in parts:
        if p.lower() in ['heatpump', 'dishwasher', 'washingmachine', 'dryer', 'evcharger']:
            appliance = p
            break
            
    if appliance == 'unknown':
        print(f"⚠️ Could not infer appliance from {ckpt_path.name}, skipping.")
        return

    # Create output directory
    model_version = "v1-sota"
    model_id = f"{appliance.lower()}-{model_version}"
    save_dir = output_dir / "tcn_sa" / appliance.lower() / model_version
    save_dir.mkdir(parents=True, exist_ok=True)
    
    safetensors_path = save_dir / "model.safetensors"
    
    # Save as safetensors
    # Remove 'module.' prefix if from distributed training
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        clean_state_dict[k] = v.contiguous()
        
    save_file(clean_state_dict, safetensors_path)
    print(f"  [OK] Saved safetensors to {safetensors_path}")
    
    # Compute SHA
    sha = compute_sha256(safetensors_path)
    
    # Registry Entry
    entry = {
        "model_id": model_id,
        "model_version": model_version,
        "appliance_id": appliance,
        "architecture": "tcn_sa",
        "architecture_params": {
            "n_blocks": n_blocks,
            "hidden_channels": hidden_channels,
            "n_stacks": 2, # Default
            "window": window,
            "p_max_kw": p_max,
            "optimal_threshold": optimal_threshold
        },
        "artifact_path": str(safetensors_path.relative_to(output_dir)),
        "artifact_sha256": sha,
        "input_window_size": window,
        "preprocessing": {
            "type": "standard",
            "max": p_max
        },
        "is_active": True
    }
    
    registry_data["models"][model_id] = entry
    print(f"  [OK] Added to registry as {model_id}")

def main():
    parser = argparse.ArgumentParser(description="Import TCN_SA checkpoints")
    parser.add_argument("--src", type=str, default="checkpoints", help="Source directory containing .pth files")
    parser.add_argument("--dest", type=str, default="apps/backend/models", help="Destination storage directory")
    parser.add_argument("--registry", type=str, default="apps/backend/models/registry.json", help="Registry JSON path")
    
    args = parser.parse_args()
    
    src_dir = Path(args.src)
    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    registry_path = Path(args.registry)
    if registry_path.exists():
        with open(registry_path, 'r') as f:
            registry_data = json.load(f)
            # Ensure 'models' is a dict
            if isinstance(registry_data.get("models"), list):
                new_models = {}
                for m in registry_data["models"]:
                    if "model_id" in m:
                        new_models[m["model_id"]] = m
                registry_data["models"] = new_models
            elif "models" not in registry_data:
                registry_data["models"] = {}
    else:
        registry_data = {"models": {}}
        
    # Process all .pth files
    for ckpt in src_dir.glob("*.pth"):
        convert_checkpoint(ckpt, dest_dir, registry_data)
        
    # Save registry
    with open(registry_path, 'w') as f:
        json.dump(registry_data, f, indent=2)
    print(f"\nUpdated registry at {registry_path}")

if __name__ == "__main__":
    main()
