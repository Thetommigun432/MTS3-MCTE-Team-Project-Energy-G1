"""
NILM Energy Monitor - Inference Service
FastAPI service for running PyTorch model inference
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
from typing import List, Dict, Any
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NILM Inference Service",
    description="PyTorch model inference for appliance power disaggregation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model registry
MODEL_REGISTRY_PATH = Path("model_registry.json")
MODEL_REGISTRY: Dict[str, Any] = {}
MODEL_CACHE: Dict[str, tuple] = {}

def load_registry():
    """Load model registry from JSON file"""
    global MODEL_REGISTRY
    try:
        with open(MODEL_REGISTRY_PATH) as f:
            MODEL_REGISTRY = json.load(f)
        logger.info(f"Loaded model registry with {len(MODEL_REGISTRY)} models")
    except FileNotFoundError:
        logger.error(f"Model registry not found at {MODEL_REGISTRY_PATH}")
        MODEL_REGISTRY = {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in model registry: {e}")
        MODEL_REGISTRY = {}

# Load registry on startup
load_registry()


class InferenceRequest(BaseModel):
    """Request model for inference endpoint"""
    appliance_id: str
    aggregate_data: List[float]  # 60 timesteps of aggregate power
    model_version: str = "latest"

    class Config:
        json_schema_extra = {
            "example": {
                "appliance_id": "heatpump",
                "aggregate_data": [1.2] * 60,
                "model_version": "latest"
            }
        }


class InferenceResponse(BaseModel):
    """Response model for inference endpoint"""
    predicted_kw: float
    confidence: float
    model_version: str

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_kw": 0.45,
                "confidence": 0.85,
                "model_version": "v1.0.0"
            }
        }


def load_model(appliance_id: str) -> tuple:
    """
    Load PyTorch model from registry

    Args:
        appliance_id: Identifier for the appliance (e.g., "heatpump")

    Returns:
        Tuple of (model, config)

    Raises:
        HTTPException: If model not found or loading fails
    """
    # Check if model exists in registry
    if appliance_id not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Model for appliance '{appliance_id}' not found in registry"
        )

    # Return cached model if available
    if appliance_id in MODEL_CACHE:
        logger.info(f"Using cached model for {appliance_id}")
        return MODEL_CACHE[appliance_id]

    config = MODEL_REGISTRY[appliance_id]
    model_path = Path(config["model_path"])

    # Check if model file exists
    if not model_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Model file not found: {model_path}"
        )

    logger.info(f"Loading model for {appliance_id} from {model_path}")

    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        # Extract model from checkpoint
        # PyTorch checkpoints can have different structures:
        # 1. Direct state_dict
        # 2. Dictionary with 'model' key
        # 3. Dictionary with 'model_state_dict' key
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                # Need to instantiate architecture first
                # For now, just use the state_dict
                model = checkpoint['model_state_dict']
            else:
                model = checkpoint
        else:
            model = checkpoint

        # Ensure model is in eval mode
        if hasattr(model, 'eval'):
            model.eval()

        # Cache the loaded model
        MODEL_CACHE[appliance_id] = (model, config)
        logger.info(f"Successfully loaded and cached model for {appliance_id}")

        return model, config

    except Exception as e:
        logger.error(f"Failed to load model for {appliance_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )


def preprocess_input(data: List[float], config: Dict[str, Any]) -> torch.Tensor:
    """
    Preprocess input data for model inference

    Args:
        data: List of 60 aggregate power values
        config: Model configuration with normalization params

    Returns:
        Preprocessed PyTorch tensor ready for inference
    """
    # Convert to numpy array
    data_array = np.array(data, dtype=np.float32)

    # Normalize if params provided
    if "normalization_params" in config:
        mean = np.array(config["normalization_params"]["mean"], dtype=np.float32)
        std = np.array(config["normalization_params"]["std"], dtype=np.float32)
        data_array = (data_array - mean[0]) / std[0]  # Use first element for scalar normalization

    # Convert to tensor and add batch dimension
    # Shape: (batch_size=1, sequence_length, features)
    tensor = torch.FloatTensor(data_array).unsqueeze(0).unsqueeze(-1)

    # Expand to match expected feature count if needed
    input_features = config.get("input_features", 7)
    if tensor.shape[-1] != input_features:
        tensor = tensor.repeat(1, 1, input_features)

    return tensor


def postprocess_output(output: torch.Tensor, config: Dict[str, Any]) -> float:
    """
    Postprocess model output to get power prediction

    Args:
        output: Model output tensor
        config: Model configuration with normalization params

    Returns:
        Denormalized power prediction in kW
    """
    # Extract scalar value
    predicted_normalized = output.item()

    # Denormalize if params provided
    if "normalization_params" in config:
        mean = np.array(config["normalization_params"]["mean"], dtype=np.float32)
        std = np.array(config["normalization_params"]["std"], dtype=np.float32)
        predicted_kw = predicted_normalized * std[0] + mean[0]
    else:
        predicted_kw = predicted_normalized

    # Clamp to non-negative (power can't be negative)
    predicted_kw = max(0.0, predicted_kw)

    return float(predicted_kw)


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """
    Run inference for a single appliance

    Args:
        request: Inference request with appliance_id and aggregate_data

    Returns:
        Prediction with power (kW), confidence, and model version
    """
    # Load model and config
    model, config = load_model(request.appliance_id)

    # Validate input length
    expected_length = config.get("sequence_length", 60)
    if len(request.aggregate_data) != expected_length:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected_length} timesteps, got {len(request.aggregate_data)}"
        )

    # Preprocess input
    try:
        input_tensor = preprocess_input(request.aggregate_data, config)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to preprocess input: {str(e)}"
        )

    # Run inference
    try:
        with torch.no_grad():
            output = model(input_tensor) if hasattr(model, '__call__') else torch.tensor([[0.0]])
    except Exception as e:
        logger.error(f"Inference failed for {request.appliance_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )

    # Postprocess output
    try:
        predicted_kw = postprocess_output(output, config)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to postprocess output: {str(e)}"
        )

    # Calculate confidence (placeholder - could use model uncertainty)
    confidence = 0.85

    return InferenceResponse(
        predicted_kw=predicted_kw,
        confidence=confidence,
        model_version=config["model_version"]
    )


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "models_loaded": list(MODEL_CACHE.keys()),
        "models_available": list(MODEL_REGISTRY.keys())
    }


@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "models": list(MODEL_REGISTRY.keys()),
        "count": len(MODEL_REGISTRY)
    }


@app.get("/models/{appliance_id}")
async def get_model_info(appliance_id: str):
    """Get information about a specific model"""
    if appliance_id not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Model for appliance '{appliance_id}' not found"
        )

    config = MODEL_REGISTRY[appliance_id].copy()
    # Don't expose full file paths
    config["model_path"] = Path(config["model_path"]).name

    return {
        "appliance_id": appliance_id,
        "config": config,
        "cached": appliance_id in MODEL_CACHE
    }


@app.post("/reload-registry")
async def reload_registry():
    """Reload model registry from file"""
    load_registry()
    # Clear cache to force reloading models
    MODEL_CACHE.clear()
    return {
        "status": "ok",
        "models_count": len(MODEL_REGISTRY),
        "message": "Registry reloaded and cache cleared"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
