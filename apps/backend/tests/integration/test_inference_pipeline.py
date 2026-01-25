
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from app.domain.inference.engine import InferenceEngine

@pytest.mark.asyncio
async def test_end_to_end_inference_flow():
    """
    Test the complete flow from Ingest -> Worker -> Inference -> Result.
    This uses mocks for external services (Redis, Influx) but exercises the domain logic.
    """
    # 1. Setup Data
    reading = {
        "timestamp": "2026-01-25T12:00:00Z",
        "power": 1500.0, # Main consumption
        "voltage": 230.0
    }
    
    # 2. Mock Dependencies
    # We need to mock the Redis worker processing logic or the engine directly if testing components
    # Since we can't easily spin up a real Redis worker in this unit/integration test without Docker,
    # we will test the 'InferenceEngine' call which is the core logic the worker invokes.
    
    mock_registry = MagicMock()
    mock_model_entry = MagicMock()
    mock_model_entry.input_window_size = 100
    mock_model_entry.preprocessing.type = "minmax"
    mock_model_entry.preprocessing.min_val = 0.0
    mock_model_entry.preprocessing.max_val = 1.0
    
    # Mock model
    mock_model = MagicMock()
    # tensor output mocks
    import torch
    mock_model.return_value = torch.tensor([[0.5]]) # scaled output
    
    # Engine
    engine = InferenceEngine()
    engine.get_model = MagicMock(return_value=(mock_model, mock_model_entry))
    
    # Test execution
    # Simulate a window of data
    window = [1000.0] * 100
    
    # Call run_inference
    predicted, confidence = engine.run_inference(mock_model, mock_model_entry, window)
    
    # Assertions
    assert isinstance(predicted, float)
    assert predicted >= 0.0
    assert 0.0 <= confidence <= 1.0
    
    # Verify model was called
    mock_model.assert_called_once()
    
    print(f"\nInference successful: {predicted} kW (conf: {confidence})")

