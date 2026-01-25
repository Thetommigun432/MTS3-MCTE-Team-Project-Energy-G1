"""
NILM Real-Time Inference Service
=================================

Production service that:
1. Receives raw building data (1 sample/second)
2. Buffers data in Redis sliding window
3. When window is complete (1536 samples), runs inference on ALL 11 appliances
4. Publishes predictions to Redis for downstream consumers

Architecture:
    Raw Data → Preprocessor → Redis Buffer → Inference Engine → Predictions

Usage:
    # Start the service
    python -m production.inference_service --redis-host localhost --redis-port 6379
    
    # Or with Docker
    docker run -e REDIS_HOST=redis -e REDIS_PORT=6379 nilm-inference

Environment Variables:
    REDIS_HOST: Redis server hostname (default: localhost)
    REDIS_PORT: Redis server port (default: 6379)
    MODEL_DIR: Directory with trained models (default: ./checkpoints)
    WINDOW_SIZE: Inference window size (default: 1536)
    INFERENCE_INTERVAL: Seconds between inferences (default: 60)
"""

import os
import sys
import json
import time
import pickle
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import torch
import torch.nn as nn
import redis

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RawSample:
    """Raw input sample from building sensors."""
    timestamp: float  # Unix timestamp
    power_total: float  # Total power consumption (W)
    voltage: Optional[float] = None
    current: Optional[float] = None
    power_factor: Optional[float] = None
    reactive_power: Optional[float] = None
    # Time features (will be computed)
    hour: Optional[int] = None
    day_of_week: Optional[int] = None


@dataclass 
class Prediction:
    """Prediction output for a single appliance."""
    appliance: str
    power_watts: float  # Predicted power consumption
    probability: float  # Probability appliance is ON
    is_on: bool  # Binary ON/OFF state
    confidence: float  # Model confidence (0-1)


@dataclass
class InferenceResult:
    """Complete inference result for all appliances."""
    timestamp: float
    window_start: float
    window_end: float
    total_power: float
    predictions: List[Prediction]
    inference_time_ms: float


# =============================================================================
# PREPROCESSOR
# =============================================================================

class DataPreprocessor:
    """
    Preprocesses raw sensor data into model-ready features.
    
    Features (7 total) - MUST MATCH TRAINING DATA:
        0: Aggregate (power_normalized 0-1, scaled by P_MAX)
        1: hour_sin   = sin(2π × hour/24)
        2: hour_cos   = cos(2π × hour/24)
        3: dow_sin    = sin(2π × day_of_week/7)
        4: dow_cos    = cos(2π × day_of_week/7)
        5: month_sin  = sin(2π × month/12)
        6: month_cos  = cos(2π × month/12)
    
    NOTE: Temporal features are in [-1, 1] (NO SCALING needed).
    NOTE: P_MAX is in WATTS (same as metadata from training)!
    """
    
    def __init__(self, P_MAX: float = 15000.0):
        """
        Args:
            P_MAX: Maximum power in WATTS for normalization (from training metadata)
                   Default ~15kW, but should be loaded from metadata.pkl
        """
        self.P_MAX = P_MAX  # WATTS (NOT kW!)
        
    def process_sample(self, sample: RawSample) -> np.ndarray:
        """
        Process a single raw sample into feature vector.
        
        Args:
            sample: RawSample with power_total in WATTS
        
        Returns:
            np.ndarray: Feature vector of shape (7,)
        """
        # 1. Normalize Aggregate power: Aggregate_scaled = Aggregate / P_MAX
        #    SAME as preprocessing notebook!
        aggregate_norm = np.clip(sample.power_total / self.P_MAX, 0, 1)
        
        # 2. Extract time components
        dt = datetime.fromtimestamp(sample.timestamp, tz=timezone.utc)
        hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        day_of_week = dt.weekday()  # 0=Monday, 6=Sunday
        month = dt.month - 1 + dt.day / 31.0  # 0-11 continuous
        
        # 3. Cyclical encoding (sin/cos)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin = np.sin(2 * np.pi * day_of_week / 7)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        # Return in EXACT order as training data
        return np.array([
            aggregate_norm,  # 0: Aggregate
            hour_sin,        # 1: hour_sin
            hour_cos,        # 2: hour_cos
            dow_sin,         # 3: dow_sin
            dow_cos,         # 4: dow_cos
            month_sin,       # 5: month_sin
            month_cos        # 6: month_cos
        ], dtype=np.float32)
    
    def reset(self):
        """Reset preprocessor state (stateless, but kept for API compatibility)."""
        pass


# =============================================================================
# REDIS BUFFER MANAGER
# =============================================================================

class RedisBufferManager:
    """
    Manages sliding window buffer in Redis.
    
    Stores:
        - Feature buffer: List of preprocessed feature vectors
        - Raw power buffer: For total power tracking
        - Timestamps: For window timing
    """
    
    def __init__(
        self,
        redis_client: redis.Redis,
        building_id: str = "building_1",
        window_size: int = 1536,
        key_prefix: str = "nilm"
    ):
        self.redis = redis_client
        self.building_id = building_id
        self.window_size = window_size
        self.key_prefix = key_prefix
        
        # Redis keys
        self.features_key = f"{key_prefix}:{building_id}:features"
        self.timestamps_key = f"{key_prefix}:{building_id}:timestamps"
        self.power_key = f"{key_prefix}:{building_id}:power"
        self.lock_key = f"{key_prefix}:{building_id}:lock"
        
    def add_sample(self, features: np.ndarray, timestamp: float, power: float):
        """
        Add a new sample to the buffer.
        
        Uses Redis pipeline for atomic operation.
        """
        pipe = self.redis.pipeline()
        
        # Serialize features
        features_bytes = features.tobytes()
        
        # Add to lists (RPUSH)
        pipe.rpush(self.features_key, features_bytes)
        pipe.rpush(self.timestamps_key, timestamp)
        pipe.rpush(self.power_key, power)
        
        # Trim to window size (keep last N elements)
        pipe.ltrim(self.features_key, -self.window_size, -1)
        pipe.ltrim(self.timestamps_key, -self.window_size, -1)
        pipe.ltrim(self.power_key, -self.window_size, -1)
        
        pipe.execute()
        
    def get_window(self) -> Optional[Tuple[np.ndarray, List[float], List[float]]]:
        """
        Get the current window if complete.
        
        Returns:
            Tuple of (features_array, timestamps, powers) or None if incomplete
        """
        # Check buffer length
        length = self.redis.llen(self.features_key)
        
        if length < self.window_size:
            return None
        
        # Get all data atomically
        pipe = self.redis.pipeline()
        pipe.lrange(self.features_key, 0, -1)
        pipe.lrange(self.timestamps_key, 0, -1)
        pipe.lrange(self.power_key, 0, -1)
        features_bytes, timestamps_bytes, powers_bytes = pipe.execute()
        
        # Deserialize
        n_features = 7  # Must match preprocessor
        features = np.array([
            np.frombuffer(fb, dtype=np.float32)
            for fb in features_bytes
        ])
        timestamps = [float(t) for t in timestamps_bytes]
        powers = [float(p) for p in powers_bytes]
        
        return features, timestamps, powers
    
    def get_buffer_status(self) -> Dict:
        """Get current buffer status."""
        length = self.redis.llen(self.features_key)
        return {
            "building_id": self.building_id,
            "buffer_size": length,
            "window_size": self.window_size,
            "ready": length >= self.window_size,
            "fill_percent": min(100, length / self.window_size * 100)
        }
    
    def clear(self):
        """Clear the buffer (for testing/reset)."""
        self.redis.delete(self.features_key, self.timestamps_key, self.power_key)


# =============================================================================
# MULTI-MODEL INFERENCE ENGINE
# =============================================================================

class MultiModelInferenceEngine:
    """
    Loads and manages all 11 appliance models for parallel inference.
    
    Features:
        - Lazy model loading
        - GPU batch inference
        - Concurrent prediction for all appliances
    """
    
    # All 11 appliances in the NILM system
    APPLIANCES = [
        "HeatPump",
        "Dishwasher", 
        "WashingMachine",
        "Dryer",
        "Freezer",
        "Refrigerator",
        "ElectricOven",
        "Microwave",
        "ElectricKettle",
        "Computer",
        "Television"
    ]
    
    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        default_threshold: float = 0.5
    ):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.default_threshold = default_threshold
        
        self.models: Dict[str, nn.Module] = {}
        self.thresholds: Dict[str, float] = {}
        self.P_MAX = 15.0  # Will be loaded from metadata
        
        logger.info(f"Inference Engine initialized on {self.device}")
        
    def load_models(self):
        """Load all available appliance models."""
        logger.info(f"Loading models from {self.model_dir}...")
        
        # Load metadata if available
        meta_path = self.model_dir / "metadata.pkl"
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
                self.P_MAX = meta.get('scaling', {}).get('P_MAX', 15.0)
                logger.info(f"Loaded metadata: P_MAX={self.P_MAX}")
        
        # Import model architecture
        try:
            sys.path.insert(0, str(self.model_dir.parent.parent))
            from nilm.causal.wavenilm_v3 import WaveNILM_v3
        except ImportError as ie:
            logger.error(f"Failed to import WaveNILM_v3: {ie}. Running in NO-MODEL mode.")
            return

        loaded = 0
        for appliance in self.APPLIANCES:
            # ... (rest of loop logic would go here but we return early above if import fails)
            # Actually, we should try to load *something* or just return if import fails.
            pass

        # If we are here, we imported successfully, proceed with loading loop
        # But for brevity in this patch, I'll keep the logic clean.
        # The user wants "test preprocessing pipeline", which happens BEFORE inference.
        # So having 0 models loaded is fine if we handle it in predict_all.
        
        for appliance in self.APPLIANCES:
            # Try different checkpoint naming conventions
            ckpt_paths = [
                self.model_dir / f"wavenilm_v3_SOTA_{appliance}_best.pth",
                self.model_dir / f"wavenilm_v3_{appliance}_best.pth",
                self.model_dir / f"{appliance}_best.pth",
            ]
            
            ckpt_path = None
            for p in ckpt_paths:
                if p.exists():
                    ckpt_path = p
                    break
            
            if ckpt_path is None:
                logger.warning(f"No checkpoint found for {appliance}")
                continue
            
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                
                # Get model config from checkpoint
                n_blocks = ckpt.get('n_blocks', 9)
                hidden_channels = ckpt.get('hidden_channels', 64)
                window = ckpt.get('window', 1536)
                
                # Create model
                model = WaveNILM_v3(
                    n_input_features=7,
                    hidden_channels=hidden_channels,
                    n_blocks=n_blocks,
                    n_stacks=2,
                    use_attention=False,
                    use_mtl=True,
                    dropout=0.0  # No dropout at inference
                )
                model.load_state_dict(ckpt['model'])
                model.to(self.device)
                model.eval()
                
                self.models[appliance] = model
                self.thresholds[appliance] = ckpt.get('optimal_threshold', self.default_threshold)
                
                logger.info(f"  ✓ {appliance}: {n_blocks} blocks, hidden={hidden_channels}, thresh={self.thresholds[appliance]:.2f}")
                loaded += 1
                
            except Exception as e:
                logger.error(f"  ✗ Failed to load {appliance}: {e}")
        
        logger.info(f"Loaded {loaded}/{len(self.APPLIANCES)} models")
        
        if loaded == 0:
             logger.warning("No models loaded! Service will run but produce empty predictions.")

        
    @torch.no_grad()
    def predict_all(self, features: np.ndarray) -> List[Prediction]:
        """
        Run inference on all loaded models.
        
        Args:
            features: Input array of shape (window_size, n_features)
            
        Returns:
            List of Prediction objects for each appliance
        """
        predictions = []
        
        # Prepare input tensor
        x = torch.from_numpy(features).float()
        x = x.unsqueeze(0)  # Add batch dimension: (1, window, features)
        x = x.to(self.device)
        
        # Run inference on each model
        for appliance, model in self.models.items():
            try:
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.device.type=='cuda'):
                    power_pred, prob_pred = model(x)
                
                # Extract predictions (last timestep of window)
                power_watts = power_pred[0, -1, 0].item() * self.P_MAX * 1000
                probability = prob_pred[0, -1, 0].item()
                
                # Apply threshold
                threshold = self.thresholds.get(appliance, self.default_threshold)
                is_on = probability > threshold
                
                # Confidence based on distance from threshold
                confidence = abs(probability - threshold) / max(threshold, 1 - threshold)
                confidence = min(1.0, confidence)
                
                # Clamp power
                power_watts = max(0, power_watts)
                if not is_on:
                    power_watts = 0  # Hard gating when OFF
                
                predictions.append(Prediction(
                    appliance=appliance,
                    power_watts=round(power_watts, 1),
                    probability=round(probability, 4),
                    is_on=is_on,
                    confidence=round(confidence, 3)
                ))
                
            except Exception as e:
                logger.error(f"Inference error for {appliance}: {e}")
                predictions.append(Prediction(
                    appliance=appliance,
                    power_watts=0,
                    probability=0,
                    is_on=False,
                    confidence=0
                ))
        
        return predictions
    
    def predict_batch(self, features: np.ndarray) -> List[Prediction]:
        """
        Optimized batch inference (all models share same input).
        Uses threading for parallel model execution.
        """
        predictions = []
        lock = threading.Lock()
        
        # Prepare input once
        x = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        
        def run_model(appliance: str, model: nn.Module):
            try:
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.device.type=='cuda'):
                        power_pred, prob_pred = model(x)
                
                power_watts = power_pred[0, -1, 0].item() * self.P_MAX * 1000
                probability = prob_pred[0, -1, 0].item()
                threshold = self.thresholds.get(appliance, self.default_threshold)
                is_on = probability > threshold
                confidence = min(1.0, abs(probability - threshold) / max(threshold, 1 - threshold))
                
                power_watts = max(0, power_watts) if is_on else 0
                
                pred = Prediction(
                    appliance=appliance,
                    power_watts=round(power_watts, 1),
                    probability=round(probability, 4),
                    is_on=is_on,
                    confidence=round(confidence, 3)
                )
                
                with lock:
                    predictions.append(pred)
                    
            except Exception as e:
                logger.error(f"Batch inference error for {appliance}: {e}")
        
        # Run models in parallel (GPU handles actual parallelism)
        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            futures = [
                executor.submit(run_model, app, model)
                for app, model in self.models.items()
            ]
            for f in futures:
                f.result()  # Wait for completion
        
        # Sort by appliance name for consistent output
        predictions.sort(key=lambda p: p.appliance)
        return predictions


# =============================================================================
# MAIN SERVICE
# =============================================================================

class NILMInferenceService:
    """
    Main service orchestrating the entire pipeline.
    
    Flow:
        1. Receive raw sample
        2. Preprocess
        3. Buffer in Redis
        4. If window complete, run inference
        5. Publish results
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        model_dir: str = "./checkpoints",
        building_id: str = "building_1",
        window_size: int = 1536,
        inference_interval: int = 60,  # Seconds between inferences
        P_MAX: float = 15.0
    ):
        # Redis connection
        self.redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=False  # Binary mode for numpy
        )
        
        # Components
        self.preprocessor = DataPreprocessor(P_MAX=P_MAX)
        self.buffer = RedisBufferManager(
            self.redis, 
            building_id=building_id,
            window_size=window_size
        )
        self.engine = MultiModelInferenceEngine(model_dir)
        
        # Config
        self.building_id = building_id
        self.window_size = window_size
        self.inference_interval = inference_interval
        self.last_inference_time = 0
        
        # Redis pub/sub keys
        self.input_channel = f"nilm:{building_id}:input"
        self.output_channel = f"nilm:{building_id}:predictions"
        self.status_key = f"nilm:{building_id}:status"
        
        logger.info(f"NILM Service initialized for {building_id}")
        logger.info(f"  Window: {window_size} samples (~{window_size*5/60:.0f} min @ 5sec resolution)")
        logger.info(f"  Inference interval: {inference_interval}s")
        
    def start(self):
        """Initialize and start the service."""
        logger.info("Starting NILM Inference Service...")
        
        # Load models
        self.engine.load_models()
        
        # Clear old buffer (optional, for clean start)
        # self.buffer.clear()
        
        # Update status
        self._update_status("running")
        
        logger.info("Service ready! Waiting for data...")
        
    def process_sample(self, raw_sample: RawSample) -> Optional[InferenceResult]:
        """
        Process a single incoming sample.
        
        Returns:
            InferenceResult if inference was triggered, None otherwise
        """
        # 1. Preprocess
        features = self.preprocessor.process_sample(raw_sample)
        
        # 2. Add to buffer
        self.buffer.add_sample(features, raw_sample.timestamp, raw_sample.power_total)
        
        # 3. Check if we should run inference
        now = time.time()
        time_since_last = now - self.last_inference_time
        
        if time_since_last < self.inference_interval:
            return None  # Too soon
        
        # 4. Get window
        window_data = self.buffer.get_window()
        if window_data is None:
            return None  # Buffer not full
        
        features_array, timestamps, powers = window_data
        
        # 5. Run inference
        t0 = time.time()
        predictions = self.engine.predict_all(features_array)
        inference_time_ms = (time.time() - t0) * 1000
        
        # 6. Create result
        result = InferenceResult(
            timestamp=now,
            window_start=timestamps[0],
            window_end=timestamps[-1],
            total_power=sum(powers[-60:]) / 60,  # Average last minute
            predictions=predictions,
            inference_time_ms=round(inference_time_ms, 2)
        )
        
        self.last_inference_time = now
        
        # 7. Publish to Redis
        self._publish_result(result)
        
        return result
    
    def process_batch(self, samples: List[RawSample]) -> Optional[InferenceResult]:
        """Process multiple samples at once (for batch ingestion)."""
        result = None
        for sample in samples:
            result = self.process_sample(sample)
        return result
    
    def _publish_result(self, result: InferenceResult):
        """Publish inference result to Redis."""
        # Convert to JSON-serializable dict
        result_dict = {
            "timestamp": result.timestamp,
            "window_start": result.window_start,
            "window_end": result.window_end,
            "total_power": result.total_power,
            "inference_time_ms": result.inference_time_ms,
            "predictions": [asdict(p) for p in result.predictions]
        }
        
        # Publish to channel
        self.redis.publish(self.output_channel, json.dumps(result_dict))
        
        # Also store latest result
        self.redis.set(
            f"nilm:{self.building_id}:latest",
            json.dumps(result_dict),
            ex=3600  # Expire in 1 hour
        )
        
        logger.info(f"Published predictions: {len(result.predictions)} appliances, {result.inference_time_ms:.1f}ms")
    
    def _update_status(self, status: str):
        """Update service status in Redis."""
        self.redis.hset(self.status_key, mapping={
            "status": status,
            "updated": datetime.now(timezone.utc).isoformat(),
            "building_id": self.building_id,
            "models_loaded": len(self.engine.models)
        })
    
    def run_subscriber(self):
        """
        Run as subscriber listening for raw data on Redis channel.
        
        Data format (JSON):
            {
                "timestamp": 1706198400.0,
                "power_total": 3500.0,
                "voltage": 230.0,
                "current": 15.2
            }
        """
        pubsub = self.redis.pubsub()
        pubsub.subscribe(self.input_channel)
        
        logger.info(f"Subscribed to {self.input_channel}")
        
        for message in pubsub.listen():
            if message['type'] != 'message':
                continue
            
            try:
                data = json.loads(message['data'])
                sample = RawSample(
                    timestamp=data['timestamp'],
                    power_total=data['power_total'],
                    voltage=data.get('voltage'),
                    current=data.get('current'),
                    power_factor=data.get('power_factor'),
                    reactive_power=data.get('reactive_power')
                )
                
                result = self.process_sample(sample)
                
                if result:
                    self._log_predictions(result)
                    
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    def _log_predictions(self, result: InferenceResult):
        """Log prediction summary."""
        active = [p for p in result.predictions if p.is_on]
        total_predicted = sum(p.power_watts for p in result.predictions)
        
        logger.info(f"═══ Inference Result ═══")
        logger.info(f"  Total Power: {result.total_power:.0f}W")
        logger.info(f"  Predicted:   {total_predicted:.0f}W ({len(active)} active)")
        
        for p in sorted(result.predictions, key=lambda x: -x.power_watts):
            status = "🟢" if p.is_on else "⚫"
            logger.info(f"    {status} {p.appliance:15s}: {p.power_watts:6.0f}W (p={p.probability:.2f})")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NILM Real-Time Inference Service")
    
    parser.add_argument('--redis-host', type=str, 
                        default=os.environ.get('REDIS_HOST', 'localhost'))
    parser.add_argument('--redis-port', type=int,
                        default=int(os.environ.get('REDIS_PORT', 6379)))
    parser.add_argument('--model-dir', type=str,
                        default=os.environ.get('MODEL_DIR', './checkpoints'))
    parser.add_argument('--building-id', type=str,
                        default=os.environ.get('BUILDING_ID', 'building_1'))
    parser.add_argument('--window-size', type=int,
                        default=int(os.environ.get('WINDOW_SIZE', 1536)))
    parser.add_argument('--inference-interval', type=int,
                        default=int(os.environ.get('INFERENCE_INTERVAL', 60)),
                        help='Seconds between inferences (default: 60)')
    parser.add_argument('--mode', type=str, choices=['subscriber', 'api', 'test'],
                        default='subscriber',
                        help='Run mode: subscriber (Redis pub/sub), api (REST), test (demo)')
    
    args = parser.parse_args()
    
    # Initialize service
    service = NILMInferenceService(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        model_dir=args.model_dir,
        building_id=args.building_id,
        window_size=args.window_size,
        inference_interval=args.inference_interval
    )
    
    # Start service
    service.start()
    
    if args.mode == 'subscriber':
        # Run as Redis subscriber
        service.run_subscriber()
        
    elif args.mode == 'test':
        # Demo mode with synthetic data
        logger.info("Running in TEST mode with synthetic data...")
        
        import random
        
        for i in range(args.window_size + 100):
            # Generate synthetic sample
            sample = RawSample(
                timestamp=time.time(),
                power_total=2000 + random.gauss(0, 500) + (1500 if i % 300 < 100 else 0),
                voltage=230 + random.gauss(0, 2),
                current=10 + random.gauss(0, 2)
            )
            
            result = service.process_sample(sample)
            
            if result:
                service._log_predictions(result)
            
            # Progress
            if i % 100 == 0:
                status = service.buffer.get_buffer_status()
                logger.info(f"Buffer: {status['fill_percent']:.0f}% ({status['buffer_size']}/{status['window_size']})")
            
            time.sleep(0.01)  # Fast simulation
        
        logger.info("Test complete!")


if __name__ == "__main__":
    main()
