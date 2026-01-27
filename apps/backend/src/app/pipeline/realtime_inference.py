"""
NILM Real-Time Inference Service
=================================

Production service that:
1. Receives raw building data (1 sample/second via Redis)
2. Preprocesses with 8 features (including ΔP)
3. Buffers data in Redis sliding window
4. When window is complete, runs inference on ALL appliances
5. Validates predictions (sum <= building total)
6. Publishes predictions to Redis for downstream consumers

Flow:
    producer.py (reads CSV every 1s) 
        → Redis pub/sub (raw data)
        → DataPreprocessor (8 features)
        → RedisBufferManager (sliding window)
        → MultiModelInferenceEngine (per-appliance models)
        → PredictionValidator (sum check)
        → Redis pub/sub (predictions)
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
    total_power: float  # Building aggregate
    sum_predictions: float  # Sum of all predictions
    residual_power: float  # Ghost load (unmonitored appliances)
    predictions: List[Prediction]
    inference_time_ms: float
    is_valid: bool  # True if sum <= total_power (no scaling needed)
    coverage_ratio: float  # sum / total_power


# =============================================================================
# PREPROCESSOR (8 features with ΔP)
# =============================================================================

class DataPreprocessor:
    """
    Preprocesses raw sensor data into model-ready features.
    
    Features (8 total) - MUST MATCH TRAINING DATA:
        0: Aggregate (normalized by p95, scaled to [-1, 3])
        1: hour_sin   = sin(2π × hour/24)
        2: hour_cos   = cos(2π × hour/24)
        3: dow_sin    = sin(2π × day_of_week/7)
        4: dow_cos    = cos(2π × day_of_week/7)
        5: month_sin  = sin(2π × month/12)
        6: month_cos  = cos(2π × month/12)
        7: ΔP         = delta power (clipped to ±5kW, scaled to [-1, 1])
    """
    
    def __init__(self, agg_p95: float = 8000.0):
        """
        Args:
            agg_p95: 95th percentile of Aggregate in WATTS (from training metadata)
        """
        self.agg_p95 = agg_p95
        self.prev_power = None
        
    def process_sample(self, sample: RawSample) -> np.ndarray:
        """Process a single raw sample into feature vector (8 features)."""
        # 1. Normalize Aggregate power using p95
        aggregate_norm = np.clip(sample.power_total / self.agg_p95, 0, 2) * 2 - 1
        
        # 2. Extract time components
        dt = datetime.fromtimestamp(sample.timestamp, tz=timezone.utc)
        hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        day_of_week = dt.weekday()
        month = dt.month - 1 + dt.day / 31.0
        
        # 3. Cyclical encoding
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin = np.sin(2 * np.pi * day_of_week / 7)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        # 4. Calculate ΔP
        if self.prev_power is None:
            delta_p = 0.0
        else:
            delta_p_raw = sample.power_total - self.prev_power
            delta_p = np.clip(delta_p_raw / 5000.0, -1, 1)
        
        self.prev_power = sample.power_total
        
        return np.array([
            aggregate_norm, hour_sin, hour_cos, dow_sin, dow_cos, 
            month_sin, month_cos, delta_p
        ], dtype=np.float32)
    
    def reset(self):
        """Reset state."""
        self.prev_power = None


# =============================================================================
# REDIS BUFFER MANAGER
# =============================================================================

class RedisBufferManager:
    """
    Manages sliding window buffer in Redis.
    Supports different window sizes per model.
    """
    
    def __init__(
        self,
        redis_client: redis.Redis,
        building_id: str = "building_1",
        max_window_size: int = 4100,  # Keep enough for largest model (4096 + margin)
        key_prefix: str = "nilm"
    ):
        self.redis = redis_client
        self.building_id = building_id
        self.max_window_size = max_window_size
        
        # Redis keys
        self.features_key = f"{key_prefix}:{building_id}:features"
        self.timestamps_key = f"{key_prefix}:{building_id}:timestamps"
        self.power_key = f"{key_prefix}:{building_id}:power"
        
    def add_sample(self, features: np.ndarray, timestamp: float, power: float):
        """Add a preprocessed sample to the buffer."""
        # Serialize feature vector
        features_bytes = features.tobytes()
        
        # Pipeline for atomic operations
        pipe = self.redis.pipeline()
        
        # Add to lists
        pipe.rpush(self.features_key, features_bytes)
        pipe.rpush(self.timestamps_key, str(timestamp))
        pipe.rpush(self.power_key, str(power))
        
        # Trim to max window size
        pipe.ltrim(self.features_key, -self.max_window_size, -1)
        pipe.ltrim(self.timestamps_key, -self.max_window_size, -1)
        pipe.ltrim(self.power_key, -self.max_window_size, -1)
        
        pipe.execute()
    
    def get_window(self, window_size: int) -> Optional[Tuple[np.ndarray, List[float], List[float]]]:
        """
        Get the last window_size samples.
        
        Returns:
            Tuple of (features, timestamps, powers) or None if not enough data
        """
        # Get buffer length
        buf_len = self.redis.llen(self.features_key)
        
        if buf_len < window_size:
            return None
        
        # Get last window_size elements
        start_idx = buf_len - window_size
        
        pipe = self.redis.pipeline()
        pipe.lrange(self.features_key, start_idx, -1)
        pipe.lrange(self.timestamps_key, start_idx, -1)
        pipe.lrange(self.power_key, start_idx, -1)
        
        features_bytes, timestamps_raw, powers_raw = pipe.execute()
        
        # Reconstruct arrays
        features_list = [np.frombuffer(f, dtype=np.float32) for f in features_bytes]
        features_array = np.stack(features_list)  # (window_size, 8)
        
        timestamps = [float(t) for t in timestamps_raw]
        powers = [float(p) for p in powers_raw]
        
        return features_array, timestamps, powers
    
    def get_buffer_status(self) -> Dict:
        """Get current buffer status."""
        buf_len = self.redis.llen(self.features_key)
        return {
            "buffer_size": buf_len,
            "max_window_size": self.max_window_size,
            "fill_percent": (buf_len / self.max_window_size) * 100
        }
    
    def clear(self):
        """Clear the buffer."""
        self.redis.delete(self.features_key, self.timestamps_key, self.power_key)


# =============================================================================
# MULTI-MODEL INFERENCE ENGINE
# =============================================================================

class MultiModelInferenceEngine:
    """
    Loads and manages all appliance models.
    Each model can have different window size.
    """
    
    # All appliances with their expected window sizes
    APPLIANCES_CONFIG = {
        "HeatPump": {"window": 1536},
        "Dishwasher": {"window": 1536},
        "WashingMachine": {"window": 1536},
        "Dryer": {"window": 1536},
        "Oven": {"window": 1536},
        "Stove": {"window": 1536},
        "RangeHood": {"window": 1536},
        "EVCharger": {"window": 1536},
        "EVSocket": {"window": 1536},
        "RainwaterPump": {"window": 1536}
    }
    
    # P_MAX per appliance (Watts)
    PMAX_DICT = {
        "HeatPump": 3000,
        "Dishwasher": 2500,
        "WashingMachine": 2000,
        "Dryer": 3000,
        "Oven": 3500,
        "Stove": 2500,
        "RangeHood": 300,
        "EVCharger": 7000,
        "EVSocket": 3700,
        "RainwaterPump": 500
    }
    
    def __init__(
        self,
        model_dir: str,
        device: str = "cuda"
    ):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.models: Dict[str, nn.Module] = {}
        self.model_windows: Dict[str, int] = {}
        
        logger.info(f"Inference Engine initialized on {self.device}")
        
    def load_models(self):
        """Load all available appliance models (TCN_SA_*.pt format)."""
        logger.info(f"Loading models from {self.model_dir}...")
        
        # Import model architecture
        from app.domain.inference.architectures import TCN_SA
        
        loaded = 0
        for appliance, config in self.APPLIANCES_CONFIG.items():
            # Try to find checkpoint with TCN_SA naming
            patterns = [
                f"TCN_SA_{appliance}*.pt",
                f"TCN_SA_{appliance.lower()}*.pt",
            ]
            
            ckpt_path = None
            for pattern in patterns:
                matches = list(self.model_dir.glob(pattern))
                if matches:
                    ckpt_path = matches[0]
                    break
            
            if ckpt_path is None:
                logger.warning(f"No checkpoint found for {appliance}")
                continue
            
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                
                # Get model config from checkpoint
                n_blocks = ckpt.get('n_blocks', 11)
                hidden_channels = ckpt.get('hidden_channels', 64)
                window = ckpt.get('window', config['window'])
                lookahead = ckpt.get('lookahead', 0)
                
                # Create model (TCN_SA architecture)
                model = TCN_SA(
                    n_features=8,
                    n_appliances=1,
                    hidden_channels=hidden_channels,
                    n_blocks=n_blocks,
                    kernel_size=2,
                    stem_kernel=7,
                    spatial_dropout=0.0,
                    head_dropout=0.0,
                    use_psa=True,
                    use_inception=True,
                    lookahead=lookahead
                )
                model.load_state_dict(ckpt['model_state_dict'])
                model.to(self.device)
                model.eval()
                
                self.models[appliance] = model
                self.model_windows[appliance] = window
                
                logger.info(f"  ✓ {appliance}: window={window}, blocks={n_blocks}")
                loaded += 1
                
            except Exception as e:
                logger.error(f"  ✗ Failed to load {appliance}: {e}")
        
        logger.info(f"Loaded {loaded}/{len(self.APPLIANCES_CONFIG)} models")
        
        if loaded == 0:
            raise RuntimeError("No models loaded! Check model_dir path.")
    
    def get_max_window(self) -> int:
        """Get the maximum window size needed across all models."""
        if not self.model_windows:
            return 1536
        return max(self.model_windows.values())
    
    @torch.no_grad()
    def predict_all(self, buffer_manager: RedisBufferManager) -> List[Prediction]:
        """
        Run inference on all loaded models.
        Each model gets its required window size from the buffer.
        """
        predictions = []
        
        for appliance, model in self.models.items():
            window_size = self.model_windows.get(appliance, 1536)
            
            # Get window for this model
            window_data = buffer_manager.get_window(window_size)
            if window_data is None:
                logger.debug(f"Not enough data for {appliance} (need {window_size})")
                continue
            
            features_array, timestamps, powers = window_data
            
            try:
                # Prepare input
                x = torch.from_numpy(features_array).float()
                x = x.unsqueeze(0).to(self.device)
                
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.device.type=='cuda'):
                    gate, power_pred = model(x, target_timestep=-1)
                
                probability = gate[0, 0].item()
                power_normalized = power_pred[0, 0].item()
                
                # Scale to watts
                pmax = self.PMAX_DICT.get(appliance, 2000)
                power_watts = max(0, power_normalized * pmax)
                
                # ON/OFF threshold
                is_on = probability > 0.5
                confidence = abs(probability - 0.5) / 0.5
                
                predictions.append(Prediction(
                    appliance=appliance,
                    power_watts=round(power_watts, 1),
                    probability=round(probability, 4),
                    is_on=is_on,
                    confidence=round(min(1.0, confidence), 3)
                ))
                
            except Exception as e:
                logger.error(f"Inference error for {appliance}: {e}")
        
        return predictions


# =============================================================================
# PREDICTION VALIDATOR
# =============================================================================

class PredictionValidator:
    """
    Validates that predictions make physical sense (Summation Constraint).
    
    Physics Rule: Σ(appliance_power) ≤ building_total_power
    
    Methods:
    1. Proportional Scaling: If sum > building, scale all predictions down
    2. Ghost Load: If sum << building, the difference is unmonitored load (lights, TV, etc.)
       - We do NOT inflate predictions to match the building total
       - This is expected behavior (~25% of house consumption is unmonitored)
    """
    
    def __init__(self, tolerance: float = 1.1, min_coverage_ratio: float = 0.10):
        """
        Args:
            tolerance: Allow predictions to exceed building by this factor (10% default)
            min_coverage_ratio: Minimum expected coverage (below this = anomaly, default 10%)
        """
        self.tolerance = tolerance
        self.min_coverage_ratio = min_coverage_ratio
    
    def validate(
        self, 
        predictions: List[Prediction], 
        building_power: float
    ) -> Tuple[List[Prediction], bool, float, float]:
        """
        Validate and optionally scale predictions.
        
        Returns:
            Tuple of (validated_predictions, is_valid, coverage_ratio, residual_power)
            - is_valid: False if scaling was applied
            - coverage_ratio: sum_predictions / building_power
            - residual_power: building_power - sum_predictions (ghost load)
        """
        sum_predictions = sum(p.power_watts for p in predictions if p.is_on)
        
        if building_power <= 0:
            return predictions, True, 0.0, 0.0
        
        coverage_ratio = sum_predictions / building_power
        residual_power = max(0, building_power - sum_predictions)
        
        # Case 1: Predictions exceed building power (VIOLATION - must fix)
        if coverage_ratio > self.tolerance:
            scale_factor = building_power / sum_predictions
            logger.warning(
                f"⚠ Summation constraint violated: {sum_predictions:.0f}W > {building_power:.0f}W, "
                f"scaling by {scale_factor:.3f}"
            )
            
            scaled_predictions = []
            for p in predictions:
                if p.is_on:
                    scaled_power = p.power_watts * scale_factor
                    scaled_predictions.append(Prediction(
                        appliance=p.appliance,
                        power_watts=round(scaled_power, 1),
                        probability=p.probability,
                        is_on=p.is_on,
                        confidence=p.confidence
                    ))
                else:
                    scaled_predictions.append(p)
            
            # Recalculate after scaling
            sum_scaled = sum(p.power_watts for p in scaled_predictions if p.is_on)
            residual_power = max(0, building_power - sum_scaled)
            
            return scaled_predictions, False, scale_factor, residual_power
        
        # Case 2: Predictions are very low compared to building (Ghost Load)
        # This is EXPECTED - we don't monitor ~25% of house consumption
        # Do NOT inflate predictions
        if coverage_ratio < self.min_coverage_ratio and building_power > 100:
            logger.debug(
                f"Low coverage: {sum_predictions:.0f}W / {building_power:.0f}W = {coverage_ratio:.1%}. "
                f"Ghost load (unmonitored): {residual_power:.0f}W"
            )
        
        return predictions, True, coverage_ratio, residual_power


# =============================================================================
# MAIN SERVICE
# =============================================================================

class NILMRealtimeService:
    """
    Main service orchestrating the entire pipeline.
    
    Flow:
        1. Receive raw sample from Redis (1/second)
        2. Preprocess (8 features with ΔP)
        3. Buffer in Redis
        4. If enough samples, run inference on all models
        5. Validate predictions
        6. Publish results
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        model_dir: str = "./checkpoints",
        building_id: str = "building_1",
        inference_interval: int = 60,
        agg_p95: float = 8000.0
    ):
        # Redis connection
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
        
        # Components
        self.preprocessor = DataPreprocessor(agg_p95=agg_p95)
        self.engine = MultiModelInferenceEngine(model_dir)
        self.validator = PredictionValidator(tolerance=1.1)
        
        # Config
        self.building_id = building_id
        self.inference_interval = inference_interval
        self.last_inference_time = 0
        
        # Redis channels
        self.input_channel = f"nilm:{building_id}:input"
        self.output_channel = f"nilm:{building_id}:predictions"
        
        logger.info(f"NILM Service initialized for {building_id}")
        logger.info(f"  Inference interval: {inference_interval}s")
        
    def start(self):
        """Initialize and start the service."""
        logger.info("Starting NILM Realtime Inference Service...")
        
        # Load models
        self.engine.load_models()
        
        # Create buffer manager with max window from models
        max_window = self.engine.get_max_window()
        self.buffer = RedisBufferManager(
            self.redis, 
            building_id=self.building_id,
            max_window_size=max_window
        )
        
        logger.info(f"  Max window size: {max_window}")
        logger.info("Service ready! Waiting for data...")
        
    def process_sample(self, raw_sample: RawSample) -> Optional[InferenceResult]:
        """Process a single incoming sample."""
        # 1. Preprocess
        features = self.preprocessor.process_sample(raw_sample)
        
        # 2. Add to buffer
        self.buffer.add_sample(features, raw_sample.timestamp, raw_sample.power_total)
        
        # 3. Check if we should run inference
        now = time.time()
        if now - self.last_inference_time < self.inference_interval:
            return None
        
        # 4. Run inference on all models
        t0 = time.time()
        predictions = self.engine.predict_all(self.buffer)
        
        if not predictions:
            return None
        
        inference_time_ms = (time.time() - t0) * 1000
        
        # 5. Validate predictions (enforce summation constraint)
        building_power = raw_sample.power_total
        validated_predictions, is_valid, coverage_ratio, residual_power = self.validator.validate(
            predictions, building_power
        )
        
        sum_predictions = sum(p.power_watts for p in validated_predictions if p.is_on)
        
        # 6. Create result
        status = self.buffer.get_buffer_status()
        
        result = InferenceResult(
            timestamp=now,
            window_start=now - status['buffer_size'],
            window_end=now,
            total_power=building_power,
            sum_predictions=sum_predictions,
            residual_power=round(residual_power, 1),
            predictions=validated_predictions,
            inference_time_ms=round(inference_time_ms, 2),
            is_valid=is_valid,
            coverage_ratio=round(coverage_ratio, 3)
        )
        
        self.last_inference_time = now
        
        # 7. Publish to Redis
        self._publish_result(result)
        
        return result
    
    def _publish_result(self, result: InferenceResult):
        """Publish inference result to Redis."""
        result_dict = {
            "timestamp": result.timestamp,
            "total_power": result.total_power,
            "sum_predictions": result.sum_predictions,
            "residual_power": result.residual_power,
            "is_valid": result.is_valid,
            "coverage_ratio": result.coverage_ratio,
            "inference_time_ms": result.inference_time_ms,
            "predictions": [asdict(p) for p in result.predictions]
        }
        
        self.redis.publish(self.output_channel, json.dumps(result_dict))
        self.redis.set(
            f"nilm:{self.building_id}:latest",
            json.dumps(result_dict),
            ex=3600
        )
        
        logger.info(
            f"Published: {len(result.predictions)} appliances, "
            f"sum={result.sum_predictions:.0f}W + ghost={result.residual_power:.0f}W = building={result.total_power:.0f}W "
            f"({'✓' if result.is_valid else '⚠ scaled'})"
        )
    
    def run_subscriber(self):
        """Run as subscriber listening for raw data on Redis channel."""
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
                    current=data.get('current')
                )
                
                result = self.process_sample(sample)
                
                if result:
                    self._log_predictions(result)
                    
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    def _log_predictions(self, result: InferenceResult):
        """Log prediction summary."""
        active = [p for p in result.predictions if p.is_on]
        
        logger.info(f"═══ Inference Result ═══")
        logger.info(f"  Building:    {result.total_power:.0f}W")
        logger.info(f"  Predicted:   {result.sum_predictions:.0f}W ({len(active)} active)")
        logger.info(f"  Ghost Load:  {result.residual_power:.0f}W (unmonitored)")
        logger.info(f"  Coverage:    {result.coverage_ratio:.1%} {'✓' if result.is_valid else '⚠ scaled'}")
        
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
    parser.add_argument('--inference-interval', type=int,
                        default=int(os.environ.get('INFERENCE_INTERVAL', 60)),
                        help='Seconds between inferences (default: 60)')
    parser.add_argument('--mode', type=str, choices=['subscriber', 'test'],
                        default='subscriber',
                        help='Run mode: subscriber (Redis pub/sub), test (demo)')
    
    args = parser.parse_args()
    
    service = NILMRealtimeService(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        model_dir=args.model_dir,
        building_id=args.building_id,
        inference_interval=args.inference_interval
    )
    
    service.start()
    
    if args.mode == 'subscriber':
        service.run_subscriber()
    elif args.mode == 'test':
        logger.info("Running in TEST mode...")
        import random
        
        for i in range(2000):
            sample = RawSample(
                timestamp=time.time(),
                power_total=2000 + random.gauss(0, 500),
            )
            result = service.process_sample(sample)
            
            if i % 100 == 0:
                status = service.buffer.get_buffer_status()
                logger.info(f"Buffer: {status['fill_percent']:.0f}%")
            
            time.sleep(0.01)


if __name__ == "__main__":
    main()
