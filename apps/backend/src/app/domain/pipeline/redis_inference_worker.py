"""
Redis Inference Worker.
Consumes power readings from Redis Stream, buffers them, and triggers inference
updates to InfluxDB predictions bucket.
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, List, Tuple
from collections import deque

from app.core.config import get_settings
from app.core.logging import get_logger
from app.domain.inference import get_inference_engine, get_model_registry
from app.infra.influx import get_influx_client
from app.infra.redis.streams import ack, ensure_group, read_group

logger = get_logger(__name__)


class RedisInferenceWorker:
    """
    Background worker that processes readings from Redis Streams.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.running = False
        self.consumer_name = f"worker-{datetime.now().timestamp()}"
        
        # Buffer: building_id -> deque of (ts, aggregate_kw)
        self.buffers: Dict[str, deque[Tuple[float, float]]] = {}
        # Counter: building_id -> items since last inference
        self.counters: Dict[str, int] = {}
        
    async def start(self) -> None:
        """Start the worker loop."""
        logger.info("Starting Redis Inference Worker")
        await ensure_group(
            self.settings.redis_stream_key, self.settings.redis_consumer_group
        )
        self.running = True
        
        while self.running:
            try:
                await self._process_batch()
            except Exception as e:
                logger.error(f"Worker iteration failed: {e}")
                await asyncio.sleep(5)  # Backoff on error
    
    async def stop(self) -> None:
        """Stop the worker."""
        self.running = False
        logger.info("Stopping Redis Inference Worker")

    async def _process_batch(self) -> None:
        """Process a batch of messages from Redis."""
        messages = await read_group(
            self.settings.redis_stream_key,
            self.settings.redis_consumer_group,
            self.consumer_name,
            count=100,
            block_ms=2000,
        )

        if not messages:
            return

        msg_ids_to_ack = []
        
        for msg_id, fields in messages:
            try:
                building_id = fields.get("building_id")
                # Redis streams return strings, parse them
                # ts might be ISO string
                ts_str = fields.get("ts")
                agg_kw = float(fields.get("aggregate_kw", 0.0))
                
                if not building_id or not ts_str:
                    logger.warning(f"Malformed message {msg_id}: {fields}")
                    msg_ids_to_ack.append(msg_id)
                    continue

                # Parse timestamp to float timestamp for windowing
                try:
                    dt = datetime.fromisoformat(str(ts_str))
                    ts = dt.timestamp()
                except ValueError:
                    logger.warning(f"Invalid timestamp in message {msg_id}: {ts_str}")
                    msg_ids_to_ack.append(msg_id)
                    continue

                # Add to buffer
                await self._add_to_buffer(building_id, ts, agg_kw, dt)
                msg_ids_to_ack.append(msg_id)

            except Exception as e:
                logger.error(f"Failed to process message {msg_id}: {e}")
                # We ack poison messages to verify they don't block
                msg_ids_to_ack.append(msg_id)

        # Batch ACK
        if msg_ids_to_ack:
            await ack(
                self.settings.redis_stream_key,
                self.settings.redis_consumer_group,
                msg_ids_to_ack,
            )

    async def _add_to_buffer(
        self, building_id: str, ts: float, agg_kw: float, dt_obj: datetime
    ) -> None:
        """Add reading to buffer and trigger inference if needed."""
        if building_id not in self.buffers:
            self.buffers[building_id] = deque(maxlen=self.settings.pipeline_max_buffer)
            self.counters[building_id] = 0

        self.buffers[building_id].append((ts, agg_kw))
        self.counters[building_id] += 1

        # Check inference trigger
        # Stride: Run every N new samples
        if self.counters[building_id] >= self.settings.pipeline_stride:
            await self._try_inference(building_id, dt_obj)
            self.counters[building_id] = 0

    async def _try_inference(self, building_id: str, current_dt: datetime) -> None:
        """Attempt to run inference for a building."""
        engine = get_inference_engine()
        registry = get_model_registry()
        
        # Determine active model and required window
        # For simplicity, we grab the first active model enabled for pipeline
        active_models = registry.list_active_models()
        if not active_models:
            return

        # Use the first one (e.g., transformer-hybrid)
        model_meta = active_models[0]
        window_size = model_meta.input_window_size
        
        # Check if we have enough data
        # We need `window_size` samples ending at the current timestamp
        # In a real system, we'd ensure contiguous timestamps.
        # Here we just take the last N samples from the buffer.
        
        buffer = self.buffers[building_id]
        if len(buffer) < window_size:
            # Not enough data yet
            return

        # Extract window data
        # Takes the *last* window_size elements
        window_data = list(buffer)[-window_size:]
        
        # Prepare input for engine: list of floats
        # Engine expects just the values
        aggregate_window = [kw for _, kw in window_data]
        
        # Run inference
        try:
            logger.debug(f"Running inference for {building_id} with model {model_meta.model_id}")
            results = await engine.run_inference_multi_head(
                model_id=model_meta.model_id,
                aggregate_window=aggregate_window,
            )
            
            # Persist predictions
            # Results is { appliance_key: (watts, confidence) }
            # Convert to Influx fields
            fields = {}
            for appliance, (kw, conf) in results.items():
                fields[f"predicted_kw_{appliance}"] = kw
                if conf is not None:
                    fields[f"confidence_{appliance}"] = conf

            influx = get_influx_client()
            await influx.write_predictions_wide(
                building_id=building_id,
                model_version=model_meta.model_version,
                timestamp=current_dt,
                fields=fields,
            )
            logger.info(f"Persisted inference for {building_id} @ {current_dt.isoformat()}")

        except Exception as e:
            logger.error(f"Inference failed for {building_id}: {e}")
