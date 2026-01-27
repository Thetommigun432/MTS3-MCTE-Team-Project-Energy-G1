"""
Redis Inference Worker.
Consumes power readings from Redis Stream, reads rolling window from Redis,
runs inference for each active model, and writes predictions to InfluxDB.

Pipeline flow:
1. Stream event arrives (building_id, ts)
2. Read rolling window from Redis (already maintained by ingest endpoint)
3. For each active model, run inference if window is large enough
4. Write predictions to InfluxDB
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List

from app.core.config import get_settings
from app.core.logging import get_logger
from app.domain.inference.engine import get_inference_engine
from app.domain.inference.registry import get_model_registry, ModelEntry
from app.domain.inference.postprocessing import enforce_sum_constraint_smart, compute_residual
from app.infra.influx import get_influx_client, init_influx_client
from app.infra.redis.streams import ack, ensure_group, read_group
from app.infra.redis.rolling_window import get_window_values, get_window_samples, get_window_length

logger = get_logger(__name__)


class RedisInferenceWorker:
    """
    Background worker that processes readings from Redis Streams.

    Unlike the previous implementation that buffered in-process,
    this worker reads from the shared Redis rolling window maintained
    by the ingest endpoint. This enables horizontal scaling of workers.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.running = False
        self.consumer_name = f"worker-{int(datetime.now().timestamp())}"
        # Track last inference time per building to implement coalescing
        self._last_inference: Dict[str, float] = {}
        self._influx_initialized = False

    async def start(self) -> None:
        """Start the worker loop."""
        logger.info("Starting Redis Inference Worker")

        # Initialize InfluxDB connection (CRITICAL - worker needs this)
        if not self._influx_initialized:
            try:
                await init_influx_client()
                self._influx_initialized = True
                logger.info("InfluxDB client initialized for worker")
            except Exception as e:
                logger.error(f"Failed to initialize InfluxDB: {e}")
                raise

        # Load model registry
        try:
            registry = get_model_registry()
            if not registry.is_loaded:
                registry.load()
            logger.info(f"Model registry loaded: {len(registry.list_all())} models")
        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")
            raise

        # Ensure consumer group exists
        await ensure_group(
            self.settings.redis_stream_key, self.settings.redis_consumer_group
        )

        self.running = True
        logger.info(
            f"Worker ready: stream={self.settings.redis_stream_key}, "
            f"group={self.settings.redis_consumer_group}, "
            f"consumer={self.consumer_name}"
        )

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
        """Process a batch of messages from Redis Stream."""
        # Health check: ensure Redis connection is alive
        try:
            from app.infra.redis.client import get_redis_cache
            cache = get_redis_cache()
            if cache._redis:
                await cache._redis.ping()
        except Exception as e:
            logger.warning(f"Redis health check failed, attempting reconnect: {e}")
            try:
                from app.infra.redis.client import init_redis_cache
                await init_redis_cache()
            except Exception as e2:
                logger.error(f"Redis reconnect failed: {e2}")
                raise

        messages = await read_group(
            self.settings.redis_stream_key,
            self.settings.redis_consumer_group,
            self.consumer_name,
            count=100,
            block_ms=2000,
        )

        if not messages:
            return

        # Group messages by building_id and take only the latest per building
        # This implements message coalescing to avoid redundant inference
        latest_by_building: Dict[str, tuple] = {}
        all_msg_ids = []

        for msg_id, fields in messages:
            all_msg_ids.append(msg_id)
            building_id = fields.get("building_id")
            if building_id:
                latest_by_building[building_id] = (msg_id, fields)

        # Process only the latest message per building
        for building_id, (msg_id, fields) in latest_by_building.items():
            try:
                ts_str = fields.get("ts")
                if not ts_str:
                    logger.warning(f"Missing timestamp in message {msg_id}")
                    continue

                # Parse timestamp
                try:
                    dt = datetime.fromisoformat(str(ts_str))
                except ValueError:
                    logger.warning(f"Invalid timestamp: {ts_str}")
                    continue

                # Run inference for this building
                await self._run_inference_for_building(building_id, dt)

            except Exception as e:
                logger.error(f"Failed to process building {building_id}: {e}")

        # ACK all messages (including coalesced ones)
        if all_msg_ids:
            await ack(
                self.settings.redis_stream_key,
                self.settings.redis_consumer_group,
                all_msg_ids,
            )

    async def _run_inference_for_building(
        self, building_id: str, event_time: datetime
    ) -> None:
        """
        Run inference for a building using the Redis rolling window.

        This reads the window from Redis (shared state) rather than
        maintaining an in-process buffer.
        """
        registry = get_model_registry()
        engine = get_inference_engine()
        influx = get_influx_client()

        # Get all active models
        active_models = [m for m in registry.list_all() if m.is_active]
        if not active_models:
            logger.debug("No active models in registry")
            return

        # Get current window length
        try:
            window_len = await get_window_length(building_id)
        except Exception as e:
            logger.warning(f"Failed to get window length: {e}")
            return

        # Collect all predictions from all active models
        all_predictions: Dict[str, Tuple[float, float]] = {}
        aggregate_power_kw = 0.0  # Will be extracted from window

        # Run inference for each active model
        for model_entry in active_models:
            required_size = model_entry.input_window_size

            if window_len < required_size:
                logger.debug(
                    f"Warming up: building={building_id}, "
                    f"model={model_entry.model_id}, "
                    f"window={window_len}/{required_size}"
                )
                continue

            try:
                # Read the window values from Redis (full samples for temporal features)
                window_samples = await get_window_samples(
                    building_id, last_n=required_size
                )

                if len(window_samples) < required_size:
                    logger.debug(f"Insufficient window data: {len(window_samples)}/{required_size}")
                    continue

                # Extract power values for backward compatibility and basic scaling
                window_values = [s[1] for s in window_samples]

                # Get aggregate power from the MOST RECENT sample (for constraint enforcement)
                if window_samples:
                    aggregate_power_kw = window_samples[-1][1]  # Most recent power reading

                # Load the model and run inference
                model, entry = engine.get_model(model_entry.model_id, model_entry.appliance_id)

                # Run multi-head inference
                predictions = engine.run_inference_multi_head(
                    model=model,
                    entry=entry,
                    window=window_values,
                    samples=window_samples,
                )

                # Accumulate predictions (one model = one appliance typically)
                # Results format: {field_key: (predicted_kw, confidence)}
                all_predictions.update(predictions)

            except Exception as e:
                logger.error(
                    f"Inference failed: building={building_id}, "
                    f"model={model_entry.model_id}, error={e}"
                )

        # Apply Smart Priority Scaling post-processing
        if all_predictions:
            corrected_predictions = enforce_sum_constraint_smart(
                predictions=all_predictions,
                aggregate_power_kw=aggregate_power_kw,
            )

            # Compute residual (ghost load)
            residual_kw = compute_residual(corrected_predictions, aggregate_power_kw)
            logger.debug(
                f"Residual power: {residual_kw:.3f}kW "
                f"({residual_kw/aggregate_power_kw*100:.1f}% of aggregate)"
            )

            # Write predictions to InfluxDB
            try:
                await influx.write_predictions_wide(
                    building_id=building_id,
                    predictions=corrected_predictions,
                    model_version="ensemble-v1",  # Multiple models
                    user_id="pipeline",
                    request_id=f"worker-{self.consumer_name}",
                    latency_ms=0.0,  # Could measure actual latency
                    timestamp=event_time,
                )
                logger.info(
                    f"Predictions written: building={building_id}, "
                    f"appliances={len(corrected_predictions)}, "
                    f"aggregate={aggregate_power_kw:.3f}kW, "
                    f"sum={sum(p for p, _ in corrected_predictions.values()):.3f}kW"
                )
            except Exception as e:
                logger.error(f"Failed to write predictions to InfluxDB: {e}")

