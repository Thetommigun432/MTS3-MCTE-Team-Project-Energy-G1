"""
NILM Inference Worker - Standalone Entrypoint
==============================================

This module provides a dedicated entrypoint for running the Redis inference
worker as a separate service from the API. This is required for Railway
deployment where API and Worker run as separate services.

Usage:
    python -m app.worker_main

Environment Variables:
    REDIS_URL: Redis connection URL
    INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG: InfluxDB configuration
    MODEL_REGISTRY_PATH: Path to registry.json (default: models/registry.json)
"""

import asyncio
import signal
import sys
from typing import NoReturn

from app.core.config import get_settings
from app.core.logging import get_logger
from app.domain.pipeline.redis_inference_worker import RedisInferenceWorker

logger = get_logger(__name__)


class WorkerRunner:
    """Manages the worker lifecycle with graceful shutdown."""

    def __init__(self):
        self.worker: RedisInferenceWorker | None = None
        self.shutdown_event = asyncio.Event()

    async def start(self) -> NoReturn:
        """Start the worker and run until shutdown."""
        settings = get_settings()

        logger.info(
            "Starting NILM Inference Worker",
            extra={
                "redis_url": settings.redis_url[:20] + "..." if settings.redis_url else "not set",
                "influx_url": settings.influx_url,
            },
        )
        
        # Validate dataset and models
        import os
        if os.path.exists(settings.dataset_path):
            size_mb = os.path.getsize(settings.dataset_path) / (1024 * 1024)
            logger.info(f"Dataset found at {settings.dataset_path} ({size_mb:.2f} MB)")
        else:
            logger.error(f"Dataset NOT found at {settings.dataset_path}")
            # Worker strictly needs data/models? Maybe not strict for startup, but good to know.
            
        if os.path.isdir(settings.models_dir):
            logger.info(f"Models directory found at {settings.models_dir}")
        else:
            logger.error(f"Models directory NOT found at {settings.models_dir}")

        # Initialize Redis connection (CRITICAL FIX)
        from app.infra.redis.client import init_redis_cache, close_redis_cache
        await init_redis_cache()

        # Create worker
        self.worker = RedisInferenceWorker()

        # Setup signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._signal_handler)

        try:
            # Start worker
            # Note: worker.start() contains the loop, so we run it directly.
            # However, looking at RedisInferenceWorker.start(), it runs a while loop.
            # If we await it, it blocks until stop() is called. Used task here or await directly?
            # RedisInferenceWorker.start() is a while loop. We should let it run.
            # But we also need to listen for shutdown event.
            # Better pattern: Run worker in a task.
            # Start worker task
            worker_task = asyncio.create_task(self.worker.start())
            
            # Wait for shutdown signal (or worker failure)
            done, pending = await asyncio.wait(
                [asyncio.create_task(self.shutdown_event.wait()), worker_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            if worker_task in done:
                # Worker finished (error or break), re-raise if exception
                worker_task.result()
                
        except Exception as e:
            logger.error(f"Worker error: {e}")
            raise
        finally:
            await self._shutdown()
            await close_redis_cache()

    def _signal_handler(self):
        """Handle shutdown signals."""
        logger.info("Received shutdown signal")
        self.shutdown_event.set()

    async def _shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down worker...")
        if self.worker:
            await self.worker.stop()
        logger.info("Worker stopped")


def main():
    """Main entry point."""
    runner = WorkerRunner()

    try:
        asyncio.run(runner.start())
    except KeyboardInterrupt:
        logger.info("Worker interrupted")
    except Exception as e:
        logger.error(f"Worker failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
