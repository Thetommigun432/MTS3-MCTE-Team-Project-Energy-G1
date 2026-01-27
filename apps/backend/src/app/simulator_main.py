"""
NILM Simulator - Parquet Data Ingestion Service
================================================

This module provides a standalone service that simulates live sensor ingestion
by reading rows from a parquet dataset and posting them to the backend API.

The simulator:
1. Waits for the backend to be ready (/ready returns 200)
2. Reads the parquet dataset row by row
3. Posts each row to the ingest API at configurable speed
4. Runs continuously until stopped or dataset exhausted

Usage:
    python -m app.simulator_main

Environment Variables:
    BACKEND_URL: Backend base URL (default: http://backend:8000)
    PARQUET_PATH: Path to parquet file (default from config)
    BUILDING_ID: Building identifier (default: building-1)
    DATA_POWER_COLUMN: Column name for aggregate power (default: aggregate)
    SIM_SPEEDUP: Speedup factor (default: 1, meaning 1 row/sec)
    SIM_DURATION_SECONDS: Optional max duration (default: unlimited)
"""

import asyncio
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import NoReturn

import httpx
import pyarrow.parquet as pq

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging

logger = get_logger(__name__)


class DataSimulator:
    """Simulates live data ingestion from parquet file."""

    def __init__(self):
        self.settings = get_settings()
        self.running = False
        self.shutdown_event = asyncio.Event()

        # Configuration from environment
        self.backend_url = os.getenv("BACKEND_URL", "http://backend:8000")
        self.parquet_path = os.getenv("PARQUET_PATH", self.settings.dataset_path)
        self.building_id = os.getenv("BUILDING_ID", "building-1")
        self.power_column = os.getenv("DATA_POWER_COLUMN", "aggregate")
        self.speedup = float(os.getenv("SIM_SPEEDUP", "1"))
        self.duration_seconds = int(os.getenv("SIM_DURATION_SECONDS", "0"))  # 0 = unlimited

        # Stats
        self.rows_sent = 0
        self.errors = 0
        self.start_time = None

    async def wait_for_backend(self, timeout: int = 300) -> bool:
        """Wait for backend to be ready."""
        logger.info(f"Waiting for backend at {self.backend_url}/ready...")

        start = time.time()
        async with httpx.AsyncClient() as client:
            while time.time() - start < timeout:
                try:
                    resp = await client.get(f"{self.backend_url}/ready", timeout=5.0)
                    if resp.status_code == 200:
                        logger.info("Backend is ready!")
                        return True
                except Exception as e:
                    logger.debug(f"Backend not ready yet: {e}")

                await asyncio.sleep(2)

        logger.error(f"Backend did not become ready within {timeout}s")
        return False

    def load_parquet(self):
        """Load the parquet file and validate columns."""
        logger.info(f"Loading parquet from {self.parquet_path}")

        if not os.path.exists(self.parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")

        table = pq.read_table(self.parquet_path)
        df = table.to_pandas()

        logger.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

        # Find the power column
        if self.power_column not in df.columns:
            # Try common alternatives
            alternatives = ["aggregate", "aggregate_power", "power", "Aggregate", "P"]
            for alt in alternatives:
                if alt in df.columns:
                    self.power_column = alt
                    break
            else:
                raise ValueError(
                    f"Power column '{self.power_column}' not found. "
                    f"Available: {list(df.columns)}"
                )

        logger.info(f"Using power column: {self.power_column}")
        return df

    async def send_reading(
        self, client: httpx.AsyncClient, row_idx: int, power_value: float, timestamp: datetime
    ) -> bool:
        """Send a single reading to the backend."""
        try:
            payload = {
                "readings": [
                    {
                        "building_id": self.building_id,
                        "aggregate_kw": power_value / 1000.0 if power_value > 100 else power_value,  # Convert W to kW if needed
                        "ts": timestamp.isoformat(),
                    }
                ]
            }

            resp = await client.post(
                f"{self.backend_url}/api/ingest/readings",
                json=payload,
                timeout=10.0,
            )

            if resp.status_code in (200, 202):
                return True
            else:
                logger.warning(f"Ingest returned {resp.status_code}: {resp.text[:200]}")
                return False

        except Exception as e:
            logger.error(f"Failed to send reading {row_idx}: {e}")
            return False

    async def run(self) -> NoReturn:
        """Main simulation loop."""
        logger.info("=" * 60)
        logger.info("NILM Data Simulator Starting")
        logger.info("=" * 60)
        logger.info(f"Backend URL: {self.backend_url}")
        logger.info(f"Parquet path: {self.parquet_path}")
        logger.info(f"Building ID: {self.building_id}")
        logger.info(f"Power column: {self.power_column}")
        logger.info(f"Speedup: {self.speedup}x")
        logger.info("=" * 60)

        # Wait for backend
        if not await self.wait_for_backend():
            logger.error("Backend not available, exiting")
            sys.exit(1)

        # Load data
        df = self.load_parquet()
        total_rows = len(df)

        # Calculate interval
        interval = 1.0 / self.speedup
        logger.info(f"Interval: {interval:.3f}s between rows ({self.speedup} rows/sec)")

        self.running = True
        self.start_time = time.time()
        row_idx = 0

        async with httpx.AsyncClient() as client:
            while self.running:
                # Check duration limit
                if self.duration_seconds > 0:
                    elapsed = time.time() - self.start_time
                    if elapsed >= self.duration_seconds:
                        logger.info(f"Duration limit reached ({self.duration_seconds}s)")
                        break

                # Get current row (loop if needed)
                actual_idx = row_idx % total_rows
                row = df.iloc[actual_idx]

                # Get power value
                power_value = float(row[self.power_column])

                # Generate timestamp (current time or from dataset if available)
                if "timestamp" in df.columns:
                    ts = row["timestamp"]
                    if isinstance(ts, str):
                        timestamp = datetime.fromisoformat(ts)
                    else:
                        timestamp = ts.to_pydatetime()
                    # Make timezone aware if needed
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                else:
                    timestamp = datetime.now(timezone.utc)

                # Send reading
                success = await self.send_reading(client, row_idx, power_value, timestamp)

                if success:
                    self.rows_sent += 1
                else:
                    self.errors += 1

                # Log progress periodically
                if row_idx > 0 and row_idx % 60 == 0:
                    elapsed = time.time() - self.start_time
                    rate = self.rows_sent / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Progress: {self.rows_sent} sent, {self.errors} errors, "
                        f"rate={rate:.2f}/s, elapsed={elapsed:.1f}s"
                    )

                row_idx += 1

                # Wait for next interval
                await asyncio.sleep(interval)

                # Check for shutdown
                if self.shutdown_event.is_set():
                    break

        # Final stats
        elapsed = time.time() - self.start_time if self.start_time else 0
        logger.info("=" * 60)
        logger.info("Simulator stopped")
        logger.info(f"Total sent: {self.rows_sent}")
        logger.info(f"Total errors: {self.errors}")
        logger.info(f"Elapsed time: {elapsed:.1f}s")
        logger.info("=" * 60)

    def stop(self):
        """Signal shutdown."""
        self.running = False
        self.shutdown_event.set()


async def main():
    """Entry point."""
    setup_logging("INFO")

    simulator = DataSimulator()

    # Setup signal handlers
    loop = asyncio.get_running_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        simulator.stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    try:
        await simulator.run()
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.error(f"Simulator error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
