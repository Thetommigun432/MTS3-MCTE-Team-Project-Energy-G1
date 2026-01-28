"""
NILM Simulator - Data Ingestion Service
========================================

This module provides a standalone service that simulates live sensor ingestion
by reading rows from either a parquet dataset or InfluxDB raw bucket
and posting them to the backend API.

The simulator:
1. Waits for the backend to be ready (/ready returns 200)
2. Loads data from parquet file (local) or InfluxDB raw bucket (cloud)
3. Posts each reading to the ingest API at configurable speed
4. Runs continuously until stopped, optionally looping through data

Usage:
    python -m app.simulator_main

Environment Variables:
    BACKEND_URL: Backend base URL (default: http://backend:8000)
    BUILDING_ID: Building identifier (default: building-1)
    SIM_SPEEDUP: Speedup factor (default: 1, meaning 1 row/sec)
    SIM_DURATION_SECONDS: Optional max duration (default: 0 = unlimited)
    SIM_LOOP: Loop through data forever (default: true)

    # Data Source Selection
    RAW_DATA_SOURCE: "local" (parquet) or "influx" (InfluxDB raw bucket)

    # For RAW_DATA_SOURCE=local:
    PARQUET_PATH: Path to parquet file (default from config)
    DATA_POWER_COLUMN: Column name for aggregate power (default: aggregate)

    # For RAW_DATA_SOURCE=influx:
    INFLUX_URL: InfluxDB URL
    INFLUX_TOKEN: InfluxDB token
    INFLUX_ORG: InfluxDB organization
    INFLUX_BUCKET_RAW: Raw readings bucket name
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
    """Simulates live data ingestion from parquet file or InfluxDB."""

    def __init__(self):
        self.settings = get_settings()
        self.running = False
        self.shutdown_event = asyncio.Event()

        # Configuration from environment
        self.backend_url = os.getenv("BACKEND_URL", "http://backend:8000")
        self.building_id = os.getenv("BUILDING_ID", "building-1")
        self.speedup = float(os.getenv("SIM_SPEEDUP", "1"))
        self.duration_seconds = int(os.getenv("SIM_DURATION_SECONDS", "0"))  # 0 = unlimited
        self.loop_forever = os.getenv("SIM_LOOP", "true").lower() in ("true", "1", "yes")

        # Data source selection
        self.raw_data_source = os.getenv("RAW_DATA_SOURCE", self.settings.raw_data_source)

        # Parquet-specific config
        self.parquet_path = os.getenv("PARQUET_PATH", self.settings.dataset_path)
        self.power_column = os.getenv("DATA_POWER_COLUMN", "aggregate")

        # Stats
        self.rows_sent = 0
        self.errors = 0
        self.start_time = None
        self.loop_count = 0

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

    def load_parquet(self) -> list[tuple[datetime, float]]:
        """Load the parquet file and return list of (timestamp, power_kw) tuples."""
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

        # Convert to list of (timestamp, power_kw) tuples
        readings: list[tuple[datetime, float]] = []

        for idx, row in df.iterrows():
            # Get power value and convert to kW if needed
            power_value = float(row[self.power_column])
            # Assume values > 100 are in Watts, convert to kW
            power_kw = power_value / 1000.0 if power_value > 100 else power_value

            # Get timestamp
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
                # Generate synthetic timestamps at 1Hz starting from now
                timestamp = datetime.now(timezone.utc)

            readings.append((timestamp, power_kw))

        return readings

    async def load_from_influx(self) -> list[tuple[datetime, float]]:
        """Load raw readings from InfluxDB raw bucket."""
        logger.info("Loading data from InfluxDB raw bucket...")

        from app.infra.influx.client import init_influx_client, get_influx_client, close_influx_client

        try:
            await init_influx_client()
            influx = get_influx_client()

            # Check how many readings exist
            count = await influx.count_raw_readings(self.building_id)
            logger.info(f"Found {count} raw readings for building {self.building_id}")

            if count == 0:
                raise ValueError(
                    f"No raw readings found in InfluxDB for building '{self.building_id}'. "
                    f"Run 'python -m app.tools.ingest_raw_to_influx' first to populate the raw bucket."
                )

            # Query all data from raw bucket
            readings = await influx.query_raw_readings(
                building_id=self.building_id,
                start="-365d",  # Last year
                end="now()",
                limit=3000000,  # ~1 month at 1Hz
            )

            logger.info(f"Loaded {len(readings)} readings from InfluxDB")
            return readings

        finally:
            await close_influx_client()

    async def send_reading(
        self, client: httpx.AsyncClient, row_idx: int, power_kw: float, timestamp: datetime
    ) -> bool:
        """Send a single reading to the backend."""
        try:
            payload = {
                "readings": [
                    {
                        "building_id": self.building_id,
                        "aggregate_kw": power_kw,
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
        logger.info(f"Building ID: {self.building_id}")
        logger.info(f"Data source: {self.raw_data_source}")
        logger.info(f"Speedup: {self.speedup}x")
        logger.info(f"Loop forever: {self.loop_forever}")
        if self.raw_data_source == "local":
            logger.info(f"Parquet path: {self.parquet_path}")
            logger.info(f"Power column: {self.power_column}")
        logger.info("=" * 60)

        # Wait for backend
        if not await self.wait_for_backend():
            logger.error("Backend not available, exiting")
            sys.exit(1)

        # Load data based on source
        if self.raw_data_source == "influx":
            readings = await self.load_from_influx()
        else:
            readings = self.load_parquet()

        total_rows = len(readings)
        logger.info(f"Total readings to replay: {total_rows}")

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

                # Get current reading (loop if enabled)
                actual_idx = row_idx % total_rows
                timestamp, power_kw = readings[actual_idx]

                # For replay, use current time instead of original timestamp
                current_timestamp = datetime.now(timezone.utc)

                # Send reading
                success = await self.send_reading(client, row_idx, power_kw, current_timestamp)

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
                        f"rate={rate:.2f}/s, elapsed={elapsed:.1f}s, loop={self.loop_count}"
                    )

                row_idx += 1

                # Check if we completed a loop
                if actual_idx == total_rows - 1:
                    self.loop_count += 1
                    logger.info(f"Completed loop {self.loop_count}")
                    if not self.loop_forever:
                        logger.info("Loop disabled, stopping after first pass")
                        break

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
        logger.info(f"Loops completed: {self.loop_count}")
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
