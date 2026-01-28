"""
Ingest Raw Parquet Data to InfluxDB Raw Bucket
==============================================

This tool reads the simulation parquet file and ingests it into the InfluxDB
raw_readings bucket. This is used to populate the raw bucket for Railway
deployments where the simulator reads from InfluxDB instead of local files.

Usage:
    python -m app.tools.ingest_raw_to_influx

Environment Variables:
    INFLUX_URL: InfluxDB URL (default: http://localhost:8086)
    INFLUX_TOKEN: InfluxDB admin token (required)
    INFLUX_ORG: InfluxDB organization (default: energy-monitor)
    INFLUX_BUCKET_RAW: Raw readings bucket (default: raw_readings)
    PARQUET_PATH: Path to parquet file (default: /app/data/simulation_data.parquet)
    BUILDING_ID: Building identifier (default: building-1)
    BATCH_SIZE: Points per batch write (default: 1000)
    DRY_RUN: If "true", don't actually write to InfluxDB (default: false)

Example:
    # Local development (uses defaults)
    python -m app.tools.ingest_raw_to_influx

    # Production with explicit config
    INFLUX_URL=https://influx.railway.internal:8086 \\
    INFLUX_TOKEN=your-token \\
    python -m app.tools.ingest_raw_to_influx
"""

import asyncio
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import pyarrow.parquet as pq

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.infra.influx.client import init_influx_client, get_influx_client, close_influx_client

logger = get_logger(__name__)


async def main():
    """Main entry point for raw data ingestion."""
    setup_logging("INFO")

    settings = get_settings()

    # Configuration from environment (with overrides for this tool)
    parquet_path = os.getenv("PARQUET_PATH", settings.dataset_path)
    building_id = os.getenv("BUILDING_ID", "building-1")
    batch_size = int(os.getenv("BATCH_SIZE", "1000"))
    dry_run = os.getenv("DRY_RUN", "false").lower() in ("true", "1", "yes")
    power_column = os.getenv("DATA_POWER_COLUMN", "aggregate")

    logger.info("=" * 60)
    logger.info("NILM Raw Data Ingestion Tool")
    logger.info("=" * 60)
    logger.info(f"InfluxDB URL: {settings.influx_url}")
    logger.info(f"Organization: {settings.influx_org}")
    logger.info(f"Raw Bucket: {settings.influx_bucket_raw}")
    logger.info(f"Parquet Path: {parquet_path}")
    logger.info(f"Building ID: {building_id}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Dry Run: {dry_run}")
    logger.info("=" * 60)

    # Validate parquet file exists
    if not os.path.exists(parquet_path):
        logger.error(f"Parquet file not found: {parquet_path}")
        sys.exit(1)

    # Load parquet file
    logger.info(f"Loading parquet file: {parquet_path}")
    start_load = time.time()
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    load_time = time.time() - start_load
    logger.info(f"Loaded {len(df)} rows in {load_time:.2f}s")
    logger.info(f"Columns: {list(df.columns)}")

    # Find power column
    if power_column not in df.columns:
        alternatives = ["aggregate", "aggregate_power", "power", "Aggregate", "P"]
        for alt in alternatives:
            if alt in df.columns:
                power_column = alt
                break
        else:
            logger.error(f"Power column not found. Available: {list(df.columns)}")
            sys.exit(1)

    logger.info(f"Using power column: {power_column}")

    # Check for timestamp column
    has_timestamp = "timestamp" in df.columns
    logger.info(f"Has timestamp column: {has_timestamp}")

    # Initialize InfluxDB client
    if not dry_run:
        logger.info("Connecting to InfluxDB...")
        await init_influx_client()
        influx = get_influx_client()

        # Ensure raw bucket exists
        logger.info("Ensuring raw bucket exists...")
        await influx.ensure_buckets()

        # Check current count
        current_count = await influx.count_raw_readings(building_id)
        logger.info(f"Current raw readings count for {building_id}: {current_count}")

        if current_count > 0:
            logger.warning(
                f"Raw bucket already contains {current_count} readings for {building_id}. "
                f"New data will be added (may create duplicates if timestamps overlap)."
            )

    # Prepare readings for batch insert
    logger.info("Preparing readings for ingestion...")
    total_rows = len(df)
    total_written = 0
    start_ingest = time.time()

    # Generate base timestamp if no timestamp column
    base_time = datetime.now(timezone.utc) - timedelta(seconds=total_rows)

    for batch_start in range(0, total_rows, batch_size):
        batch_end = min(batch_start + batch_size, total_rows)
        batch_df = df.iloc[batch_start:batch_end]

        readings = []
        for idx, row in batch_df.iterrows():
            # Get power value and convert to kW if needed
            power_value = float(row[power_column])
            power_kw = power_value / 1000.0 if power_value > 100 else power_value

            # Get timestamp
            if has_timestamp:
                ts = row["timestamp"]
                if isinstance(ts, str):
                    timestamp = datetime.fromisoformat(ts)
                else:
                    timestamp = ts.to_pydatetime()
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
            else:
                # Generate sequential timestamps at 1Hz
                offset = batch_start + (idx - batch_df.index[0])
                timestamp = base_time + timedelta(seconds=offset)

            readings.append({
                "timestamp": timestamp,
                "aggregate_kw": power_kw,
            })

        # Write batch
        if not dry_run:
            written = await influx.write_raw_readings_batch(
                readings=readings,
                building_id=building_id,
                source="parquet",
            )
            total_written += written
        else:
            total_written += len(readings)

        # Progress update
        progress = (batch_end / total_rows) * 100
        elapsed = time.time() - start_ingest
        rate = total_written / elapsed if elapsed > 0 else 0
        logger.info(
            f"Progress: {batch_end}/{total_rows} ({progress:.1f}%) - "
            f"{total_written} written - {rate:.0f} pts/sec"
        )

    # Cleanup
    if not dry_run:
        await close_influx_client()

    # Final summary
    total_time = time.time() - start_ingest
    logger.info("=" * 60)
    logger.info("Ingestion Complete")
    logger.info("=" * 60)
    logger.info(f"Total rows processed: {total_rows}")
    logger.info(f"Total points written: {total_written}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Average rate: {total_written / total_time:.0f} pts/sec")
    if dry_run:
        logger.info("(DRY RUN - no data actually written)")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
