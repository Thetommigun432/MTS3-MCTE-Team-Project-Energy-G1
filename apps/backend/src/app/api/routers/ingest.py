"""
Router for high-throughput data ingestion.
Supports batching and optional server-to-server auth.
Updates Redis rolling window and enqueues to stream for worker processing.
"""

import json
from fastapi import APIRouter, Depends, Header, HTTPException, status
from typing import Annotated, Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.infra.influx import get_influx_client
from app.infra.redis.streams import xadd_reading
from app.infra.redis.rolling_window import update_rolling_window
from app.schemas.ingest import IngestBatchRequest, IngestBatchResponse

router = APIRouter(prefix="/ingest", tags=["Ingestion"])
logger = get_logger(__name__)


async def verify_ingest_token(
    x_ingest_token: Annotated[Optional[str], Header()] = None
) -> None:
    """
    Verify the ingestion token if configured.
    """
    settings = get_settings()

    # If no token configured in env (e.g. dev), allow open access
    ingest_token = settings.ingest_token

    if not ingest_token:
        return

    if x_ingest_token != ingest_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid ingestion token",
        )


@router.post(
    "/readings",
    response_model=IngestBatchResponse,
    dependencies=[Depends(verify_ingest_token)],
    status_code=status.HTTP_202_ACCEPTED,
)
async def ingest_readings(batch: IngestBatchRequest) -> IngestBatchResponse:
    """
    Ingest a batch of power readings.

    Pipeline flow:
    1. Preprocess each reading (extract aggregate power value)
    2. Update Redis rolling window (RPUSH/LTRIM to maintain 3600 samples)
    3. Enqueue to Redis Stream for worker processing
    4. Optionally write raw reading to InfluxDB
    """
    settings = get_settings()
    stream_key = settings.redis_stream_key
    rolling_window_size = settings.pipeline_rolling_window_size
    enqueue_enabled = settings.pipeline_enqueue_enabled

    success_count = 0
    errors_count = 0

    for reading in batch.readings:
        try:
            # 1. Preprocess: extract the aggregate power value
            # The reading.aggregate_kw is already the preprocessed value
            preprocessed_value = float(reading.aggregate_kw)
            ts_iso = reading.ts.isoformat()
            building_id = reading.building_id

            # 2. Update Redis rolling window (enforces 3600 max)
            if enqueue_enabled:
                try:
                    window_len = await update_rolling_window(
                        building_id=building_id,
                        timestamp=ts_iso,
                        value=preprocessed_value,
                        max_size=rolling_window_size,
                    )
                    logger.debug(
                        f"Rolling window updated: building={building_id}, len={window_len}"
                    )
                except Exception as e:
                    logger.warning(f"Rolling window update failed: {e}")
                    # Continue - we can still enqueue the event

            # 3. Enqueue to Redis Stream for worker
            if enqueue_enabled:
                try:
                    await xadd_reading(
                        stream_key=stream_key,
                        fields={
                            "building_id": building_id,
                            "aggregate_kw": str(preprocessed_value),
                            "ts": ts_iso,
                        },
                    )
                except Exception as e:
                    logger.error(f"Redis stream enqueue failed: {e}")
                    # Don't fail the whole request if stream write fails

            # 4. Optionally write to InfluxDB (raw storage)
            # For the local pipeline, we focus on predictions not raw readings
            # but we keep this for observability
            try:
                influx = get_influx_client()
                await influx.write_point(
                    measurement="sensor_reading",
                    tags={
                        "building_id": building_id,
                        "appliance_id": "aggregate",
                    },
                    fields={"aggregate_kw": preprocessed_value},
                    timestamp=reading.ts,
                )
            except Exception as e:
                logger.debug(f"Influx raw write failed (non-critical): {e}")
                # Don't fail - raw storage is optional for pipeline

            success_count += 1

        except Exception as e:
            logger.error(f"Failed to process reading: {e}")
            errors_count += 1
            continue

    if success_count == 0 and len(batch.readings) > 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process any readings",
        )

    return IngestBatchResponse(ingested=success_count, errors=errors_count)
