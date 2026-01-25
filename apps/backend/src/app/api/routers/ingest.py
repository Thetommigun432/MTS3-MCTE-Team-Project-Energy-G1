"""
Router for high-throughput data ingestion.
Supports batching and optional server-to-server auth.
"""

from fastapi import APIRouter, Depends, Header, HTTPException, status
from typing import Annotated, Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.infra.influx import get_influx_client
from app.infra.redis.streams import xadd_reading
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
    # You might want to remove this fallback in PROD
    ingest_token = getattr(settings, "ingest_token", None)
    
    if not ingest_token:
        # Warning only once per boot usually, but acceptable for dev
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
    Writes to Redis Stream for pipeline processing AND InfluxDB for raw storage.
    """
    settings = get_settings()
    stream_key = getattr(settings, "redis_stream_key", "nilm:readings")
    influx = get_influx_client()

    success_count = 0
    errors_count = 0

    # 1. Write to InfluxDB (Raw Storage)
    # We do this synchronously (in async sense) to ensure data safety
    points = []
    for reading in batch.readings:
        # Construct Point object or dictionary for Influx client
        # Assuming client.write_point accepts dict-like structure or specific args
        # We'll use write_point from existing client if available, or bulk write helper
        
        # Check if batch write is available (it should be in a robust client)
        # For now, we'll collect them. In a real scenario, InfluxClient should support batching.
        # Let's write individually for simplicity unless client has batch method
        try:
            await influx.write_point(
                measurement="sensor_reading",
                tags={
                    "building_id": reading.building_id,
                    "appliance_id": "aggregate",
                },
                fields={"aggregate_kw": reading.aggregate_kw},
                timestamp=reading.ts,
            )
            success_count += 1
        except Exception as e:
            logger.error(f"Influx write failed: {e}")
            errors_count += 1
            # Decide if we want to continue or fail. 
            # Usually we want partial success for high throughput.
            continue

        # 2. Write to Redis Stream (Pipeline)
        if getattr(settings, "pipeline_enabled", True):
            try:
                await xadd_reading(
                    stream_key=stream_key,
                    fields={
                        "building_id": reading.building_id,
                        "aggregate_kw": reading.aggregate_kw,
                        "ts": reading.ts.isoformat(),
                    },
                )
            except Exception as e:
                # If Redis fails, we log but don't fail the request if Influx succeeded
                # This means "stored but not processed yet"
                logger.error(f"Redis pipeline enqueue failed: {e}")
                # We don't increment errors_count here since the primary storage (Influx) worked
                pass

    if success_count == 0 and len(batch.readings) > 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist any readings",
        )

    return IngestBatchResponse(ingested=success_count, errors=errors_count)
