"""
E2E probe endpoints for Railway pipeline testing.

These endpoints allow testing the deployed pipeline without exposing
internal infrastructure (Redis, InfluxDB) publicly.

Security:
- All endpoints require E2E_PROBES_ENABLED=true
- All endpoints require valid X-E2E-Token header
- Endpoints are hidden from OpenAPI documentation
"""

import json
import os
import time
from typing import Any

import redis
from fastapi import APIRouter, Depends, Query

from app.api.deps import RequestIdDep, require_e2e_token
from app.core.config import get_settings
from app.core.logging import get_logger
from app.domain.inference.preprocessing import DataPreprocessor
from app.infra.influx import get_influx_client
from app.schemas.e2e import (
    E2EInfluxStatusResponse,
    E2EInjectRequest,
    E2EInjectResponse,
    E2EPreprocessRequest,
    E2EPreprocessResponse,
    E2ERedisBufferResponse,
)

logger = get_logger(__name__)

router = APIRouter(
    prefix="/e2e",
    tags=["E2E Testing"],
    dependencies=[Depends(require_e2e_token)],
    include_in_schema=False,  # Hide from OpenAPI docs
)


@router.post("/inject", response_model=E2EInjectResponse)
async def inject_sample(
    request: E2EInjectRequest,
    request_id: RequestIdDep,
) -> E2EInjectResponse:
    """
    Inject a sample into the REAL pipeline via Redis pub/sub.

    This endpoint:
    1. Preprocesses the sample using the SAME DataPreprocessor as the inference worker
    2. Publishes to the SAME Redis channel the producer uses
    3. Includes run_id in the message for correlation tracking

    The sample will flow through the full pipeline:
    inject -> Redis pub/sub -> inference worker -> persister -> InfluxDB
    """
    settings = get_settings()

    # 1. Preprocess using the REAL preprocessing logic
    preprocessor = DataPreprocessor(P_MAX=15000.0)
    features = preprocessor.process_sample(request.timestamp, request.power_watts)

    # 2. Build message in the same format as producer.py
    channel = f"nilm:{request.building_id}:input"
    message = {
        "timestamp": request.timestamp,
        "power_total": request.power_watts,
        "voltage": request.voltage,
        "current": request.current or (request.power_watts / request.voltage),
        "power_factor": request.power_factor,
        "run_id": request.run_id,  # Propagate for tracking through pipeline
    }

    # 3. Publish to Redis (same channel as producer)
    redis_published = False
    redis_url = settings.redis_url

    if redis_url:
        try:
            r = redis.from_url(redis_url)
            r.publish(channel, json.dumps(message))
            redis_published = True
            logger.info(
                f"E2E inject: published to {channel}",
                extra={"run_id": request.run_id, "request_id": request_id},
            )
        except Exception as e:
            logger.error(f"E2E inject: Redis publish failed: {e}")
    else:
        logger.warning("E2E inject: REDIS_URL not configured")

    return E2EInjectResponse(
        status="ok",
        run_id=request.run_id,
        preprocessed=features.tolist(),
        redis_published=redis_published,
        channel=channel,
    )


@router.post("/preprocess", response_model=E2EPreprocessResponse)
async def preprocess_sample(
    request: E2EPreprocessRequest,
    request_id: RequestIdDep,
) -> E2EPreprocessResponse:
    """
    Test preprocessing without side effects.

    Uses the SAME DataPreprocessor as the inference worker.
    Returns the 7-element feature vector that would be used for inference.
    """
    preprocessor = DataPreprocessor(P_MAX=15000.0)
    features = preprocessor.process_sample(request.timestamp, request.power_watts)

    return E2EPreprocessResponse(
        features=features.tolist(),
        feature_names=[
            "aggregate_norm",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
        ],
    )


@router.get("/redis-buffer", response_model=E2ERedisBufferResponse)
async def redis_buffer_status(
    request_id: RequestIdDep,
    building_id: str = Query(default="building_1"),
) -> E2ERedisBufferResponse:
    """
    Check Redis buffer status for a building.

    Returns information about the inference buffer:
    - Current buffer length
    - Window size requirement
    - Oldest/newest timestamps
    """
    settings = get_settings()

    features_key = f"nilm:{building_id}:features"
    timestamps_key = f"nilm:{building_id}:timestamps"
    window_size = int(os.environ.get("WINDOW_SIZE", "3600"))

    if not settings.redis_url:
        return E2ERedisBufferResponse(
            building_id=building_id,
            features_key=features_key,
            buffer_length=0,
            window_size=window_size,
            buffer_full=False,
            oldest_timestamp=None,
            newest_timestamp=None,
        )

    try:
        r = redis.from_url(settings.redis_url)
        buffer_length = r.llen(features_key)

        oldest_ts = None
        newest_ts = None

        if buffer_length > 0:
            oldest_bytes = r.lindex(timestamps_key, 0)
            newest_bytes = r.lindex(timestamps_key, -1)
            if oldest_bytes:
                oldest_ts = float(oldest_bytes)
            if newest_bytes:
                newest_ts = float(newest_bytes)

        return E2ERedisBufferResponse(
            building_id=building_id,
            features_key=features_key,
            buffer_length=buffer_length,
            window_size=window_size,
            buffer_full=buffer_length >= window_size,
            oldest_timestamp=oldest_ts,
            newest_timestamp=newest_ts,
        )

    except Exception as e:
        logger.error(f"E2E redis-buffer: error checking buffer: {e}")
        return E2ERedisBufferResponse(
            building_id=building_id,
            features_key=features_key,
            buffer_length=0,
            window_size=window_size,
            buffer_full=False,
            oldest_timestamp=None,
            newest_timestamp=None,
        )


@router.get("/influx-status", response_model=E2EInfluxStatusResponse)
async def influx_status(
    request_id: RequestIdDep,
    run_id: str = Query(..., description="Run ID to search for"),
) -> E2EInfluxStatusResponse:
    """
    Check InfluxDB for predictions with given run_id.

    Queries the predictions bucket for records tagged with the specified run_id.
    Used to verify that injected samples flow through the entire pipeline.
    """
    settings = get_settings()
    influx = get_influx_client()

    t0 = time.time()

    # Build Flux query filtering by run_id tag
    query = f'''
        from(bucket: "{settings.influx_bucket_pred}")
        |> range(start: -24h)
        |> filter(fn: (r) => r["_measurement"] == "prediction")
        |> filter(fn: (r) => r["run_id"] == "{run_id}")
        |> limit(n: 5)
    '''

    records: list[dict[str, Any]] = []

    try:
        query_api = influx._client.query_api()
        tables = query_api.query(query)

        for table in tables:
            for record in table.records:
                records.append(
                    {
                        "time": record.get_time().isoformat() if record.get_time() else None,
                        "field": record.get_field(),
                        "value": record.get_value(),
                        "building_id": record.values.get("building_id"),
                        "model_version": record.values.get("model_version"),
                    }
                )

    except Exception as e:
        logger.error(f"E2E influx-status: query failed: {e}")

    query_time_ms = (time.time() - t0) * 1000

    return E2EInfluxStatusResponse(
        found=len(records) > 0,
        run_id=run_id,
        records_count=len(records),
        sample_record=records[0] if records else None,
        query_time_ms=query_time_ms,
    )
