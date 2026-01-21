"""
InfluxDB async client wrapper.
Provides query and write operations with error handling and retries.
"""

import time
from datetime import datetime, timezone
from typing import Any

from influxdb_client import Point
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from influxdb_client.client.write_api_async import WriteApiAsync

from app.core.config import get_settings
from app.core.errors import ErrorCode, InfluxError
from app.core.logging import get_logger
from app.core.telemetry import (
    INFLUX_QUERY_LATENCY,
    INFLUX_WRITE_COUNT,
    INFLUX_WRITE_LATENCY,
)
from app.infra.influx.queries import build_predictions_query, build_readings_query
from app.schemas.analytics import DataPoint, PredictionPoint, Resolution

logger = get_logger(__name__)


class InfluxClient:
    """Async InfluxDB client wrapper."""

    def __init__(self) -> None:
        self._client: InfluxDBClientAsync | None = None
        self._write_api: WriteApiAsync | None = None

    async def connect(self) -> None:
        """Initialize the InfluxDB client connection."""
        settings = get_settings()

        if not settings.influx_token:
            logger.warning("INFLUX_TOKEN not set, InfluxDB operations will fail")

        self._client = InfluxDBClientAsync(
            url=settings.influx_url,
            token=settings.influx_token,
            org=settings.influx_org,
            timeout=settings.influx_timeout_ms,
        )
        self._write_api = self._client.write_api()
        logger.info("InfluxDB client connected", extra={"url": settings.influx_url})

    async def close(self) -> None:
        """Close the InfluxDB client connection."""
        if self._client:
            await self._client.close()
            self._client = None
            self._write_api = None
            logger.info("InfluxDB client closed")

    async def ping(self) -> bool:
        """Check if InfluxDB is reachable."""
        if not self._client:
            return False
        try:
            return await self._client.ping()
        except Exception as e:
            logger.error("InfluxDB ping failed", extra={"error": str(e)})
            return False

    async def verify_setup(self) -> dict[str, bool]:
        """
        Verify InfluxDB setup (connection + buckets).
        Returns a dict of status checks.
        """
        status = {
            "connected": False,
            "bucket_raw": False,
            "bucket_pred": False
        }
        
        # 1. Check connection
        if not await self.ping():
            return status
        status["connected"] = True

        # 2. Check buckets
        settings = get_settings()
        try:
            buckets_api = self._client.buckets_api()
            
            # Get all buckets once to reduce API calls
            buckets = await buckets_api.find_buckets()
            bucket_names = {b.name for b in buckets.buckets}
            
            status["bucket_raw"] = settings.influx_bucket_raw in bucket_names
            status["bucket_pred"] = settings.influx_bucket_pred in bucket_names
            
        except Exception as e:
            logger.error("Failed to list buckets during verification", extra={"error": str(e)})
            
        return status

    async def bucket_exists(self, bucket_name: str) -> bool:
        """Check if a bucket exists."""
        if not self._client:
            return False
        try:
            buckets_api = self._client.buckets_api()
            bucket = await buckets_api.find_bucket_by_name(bucket_name)
            return bucket is not None
        except Exception as e:
            logger.error("Bucket check failed", extra={"bucket": bucket_name, "error": str(e)})
            return False

    async def ensure_predictions_bucket(self) -> bool:
        """
        Ensure the predictions bucket exists, creating it if necessary.
        
        This is called during startup to ensure the bucket exists before
        the readiness check runs.
        
        Returns:
            True if bucket exists or was created successfully
        """
        settings = get_settings()
        bucket_name = settings.influx_bucket_pred
        
        if not self._client:
            logger.error("Cannot ensure predictions bucket: client not connected")
            return False
        
        try:
            buckets_api = self._client.buckets_api()
            
            # Check if bucket already exists
            existing = await buckets_api.find_bucket_by_name(bucket_name)
            if existing:
                logger.debug("Predictions bucket already exists", extra={"bucket": bucket_name})
                return True
            
            # Get organization ID
            orgs_api = self._client.organizations_api()
            orgs = await orgs_api.find_organizations(org=settings.influx_org)
            if not orgs:
                logger.error("Organization not found", extra={"org": settings.influx_org})
                return False
            
            org_id = orgs[0].id
            
            # Create the bucket with infinite retention (0 = never expire)
            from influxdb_client import BucketRetentionRules
            await buckets_api.create_bucket(
                bucket_name=bucket_name,
                org_id=org_id,
                retention_rules=[BucketRetentionRules(type="expire", every_seconds=0)]
            )
            
            logger.info("Created predictions bucket", extra={"bucket": bucket_name, "org": settings.influx_org})
            return True
            
        except Exception as e:
            logger.error("Failed to ensure predictions bucket", extra={"bucket": bucket_name, "error": str(e)})
            return False

    async def query_readings(
        self,
        building_id: str,
        appliance_id: str | None,
        start: str,
        end: str,
        resolution: Resolution,
    ) -> list[DataPoint]:
        """
        Query sensor readings from InfluxDB.

        Returns:
            List of DataPoint objects
        """
        settings = get_settings()
        start_time = time.time()

        try:
            if not self._client:
                raise InfluxError(
                    code=ErrorCode.INFLUX_CONNECTION_ERROR,
                    message="InfluxDB client not connected",
                )

            query = build_readings_query(
                bucket=settings.influx_bucket_raw,
                building_id=building_id,
                appliance_id=appliance_id,
                start=start,
                end=end,
                resolution=resolution,
            )

            query_api = self._client.query_api()
            tables = await query_api.query(query, org=settings.influx_org)

            data_points: list[DataPoint] = []
            for table in tables:
                for record in table.records:
                    # Extract value from pivoted data
                    value = record.values.get("aggregate_kw") or record.values.get("power_kw") or 0.0
                    data_points.append(
                        DataPoint(
                            time=record.get_time().isoformat() if record.get_time() else "",
                            value=float(value),
                        )
                    )

            logger.debug(
                "Query readings completed",
                extra={"building_id": building_id, "count": len(data_points)},
            )
            return data_points

        except InfluxError:
            raise
        except Exception as e:
            logger.error("Query readings failed", extra={"error": str(e)})
            raise InfluxError(
                code=ErrorCode.INFLUX_QUERY_ERROR,
                message=f"Failed to query readings: {e}",
            )
        finally:
            duration = time.time() - start_time
            INFLUX_QUERY_LATENCY.labels(query_type="readings").observe(duration)

    async def query_predictions(
        self,
        building_id: str,
        appliance_id: str | None,
        start: str,
        end: str,
        resolution: Resolution,
    ) -> list[PredictionPoint]:
        """
        Query predictions from InfluxDB.

        Returns:
            List of PredictionPoint objects
        """
        settings = get_settings()
        start_time = time.time()

        try:
            if not self._client:
                raise InfluxError(
                    code=ErrorCode.INFLUX_CONNECTION_ERROR,
                    message="InfluxDB client not connected",
                )

            query = build_predictions_query(
                bucket=settings.influx_bucket_pred,
                building_id=building_id,
                appliance_id=appliance_id,
                start=start,
                end=end,
                resolution=resolution,
            )

            query_api = self._client.query_api()
            tables = await query_api.query(query, org=settings.influx_org)

            data_points: list[PredictionPoint] = []
            for table in tables:
                for record in table.records:
                    data_points.append(
                        PredictionPoint(
                            time=record.get_time().isoformat() if record.get_time() else "",
                            predicted_kw=float(record.values.get("predicted_kw", 0.0)),
                            confidence=record.values.get("confidence"),
                            model_version=record.values.get("model_version"),
                        )
                    )

            logger.debug(
                "Query predictions completed",
                extra={"building_id": building_id, "count": len(data_points)},
            )
            return data_points

        except InfluxError:
            raise
        except Exception as e:
            logger.error("Query predictions failed", extra={"error": str(e)})
            raise InfluxError(
                code=ErrorCode.INFLUX_QUERY_ERROR,
                message=f"Failed to query predictions: {e}",
            )
        finally:
            duration = time.time() - start_time
            INFLUX_QUERY_LATENCY.labels(query_type="predictions").observe(duration)

    async def write_prediction(
        self,
        building_id: str,
        appliance_id: str,
        predicted_kw: float,
        confidence: float,
        model_version: str,
        user_id: str,
        request_id: str,
        latency_ms: float,
        timestamp: datetime | None = None,
        max_retries: int = 3,
    ) -> bool:
        """
        Write a prediction point to InfluxDB.

        Uses bounded retries with exponential backoff.

        Returns:
            True if write succeeded

        Raises:
            InfluxError: If all retries fail
        """
        settings = get_settings()
        start_time = time.time()

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        point = (
            Point("prediction")
            .tag("building_id", building_id)
            .tag("appliance_id", appliance_id)
            .tag("model_version", model_version)
            .field("predicted_kw", predicted_kw)
            .field("confidence", confidence)
            .field("user_id", user_id)
            .field("request_id", request_id)
            .field("latency_ms", latency_ms)
            .time(timestamp)
        )

        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                if not self._write_api:
                    raise InfluxError(
                        code=ErrorCode.INFLUX_CONNECTION_ERROR,
                        message="InfluxDB write API not initialized",
                    )

                await self._write_api.write(
                    bucket=settings.influx_bucket_pred,
                    org=settings.influx_org,
                    record=point,
                )

                duration = time.time() - start_time
                INFLUX_WRITE_LATENCY.labels(bucket=settings.influx_bucket_pred).observe(duration)
                INFLUX_WRITE_COUNT.labels(
                    bucket=settings.influx_bucket_pred, status="success"
                ).inc()

                logger.debug(
                    "Prediction written to InfluxDB",
                    extra={
                        "building_id": building_id,
                        "appliance_id": appliance_id,
                        "predicted_kw": predicted_kw,
                    },
                )
                return True

            except Exception as e:
                last_error = e
                logger.warning(
                    "InfluxDB write attempt failed",
                    extra={"attempt": attempt + 1, "max_retries": max_retries, "error": str(e)},
                )

                if attempt < max_retries - 1:
                    # Exponential backoff: 100ms, 200ms, 400ms
                    await self._sleep(0.1 * (2**attempt))

        # All retries failed
        INFLUX_WRITE_COUNT.labels(bucket=settings.influx_bucket_pred, status="failure").inc()

        raise InfluxError(
            code=ErrorCode.INFLUX_WRITE_FAILED,
            message=f"Failed to write prediction after {max_retries} attempts",
            details={"last_error": str(last_error) if last_error else None},
        )

    async def write_predictions_wide(
        self,
        building_id: str,
        predictions: dict[str, tuple[float, float]],  # {field_key: (predicted_kw, confidence)}
        model_version: str,
        user_id: str,
        request_id: str,
        latency_ms: float,
        timestamp: datetime | None = None,
        max_retries: int = 3,
    ) -> bool:
        """
        Write multi-head predictions as a single WIDE point to InfluxDB.

        Creates ONE point per timestamp with fields:
        - predicted_kw_<field_key> for each head
        - confidence_<field_key> for each head
        - user_id, request_id, latency_ms

        Tags: building_id, model_version (no appliance_id tag for wide format)

        Args:
            building_id: Building identifier
            predictions: Dict mapping field_key to (predicted_kw, confidence)
            model_version: Model version string
            user_id: User ID who made the request
            request_id: Request ID for tracing
            latency_ms: Inference latency in milliseconds
            timestamp: Optional timestamp (defaults to now)
            max_retries: Retry count on failure

        Returns:
            True if write succeeded

        Raises:
            InfluxError: If all retries fail
        """
        settings = get_settings()
        start_time = time.time()

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Build wide point with all predictions as fields
        point = (
            Point("predictions")  # Note: plural measurement for wide format
            .tag("building_id", building_id)
            .tag("model_version", model_version)
            .field("user_id", user_id)
            .field("request_id", request_id)
            .field("latency_ms", latency_ms)
            .time(timestamp)
        )

        # Add prediction fields for each head
        for field_key, (predicted_kw, confidence) in predictions.items():
            point = point.field(f"predicted_kw_{field_key}", max(0.0, predicted_kw))
            point = point.field(f"confidence_{field_key}", min(1.0, max(0.0, confidence)))

        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                if not self._write_api:
                    raise InfluxError(
                        code=ErrorCode.INFLUX_CONNECTION_ERROR,
                        message="InfluxDB write API not initialized",
                    )

                await self._write_api.write(
                    bucket=settings.influx_bucket_pred,
                    org=settings.influx_org,
                    record=point,
                )

                duration = time.time() - start_time
                INFLUX_WRITE_LATENCY.labels(bucket=settings.influx_bucket_pred).observe(duration)
                INFLUX_WRITE_COUNT.labels(
                    bucket=settings.influx_bucket_pred, status="success"
                ).inc()

                logger.debug(
                    "Wide predictions written to InfluxDB",
                    extra={
                        "building_id": building_id,
                        "heads_count": len(predictions),
                        "field_keys": list(predictions.keys()),
                    },
                )
                return True

            except Exception as e:
                last_error = e
                logger.warning(
                    "InfluxDB wide write attempt failed",
                    extra={"attempt": attempt + 1, "max_retries": max_retries, "error": str(e)},
                )

                if attempt < max_retries - 1:
                    await self._sleep(0.1 * (2**attempt))

        INFLUX_WRITE_COUNT.labels(bucket=settings.influx_bucket_pred, status="failure").inc()

        raise InfluxError(
            code=ErrorCode.INFLUX_WRITE_FAILED,
            message=f"Failed to write wide predictions after {max_retries} attempts",
            details={"last_error": str(last_error) if last_error else None},
        )

    async def _sleep(self, seconds: float) -> None:
        """Async sleep helper."""
        import asyncio
        await asyncio.sleep(seconds)


# Global client instance (initialized during app lifespan)
_influx_client: InfluxClient | None = None


def get_influx_client() -> InfluxClient:
    """Get the global InfluxDB client instance."""
    global _influx_client
    if _influx_client is None:
        _influx_client = InfluxClient()
    return _influx_client


async def init_influx_client() -> InfluxClient:
    """Initialize the global InfluxDB client."""
    client = get_influx_client()
    await client.connect()
    return client


async def close_influx_client() -> None:
    """Close the global InfluxDB client."""
    global _influx_client
    if _influx_client:
        await _influx_client.close()
        _influx_client = None
