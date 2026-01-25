"""
Prometheus metrics and telemetry for observability.
Exposes metrics endpoint and provides instrumentation helpers.
"""

from prometheus_client import Counter, Gauge, Histogram


# =============================================================================
# Request Metrics
# =============================================================================

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "route", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "route"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)


# =============================================================================
# Inference Metrics
# =============================================================================

INFERENCE_LATENCY = Histogram(
    "inference_duration_seconds",
    "Model inference latency in seconds",
    ["model_id"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

INFERENCE_COUNT = Counter(
    "inference_requests_total",
    "Total inference requests",
    ["model_id", "status"],
)


# =============================================================================
# InfluxDB Metrics
# =============================================================================

INFLUX_WRITE_LATENCY = Histogram(
    "influx_write_duration_seconds",
    "InfluxDB write latency in seconds",
    ["bucket"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

INFLUX_WRITE_COUNT = Counter(
    "influx_writes_total",
    "Total InfluxDB write operations",
    ["bucket", "status"],
)

INFLUX_QUERY_LATENCY = Histogram(
    "influx_query_duration_seconds",
    "InfluxDB query latency in seconds",
    ["query_type"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)


# =============================================================================
# Auth Metrics
# =============================================================================

AUTH_VERIFY_LATENCY = Histogram(
    "auth_verify_duration_seconds",
    "JWT verification latency in seconds",
    ["algorithm"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
)

AUTHZ_CHECK_LATENCY = Histogram(
    "authz_check_duration_seconds",
    "Authorization check latency in seconds",
    ["check_type"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)

AUTHZ_CACHE_HIT = Counter(
    "authz_cache_hits_total",
    "AuthZ cache hits",
)

AUTHZ_CACHE_MISS = Counter(
    "authz_cache_misses_total",
    "AuthZ cache misses",
)


# =============================================================================
# Model Cache Metrics
# =============================================================================

MODEL_CACHE_SIZE = Gauge(
    "model_cache_size",
    "Number of models loaded in cache",
)

MODEL_CACHE_BYTES = Gauge(
    "model_cache_bytes",
    "Estimated memory usage of model cache in bytes",
)


# =============================================================================
# Idempotency Metrics
# =============================================================================

IDEMPOTENCY_CACHE_HIT = Counter(
    "idempotency_cache_hits_total",
    "Idempotency cache hits (repeated requests)",
)

IDEMPOTENCY_CACHE_SIZE = Gauge(
    "idempotency_cache_size",
    "Number of entries in idempotency cache",
)


# =============================================================================
# Rate Limiting Metrics
# =============================================================================

RATE_LIMIT_HIT = Counter(
    "rate_limit_hits_total",
    "Rate limit violations",
    ["key_type"],  # "user" or "ip"
)


# =============================================================================
# Redis Cache Metrics
# =============================================================================

REDIS_UNAVAILABLE = Counter(
    "redis_unavailable_total",
    "Number of times Redis was unavailable",
)

CACHE_FALLBACK_IN_USE = Gauge(
    "cache_fallback_in_use",
    "Whether in-memory fallback cache is in use (1=yes, 0=no)",
)
