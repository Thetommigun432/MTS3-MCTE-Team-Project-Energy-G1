"""
Core configuration module using pydantic-settings.
All settings are loaded from environment variables with sensible defaults.
"""

from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=('settings_',),
    )

    # ==========================================================================
    # Environment
    # ==========================================================================
    env: Literal["dev", "test", "prod"] = Field(
        default="dev",
        description="Environment: dev, test, or prod",
    )

    # ==========================================================================
    # Server
    # ==========================================================================
    port: int = Field(default=8000, description="Server port (Railway injects PORT)")
    host: str = Field(default="0.0.0.0", description="Server host")
    debug: bool = Field(default=False, description="Enable debug mode")

    # ==========================================================================
    # CORS
    # ==========================================================================
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:8080",
        description="Comma-separated list of allowed CORS origins",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    # ==========================================================================
    # InfluxDB
    # ==========================================================================
    influx_url: str = Field(
        default="http://localhost:8086",
        description="InfluxDB URL",
    )
    influx_token: str = Field(
        default="",
        description="InfluxDB admin token",
    )
    influx_org: str = Field(
        default="energy-monitor",
        description="InfluxDB organization",
    )

    influx_bucket_pred: str = Field(
        default="predictions",
        description="InfluxDB bucket for predictions",
    )
    influx_timeout_ms: int = Field(
        default=10000,
        description="InfluxDB client timeout in milliseconds",
    )

    # ==========================================================================
    # Supabase
    # ==========================================================================
    supabase_url: str = Field(
        default="",
        description="Supabase project URL",
    )
    supabase_publishable_key: str = Field(
        default="",
        description="Supabase publishable key (preferred)",
    )
    supabase_anon_key: str = Field(
        default="",
        description="Supabase anonymous/public key (legacy fallback)",
    )

    @field_validator("supabase_publishable_key")
    @classmethod
    def validate_supabase_keys(cls, v: str, info: Any) -> str:
        """Fallback to anon_key if publishable_key is missing."""
        if v:
            return v
        # Try to get anon_key from values if available
        values = info.data
        if "supabase_anon_key" in values and values["supabase_anon_key"]:
            return values["supabase_anon_key"]
        return v
    supabase_jwt_secret: str = Field(
        default="",
        description="Supabase JWT secret for HS256 verification",
    )
    supabase_jwks_url: str | None = Field(
        default=None,
        description="Optional JWKS URL for RS256 verification (auto-derived if empty)",
    )

    @field_validator("supabase_jwks_url")
    @classmethod
    def set_default_jwks_url(cls, v: str | None, info: Any) -> str | None:
        """Derive JWKS URL from Supabase URL if not explicitly set."""
        if v:
            return v
        values = info.data
        if "supabase_url" in values and values["supabase_url"]:
            base_url = values["supabase_url"].rstrip("/")
            return f"{base_url}/auth/v1/.well-known/jwks.json"
        return None

    # ==========================================================================
    # Authentication
    # ==========================================================================
    auth_verify_aud: bool = Field(
        default=True,
        description="Verify JWT audience claim",
    )
    test_jwt_secret: str = Field(
        default="",
        description="Test JWT secret (only used when ENV=test)",
    )

    # ==========================================================================
    # Rate Limiting
    # ==========================================================================
    rate_limit_per_user: str = Field(
        default="60/minute",
        description="Rate limit per authenticated user",
    )
    rate_limit_per_ip: str = Field(
        default="120/minute",
        description="Rate limit per IP for unauthenticated requests",
    )

    @field_validator("rate_limit_per_user", "rate_limit_per_ip")
    @classmethod
    def validate_rate_limit(cls, v: str) -> str:
        """Validate rate limit format (e.g., '60/minute')."""
        parts = v.split("/")
        if len(parts) != 2:
            raise ValueError("Rate limit must be in format 'N/interval' (e.g., '60/minute')")
        try:
            int(parts[0])
        except ValueError:
            raise ValueError(f"Invalid rate limit count: {parts[0]}")
        if parts[1] not in ("second", "minute", "hour"):
            raise ValueError(f"Invalid rate limit interval: {parts[1]}")
        return v

    # ==========================================================================
    # Request Limits
    # ==========================================================================
    max_body_bytes: int = Field(
        default=262144,  # 256KB
        description="Maximum request body size in bytes",
    )

    # ==========================================================================
    # Admin
    # ==========================================================================
    admin_token: str | None = Field(
        default=None,
        description="Admin token for protected admin endpoints (required in prod)",
    )

    # ==========================================================================
    # Model Registry
    # ==========================================================================
    model_registry_path: str = Field(
        default="/app/models/registry.json",
        description="Path to model registry JSON file",
    )
    models_dir: str = Field(
        default="/app/models",
        description="Directory containing model artifacts",
    )
    dataset_path: str = Field(
        default="/app/data/simulation_data.parquet",
        description="Path to simulation dataset",
    )

    # ==========================================================================
    # Redis
    # ==========================================================================
    redis_url: str = Field(
        default="",
        description="Redis URL for caching (optional, falls back to in-memory)",
    )
    redis_pool_size: int = Field(
        default=10,
        description="Redis connection pool size",
    )
    redis_connect_timeout_ms: int = Field(
        default=5000,
        description="Redis connection timeout in milliseconds",
    )
    redis_stream_key: str = Field(
        default="nilm:readings",
        description="Redis stream key for ingestion",
    )
    redis_consumer_group: str = Field(
        default="nilm-infer",
        description="Redis consumer group name",
    )

    # ==========================================================================
    # Pipeline
    # ==========================================================================
    pipeline_enabled: bool = Field(
        default=True,
        description="Enable background inference pipeline",
    )
    pipeline_stride: int = Field(
        default=30,
        description="Run inference every N samples",
    )
    pipeline_max_buffer: int = Field(
        default=2048,
        description="Max per-building buffer size",
    )
    ingest_token: str | None = Field(
        default=None,
        description="Optional token for server-to-server ingestion auth",
    )

    # ==========================================================================
    # Model Artifacts
    # ==========================================================================
    model_artifact_base_url: str | None = Field(
        default=None,
        description="Base URL for fetching missing model artifacts (e.g. https://models.example.com)",
    )
    model_artifact_timeout_sec: int = Field(
        default=60,
        description="Timeout in seconds for downloading model artifacts",
    )
    model_artifact_max_mb: int = Field(
        default=500,
        description="Maximum size in MB for downloaded model artifacts",
    )

    # ==========================================================================
    # E2E Testing
    # ==========================================================================
    e2e_probes_enabled: bool = Field(
        default=False,
        description="Enable E2E probe endpoints (disabled by default, Railway testing only)",
    )
    e2e_token: str | None = Field(
        default=None,
        description="Secret token for E2E probe authentication",
    )

    # ==========================================================================
    # Caching
    # ==========================================================================
    authz_cache_ttl_seconds: int = Field(
        default=60,
        description="AuthZ permission cache TTL in seconds",
    )
    idempotency_cache_ttl_seconds: int = Field(
        default=600,
        description="Idempotency cache TTL in seconds (10 minutes)",
    )
    jwks_cache_ttl_hours: int = Field(
        default=6,
        description="JWKS cache TTL in hours",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


def validate_production_settings(settings: Settings) -> list[str]:
    """
    Validate critical settings for production environment.
    Returns a list of error messages.
    """
    if settings.env != "prod":
        return []

    errors = []

    # 1. Critical Services
    if not settings.supabase_url:
        errors.append("Missing SUPABASE_URL")
    if not settings.supabase_publishable_key and not settings.supabase_anon_key:
        errors.append("Missing SUPABASE_PUBLISHABLE_KEY")
    if not settings.influx_url or "localhost" in settings.influx_url:
        errors.append("INFLUX_URL must be set and not localhost in prod")
    if not settings.influx_token:
        errors.append("Missing INFLUX_TOKEN")

    # 2. CORS
    if not settings.cors_origins or settings.cors_origins == "*":
        errors.append("CORS_ORIGINS must be set and cannot be wildcard '*' in prod")
    
    for origin in settings.cors_origins_list:
        if origin == "*":
            errors.append("Generic wildcard '*' in CORS_ORIGINS is not allowed in prod")
        if "localhost" in origin:
            errors.append(f"Localhost origin '{origin}' is not allowed in prod")

    # 3. Security
    if not settings.auth_verify_aud:
        errors.append("AUTH_VERIFY_AUD must be True in prod")

    return errors
