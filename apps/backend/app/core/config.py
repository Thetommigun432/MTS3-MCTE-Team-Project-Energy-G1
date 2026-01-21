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
    influx_bucket_raw: str = Field(
        default="raw_sensor_data",
        description="InfluxDB bucket for raw sensor data",
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
