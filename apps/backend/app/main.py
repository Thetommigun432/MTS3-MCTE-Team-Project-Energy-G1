"""
NILM Energy Monitor - Unified Python Backend
FastAPI application entry point with lifespan management.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.api.middleware import (
    MetricsMiddleware,
    RateLimitMiddleware,
    RequestIdMiddleware,
    RequestSizeLimitMiddleware,
)
from app.api.routers import admin, analytics, health, inference
from app.core.config import get_settings
from app.core.errors import AppError, ErrorCode, error_response
from app.core.logging import get_logger, request_id_ctx, setup_logging
from app.domain.inference import init_model_registry
from app.infra.influx import close_influx_client, init_influx_client
from app.infra.supabase import init_supabase_client

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Handles startup and shutdown of dependencies.
    """
    settings = get_settings()

    # Startup
    logger.info("Starting NILM Backend", extra={"env": settings.env})

    # Setup logging
    log_level = "DEBUG" if settings.debug else "INFO"
    setup_logging(log_level)

    # Initialize InfluxDB client
    try:
        await init_influx_client()
    except Exception as e:
        logger.error("Failed to connect to InfluxDB", extra={"error": str(e)})

    # Initialize Supabase client
    try:
        init_supabase_client()
    except Exception as e:
        logger.error("Failed to connect to Supabase", extra={"error": str(e)})

    # Load model registry
    try:
        registry = init_model_registry()
        errors = registry.validate()
        if errors:
            for err in errors:
                logger.warning("Registry validation error", extra={"error": err})
    except Exception as e:
        logger.error("Failed to load model registry", extra={"error": str(e)})

    logger.info("NILM Backend started")

    yield

    # Shutdown
    logger.info("Shutting down NILM Backend")
    await close_influx_client()
    logger.info("NILM Backend stopped")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="NILM Energy Monitor Backend",
        description="Unified Python backend for energy monitoring and NILM inference",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.env != "prod" else None,
        redoc_url="/redoc" if settings.env != "prod" else None,
    )

    # Add ASGI middleware (order matters - first added = outermost)
    app = RequestSizeLimitMiddleware(app)  # type: ignore

    # Add Starlette middleware
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RequestIdMiddleware)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    # Register exception handlers
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        """Handle application errors with consistent schema."""
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response(exc, request_id),
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected errors."""
        request_id = getattr(request.state, "request_id", None)
        logger.error("Unhandled exception", extra={"error": str(exc), "type": type(exc).__name__})

        from app.core.errors import AppError
        error = AppError(
            code=ErrorCode.INTERNAL_ERROR,
            message="Internal server error",
            status_code=500,
        )
        return JSONResponse(
            status_code=500,
            content=error_response(error, request_id),
        )

    # Register routers
    app.include_router(health.router)
    app.include_router(inference.router)
    app.include_router(analytics.router)
    app.include_router(admin.router)

    # Metrics endpoint
    @app.get("/metrics", tags=["Metrics"])
    async def metrics() -> PlainTextResponse:
        """Prometheus metrics endpoint."""
        return PlainTextResponse(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.env == "dev",
        log_level="debug" if settings.debug else "info",
    )
