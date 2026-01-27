"""
NILM Energy Monitor - Unified Python Backend
FastAPI application entry point with lifespan management.
"""

import asyncio
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
from app.api.routers import admin, analytics, health, inference, ingest
from app.core.config import get_settings, validate_production_settings
from app.core.errors import AppError, ErrorCode, error_response
from app.core.logging import get_logger, request_id_ctx, setup_logging
from app.domain.inference import init_model_registry
from app.infra.influx import close_influx_client, get_influx_client, init_influx_client
from app.infra.redis import close_redis_cache, init_redis_cache
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

    # Validate dataset and models
    import os
    if os.path.exists(settings.dataset_path):
        size_mb = os.path.getsize(settings.dataset_path) / (1024 * 1024)
        logger.info(f"Dataset found at {settings.dataset_path} ({size_mb:.2f} MB)")
    else:
        logger.warning(f"Dataset NOT found at {settings.dataset_path}")

    if os.path.isdir(settings.models_dir):
        logger.info(f"Models directory found at {settings.models_dir}")
        model_files = []
        for root, _, files in os.walk(settings.models_dir):
            for f in files:
                if f.endswith(('.pt', '.pth', '.onnx', '.safetensors', '.json')):
                    model_files.append(os.path.join(root, f))
        logger.info(f"Found {len(model_files)} model files")
    else:
        logger.warning(f"Models directory NOT found at {settings.models_dir}")

    # Validate production settings
    if settings.env == "prod":
        config_errors = validate_production_settings(settings)
        if config_errors:
            logger.error("Configuration validation failed", extra={"errors": config_errors})
            # We don't exit here to allow /ready to report the issue, 
            # but we definitely log it loudly.

    # Setup logging
    log_level = "DEBUG" if settings.debug else "INFO"
    setup_logging(log_level)

    # Initialize InfluxDB client
    try:
        await init_influx_client()
        # Ensure required buckets exist (creates if missing)
        try:
            influx = get_influx_client()
            bucket_results = await influx.ensure_buckets()
            for bucket_name, success in bucket_results.items():
                if success:
                    logger.info(f"InfluxDB bucket ready: {bucket_name}")
                else:
                    logger.warning(f"InfluxDB bucket may not exist: {bucket_name}")
        except Exception as e:
            logger.error("Failed to ensure InfluxDB buckets", extra={"error": str(e)})
    except Exception as e:
        logger.error("Failed to connect to InfluxDB", extra={"error": str(e)})


    # Initialize Supabase client
    try:
        init_supabase_client()
    except Exception as e:
        logger.error("Failed to connect to Supabase", extra={"error": str(e)})

    # Initialize Redis cache (graceful fallback to in-memory)
    try:
        await init_redis_cache()
    except Exception as e:
        logger.warning("Redis initialization failed, using in-memory fallback", extra={"error": str(e)})

    # Load model registry
    try:
        registry = init_model_registry()
        errors = registry.validate()
        if errors:
            for err in errors:
                logger.warning("Registry validation error", extra={"error": err})
    except Exception as e:
        logger.error("Failed to load model registry", extra={"error": str(e)})

    # Pipeline worker
    worker = None
    worker_task = None
    if settings.pipeline_enabled and settings.redis_url:
        try:
            from app.domain.pipeline.redis_inference_worker import RedisInferenceWorker
            worker = RedisInferenceWorker()
            worker_task = asyncio.create_task(worker.start())
            logger.info("Pipeline worker started")
        except Exception as e:
            logger.error(f"Failed to start pipeline worker: {e}")

    yield

    # Shutdown
    logger.info("Shutting down NILM Backend")
    
    if worker:
        await worker.stop()
        if worker_task:
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Pipeline worker stopped")

    await close_redis_cache()
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

    # Add ASGI middleware
    # Note: add_middleware adds to the stack in reverse order (last added = outermost)
    app.add_middleware(RequestSizeLimitMiddleware)


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
    # Register routers (mounted under /api)
    app.include_router(health.router)  # Health check stays at root /health and /live
    app.include_router(inference.router, prefix="/api")
    app.include_router(analytics.router, prefix="/api")
    app.include_router(ingest.router, prefix="/api")
    app.include_router(admin.router, prefix="/api")

    # Conditionally register E2E router (Railway testing)
    if settings.e2e_probes_enabled:
        from app.api.routers import e2e

        app.include_router(e2e.router)
        logger.info("E2E probe endpoints enabled")

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
