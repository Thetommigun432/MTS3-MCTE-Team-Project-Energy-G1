# =============================================================================
# NILM Backend - Dockerfile
# Production-ready Python 3.12 image with Railway support
# =============================================================================

FROM python:3.12-slim-bookworm AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# =============================================================================
# Builder Stage (shared by test and production)
# =============================================================================
FROM base AS builder

# Copy project definition
COPY apps/backend/pyproject.toml ./

# Copy source code
COPY apps/backend/src/ ./src/

# Install the application (and dependencies)
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir .

# Copy models directory
COPY apps/backend/models/ ./models/

# Change ownership to non-root user
RUN chown -R appuser:appgroup /app

# =============================================================================
# Test Stage (for E2E testing - extends builder, adds tests)
# IMPORTANT: This must come BEFORE production so production is the default
# =============================================================================
FROM builder AS test

# Copy tests (includes fixtures/simulation-data.parquet)
COPY apps/backend/tests/ ./tests/

# Command for testing
CMD ["pytest", "tests", "-v"]

# =============================================================================
# Production Stage (FINAL - this is what Railway builds by default)
# =============================================================================
FROM builder AS production

# Switch to non-root user
USER appuser

# Expose port (Railway injects PORT env var)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/live || exit 1

# Run the application
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
