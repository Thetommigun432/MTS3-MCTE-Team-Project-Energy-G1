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
# Dependencies Stage
# =============================================================================
# =============================================================================
# Dependencies Stage
# =============================================================================
FROM base AS deps

# Copy pyproject.toml
COPY apps/backend/pyproject.toml ./

# Install dependencies (including torch)
# We use pyproject.toml, but we might need to recreate requirements or use pip install .
# Since src isn't copied yet, we can't install . fully, but we can install deps if we parse it
# OR we simply copy src in production stage and install . there.
# To keep cache, we can keep using requirements.txt IF we generate it, but user said "don't maintain manually"
# So let's rely on pip installing from pyproject if possible, or just install everything in production stage?
# Better: Just use pip install . in production stage.
# For caching deps, we can install the heavy libs manually here or use a requirements.txt generated.
# Given the user constraints, let's keep the manual requirements.txt usage for caching efficienty BUT
# generated/maintained via pyproject is hard without tools.
# Let's pivot: Copy pyproject.toml and install it.
# To allow pip install . without src, we need a dummy src or just allow failure and only install deps?
# No, lets standardise: COPY src in dev/prod and install.
# To optimize cache for torch:
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies from pyproject.toml (using a trick or just installing . in next stage)
# For now, let's assume valid cache is less critical than valid structure for this refactor step.
# We will do the install in the final stage or intermediate.

FROM base AS production

# Copy project definition
COPY apps/backend/pyproject.toml ./

# Copy source code
COPY apps/backend/src/ ./src/

# Install the application (and dependencies)
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir .

# Copy models directory (if present - can be mounted as volume)
COPY apps/backend/models/ ./models/

# Change ownership to non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose port (Railway injects PORT env var)
EXPOSE 8000

# Health check (updated port ref)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/live || exit 1

# Run the application
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

# =============================================================================
# Test Stage
# =============================================================================
FROM production AS test

# Copy tests (includes fixtures/simulation-data.parquet)
COPY apps/backend/tests/ ./tests/

# Reset command for testing
CMD ["pytest", "tests/integration", "-v"]
