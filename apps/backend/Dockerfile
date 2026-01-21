# =============================================================================
# NILM Backend - Dockerfile
# Production-ready Python 3.12 image with Railway support
# =============================================================================

FROM python:3.12-slim AS base

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
FROM base AS deps

# Copy requirements
COPY apps/backend/requirements.txt apps/backend/constraints.txt ./

# Install Python dependencies
# Install CPU-only PyTorch first, then other requirements
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Production Stage
# =============================================================================
FROM deps AS production

# Copy application code
COPY apps/backend/app/ ./app/

# Copy models directory (if present - can be mounted as volume)
# Models should be bundled for Railway or mounted for local dev
COPY apps/backend/models/ ./models/

# Change ownership to non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose port (Railway injects PORT env var)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/live || exit 1

# Run the application
# Railway sets PORT, default to 8000 for local dev
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
