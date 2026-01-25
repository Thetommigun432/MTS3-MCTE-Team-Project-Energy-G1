# NILM Real-Time Inference Service
# =================================
# Production container for real-time appliance disaggregation

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY production/requirements-production.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY production/ /app/production/
# COPY nilm/ /app/nilm/ (Not present in repo)

COPY production/checkpoints/ /app/checkpoints/


# Environment variables
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV MODEL_DIR=/app/checkpoints
ENV WINDOW_SIZE=1536
ENV INFERENCE_INTERVAL=60
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python -c "import redis; r=redis.Redis(host='${REDIS_HOST}'); r.ping()"

# Run the service
CMD ["python", "-m", "production.inference_service", "--mode", "subscriber"]
