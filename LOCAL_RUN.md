# Local Development Setup

This guide explains how to run the NILM Energy Monitor locally using Docker Compose.

## Prerequisites

- Docker and Docker Compose v2+
- At least 4GB RAM available for containers
- Dataset file: `apps/backend/data/simulation_data.parquet`

## Quick Start

```bash
# 1. Start the full stack (builds all images)
docker compose up --build -d

# 2. Watch the logs
docker compose logs -f

# 3. Run smoke test (after ~30 seconds)
./scripts/local_smoke.sh
```

## Architecture

The local stack consists of 5 services:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Docker Compose Stack                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │Simulator │───▶│ Backend  │───▶│  Redis   │◀───│  Worker  │  │
│  │(parquet) │    │  (API)   │    │(window)  │    │(inference│  │
│  └──────────┘    └────┬─────┘    └──────────┘    └────┬─────┘  │
│                       │                                │        │
│                       ▼                                ▼        │
│                  ┌──────────┐                    ┌──────────┐   │
│                  │InfluxDB  │◀───────────────────│InfluxDB  │   │
│                  │(raw data)│                    │(predict) │   │
│                  └──────────┘                    └──────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Service Roles

| Service | Purpose |
|---------|---------|
| `influxdb` | Time-series database for predictions |
| `backend` | FastAPI API server (port 8000) |
| `worker` | Inference worker (consumes Redis Stream) |
| `simulator` | Reads parquet and posts to ingest API |
| `redis` | Rolling window storage + Stream queue |

## Pipeline Flow

1. **Simulator** reads `simulation_data.parquet` row by row
2. Posts each row to `POST /api/ingest/readings`
3. **Backend** updates Redis rolling window (RPUSH/LTRIM)
4. **Backend** enqueues event to Redis Stream
5. **Worker** consumes Stream events
6. **Worker** reads rolling window from Redis
7. **Worker** runs inference using loaded models
8. **Worker** writes predictions to InfluxDB

## Environment Variables

### Simulator Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SIM_SPEEDUP` | 10 | Rows per second (10 = 10x realtime) |
| `SIM_DURATION_SECONDS` | 0 | Max duration (0 = unlimited) |
| `BUILDING_ID` | building-1 | Building identifier |
| `DATA_POWER_COLUMN` | aggregate | Column name for power values |

### Backend Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PIPELINE_ENQUEUE_ENABLED` | true | Enable Redis Stream enqueueing |
| `PIPELINE_ROLLING_WINDOW_SIZE` | 3600 | Rolling window size (1 hour) |

## Expected Behavior

After starting the stack:

1. **Within 30 seconds:**
   - `/live` and `/ready` return 200
   - `/api/models` returns model registry

2. **Within 1-2 minutes:**
   - Redis window starts filling
   - `redis-cli LLEN nilm:building-1:window` shows count

3. **After window warmup (~26 minutes at 10x speed):**
   - Worker logs show "Prediction written"
   - `/api/analytics/predictions` returns data
   - InfluxDB contains predictions

## Monitoring

### Check service health
```bash
docker compose ps
```

### View all logs
```bash
docker compose logs -f
```

### View specific service logs
```bash
docker compose logs -f simulator
docker compose logs -f worker
docker compose logs -f backend
```

### Check Redis window
```bash
docker exec nilm-redis redis-cli LLEN nilm:building-1:window
```

### Query InfluxDB
```bash
docker exec -it $(docker compose ps -q influxdb) influx query \
  'from(bucket: "predictions") |> range(start: -5m) |> limit(n: 5)' \
  --org energy-monitor --token admin-token
```

## Troubleshooting

### "No predictions" in dashboard

1. Check if window is large enough:
   ```bash
   docker exec nilm-redis redis-cli LLEN nilm:building-1:window
   ```
   Models require 1536 samples. At 10x speed, this takes ~2.5 minutes.

2. Check worker logs for "warming up" messages:
   ```bash
   docker compose logs worker | grep -i warm
   ```

3. Check for inference errors:
   ```bash
   docker compose logs worker | grep -i error
   ```

### Simulator not starting

1. Check if parquet file exists:
   ```bash
   ls -la apps/backend/data/simulation_data.parquet
   ```

2. Check simulator logs:
   ```bash
   docker compose logs simulator
   ```

### Backend returns 500

1. Check backend logs:
   ```bash
   docker compose logs backend
   ```

2. Verify InfluxDB is healthy:
   ```bash
   curl http://localhost:8086/health
   ```

### Clean restart

```bash
# Stop and remove all containers and volumes
docker compose down -v

# Rebuild and start
docker compose up --build -d
```

## Frontend Development

The frontend can run separately from the Docker stack:

```bash
cd apps/web
npm install
npm run dev
```

With Vite proxy configured, API calls to `/api/*` are forwarded to `http://localhost:8000`.

Set `VITE_BACKEND_URL` for different backends:
- Local proxy: `VITE_BACKEND_URL=/api`
- Direct: `VITE_BACKEND_URL=http://localhost:8000`

## Success Signals

When everything is working correctly, you should see:

1. **Smoke test passes:**
   ```bash
   ./scripts/local_smoke.sh
   # All checks show GREEN
   ```

2. **Redis window at capacity:**
   ```bash
   docker exec nilm-redis redis-cli LLEN nilm:building-1:window
   # Returns: 3600
   ```

3. **Worker producing predictions:**
   ```bash
   docker compose logs worker | tail -20
   # Shows: "Prediction written: building=building-1, model=..."
   ```

4. **Predictions endpoint returns data:**
   ```bash
   curl "http://localhost:8000/api/analytics/predictions?building_id=building-1&start=-5m&end=now()"
   # Returns JSON with count > 0
   ```
