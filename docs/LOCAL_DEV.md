# Local Development

This guide is the single source of truth for running the NILM stack locally.

## Prerequisites
- Node.js 22+
- Docker Desktop with Docker Compose v2
- (Optional) Python 3.12+ if running backend outside Docker

## Quick Start (Docker + Frontend)

```bash
# 1) Configure shared env for Docker Compose
cp .env.local.example .env.local

# 2) Start the backend stack (API + Worker + Redis + Influx + Simulator)
docker compose up -d --build

# 3) Install frontend deps at the repo root (npm workspace)
npm install

# 4) Start frontend dev server (port 8080)
npm run dev:web
```

Open:
- Frontend: http://localhost:8080
- Backend API: http://localhost:8000
- InfluxDB UI: http://localhost:8086

## Environment Files

| File | Purpose |
|------|---------|
| `.env.local` | Shared local secrets for Compose (InfluxDB token/org/bucket). |
| `apps/web/.env` | Frontend build-time variables (Supabase + backend URL). |

### Required local variables

**Root (.env.local)**
- `INFLUX_TOKEN` (set to a local token)
- `INFLUX_ORG` (default: `energy-monitor`)
- `INFLUX_BUCKET_PRED` (default: `predictions`)

**Frontend (apps/web/.env)**
- `VITE_SUPABASE_URL` (if using auth)
- `VITE_SUPABASE_PUBLISHABLE_KEY` (preferred; falls back to `VITE_SUPABASE_ANON_KEY`)
- `VITE_BACKEND_URL` (defaults to `/api`, so you can leave it as-is for local dev)

## Local Service Ports

| Service | URL | Notes |
|--------|-----|------|
| Frontend (Vite) | http://localhost:8080 | `npm run dev:web` |
| Backend API | http://localhost:8000 | Docker Compose service `backend` |
| InfluxDB | http://localhost:8086 | Docker Compose service `influxdb` |
| Redis | redis://localhost:6379 | Docker Compose service `redis` |

## Local Dataflow (end-to-end)

1. Simulator reads `apps/backend/data/simulation_data.parquet`.
2. Simulator posts readings to `POST /ingest/readings` on the backend.
3. Backend stores a rolling window in Redis and enqueues a Redis Stream event.
4. Worker consumes the stream, builds an inference window, runs the model.
5. Worker writes predictions to InfluxDB (`predictions` bucket).

## Verification (Local)

### Health endpoints
```bash
curl http://localhost:8000/live
curl http://localhost:8000/ready
```

### Verify Redis stream activity
```bash
docker exec nilm-redis redis-cli XLEN nilm:readings
```

### Verify predictions in InfluxDB
```bash
docker exec -it $(docker compose ps -q influxdb) influx query \
  'from(bucket: "predictions") |> range(start: -5m) |> limit(n: 5)' \
  --org energy-monitor --token $INFLUX_TOKEN
```

## Common Commands

```bash
# View service status
docker compose ps

# Tail logs
docker compose logs -f backend

# Restart everything
docker compose down
```

## Troubleshooting

- **No predictions**: The model requires a full window; wait for the simulator to fill the buffer.
- **API unreachable**: Ensure Docker containers are healthy (`docker compose ps`).
- **CORS issues**: Use Vite proxy via `/api` or set `CORS_ORIGINS` in `.env.local`.
