# Local Development

Quick guide for running the NILM stack locally.

## Prerequisites

- Docker Desktop with Compose v2
- Node.js 20+

## Quick Start

```bash
# Start backend stack
docker compose up -d

# Start frontend
cd apps/web && npm install && npm run dev
```

Open: http://localhost:8080/live (no login required)

---

## Service URLs

| Service | URL |
|---------|-----|
| Frontend | http://localhost:8080 |
| Public Dashboard | http://localhost:8080/live |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| InfluxDB UI | http://localhost:8086 |

---

## Operating Modes

### Simulator Mode (Default)

Replays data from a parquet file at configurable speed.

```bash
docker compose up -d
```

### MQTT Realtime Mode

Connects to Howest Energy Lab live MQTT broker.

```bash
docker compose -f compose.realtime.yaml up -d
```

---

## Useful Commands

```bash
# View all containers
docker compose ps

# View logs
docker compose logs -f backend
docker compose logs -f worker
docker compose logs -f mqtt-ingestor  # MQTT mode only

# Restart everything
docker compose down && docker compose up -d

# Rebuild after code changes
docker compose up -d --build
```

---

## Data Flow

```
Simulator/MQTT → Backend API → Redis (window) → Worker (PyTorch) → InfluxDB → Frontend
```

1. **Simulator/MQTT** sends power readings to Backend
2. **Backend** stores readings in Redis rolling window (4096 samples)
3. **Worker** consumes stream, runs 10 NILM models
4. **Worker** writes predictions to InfluxDB
5. **Frontend** polls predictions via Backend API

---

## Environment Variables

All have defaults - no configuration needed for local dev.

| Variable | Default | Description |
|----------|---------|-------------|
| INFLUX_TOKEN | admin-token | InfluxDB auth |
| INFLUX_ORG | energy-monitor | Organization |
| SIM_SPEEDUP | 100 | Simulator speed multiplier |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No predictions | Wait ~40s for buffer to fill (4096 samples) |
| API unreachable | Run `docker compose ps`, check all healthy |
| Frontend errors | `rm -rf node_modules && npm install` |
| MQTT no data | Check `docker logs mqtt-ingestor --tail 20` |

---

## Verification

```bash
# API health
curl http://localhost:8000/health

# Check predictions exist
curl "http://localhost:8000/api/analytics/predictions?building_id=building-1&start=2026-01-01&end=2026-12-31"
```
