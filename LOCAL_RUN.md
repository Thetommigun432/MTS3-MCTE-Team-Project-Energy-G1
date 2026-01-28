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

Open: http://localhost:8080/live

---

## Service URLs

| Service | URL |
|---------|-----|
| Frontend | http://localhost:8080 |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| InfluxDB UI | http://localhost:8086 |

---

## Useful Commands

```bash
# View all containers
docker compose ps

# View logs
docker compose logs -f backend
docker compose logs -f worker

# Restart everything
docker compose down && docker compose up -d

# Rebuild after code changes
docker compose up -d --build
```

---

## MQTT Realtime Mode

For live data from Howest Energy Lab:

```bash
docker compose -f compose.realtime.yaml up -d
```

Verify data flow:
```bash
docker logs mqtt-ingestor --tail 20
# Should show: "Sent X readings | Latest: XXX W"
```

---

## Data Flow

1. **Simulator/MQTT** sends readings to Backend
2. **Backend** stores in Redis rolling window
3. **Worker** runs NILM inference (10 models)
4. **Worker** writes predictions to InfluxDB
5. **Frontend** polls predictions from Backend

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No predictions | Wait ~40s for buffer to fill |
| API unreachable | Check `docker compose ps` |
| Frontend errors | Delete node_modules, npm install |
