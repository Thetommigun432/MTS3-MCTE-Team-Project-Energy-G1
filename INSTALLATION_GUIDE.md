# NILM Energy Monitor - Installation Guide

Complete guide to set up and run the NILM Energy Monitor.

---

## Quick Start (Recommended)

```bash
git clone https://github.com/Thetommigun432/MTS3-MCTE-Team-Project-Energy-G1.git
cd MTS3-MCTE-Team-Project-Energy-G1
docker compose up -d
cd apps/web && npm install && npm run dev
```

**Open:** http://localhost:8080/live ← **No login required!**

> 💡 The `/live` route is a public dashboard that works without authentication.
> Perfect for demos and testing.

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8 GB | 16 GB |
| **Storage** | 10 GB | 20 GB |
| **CPU** | 4 cores | 8 cores |
| **OS** | Windows 10/11, macOS 12+, Linux |

---

## Prerequisites

### Docker Desktop

```bash
docker --version          # v24+
docker compose version    # v2+
```

Install: https://www.docker.com/products/docker-desktop/

### Node.js 20+

```bash
node --version   # v20+
npm --version    # v10+
```

Install via nvm: https://github.com/nvm-sh/nvm

---

## Running Modes

### Option A: Simulator Mode (Default)

Uses pre-recorded data from a parquet file.

```bash
# Start all services
docker compose up -d

# Start frontend
cd apps/web && npm install && npm run dev
```

### Option B: MQTT Realtime Mode

Connects to Howest Energy Lab live MQTT broker.

```bash
# Start with MQTT ingestor
docker compose -f compose.realtime.yaml up -d

# Start frontend (same as before)
cd apps/web && npm install && npm run dev
```

---

## Services Overview

| Service | Port | Description |
|---------|------|-------------|
| Frontend | 8080 | React dashboard |
| Backend | 8000 | FastAPI REST API |
| InfluxDB | 8086 | Time-series database |
| Redis | 6379 | Cache + message queue |

---

## Verification

### Check containers are running

```bash
docker compose ps
```

All services should show "Up" or "healthy".

### Check API health

```bash
curl http://localhost:8000/health
```

### Check predictions

```bash
curl "http://localhost:8000/api/analytics/predictions?building_id=building-1&start=2026-01-01T00:00:00Z&end=2026-12-31T23:59:59Z"
```

---

## Troubleshooting

### No predictions appearing

The NILM models need 4096 samples (~40 seconds with speedup=100) to fill the rolling window. Wait for the buffer to fill.

### Docker containers not starting

```bash
docker compose down
docker compose up -d --build
```

### Frontend not loading

```bash
cd apps/web
rm -rf node_modules
npm install
npm run dev
```

### MQTT mode not receiving data

Check the mqtt-ingestor logs:
```bash
docker logs mqtt-ingestor --tail 50
```

---

## Default Credentials

| Service | Username | Password |
|---------|----------|----------|
| InfluxDB | admin | admin12345 |

---

## Environment Variables (Optional)

All values have sensible defaults. Override only if needed:

| Variable | Default | Description |
|----------|---------|-------------|
| INFLUX_TOKEN | admin-token | InfluxDB auth token |
| INFLUX_ORG | energy-monitor | InfluxDB organization |
| SIM_SPEEDUP | 100 | Simulator speed (1=realtime) |

---

## Next Steps

| URL | Auth Required | Description |
|-----|---------------|-------------|
| http://localhost:8080/live | ❌ No | Public dashboard (recommended) |
| http://localhost:8080 | ✅ Yes | Full app (needs Supabase) |
| http://localhost:8000/docs | ❌ No | API documentation |
| http://localhost:8086 | ❌ No | InfluxDB UI |
