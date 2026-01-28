# NILM Energy Monitor - Installation Guide

Complete guide to set up and run the NILM Energy Monitor on a fresh system.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Prerequisites](#2-prerequisites)
3. [Project Setup](#3-project-setup)
4. [Running the Application](#4-running-the-application)
5. [Verification](#5-verification)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. System Requirements

| Component | Requirement |
|-----------|-------------|
| **RAM** | 8 GB minimum (4 GB for Docker) |
| **Storage** | 10 GB free disk space |
| **CPU** | 4 cores recommended |
| **OS** | Windows 10/11 (WSL2), macOS 12+, Linux (Ubuntu 22.04+) |

---

## 2. Prerequisites

### 2.1 Docker Desktop

`ash
# Verify installation
docker --version          # Docker 24+
docker compose version    # Compose v2+
`

**Install from:** https://www.docker.com/products/docker-desktop/

### 2.2 Node.js 22+ (for frontend)

`ash
# Verify installation
node --version   # v22.x.x
npm --version    # 10.x.x+
`

**Install via nvm:** https://github.com/nvm-sh/nvm

### 2.3 Git

`ash
git --version
`

---

## 3. Project Setup

### 3.1 Clone Repository

`ash
git clone https://github.com/Thetommigun432/MTS3-MCTE-Team-Project-Energy-G1.git
cd MTS3-MCTE-Team-Project-Energy-G1
`

### 3.2 Environment Configuration

`ash
cp .env.local.example .env.local
`

**Default values (no changes needed for local dev):**
| Variable | Default |
|----------|---------|
| INFLUX_TOKEN | admin-token |
| INFLUX_ORG | energy-monitor |
| INFLUX_BUCKET_PRED | predictions |

---

## 4. Running the Application

### Option A: Simulator Mode (Local Data)

`ash
# Start all services
docker compose up -d --build

# Start frontend
npm install
npm run dev:web
`

### Option B: MQTT Realtime Mode (Howest Live Data)

`ash
# Start with MQTT ingestor
docker compose -f compose.realtime.yaml up -d --build

# Start frontend
npm install
npm run dev:web
`

### Services Started

| Service | Port | Description |
|---------|------|-------------|
| backend | 8000 | FastAPI REST API |
| worker | - | ML inference (10 models) |
| influxdb | 8086 | Time-series database |
| redis | 6379 | Rolling window cache |
| simulator/mqtt-ingestor | - | Data source |

### Access Points

| URL | Description |
|-----|-------------|
| http://localhost:8080/live | **Public Dashboard** (no login) |
| http://localhost:8080/app | Full app with auth |
| http://localhost:8000/api/docs | Swagger API docs |
| http://localhost:8086 | InfluxDB UI |

---

## 5. Verification

### 5.1 Health Checks

`ash
# API health
curl http://localhost:8000/live
# Expected: {"status":"ok"}

curl http://localhost:8000/ready
# Expected: {"status":"ready","checks":{"influxdb":"ok","redis":"ok"}}
`

### 5.2 Check Models

`ash
curl http://localhost:8000/api/models
# Should return 10 loaded models
`

### 5.3 Check Data Flow

`ash
# Simulator mode
docker compose logs simulator | tail -5

# MQTT mode
docker compose -f compose.realtime.yaml logs mqtt-ingestor | tail -5
# Should show: "Sent X readings | Latest: XXX W"

# Worker predictions
docker compose logs worker | tail -5
# Should show: "Predictions written: building=building-1, appliances=10"
`

### 5.4 Query API

`ash
# Get buildings
curl http://localhost:8000/api/analytics/buildings
# Expected: {"buildings":["building-1"]}

# Get readings (last 5 min)
curl "http://localhost:8000/api/analytics/readings?building_id=building-1&start=-5m&end=now()"
`

---

## 6. Troubleshooting

### Containers won't start

`ash
docker compose down -v
docker compose up --build -d
`

### Port already in use

`ash
# Windows
netstat -ano | findstr :8000

# macOS/Linux
lsof -i :8000
`

### No predictions appearing

1. **Check window size:**
   `ash
   docker exec nilm-redis redis-cli LLEN nilm:building-1:window
   # Needs 4096 samples for first prediction
   `

2. **Speed up simulator:**
   `ash
   docker compose down
   SIM_SPEEDUP=100 docker compose up -d
   `

### Frontend can't connect

- Verify backend is running: `docker compose ps`
- Check `VITE_BACKEND_URL` in `apps/web/.env`

### Out of memory

- Increase Docker Desktop memory (Settings > Resources > 4GB minimum)

---

## Quick Commands

`ash
# Start
docker compose up -d --build

# Stop
docker compose down

# Full reset
docker compose down -v

# Logs
docker compose logs -f

# Restart service
docker compose restart backend

# Check status
docker compose ps
`

---

## Architecture

`
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                     │
│                                                             │
│  ┌──────────┐   ┌─────────┐   ┌───────┐   ┌──────────┐    │
│  │Simulator │──▶│ Backend │──▶│ Redis │◀──│  Worker  │    │
│  │  /MQTT   │   │  (API)  │   │       │   │ (PyTorch)│    │
│  └──────────┘   └────┬────┘   └───────┘   └────┬─────┘    │
│                      │                          │          │
│                      ▼                          ▼          │
│                 ┌──────────────────────────────────┐       │
│                 │           InfluxDB               │       │
│                 │    (predictions time-series)     │       │
│                 └──────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
         ▲
         │ HTTP API (port 8000)
    ┌────┴────┐
    │Frontend │ (port 8080)
    └─────────┘
`

---

*Last updated: January 2026*
