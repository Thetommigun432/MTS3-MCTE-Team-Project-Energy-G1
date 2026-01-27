# NILM Energy Monitor - Installation Guide

A comprehensive guide to set up and run the NILM Energy Monitor project from source code on a fresh system.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Prerequisites Installation](#2-prerequisites-installation)
3. [Project Setup](#3-project-setup)
4. [Running the Application](#4-running-the-application)
5. [Verifying the Installation](#5-verifying-the-installation)
6. [Frontend Development (Optional)](#6-frontend-development-optional)
7. [Troubleshooting](#7-troubleshooting)
8. [Configuration Reference](#8-configuration-reference)

---

## 1. System Requirements

### Minimum Hardware
| Component | Requirement |
|-----------|-------------|
| **RAM** | 8 GB (4 GB available for Docker containers) |
| **Storage** | 10 GB free disk space |
| **CPU** | 4 cores recommended |

### Supported Operating Systems
- **Windows 10/11** (with WSL2 for Docker)
- **macOS** 12+ (Monterey or later)
- **Linux** (Ubuntu 22.04+, Debian 11+)

---

## 2. Prerequisites Installation

### 2.1 Install Docker Desktop

Docker is required to run all backend services (API, database, Redis, ML inference worker).

#### Windows
1. Download [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. Run the installer and follow prompts
3. **Enable WSL2 backend** during installation (recommended)
4. Restart your computer when prompted
5. Open Docker Desktop and wait for it to start

#### macOS
1. Download [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)
2. Drag to Applications folder and launch
3. Grant permissions when prompted

#### Linux (Ubuntu/Debian)
```bash
# Install Docker Engine
curl -fsSL https://get.docker.com | sh

# Add your user to docker group (avoids needing sudo)
sudo usermod -aG docker $USER

# Reboot or log out/in for group changes to apply
```

#### Verify Docker Installation
```bash
docker --version          # Should show Docker version 24+
docker compose version    # Should show Docker Compose v2+
```

---

### 2.2 Install Node.js (for Frontend Development)

Only required if you want to run the frontend in development mode.

#### Recommended: Use Node Version Manager (nvm)

**Windows (PowerShell as Administrator):**
```powershell
# Install nvm-windows from: https://github.com/coreybutler/nvm-windows/releases
# Then:
nvm install 22
nvm use 22
```

**macOS / Linux:**
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc  # or ~/.zshrc
nvm install 22
nvm use 22
```

#### Verify Node.js Installation
```bash
node --version   # Should show v22.x.x
npm --version    # Should show 10.x.x or higher
```

---

### 2.3 Install Git

#### Windows
Download and install [Git for Windows](https://git-scm.com/download/win)

#### macOS
```bash
xcode-select --install
```

#### Linux
```bash
sudo apt update && sudo apt install git -y
```

---

## 3. Project Setup

### 3.1 Clone the Repository

```bash
git clone https://github.com/Thetommigun432/MTS3-MCTE-Team-Project-Energy-G1.git nilm-energy
cd nilm-energy
```

### 3.2 Verify Project Structure

Ensure these key files/folders exist:
```
nilm-energy/
├── compose.yaml              # Docker Compose orchestration
├── apps/
│   ├── backend/
│   │   ├── Dockerfile        # Backend API container
│   │   ├── Dockerfile.worker # ML inference worker
│   │   ├── data/
│   │   │   └── simulation_data.parquet  # Demo dataset
│   │   └── models/
│   │       ├── registry.json # Model configuration
│   │       └── tcn_sa/       # Pre-trained TCN-SA models (10 appliances)
│   └── web/                  # React frontend
└── checkpoints/              # Legacy checkpoint location (models are in apps/backend/models/)
```

### 3.3 Environment Configuration (Optional)

For most local testing, defaults work out of the box. To customize:

```bash
# Copy example environment files
cp .env.local.example .env.local        # Root settings (optional)
cp apps/web/.env.example apps/web/.env  # Frontend settings (optional)
```

**Default values used if `.env.local` is not present:**
| Variable | Default Value |
|----------|---------------|
| `INFLUX_TOKEN` | `admin-token` |
| `INFLUX_ORG` | `energy-monitor` |
| `INFLUX_BUCKET_PRED` | `predictions` |

---

## 4. Running the Application

### 4.1 Start the Full Stack (Recommended)

This command builds and starts all 5 services:

```bash
docker compose up --build -d
```

**Services Started:**
| Service | Port | Description |
|---------|------|-------------|
| `backend` | 8000 | FastAPI REST API |
| `worker` | - | ML inference worker (no exposed port) |
| `simulator` | - | Reads demo data & feeds pipeline |
| `influxdb` | 8086 | Time-series database |
| `redis` | 6379 | Rolling window cache |

### 4.2 Monitor Startup Progress

```bash
# Watch all logs
docker compose logs -f

# Or watch specific services
docker compose logs -f backend worker
```

**Expected startup sequence:**
1. **InfluxDB** initializes (~10s)
2. **Redis** becomes healthy (~5s)
3. **Backend** starts API server (~15s)
4. **Worker** loads ML models (~20-30s)
5. **Simulator** begins streaming data

### 4.3 Wait for Models to Warm Up

The ML models require a window of historical data before producing predictions.

```bash
# Check rolling window size (needs 4096 samples - the largest model window)
docker exec nilm-redis redis-cli LLEN nilm:building-1:window
```

At default speed (1 sample/second), expect ~68 minutes for first predictions (4096 samples).

**Speed up for demo:**
```bash
# Stop and restart with faster simulation
docker compose down
SIM_SPEEDUP=100 docker compose up -d
```
At 100x speed, predictions appear in ~1 minute (4096 samples / 100 = ~41 seconds).

---

## 5. Verifying the Installation

### 5.1 Health Checks

```bash
# Backend API is running
curl http://localhost:8000/live
# Expected: {"status":"ok"}

# Backend is ready for requests
curl http://localhost:8000/ready
# Expected: {"status":"ready","checks":{"influxdb":"ok","redis":"ok"}}
```

### 5.2 Check Model Registry

```bash
curl http://localhost:8000/api/models | python -m json.tool
```

Should return a list of 10 loaded appliance models (HeatPump, Dryer, etc.)

### 5.3 Verify Data Pipeline

```bash
# Check simulator is sending data
docker compose logs simulator | tail -10
# Should show: "Sent reading: ts=..."

# Check Redis window is filling
docker exec nilm-redis redis-cli LLEN nilm:building-1:window
# Should increase over time

# Check worker is making predictions (after warmup)
docker compose logs worker | grep -i "Prediction written"
```

### 5.4 Query Predictions

After warmup, query the disaggregation results:

```bash
curl "http://localhost:8000/api/analytics/predictions?building_id=building-1&start=-5m"
```

---

## 6. Frontend Development (Optional)

The frontend can run separately from Docker for faster development iteration.

### 6.1 Install Dependencies

```bash
cd apps/web
npm install
```

### 6.2 Configure Environment

```bash
# Create environment file
cp .env.example .env
```

Edit `apps/web/.env`:
```env
VITE_BACKEND_URL=http://localhost:8000
VITE_DEMO_MODE=true  # Use demo mode to bypass authentication
```

### 6.3 Start Development Server

```bash
npm run dev
```

### 6.4 Access the Application

Open your browser to: **http://localhost:8080**

The dashboard will display:
- Real-time aggregate power consumption
- Disaggregated appliance predictions (after pipeline warmup)
- Historical trends and analytics

---

## 7. Troubleshooting

### Problem: Docker containers fail to start

**Solution:**
```bash
# Clean restart with volume removal
docker compose down -v
docker compose up --build -d
```

### Problem: "Port already in use" error

**Solution:**
```bash
# Check what's using the port (e.g., 8000)
# Windows:
netstat -ano | findstr :8000

# macOS/Linux:
lsof -i :8000

# Stop the conflicting process or change port in compose.yaml
```

### Problem: No predictions appearing

**Causes & Solutions:**

1. **Window not full yet**
   ```bash
   docker exec nilm-redis redis-cli LLEN nilm:building-1:window
   # Need 4096 samples (largest model window size)
   ```

2. **Worker has errors**
   ```bash
   docker compose logs worker | grep -i error
   ```

3. **Simulator not running**
   ```bash
   docker compose logs simulator
   ```

### Problem: Frontend can't connect to backend

**Solution:**
- Ensure Docker backend is running: `docker compose ps`
- Check CORS settings in backend logs
- Verify `VITE_BACKEND_URL` is set correctly

### Problem: Out of memory errors

**Solution:**
- Increase Docker Desktop memory allocation (Settings > Resources)
- Minimum recommended: 4 GB for Docker

---

## 8. Configuration Reference

### Environment Variables

#### Backend (`compose.yaml` or `apps/backend/.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection string |
| `INFLUX_URL` | `http://influxdb:8086` | InfluxDB connection |
| `INFLUX_TOKEN` | `admin-token` | InfluxDB authentication token |
| `INFLUX_ORG` | `energy-monitor` | InfluxDB organization |
| `INFLUX_BUCKET_PRED` | `predictions` | Bucket for ML predictions |
| `PIPELINE_ROLLING_WINDOW_SIZE` | `4096` | Rolling window size (must match largest model window) |

#### Simulator (`compose.yaml`)

| Variable | Default | Description |
|----------|---------|-------------|
| `SIM_SPEEDUP` | `1` | Data streaming speed multiplier |
| `SIM_DURATION_SECONDS` | `0` | Max duration (0 = unlimited) |
| `BUILDING_ID` | `building-1` | Building identifier |

#### Frontend (`apps/web/.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_BACKEND_URL` | `/api` | Backend API URL |
| `VITE_DEMO_MODE` | `false` | Enable demo mode (no auth) |

---

## Quick Reference Commands

```bash
# Start everything
docker compose up --build -d

# Stop everything
docker compose down

# Stop and clean volumes (full reset)
docker compose down -v

# View logs
docker compose logs -f

# Restart single service
docker compose restart backend

# Rebuild single service
docker compose up --build backend -d

# Check service status
docker compose ps

# Check Redis window size
docker exec nilm-redis redis-cli LLEN nilm:building-1:window

# Query predictions API
curl "http://localhost:8000/api/analytics/predictions?building_id=building-1&start=-5m"
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Docker Compose Stack                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │Simulator │───▶│ Backend  │───▶│  Redis   │◀───│  Worker  │  │
│  │(parquet) │    │  (API)   │    │(window)  │    │(ML Infer)│  │
│  └──────────┘    └────┬─────┘    └──────────┘    └────┬─────┘  │
│                       │                                │        │
│                       ▼                                ▼        │
│                  ┌──────────────────────────────────────┐       │
│                  │          InfluxDB                    │       │
│                  │    (predictions time-series)         │       │
│                  └──────────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
         ▲
         │ HTTP API (port 8000)
         │
    ┌────┴────┐
    │ Frontend │ (optional, port 8080)
    │ React   │
    └─────────┘
```

---

## Tech Stack Summary

| Layer | Technology | Version |
|-------|------------|---------|
| **Frontend** | React, Vite, TypeScript | React 19, Vite 7 |
| **Backend API** | FastAPI, Python | Python 3.12, FastAPI 0.128 |
| **ML Inference** | PyTorch | PyTorch 2.5+ |
| **Time-Series DB** | InfluxDB | 2.8 |
| **Cache/Queue** | Redis | 7 |
| **Container Runtime** | Docker, Docker Compose | Docker 24+, Compose v2 |

---

## Support

For issues or questions:
1. Check the [Troubleshooting](#7-troubleshooting) section
2. Review logs: `docker compose logs -f`
3. Open an issue on the GitHub repository
4. Ask to an LLM agents integrated in your IDE :)

---

*Last updated: January 2026*
