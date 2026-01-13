# Local Development Guide - NILM Energy Monitor

Complete guide to running the NILM Energy Monitor locally with InfluxDB for prediction storage.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Architecture](#architecture)
- [Available Commands](#available-commands)
- [Troubleshooting](#troubleshooting)
- [Data Model](#data-model)

---

## Overview

This guide enables you to:
- Run the entire NILM system locally without cloud dependencies
- Store predictions in a local InfluxDB time-series database
- Generate mock predictions from CSV data for testing
- Access the dashboard, API server, and InfluxDB UI on localhost

**What's included:**
- Docker Compose for InfluxDB 2.7
- Node.js/Express backend API server
- Prediction generation from CSV dataset
- React frontend with local mode support

---

## Prerequisites

### Required Software

1. **Node.js 18+** with npm
   - Check: `node --version` (should be 18.x or higher)
   - Download: https://nodejs.org/

2. **Docker** and **Docker Compose**
   - Windows: Docker Desktop with WSL2
   - macOS: Docker Desktop
   - Linux: Docker Engine + Docker Compose plugin
   - Check: `docker --version` and `docker compose version`

3. **Git** (for cloning the repository)
   - Check: `git --version`

### Windows-Specific Requirements

If you're on Windows, you need:
- **WSL2** (Windows Subsystem for Linux 2)
- **Docker Desktop** with WSL2 backend enabled

---

## Quick Start

Get up and running in 3 commands:

```bash
# 1. Start InfluxDB
docker compose up -d

# 2. Seed predictions (from repository root)
cd frontend
npm install
npm run predictions:seed

# 3. Start development servers
npm run local:dev
```

Access the application:
- **Dashboard**: http://localhost:8080
- **InfluxDB UI**: http://localhost:8086
  - Username: `admin`
  - Password: `admin12345` (from `.env.local`)

---

## Detailed Setup

### Step 1: Clone and Configure

```bash
# Navigate to repository root
cd /path/to/MTS3-MCTE-Team-Project-Energy-G1

# Create local environment file
cp .env.local.example .env.local
```

Edit `.env.local` and set a secure token:

```env
INFLUX_TOKEN=your-unique-secure-token-min-32-characters
```

**Generate a secure token:**
```bash
# Option 1: Using openssl
openssl rand -hex 32

# Option 2: Using Node.js
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
```

### Step 2: Start InfluxDB

```bash
# Start InfluxDB in the background
docker compose up -d influxdb

# Verify it's running
docker compose ps

# Expected output:
# NAME            STATUS        PORTS
# nilm-influxdb   Up (healthy)  0.0.0.0:8086->8086/tcp
```

Wait for health check to pass (~10-30 seconds).

### Step 3: Install Dependencies

```bash
# Frontend dependencies
cd frontend
npm install

# Backend dependencies
cd ../backend
npm install
```

### Step 4: Generate and Load Predictions

```bash
# From frontend directory
cd frontend
npm run predictions:seed
```

This will:
1. Read `frontend/public/data/nilm_ready_dataset.csv` (~5.6 MB)
2. Generate predictions for 11 appliances
3. Write ~150,000 data points to InfluxDB
4. Takes ~30-60 seconds

**Expected output:**
```
========================================
  NILM Prediction Seeder
========================================

📊 Step 1: Generating predictions from CSV...
✅ Generated 165,000 predictions

📝 Step 2: Writing predictions to InfluxDB...
  [100%] Written 165,000/165,000 points (45.2s elapsed)

✅ SUCCESS
📊 Total predictions written: 165,000
⏱️  Total time: 46.5s
```

### Step 5: Verify Data

```bash
# From frontend directory
npm run predictions:verify
```

**Expected output:**
```
========================================
  InfluxDB Data Verification
========================================

✅ Found 11 appliances with predictions:

  RangeHood                      15,000 points
  Dryer                          15,000 points
  Stove                          15,000 points
  ...
  ─────────────────────────────────────────
  TOTAL                         165,000 points

✅ Time range:
  First: 2023-01-01T00:00:00.000Z
  Last:  2023-12-31T23:45:00.000Z
```

### Step 6: Start Development Servers

```bash
# From frontend directory
npm run local:dev
```

This starts:
- **Local API server** on port 3001 (backend proxy)
- **Vite dev server** on port 8080 (frontend)

**Expected output:**
```
[api]  ========================================
[api]    Local API Server for NILM Monitor
[api]  ========================================
[api]  🚀 Server running on http://localhost:3001

[vite] VITE v5.4.19 ready in 1234 ms
[vite] ➜  Local:   http://localhost:8080/
```

### Step 7: Enable Local Mode in Frontend

Create `frontend/.env.local`:

```env
# Enable local InfluxDB mode
VITE_LOCAL_MODE=true
```

Restart the dev server (`Ctrl+C` then `npm run local:dev`).

The dashboard will now fetch data from local InfluxDB instead of demo CSV or cloud API.

---

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                   CSV Dataset (Demo Data)                  │
│            frontend/public/data/nilm_ready_dataset.csv     │
│                      (~5.6 MB, 15-min intervals)           │
└─────────────────────────┬─────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────┐
│             Prediction Generator (TypeScript)              │
│              scripts/generate-predictions.ts               │
│   • Reads CSV aggregate power                             │
│   • Applies appliance weights (Dryer: 15%, HeatPump: 25%)│
│   • Adds deterministic noise (sinusoidal patterns)        │
└─────────────────────────┬─────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────┐
│                  InfluxDB 2.7 (Docker)                     │
│                    Port: 8086                              │
│   • Bucket: predictions                                   │
│   • Measurement: appliance_prediction                     │
│   • Tags: building_id, appliance_name                     │
│   • Fields: predicted_kw, confidence                      │
│   • Storage: ./influxdb-data (persistent volume)          │
└─────────────────────────┬─────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────┐
│            Local API Server (Express + TypeScript)         │
│                backend/local-server.ts                     │
│                    Port: 3001                              │
│   • Endpoint: GET /api/local/predictions                  │
│   • Queries InfluxDB using Flux language                  │
│   • Returns JSON to frontend                              │
│   • Token stays server-side (not exposed to browser)      │
└─────────────────────────┬─────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────┐
│                  Vite Dev Server (React)                   │
│                       Port: 8080                           │
│   • Proxy: /api/local/* → http://localhost:3001          │
│   • Hook: useLocalInfluxPredictions()                     │
│   • Mode: 'local' (alongside 'demo' and 'api')           │
│   • UI: Dashboard, charts, appliance status               │
└───────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Seeding Phase** (one-time):
   - Read CSV → Generate predictions → Write to InfluxDB

2. **Runtime Phase** (dashboard running):
   - Frontend (React) → Local API (Express) → InfluxDB → Response → Frontend

### Security

- InfluxDB token stored in `.env.local` (gitignored)
- Backend server queries InfluxDB on behalf of frontend
- Token never exposed to browser
- CORS restricted to `http://localhost:8080`

---

## Available Commands

### Docker Commands

```bash
# Start services
docker compose up -d

# Stop services
docker compose down

# View logs
docker compose logs influxdb

# Follow logs in real-time
docker compose logs -f influxdb

# Restart InfluxDB
docker compose restart influxdb

# Remove all data (WARNING: destructive)
docker compose down -v
rm -rf influxdb-data influxdb-config
```

### Prediction Management

```bash
# From frontend directory

# Seed predictions (run once after starting InfluxDB)
npm run predictions:seed

# Verify data in InfluxDB
npm run predictions:verify
```

### Development Servers

```bash
# From frontend directory

# Start both servers (API + frontend)
npm run local:dev

# Start API server only
npm run local:server

# Start frontend only (requires API server already running)
npm run dev
```

### Build and Quality

```bash
# From frontend directory

# Production build
npm run build

# Type checking
npm run typecheck

# Linting
npm run lint

# Format code
npm run format
```

---

## Troubleshooting

### Problem: InfluxDB container won't start

**Symptoms:**
```
docker compose ps
# STATUS shows "Exited" or "Restarting"
```

**Solution:**
```bash
# Check logs for errors
docker compose logs influxdb

# Common issues:
# 1. Port 8086 already in use
#    → Stop other services using that port
# 2. Permission denied on influxdb-data/
#    → Fix permissions: sudo chown -R $USER influxdb-data
# 3. Docker daemon not running
#    → Start Docker Desktop

# Nuclear option: reset everything
docker compose down -v
rm -rf influxdb-data influxdb-config
docker compose up -d
```

### Problem: Predictions seed fails with "ECONNREFUSED"

**Symptoms:**
```
❌ ERROR
Error writing predictions: connect ECONNREFUSED 127.0.0.1:8086
```

**Solution:**
```bash
# 1. Verify InfluxDB is running
docker compose ps
# Should show "Up (healthy)"

# 2. Wait for health check
docker compose logs influxdb | grep "ready"
# Should see: "msg"="InfluxDB is ready"

# 3. Check .env.local exists and has valid token
cat .env.local | grep INFLUX_TOKEN

# 4. Retry seeding
npm run predictions:seed
```

### Problem: Frontend shows "Cannot connect to local API server"

**Symptoms:**
- Dashboard displays error: "Cannot connect to local API server"
- No data appears in local mode

**Solution:**
```bash
# 1. Check if API server is running
curl http://localhost:3001/health
# Should return: {"status":"ok", ...}

# 2. If not running, start it
cd frontend
npm run local:server

# 3. Check Vite proxy configuration
# frontend/vite.config.ts should have:
# proxy: { '/api/local': { target: 'http://localhost:3001' } }

# 4. Restart frontend dev server
npm run dev
```

### Problem: "No data" in dashboard despite successful seed

**Symptoms:**
- Predictions seed completes successfully
- Dashboard shows "No data available"

**Solution:**
```bash
# 1. Verify data in InfluxDB
npm run predictions:verify
# Should show non-zero point counts

# 2. Check VITE_LOCAL_MODE is enabled
cat frontend/.env.local | grep VITE_LOCAL_MODE
# Should be: VITE_LOCAL_MODE=true

# 3. Check browser console for errors
# Open DevTools → Console
# Look for fetch errors or CORS issues

# 4. Try manual query
curl "http://localhost:3001/api/local/predictions?buildingId=local&start=-7d&end=now()"
# Should return JSON with data array
```

### Problem: Windows Docker Desktop issues

**WSL2 Integration:**
```bash
# 1. Enable WSL2 backend in Docker Desktop settings
# Settings → General → Use the WSL 2 based engine

# 2. Ensure WSL integration is enabled for your distro
# Settings → Resources → WSL Integration

# 3. Restart Docker Desktop

# 4. Run commands from WSL terminal, not PowerShell
wsl
cd /mnt/c/Users/YourName/path/to/project
docker compose up -d
```

### Problem: Port conflicts

**Port 8086 (InfluxDB):**
```bash
# Check what's using the port
# Windows:
netstat -ano | findstr :8086

# macOS/Linux:
lsof -i :8086

# Kill the process or change docker-compose.yml ports:
# ports: - "9086:8086"  # Map to different host port
```

**Port 3001 (API server):**
```bash
# Change in .env.local:
LOCAL_API_PORT=3002

# And update frontend/.env.local:
VITE_LOCAL_API_URL=http://localhost:3002
```

---

## Data Model

### InfluxDB Schema

**Bucket:** `predictions`

**Measurement:** `appliance_prediction`

**Tags:**
- `building_id` (string): Building identifier (default: "local")
- `appliance_name` (string): Appliance type

**Fields:**
- `predicted_kw` (float): Predicted power consumption in kilowatts
- `confidence` (float): Prediction confidence score (0.0 to 1.0)

**Timestamp:** Millisecond precision

### Appliances

The system tracks 11 appliances:

| Appliance Name         | Typical Weight | Typical Power |
|------------------------|----------------|---------------|
| RangeHood              | 5%             | Low           |
| Dryer                  | 15%            | High          |
| Stove                  | 20%            | High          |
| Dishwasher             | 10%            | Medium        |
| HeatPump               | 25%            | High          |
| Washer                 | 8%             | Medium        |
| Fridge                 | 7%             | Low           |
| Microwave              | 4%             | Medium        |
| AirConditioner         | 3%             | Medium        |
| ElectricWaterHeater    | 2%             | High          |
| Lighting               | 1%             | Low           |

### Example Flux Query

```flux
from(bucket: "predictions")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "appliance_prediction")
  |> filter(fn: (r) => r.appliance_name == "Dryer")
  |> filter(fn: (r) => r._field == "predicted_kw")
```

---

## Limitations

### Current Limitations

1. **Mock Predictions**: Predictions are generated using a deterministic algorithm, not a real ML model
2. **Single Building**: Only supports `building_id = "local"`
3. **No Real-Time**: Predictions are pre-generated, not computed in real-time
4. **CSV-Based**: Uses historical CSV data as input

### Future Enhancements

- Connect to actual NILM ML models
- Real-time prediction inference
- Multi-building support
- Integration with live sensor data
- Prediction accuracy metrics

---

## Next Steps

Once you have local development working:

1. **Explore InfluxDB UI**: http://localhost:8086
   - Data Explorer: Query and visualize data
   - Dashboards: Create custom visualizations
   - Tasks: Schedule data processing

2. **Customize Predictions**: Edit `scripts/generate-predictions.ts`
   - Adjust appliance weights
   - Modify noise patterns
   - Add new appliances

3. **Integrate with ML Models**: Replace mock generator with actual NILM model inference

4. **Deploy to Production**: See [DEPLOYMENT_STEPS.md](../DEPLOYMENT_STEPS.md) for cloud deployment

---

## Support

If you encounter issues not covered in this guide:

1. Check [INFLUX_SCHEMA.md](./INFLUX_SCHEMA.md) for detailed schema documentation
2. Check [SUPABASE_SETUP.md](./SUPABASE_SETUP.md) for production setup
3. Review InfluxDB logs: `docker compose logs influxdb`
4. Check API server logs in the terminal running `npm run local:server`
5. Open browser DevTools → Console for frontend errors

---

## Summary Checklist

- [ ] Docker and Docker Compose installed
- [ ] Node.js 18+ installed
- [ ] `.env.local` created with secure token
- [ ] InfluxDB started: `docker compose up -d`
- [ ] InfluxDB health check passed
- [ ] Dependencies installed: `npm install` (frontend and backend)
- [ ] Predictions seeded: `npm run predictions:seed`
- [ ] Data verified: `npm run predictions:verify`
- [ ] `frontend/.env.local` has `VITE_LOCAL_MODE=true`
- [ ] Dev servers running: `npm run local:dev`
- [ ] Dashboard accessible: http://localhost:8080
- [ ] InfluxDB UI accessible: http://localhost:8086

Good luck with local development! 🚀
