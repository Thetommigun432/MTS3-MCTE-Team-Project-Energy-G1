# Local Development Guide - NILM Energy Monitor

Complete guide to running the NILM Energy Monitor locally with the FastAPI backend.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Architecture](#architecture)
- [Development Workflow](#development-workflow)
- [Available Commands](#available-commands)
- [Troubleshooting](#troubleshooting)
- [Data Model](#data-model)

---

## Overview

This guide enables you to:
- Run the complete NILM system locally without cloud dependencies
- Use the unified FastAPI backend for all data and ML operations
- Store predictions and readings in local InfluxDB time-series database
- Generate and persist predictions from CSV data
- Access the React frontend, FastAPI backend, and InfluxDB UI on localhost

**What's included:**
- Docker Compose for InfluxDB 2.7 + FastAPI backend
- FastAPI (Python 3.12) backend with ML inference
- Prediction generation and seeding scripts
- React frontend (Vite) with Supabase authentication
- Local development with hot reload

---

## Prerequisites

### Required Software

1. **Docker** and **Docker Compose**
   - Windows: Docker Desktop with WSL2
   - macOS: Docker Desktop
   - Linux: Docker Engine + Docker Compose plugin
   - Check: `docker --version` and `docker compose version`

2. **Node.js 18+** with npm (for frontend development)
   - Check: `node --version` (should be 18.x or higher)
   - Download: https://nodejs.org/

3. **Python 3.12** (optional - only if running backend locally without Docker)
   - Check: `python --version`
   - Backend can run entirely in Docker, so this is optional

4. **Git** (for cloning the repository)
   - Check: `git --version`

### Windows-Specific Requirements

If you're on Windows, you need:
- **WSL2** (Windows Subsystem for Linux 2)
- **Docker Desktop** with WSL2 backend enabled

---

## Quick Start

Get up and running in 3 steps:

```bash
# 1. Configure environment
cp .env.local.example .env.local
# Edit .env.local and set INFLUX_TOKEN

# 2. Start backend and InfluxDB
docker compose up -d

# 3. Seed test data
npm run predictions:seed
```

Access the services:
- **Backend API**: http://localhost:8000
  - Health: http://localhost:8000/live
  - Docs: http://localhost:8000/docs (dev only)
- **InfluxDB UI**: http://localhost:8086
  - Username: `admin`
  - Password: `admin12345` (from `.env.local`)

### Frontend Development

```bash
cd apps/web
npm install
npm run dev
```

Access the frontend:
- **Dashboard**: http://localhost:8080

---

## Detailed Setup

### Step 1: Clone and Configure

```bash
# Clone repository
git clone <repository-url>
cd MTS3-MCTE-Team-Project-Energy-G1

# Create local environment file
cp .env.local.example .env.local
```

Edit `.env.local` and configure:

```env
# InfluxDB configuration
INFLUX_TOKEN=your-unique-secure-token-min-32-characters
INFLUX_ORG=energy-monitor
INFLUX_BUCKET=predictions
INFLUX_BUCKET_RAW=raw_sensor_data

# Backend environment
ENV=dev
DEBUG=true

# (Optional) Supabase configuration if using auth
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_PUBLISHABLE_KEY=your-key
```

**Important**: Generate a secure InfluxDB token:
```bash
# Generate a random 32-character token
openssl rand -base64 32
```

### Step 2: Start Backend Services

```bash
# Start InfluxDB + Backend
docker compose up -d

# Check services are running
docker compose ps

# View logs
docker compose logs -f backend
docker compose logs -f influxdb
```

**Verify backend is healthy:**
```bash
curl http://localhost:8000/live
# Expected: {"status":"ok","request_id":"..."}

curl http://localhost:8000/ready
# Expected: {"status":"ok","checks":{...}}
```

### Step 3: Initialize InfluxDB

The backend automatically creates buckets on startup, but you can verify:

```bash
# Access InfluxDB UI
open http://localhost:8086

# Login with credentials from .env.local
# Username: admin
# Password: admin12345
```

Navigate to **Data > Buckets** and verify:
- `predictions` bucket exists
- `raw_sensor_data` bucket exists

### Step 4: Seed Test Data

Generate and persist predictions from the NILM CSV dataset:

```bash
# From repository root
npm run predictions:seed
```

This script:
1. Reads `apps/web/public/data/nilm_ready_dataset.csv`
2. Generates predictions using ML (if backend available) or mock data
3. Writes predictions to InfluxDB `predictions` bucket
4. Takes 30-60 seconds for the full dataset

**Options:**
```bash
# Force mock mode (skip ML inference)
USE_ML_INFERENCE=false npm run predictions:seed

# Custom backend URL
BACKEND_URL=http://localhost:8000 npm run predictions:seed
```

### Step 5: Start Frontend

```bash
cd apps/web
npm install
npm run dev
```

The frontend starts at **http://localhost:8080** with:
- Vite dev server with hot module replacement
- Proxy: `/api/*` → `http://localhost:8000`
- Demo mode enabled by default (no auth required)

---

## Architecture

### Local Development Stack

```
┌──────────────────────────────────────────────────────────┐
│                    Developer (Browser)                    │
│                  http://localhost:8080                    │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │   Vite Dev       │
              │   Server         │
              │   (Port 8080)    │
              └────────┬─────────┘
                       │
         /api/* →      │ (proxy)
                       ▼
              ┌──────────────────┐
              │   FastAPI        │◄────────┐
              │   Backend        │         │ Read models
              │   (Port 8000)    │         │
              └────┬─────────────┘         │
                   │                       │
     ┌─────────────┼───────────────┐       │
     │             │               │       │
     ▼             ▼               ▼       ▼
┌─────────┐   ┌─────────┐   ┌──────────────────┐
│InfluxDB │   │Supabase │   │ Model Registry   │
│(Docker) │   │(Cloud)  │   │ (Local FS)       │
│Port 8086│   │         │   │ models/registry  │
└─────────┘   └─────────┘   └──────────────────┘
```

### Component Responsibilities

**FastAPI Backend** (`apps/backend`):
- Health checks (`/live`, `/ready`)
- ML inference endpoint (`POST /infer`)
- Analytics endpoints (`GET /analytics/readings`, `/analytics/predictions`)
- Model registry (`GET /models`)
- JWT authentication & authorization
- InfluxDB client
- Supabase client (for user data)

**React Frontend** (`apps/web`):
- Dashboard UI with charts
- Building/appliance selection
- Supabase authentication
- API client with Bearer token injection
- Demo mode (works without auth)

**InfluxDB**:
- `predictions` bucket: ML predictions with confidence scores
- `raw_sensor_data` bucket: Sensor readings (if ingesting live data)

---

## Development Workflow

### Hot Reload Development

**Backend:**
```bash
# Backend auto-reloads on code changes
docker compose logs -f backend

# To restart after dependency changes:
docker compose restart backend
```

**Frontend:**
```bash
# Frontend hot reloads automatically
cd apps/web
npm run dev
```

### Testing Changes End-to-End

1. **Update backend code** in `apps/backend/`
2. **Backend auto-reloads** (thanks to mounted volume)
3. **Update frontend code** in `apps/web/src/`
4. **Frontend hot reloads** automatically
5. **Test in browser** at http://localhost:8080

### Making API Calls from Frontend

The frontend uses a centralized API client:

```typescript
// apps/web/src/services/energy.ts
import { energyApi } from '@/services/energy';

// Get predictions
const response = await energyApi.getPredictions({
  building_id: 'building-123',
  start: '-7d',
  end: 'now()',
  resolution: '15m'
});

// Run inference
const result = await energyApi.runInference({
  building_id: 'building-123',
  appliance_id: 'fridge',
  window: [...], // 1000 floats
});
```

All calls:
- Go through Vite proxy: `/api/*` → `http://localhost:8000`
- Include `Authorization: Bearer <token>` if user is logged in
- Include `X-Request-ID` for tracing
- Parse backend error format with request_id

### Running Backend Tests

```bash
cd apps/backend
pytest

# With coverage
pytest --cov=app --cov-report=html

# Specific test file
pytest tests/test_auth.py -v
```

### Running Frontend Tests

```bash
cd apps/web
npm test

# TypeScript type checking
npm run typecheck

# Linting
npm run lint
```

---

## Available Commands

### Repository Root

```bash
# Start all services
docker compose up -d

# Stop all services
docker compose down

# View logs
docker compose logs -f

# Rebuild backend
docker compose build backend

# Seed predictions
npm run predictions:seed
```

### Backend (`apps/backend`)

```bash
# Run backend locally (without Docker)
cd apps/backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Run tests
pytest

# Type checking
mypy .

# Linting
ruff check .
```

### Frontend (`apps/web`)

```bash
cd apps/web

# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Type checking
npm run typecheck

# Linting
npm run lint
```

### Data Scripts

```bash
# Generate predictions (ML or mock)
npm run predictions:seed

# Force mock mode
USE_ML_INFERENCE=false npm run predictions:seed

# Custom backend URL
BACKEND_URL=http://localhost:8000 npm run predictions:seed
```

---

## Troubleshooting

### Backend Not Starting

**Symptom**: `docker compose up -d` fails or backend exits immediately

**Fixes**:
```bash
# Check logs
docker compose logs backend

# Common issues:
# 1. Port 8000 already in use
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# 2. InfluxDB not ready
docker compose logs influxdb

# 3. Missing environment variables
cat .env.local
# Ensure INFLUX_TOKEN is set

# Restart services
docker compose down
docker compose up -d
```

### InfluxDB Connection Errors

**Symptom**: Backend logs show `INFLUX_CONNECTION_FAILED`

**Fixes**:
```bash
# 1. Verify InfluxDB is running
docker compose ps influxdb

# 2. Check InfluxDB health
curl http://localhost:8086/health

# 3. Verify token in .env.local matches InfluxDB
# Login to http://localhost:8086
# Navigate to Settings > Tokens

# 4. Restart services in order
docker compose down
docker compose up -d influxdb
# Wait 10 seconds
docker compose up -d backend
```

### Frontend Can't Reach Backend

**Symptom**: API calls fail with CORS errors or network errors

**Fixes**:
```bash
# 1. Verify backend is running
curl http://localhost:8000/live

# 2. Check Vite proxy config
cat apps/web/vite.config.ts
# Should have: proxy: { "/api": { target: "http://localhost:8000" } }

# 3. Check CORS origins in backend
# apps/backend/.env should include:
# CORS_ORIGINS=http://localhost:8080,http://localhost:5173

# 4. Restart frontend
cd apps/web
npm run dev
```

### Predictions Not Appearing

**Symptom**: Dashboard shows "No data" after seeding

**Fixes**:
```bash
# 1. Verify seeding succeeded
npm run predictions:seed
# Should show: "✅ Successfully wrote X predictions"

# 2. Check InfluxDB has data
# Login to http://localhost:8086
# Navigate to Data Explorer
# Query: from(bucket: "predictions") |> range(start: -7d)

# 3. Verify frontend is calling correct endpoint
# Open browser console at http://localhost:8080
# Should see API calls to /api/analytics/predictions

# 4. Check backend logs
docker compose logs backend | grep predictions
```

### Authentication Errors

**Symptom**: 401 Unauthorized errors

**Fixes**:
```bash
# 1. Check if Supabase is configured
cat apps/web/.env
# Should have VITE_SUPABASE_URL and VITE_SUPABASE_PUBLISHABLE_KEY

# 2. Use demo mode (no auth required)
# Edit apps/web/.env:
# VITE_DEMO_MODE=true

# 3. Check backend JWT verification
docker compose logs backend | grep JWT
```

### Port Conflicts

**Symptom**: "Port already in use" errors

**Fixes**:
```bash
# Backend (port 8000)
lsof -i :8000
kill -9 <PID>

# Frontend (port 8080)
lsof -i :8080
kill -9 <PID>

# InfluxDB (port 8086)
lsof -i :8086
docker compose down
docker compose up -d
```

---

## Data Model

### InfluxDB Schema

#### Predictions Bucket

```
Measurement: predictions
Tags:
  - building_id (string)
  - appliance_id (string)
  - model_version (string, optional)
Fields:
  - predicted_kw (float)
  - confidence (float, 0-1)
Timestamp: time (nanosecond precision)
```

**Example Query**:
```flux
from(bucket: "predictions")
  |> range(start: -7d)
  |> filter(fn: (r) => r.building_id == "building-123")
  |> filter(fn: (r) => r.appliance_id == "fridge")
  |> aggregateWindow(every: 15m, fn: mean)
```

#### Raw Sensor Data Bucket

```
Measurement: sensor_readings
Tags:
  - building_id (string)
  - sensor_id (string)
Fields:
  - value (float)
  - unit (string)
Timestamp: time (nanosecond precision)
```

### Backend API Schemas

See complete API documentation at http://localhost:8000/docs (when backend is running in dev mode).

**Key Endpoints**:
- `POST /infer`: Run inference and persist prediction
- `GET /analytics/predictions`: Query predictions from InfluxDB
- `GET /analytics/readings`: Query raw sensor data
- `GET /models`: List available ML models

---

## Advanced Topics

### Running Backend Without Docker

```bash
cd apps/backend

# Setup virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export INFLUX_URL=http://localhost:8086
export INFLUX_TOKEN=your-token
export INFLUX_ORG=energy-monitor
export INFLUX_BUCKET_PRED=predictions
export SUPABASE_URL=your-supabase-url
export SUPABASE_PUBLISHABLE_KEY=your-key

# Run server
uvicorn app.main:app --reload --port 8000
```

### Custom Model Integration

1. Add model files to `models/` directory
2. Update `models/registry.json`:
```json
{
  "models": [
    {
      "model_id": "my-custom-model",
      "model_version": "v1",
      "appliance_id": "fridge",
      "architecture": "lstm",
      "input_window_size": 1000,
      "is_active": true,
      "artifact_path": "my_model.h5"
    }
  ]
}
```
3. Restart backend: `docker compose restart backend`

### Production Deployment

See `docs/DEPLOYMENT.md` (to be created) for production deployment guides.

---

## Additional Resources

- **Backend README**: `apps/backend/README.md`
- **Frontend README**: `apps/web/docs/README.md`
- **Integration Audit**: `docs/integration-audit.md`
- **API Documentation**: http://localhost:8000/docs (when running)

---

**Last Updated**: 2026-01-21
**Version**: 2.0 (FastAPI Unified Backend)
