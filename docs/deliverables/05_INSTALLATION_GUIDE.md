# NILM Energy Monitor - Technical Installation Guide

**Document Version:** 1.0  
**Project:** NILM Energy Monitor  
**Date:** January 2026  
**Audience:** Developers & System Administrators  

---

## Table of Contents

1. [Hardware & Software Requirements](#1-hardware--software-requirements)
2. [Software Prerequisites](#2-software-prerequisites)
3. [Dependencies Overview](#3-dependencies-overview)
4. [Configuration](#4-configuration)
5. [Environment Setup](#5-environment-setup)
6. [Project Installation](#6-project-installation)
7. [Running the Application](#7-running-the-application)
8. [Reproducing the Demo](#8-reproducing-the-demo)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Hardware & Software Requirements

### 1.1 Minimum Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 2 cores | 4+ cores |
| **RAM** | 4 GB | 8+ GB |
| **Storage** | 10 GB free | 20+ GB free (for datasets) |
| **Network** | 10 Mbps | 100 Mbps |

### 1.2 Operating System

| OS | Tested Version | Notes |
|----|---------------|-------|
| **Windows** | Windows 10/11 | Primary development OS |
| **macOS** | 12 Monterey+ | Intel & Apple Silicon |
| **Linux** | Ubuntu 22.04+ | Debian-based distros |

### 1.3 Required Services (External)

| Service | Purpose | Required For |
|---------|---------|--------------|
| **InfluxDB** | Time-series database | Data storage |
| **Supabase** | Auth & metadata | Authentication |

These can be run locally via Docker or connected to cloud instances.

---

## 2. Software Prerequisites

### 2.1 Required Software

Install the following before proceeding:

#### Node.js (v22 LTS)

**Windows:**
```powershell
# Using winget
winget install OpenJS.NodeJS.LTS

# Or download from https://nodejs.org/
```

**macOS:**
```bash
# Using Homebrew
brew install node@22
```

**Linux:**
```bash
# Using nvm (recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 22
nvm use 22
```

**Verify installation:**
```bash
node --version  # Should show v22.x.x
npm --version   # Should show 10.x.x
```

#### Python 3.12+

**Windows:**
```powershell
# Using winget
winget install Python.Python.3.12

# Or download from https://python.org/
```

**macOS:**
```bash
brew install python@3.12
```

**Linux:**
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip
```

**Verify installation:**
```bash
python --version  # or python3 --version
# Should show Python 3.12.x
```

#### Docker & Docker Compose

**Windows:**
Download and install Docker Desktop from https://www.docker.com/products/docker-desktop

**macOS:**
```bash
brew install --cask docker
# Then launch Docker Desktop
```

**Linux:**
```bash
# Install Docker Engine
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose plugin
sudo apt install docker-compose-plugin
```

**Verify installation:**
```bash
docker --version
docker compose version
```

#### Git

**Windows:**
```powershell
winget install Git.Git
```

**macOS:**
```bash
brew install git
```

**Linux:**
```bash
sudo apt install git
```

### 2.2 Optional Tools

| Tool | Purpose |
|------|---------|
| **VS Code** | Recommended IDE |
| **Postman** | API testing |
| **DBeaver** | Database management |

---

## 3. Dependencies Overview

### 3.1 Frontend Dependencies

**Core:**
- React 19
- TypeScript 5.9
- Vite 7

**UI Components:**
- Tailwind CSS 3.4
- Radix UI Primitives
- Lucide Icons
- Recharts

**State & Data:**
- @supabase/supabase-js
- react-hook-form
- date-fns

**Full list:** See `apps/web/package.json`

### 3.2 Backend Dependencies

**Core:**
- FastAPI 0.115.6
- Uvicorn 0.34.0
- Pydantic 2.10.4

**Database:**
- influxdb-client 1.47.0
- supabase 2.11.0

**ML/Inference:**
- PyTorch 2.5.1 (CPU)
- NumPy <2.0.0
- safetensors 0.4.5

**Full list:** See `apps/backend/requirements.txt`

### 3.3 Infrastructure Dependencies

| Service | Version | Docker Image |
|---------|---------|--------------|
| InfluxDB | 2.8 | `influxdb:2.8` |
| Redis | 7.x | `redis:alpine` |

---

## 4. Configuration

### 4.1 Environment Files Overview

The project uses multiple environment files:

| File | Location | Purpose |
|------|----------|---------|
| `.env.local` | Root | Shared secrets (InfluxDB) |
| `.env` | `apps/backend/` | Backend configuration |
| `.env` | `apps/web/` | Frontend configuration |

### 4.2 Root Environment (.env.local)

Create `.env.local` in the project root:

```env
# InfluxDB Admin Token (minimum 32 characters)
INFLUX_TOKEN=your-secure-token-at-least-32-characters-long

# Optional overrides
INFLUX_ORG=energy-monitor
INFLUX_BUCKET_RAW=raw_sensor_data
INFLUX_BUCKET_PREDICTIONS=predictions
```

### 4.3 Backend Environment (apps/backend/.env)

Create `.env` in `apps/backend/`:

```env
# InfluxDB Connection
INFLUX_URL=http://localhost:8086
INFLUX_TOKEN=your-secure-token-at-least-32-characters-long
INFLUX_ORG=energy-monitor
INFLUX_BUCKET_RAW=raw_sensor_data
INFLUX_BUCKET_PREDICTIONS=predictions

# Supabase Connection
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-key

# CORS (comma-separated origins)
CORS_ORIGINS=http://localhost:8080,http://localhost:5173

# Redis (optional, for caching)
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=INFO
```

### 4.4 Frontend Environment (apps/web/.env)

Create `.env` in `apps/web/`:

```env
# Backend API URL (empty for local proxy)
VITE_BACKEND_URL=

# Supabase Configuration
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key

# Demo Mode (set to true to bypass auth)
VITE_DEMO_MODE=true
```

### 4.5 Example Files

The repository includes example files:
- `.env.local.example` → Copy to `.env.local`
- `apps/backend/.env.example` → Copy to `apps/backend/.env`
- `apps/web/.env.example` → Copy to `apps/web/.env`

---

## 5. Environment Setup

### 5.1 Clone the Repository

```bash
# Clone via HTTPS
git clone https://github.com/Thetommigun432/MTS3-MCTE-Team-Project-Energy-G1.git nilm-energy-monitor

# Navigate to project
cd nilm-energy-monitor
```

### 5.2 Create Environment Files

```bash
# Root environment
cp .env.local.example .env.local

# Backend environment
cp apps/backend/.env.example apps/backend/.env

# Frontend environment
cp apps/web/.env.example apps/web/.env
```

Edit each file to add your configuration values.

### 5.3 Python Virtual Environment (Backend)

**Windows:**
```powershell
cd apps/backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**macOS/Linux:**
```bash
cd apps/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 5.4 Node.js Dependencies (Frontend)

```bash
cd apps/web
npm install
```

---

## 6. Project Installation

### 6.1 Method 1: Docker Compose (Recommended)

This method starts all services in containers.

**Step 1: Start Infrastructure**
```bash
# From project root
docker compose up -d
```

This starts:
- InfluxDB (port 8086)
- Backend API (port 8000)
- Redis (port 6379)

**Step 2: Verify Services**
```bash
docker compose ps

# Check backend health
curl http://localhost:8000/live
# Should return: {"status": "ok"}
```

**Step 3: Start Frontend**
```bash
cd apps/web
npm run dev
```

### 6.2 Method 2: Local Development

Run services individually for development:

**Step 1: Start InfluxDB (Docker)**
```bash
docker run -d \
  --name influxdb \
  -p 8086:8086 \
  -e DOCKER_INFLUXDB_INIT_MODE=setup \
  -e DOCKER_INFLUXDB_INIT_USERNAME=admin \
  -e DOCKER_INFLUXDB_INIT_PASSWORD=admin12345 \
  -e DOCKER_INFLUXDB_INIT_ORG=energy-monitor \
  -e DOCKER_INFLUXDB_INIT_BUCKET=raw_sensor_data \
  -e DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=your-token-here \
  influxdb:2.8
```

**Step 2: Start Backend**
```bash
cd apps/backend
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows
uvicorn app.main:app --reload --port 8000
```

**Step 3: Start Frontend**
```bash
cd apps/web
npm run dev
```

### 6.3 Verify Installation

After starting all services:

1. **Backend API**: http://localhost:8000
   - GET `/live` → `{"status": "ok"}`
   - GET `/ready` → `{"status": "ok", "checks": {...}}`

2. **Frontend**: http://localhost:8080
   - Should show login page or demo dashboard

3. **InfluxDB UI**: http://localhost:8086
   - Login with admin credentials

---

## 7. Running the Application

### 7.1 Development Mode

**Start Full Stack (Docker):**
```bash
# From project root
docker compose up -d
cd apps/web && npm run dev
```

**Start Backend Only:**
```bash
cd apps/backend
uvicorn app.main:app --reload --port 8000
```

**Start Frontend Only:**
```bash
cd apps/web
npm run dev
```

### 7.2 Production Build

**Build Frontend:**
```bash
cd apps/web
npm run build
# Output: apps/web/dist/
```

**Build Backend Docker Image:**
```bash
docker build -t nilm-backend:latest apps/backend/
```

### 7.3 Running Tests

**Backend Tests:**
```bash
cd apps/backend
pytest
```

**Frontend Tests:**
```bash
cd apps/web
npm run test
```

**Type Checking:**
```bash
cd apps/web
npm run typecheck
```

---

## 8. Reproducing the Demo

### 8.1 Demo Mode (Quickest)

The simplest way to see the application:

1. Set `VITE_DEMO_MODE=true` in `apps/web/.env`
2. Start the frontend: `npm run dev`
3. Open http://localhost:8080
4. The demo loads with pre-configured sample data

### 8.2 Full Pipeline Demo

To demonstrate the complete data flow:

**Step 1: Start All Services**
```bash
docker compose up -d
cd apps/web && npm run dev
```

**Step 2: Seed Sample Data**
```bash
cd apps/backend
python scripts/seed_from_y_test.py --seconds 7200 --building-id demo
```

This synthesizes 2 hours of aggregate power data and pushes it through the inference pipeline.

**Step 3: View Results**
1. Open http://localhost:8080
2. Log in or use demo mode
3. Select building "demo" from the dropdown
4. View disaggregated appliance data on the dashboard

### 8.3 Using Training Data

If you have access to the NILM training data:

**Load Model-Ready Data:**
```bash
# Ensure data is in: data/processed/15min/model_ready/[appliance]/
# Files: X_train.npy, X_val.npy, X_test.npy, y_train.npy, y_val.npy, y_test.npy

# Run training script
cd training
python train_model.py --appliance heatpump --model transformer
```

### 8.4 Accessing InfluxDB

To view raw data:
1. Open http://localhost:8086
2. Login: admin / admin12345
3. Navigate to Data Explorer
4. Select bucket: `raw_sensor_data` or `predictions`

---

## 9. Troubleshooting

### 9.1 Common Issues

#### Docker Issues

**Error: "Cannot connect to Docker daemon"**
```bash
# Windows: Ensure Docker Desktop is running
# Linux: Start Docker service
sudo systemctl start docker
```

**Error: "Port already in use"**
```bash
# Find process using port (e.g., 8000)
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # macOS/Linux

# Kill the process or use a different port
```

#### Backend Issues

**Error: "INFLUX_TOKEN is required"**
- Ensure `.env.local` exists with `INFLUX_TOKEN` set
- Token must be at least 32 characters

**Error: "Connection refused to InfluxDB"**
- Verify InfluxDB container is running: `docker ps`
- Check correct URL in backend `.env`

**Error: "Module not found"**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### Frontend Issues

**Error: "supabaseKey is required"**
- Set `VITE_DEMO_MODE=true` for demo usage, OR
- Configure valid Supabase credentials

**Error: "CORS blocked"**
- Add frontend URL to `CORS_ORIGINS` in backend `.env`
- Restart backend after changing

**Blank page on load:**
```bash
# Check browser console for errors
# Clear cache and reload
# Verify all environment variables are set
```

### 9.2 Checking Service Health

**Backend Health Endpoints:**
```bash
# Liveness (is it running?)
curl http://localhost:8000/live

# Readiness (are dependencies connected?)
curl http://localhost:8000/ready
```

**Check Docker Logs:**
```bash
# All services
docker compose logs

# Specific service
docker compose logs backend
docker compose logs influxdb
```

### 9.3 Reset Everything

If you need a clean start:

```bash
# Stop all containers and remove volumes
docker compose down -v

# Remove node_modules
rm -rf apps/web/node_modules

# Remove Python venv
rm -rf apps/backend/venv

# Reinstall everything
docker compose up -d
cd apps/web && npm install
cd apps/backend && python -m venv venv && pip install -r requirements.txt
```

### 9.4 Getting Help

If issues persist:
1. Check the [GitHub Issues](https://github.com/Thetommigun432/MTS3-MCTE-Team-Project-Energy-G1/issues)
2. Review application logs for specific error messages
3. Contact the development team

---

*Document Version: 1.0 | Last Updated: January 2026*
