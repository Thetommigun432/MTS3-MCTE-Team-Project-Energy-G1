# NILM Energy Monitor

Real-time Non-Intrusive Load Monitoring (NILM) web application with deep learning-based energy disaggregation.

## 🚀 Quick Start

### Prerequisites
- **Docker Desktop** with Docker Compose v2
- **Node.js 22+** (for frontend development)

### Setup

`ash
# 1. Clone repository
git clone https://github.com/Thetommigun432/MTS3-MCTE-Team-Project-Energy-G1.git
cd MTS3-MCTE-Team-Project-Energy-G1

# 2. Environment setup
cp .env.local.example .env.local

# 3. Start backend stack
docker compose up -d --build

# 4. Start frontend
npm install
npm run dev:web
`

### Two Modes Available

| Mode | Command | Data Source |
|------|---------|-------------|
| **Simulator** | `docker compose up -d --build` | Local simulated data |
| **MQTT Realtime** | `docker compose -f compose.realtime.yaml up -d --build` | Howest Energy Lab live data |

### Access

| URL | Description |
|-----|-------------|
| http://localhost:8080/live | **Public Dashboard** (no login required) |
| http://localhost:8080/app | Full app (requires login) |
| http://localhost:8000/api/docs | API Documentation |
| http://localhost:8086 | InfluxDB UI |

## 📚 Documentation

- [**Installation Guide**](INSTALLATION_GUIDE.md) - Complete setup instructions
- [**Local Development**](docs/LOCAL_DEV.md) - Docker Compose + npm setup
- [**API Reference**](docs/API.md) - Backend API endpoints
- [**Deployment**](docs/DEPLOYMENT.md) - Railway + Cloudflare Pages

## 🏗️ Architecture

`
MQTT/Simulator → Backend API → Redis → Worker (NILM) → InfluxDB
                                                            ↓
                                                      Frontend
`

**Services:**
- **Backend**: FastAPI REST API (port 8000)
- **Worker**: PyTorch NILM inference (10 appliance models)
- **InfluxDB**: Time-series predictions storage
- **Redis**: Rolling window cache + stream queue

## 🔧 Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 19, Vite 7, TypeScript, TailwindCSS |
| Backend | FastAPI, Python 3.12, PyTorch 2.5 |
| Database | InfluxDB 2.8, Redis 7 |
| ML Models | TCN-SA (Temporal Convolutional Network with Self-Attention) |

## 📊 NILM Models

10 appliance-specific TCN-SA models for disaggregation:

| Appliance | Confidence |
|-----------|------------|
| HeatPump | ~70% |
| Stove | ~95% |
| Dishwasher | ~57% |
| Dryer | ~68% |
| Oven | ~95% |
| EVCharger | ~95% |
| EVSocket | ~83% |
| WashingMachine | ~92% |
| RangeHood | ~92% |
| RainwaterPump | ~93% |

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.
