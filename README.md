# NILM Energy Monitor

Real-time Non-Intrusive Load Monitoring (NILM) web application with deep learning-based energy disaggregation.

## 🚀 Quick Start (Zero Configuration)

### Prerequisites
- **Docker Desktop** with Docker Compose v2
- **Node.js 20+**

### 3 Commands to Run

```bash
# Clone and start
git clone https://github.com/Thetommigun432/MTS3-MCTE-Team-Project-Energy-G1.git
cd MTS3-MCTE-Team-Project-Energy-G1

# Start backend (InfluxDB, Redis, API, Worker, Simulator)
docker compose up -d

# Start frontend
cd apps/web && npm install && npm run dev
```

**Open:** http://localhost:8080/live ← **No login required!**

> ⚠️ **Note:** First inference takes ~40 seconds (buffer filling). After that, predictions update every second.

---

## Two Operating Modes

| Mode | Command | Data Source |
|------|---------|-------------|
| **Simulator** | ``docker compose up -d`` | Replays historical parquet data |
| **MQTT Realtime** | ``docker compose -f compose.realtime.yaml up -d`` | Live data from Howest Energy Lab |

---

## Access Points

| URL | Description |
|-----|-------------|
| http://localhost:8080/live | **Public Dashboard** (no auth) |
| http://localhost:8080 | Full app (login required) |
| http://localhost:8000/health | API Health Check |
| http://localhost:8000/docs | API Documentation (Swagger) |
| http://localhost:8086 | InfluxDB UI (admin/admin12345) |

---

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Simulator  │     │    MQTT     │     │  Frontend   │
│  (Parquet)  │     │  Ingestor   │     │   (React)   │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────┐
│                   Backend API                        │
│                   (FastAPI)                          │
└──────────────────────┬──────────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│    Redis    │ │   Worker    │ │  InfluxDB   │
│   (Cache)   │ │  (PyTorch)  │ │  (Storage)  │
└─────────────┘ └─────────────┘ └─────────────┘
```

**Data Flow:** Simulator/MQTT → Backend → Redis (window) → Worker (10 NILM models) → InfluxDB → Frontend

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 19, Vite 7, TypeScript, TailwindCSS, Recharts |
| Backend | FastAPI, Python 3.12, PyTorch 2.5 |
| Database | InfluxDB 2.8 (time-series), Redis 7 (cache) |
| ML Models | TCN-SA (Temporal Convolutional Network + Self-Attention) |

---

## 📊 NILM Models (10 Appliances)

| Appliance | Max Power | Confidence |
|-----------|-----------|------------|
| HeatPump | 3.0 kW | ~70% |
| Stove | 2.5 kW | ~95% |
| Oven | 3.5 kW | ~95% |
| Dishwasher | 2.0 kW | ~57% |
| WashingMachine | 2.5 kW | ~92% |
| Dryer | 3.0 kW | ~68% |
| EVCharger | 11.0 kW | ~95% |
| EVSocket | 3.7 kW | ~83% |
| RangeHood | 0.3 kW | ~92% |
| RainwaterPump | 1.5 kW | ~93% |

---

## 📁 Project Structure

```
├── apps/
│   ├── backend/          # FastAPI + PyTorch inference
│   └── web/              # React frontend
├── checkpoints/          # Trained model weights (.pt files)
├── docs/                 # Documentation
├── compose.yaml          # Simulator mode
└── compose.realtime.yaml # MQTT realtime mode
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) | Complete setup instructions |
| [docs/LOCAL_DEV.md](docs/LOCAL_DEV.md) | Local development guide |
| [docs/API.md](docs/API.md) | Backend API reference |

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 👥 Team

Howest MCT - Team Project 2025-2026
