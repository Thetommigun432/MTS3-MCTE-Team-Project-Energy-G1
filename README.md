# NILM Energy Monitor

Real-time Non-Intrusive Load Monitoring (NILM) web application with deep learning-based energy disaggregation.

## Backend Source of Truth

> **Canonical Backend**: `apps/backend` (FastAPI Python 3.12)

| Service | Path | Status |
|---------|------|--------|
| **Backend API** | `apps/backend` | ✅ Active - FastAPI monolith |
| Web Frontend | `apps/web` | ✅ Active - React/Vite |

## Project Structure (Monorepo)

```
MTS3-MCTE-Team-Project-Energy-G1/
├── apps/
│   ├── backend/                # ✅ FastAPI backend (Python 3.12)
│   └── web/                    # ✅ React frontend (Vite + TypeScript)
├── infra/
│   └── influxdb/               # InfluxDB standalone setup
├── scripts/                    # Data seeding and utilities
├── docs/                       # Documentation
├── preprocessing/              # Python data preparation scripts
├── data/                       # Datasets (gitignored)
└── models/                     # Trained models (gitignored)
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Node.js 18+ (for frontend development)

### 1. Clone and Configure

```bash
git clone <repository-url>
cd MTS3-MCTE-Team-Project-Energy-G1

# Copy environment files
cp .env.local.example .env.local
cp apps/web/.env.example apps/web/.env
```

Edit `.env.local` with your credentials:
- `INFLUX_TOKEN` - Generate with `openssl rand -hex 32`
- `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_JWT_SECRET` - From Supabase dashboard

### 2. Start Services

```bash
# Start backend + InfluxDB
docker compose up -d

# Verify backend is running
curl http://localhost:8000/live

# Install and start frontend
cd apps/web && npm install && npm run dev
```

### 3. Access the Application

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| Backend Health | http://localhost:8000/live |
| InfluxDB UI | http://localhost:8086 |

## Backend API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/live` | GET | Liveness probe |
| `/ready` | GET | Readiness probe |
| `/infer` | POST | Run inference + persist |
| `/models` | GET | List available models |
| `/analytics/readings` | GET | Query sensor readings |
| `/analytics/predictions` | GET | Query predictions |
| `/metrics` | GET | Prometheus metrics |

See `apps/backend/README.md` for full API documentation.

## Development

### Backend (apps/backend)

```bash
cd apps/backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend (apps/web)

```bash
cd apps/web
npm install
npm run dev
```

## Operating Modes

| Mode | Description | Backend Required |
|------|-------------|------------------|
| **Demo** | Sample CSV data bundled with app | None |
| **Local** | InfluxDB + backend | Docker Compose |
| **API** | Supabase cloud backend | Supabase project |

## Available Scripts

| Script | Description |
|--------|-------------|
| `npm run dev` | Start Vite dev server |
| `npm run build` | Production build |
| `npm run local:dev` | Run frontend + legacy local server |
| `npm run predictions:seed` | Seed InfluxDB with sample data |

## Tech Stack

### Frontend
- React 19 + TypeScript
- Vite 7
- Tailwind CSS + shadcn/ui
- Recharts for visualizations

### Backend
- FastAPI (Python 3.12)
- PyTorch + safetensors (inference)
- InfluxDB 2.8 (time series)
- Supabase (auth + metadata)

## ML Pipeline (NILM Model Training)

### Model Architecture
- **CNNTransformer**: CNN feature extractor + Transformer encoder
- **CNN Seq2Seq**: Encoder-decoder for sequence-to-sequence prediction
- **U-Net 1D**: Skip connections for preserving temporal features

### Training
```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
python train_model.py --appliance heatpump --model cnn
```

## Documentation

- [**Documentation Index**](docs/README.md) - **Start Here**
- [Backend Documentation](docs/backend.md)
- [Frontend Documentation](docs/frontend.md)
- [Local Development Guide](docs/getting-started.md)
- [Railway Deployment](docs/deployment/railway.md)
- [Cloudflare Deployment](docs/deployment/cloudflare.md)

## Security Notes

- **Never commit** `.env`, `.env.local`, or `.env.production` files
- Use `.env.example` as a template
- InfluxDB tokens should be at least 32 characters
- For production, use secrets managers (GitHub Secrets, etc.)

## License

MIT
