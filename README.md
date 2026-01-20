# NILM Energy Monitor

Real-time Non-Intrusive Load Monitoring (NILM) web application with deep learning-based energy disaggregation.

## Project Structure (Monorepo)

```
MTS3-MCTE-Team-Project-Energy-G1/
├── apps/
│   ├── web/                    # React frontend (Vite + TypeScript)
│   ├── local-server/           # Express API server (InfluxDB proxy)
│   └── inference-service/      # FastAPI model inference (PyTorch)
├── infra/
│   └── influxdb/               # Docker Compose for InfluxDB
├── scripts/                    # Data seeding and utilities
├── docs/                       # Documentation
├── preprocessing/              # Python data preparation scripts
├── data/                       # Datasets (gitignored, download separately)
└── models/                     # Trained PyTorch models (gitignored)
```

## Quick Start

### Prerequisites

- Node.js 18+
- Docker and Docker Compose
- Python 3.10+ (for inference service)

### 1. Clone and Configure

```bash
git clone <repository-url>
cd MTS3-MCTE-Team-Project-Energy-G1

# Copy environment files
cp .env.local.example .env.local
cp apps/web/.env.example apps/web/.env
```

Edit `.env.local` with your InfluxDB token (generate with `openssl rand -hex 32`).

### 2. Start Services

```bash
# Start InfluxDB and inference service
docker compose up -d

# Install frontend dependencies
cd apps/web && npm install

# Seed prediction data (from apps/web directory)
npm run predictions:seed

# Start development servers
npm run local:dev
```

### 3. Access the Application

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:8080 |
| InfluxDB UI | http://localhost:8086 |
| API Health | http://localhost:3001/health |
| Inference Health | http://localhost:8000/health |

## Operating Modes

The frontend supports three data modes:

| Mode | Description | Backend Required |
|------|-------------|------------------|
| **Demo** | Sample CSV data bundled with app | None |
| **Local** | InfluxDB + inference service | Docker Compose |
| **API** | Supabase cloud backend | Supabase project |

Configure mode in `apps/web/.env`:
```env
VITE_DEMO_MODE=true      # Demo mode
VITE_LOCAL_MODE=true     # Local InfluxDB mode
```

## Available Scripts

| Script | Description |
|--------|-------------|
| `npm run dev` | Start Vite dev server |
| `npm run build` | Production build |
| `npm run preview` | Preview production build |
| `npm run lint` | Run ESLint |
| `npm run typecheck` | TypeScript type checking |
| `npm run local:server` | Start InfluxDB proxy |
| `npm run local:dev` | Run frontend + local server |
| `npm run predictions:seed` | Seed InfluxDB with sample data |

## Development

### Frontend (apps/web)

```bash
cd apps/web
npm install
npm run dev           # Start Vite dev server
npm run build         # Production build
npm run lint          # ESLint
npm run typecheck     # TypeScript check
```

### Local Server (apps/local-server)

```bash
cd apps/local-server
npm install
npm run dev           # Start with ts-node
```

### Inference Service (apps/inference-service)

```bash
cd apps/inference-service
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Tech Stack

### Frontend
- React 19 + TypeScript
- Vite 7
- Tailwind CSS + shadcn/ui
- Recharts for visualizations
- React Router 6

### Backend
- Express.js (API proxy)
- FastAPI + PyTorch (inference)
- InfluxDB 2.7 (time series)
- Supabase (auth + cloud storage)

## ML Pipeline (NILM Model Training)

### Model Architecture
- **CNN Seq2Seq**: Encoder-decoder for sequence-to-sequence prediction
- **U-Net 1D**: Skip connections for preserving temporal features
- **Input**: Multi-timestep aggregate power sequences
- **Output**: Per-appliance power predictions

### Training
```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
python train_model.py --appliance heatpump --model cnn
```

## Documentation

- [Local Development Guide](docs/LOCAL_DEVELOPMENT.md) - Full local setup
- [InfluxDB Schema](docs/INFLUX_SCHEMA.md) - Data model
- [Supabase Setup](docs/SUPABASE_SETUP.md) - Cloud backend
- [Cloudflare Deployment](apps/web/docs/DEPLOY_CLOUDFLARE_PAGES.md) - Production

## Security Notes

- **Never commit** `.env`, `.env.local`, or `.env.production` files
- Use `.env.example` as a template
- InfluxDB tokens should be at least 32 characters
- For production, use secrets managers (GitHub Secrets, etc.)

## License

MIT
