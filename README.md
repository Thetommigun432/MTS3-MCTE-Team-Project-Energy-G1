# NILM Energy Monitor

Real-time Non-Intrusive Load Monitoring (NILM) web application with deep learning-based energy disaggregation.

> **Status**: Active Development
> **Canonical Branch**: `backend` (merged with `frontend`)

## 📚 Documentation

The full documentation is located in the `docs/` directory:

- [**Start Here**](docs/PROJECT.md) - Project Goals, Architecture, and Repo Structure.
- [**Operations Guide**](docs/OPERATIONS.md) - Setup, Local Development, Deployment, and Troubleshooting.
- [**API Reference**](docs/API.md) - Backend API Endpoints and Authentication.

## 🚀 Quick Start (Local)

1.  **Clone & Setup**:
    ```bash
    git clone https://github.com/Thetommigun432/MTS3-MCTE-Team-Project-Energy-G1.git repo
    cd repo
    ```

2.  **Run with Docker Compose**:
    ```bash
    # Set shared secret
    cp .env.local.example .env.local
    
    # Start Backend + InfluxDB
    docker compose up -d
    ```

3.  **Start Frontend**:
    ```bash
    cd apps/web
    cp .env.example .env
    npm install && npm run dev
    ```

4.  **Visit**:
    - Frontend: `http://localhost:8080`
    - Backend: `http://localhost:8000`

## Tech Stack
- **Frontend**: React 19, Vite 7, TypeScript
- **Backend**: FastAPI, Python 3.12, PyTorch
- **Data**: InfluxDB 2.8, Supabase
