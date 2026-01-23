# Operations Guide

## 1. Local Development

### Prerequisites
- Node.js 22 (LTS)
- Docker & Docker Compose
- Python 3.12+ (optional, for non-Docker backend)

### Quick Start (Full Stack)
The easiest way to run the full stack locally is with Docker Compose.

1.  **Environment Setup**:
    ```bash
    cp .env.local.example .env.local
    # Set INFLUX_TOKEN (min 32 chars)
    
    cp apps/web/.env.example apps/web/.env
    # Set VITE_DEMO_MODE=true for quick testing
    ```

2.  **Start Backend & DB**:
    ```bash
    docker compose up -d
    ```

3.  **Start Frontend**:
    ```bash
    cd apps/web
    npm install
    npm run dev
    ```

4.  **Access**:
    - Frontend: `http://localhost:8080`
    - Backend API: `http://localhost:8000`
    - InfluxDB: `http://localhost:8086`

## 2. Deployment

### Backend (Railway)
- **Repo Root**: Set Railway root directory to `apps/backend`.
- **Dockerfile**: Automatically detected.
- **Port**: Auto-injected (`PORT`).
- **Health Check Path**: `/live`.
- **Environment Variables**:
    - `INFLUX_URL` (External InfluxDB cloud or hosted instance)
    - `INFLUX_TOKEN`, `INFLUX_ORG`, `INFLUX_BUCKET_*`
    - `SUPABASE_URL`, `SUPABASE_ANON_KEY`
    - `CORS_ORIGINS` (Point to your Cloudflare URL)

### Frontend (Cloudflare Pages)
- **Source**: Connect GitHub repo.
- **Build Settings**:
    - **Framework**: Vite
    - **Build command**: `npm ci && npm run build`
    - **Build output directory**: `apps/web/dist`
    - **Node Version**: `22` (Set `NODE_VERSION` env var or use `.nvmrc`)
- **Environment Variables**:
    - `VITE_BACKEND_URL`: Your Railway URL (e.g., `https://backend-production.up.railway.app`)
    - `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`

## 3. Configuration & Secrets

### Environment Variables
| Component | File | Key Variables |
|-----------|------|---------------|
| **Root** | `.env.local` | `INFLUX_TOKEN` (Shared secret) |
| **Backend** | `apps/backend/.env` | `INFLUX_*`, `SUPABASE_*`, `CORS_ORIGINS` |
| **Frontend** | `apps/web/.env` | `VITE_BACKEND_URL`, `VITE_SUPABASE_*` |

**Security Note**: Never commit `.env` files. Use secrets management in production (Railway Variables / Cloudflare Pages Variables).

## 4. Troubleshooting

### Build Failures
- **Lockfile mismatch**: Ensure you are using the root `package-lock.json`. run `npm ci` in root first.
- **Node Version**: Check that specific environments use Node 22.

### Connectivity
- **CORS Error**: Check `CORS_ORIGINS` in Railway matches your Cloudflare URL exactly (no trailing slash).
- **Influx Connection Refused**: Ensure `INFLUX_URL` is reachable from the backend container. In Docker Compose, use `http://influxdb:8086`.

### Auth Issues
- **"supabaseKey is required"**: In local dev, set `VITE_DEMO_MODE=true` to bypass Supabase, or provide valid keys.
- **Backend 401**: Ensure the JWT sent by frontend matches the Supabase project configured in Backend.
