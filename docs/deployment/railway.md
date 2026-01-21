# Railway Deployment Guide (Backend)

This guide details how to deploy the FastAPI backend to Railway.

## 1. Prerequisites
- **Railway Account**: [Sign up here](https://railway.app)
- **GitHub Repository**: Connected to Railway
- **InfluxDB**: External hosted InfluxDB (e.g. InfluxData Cloud or self-hosted VPS)
- **Supabase**: Cloud project for Authentication

---

## 2. Configuration

### 2.1 Config as Code (railway.json)
This repository uses a `railway.json` file at the root to define build policies (watch patterns, restart policy).
- **Restart Policy**: On Failure
- **Watch Patterns**: Only triggers deploy if `apps/backend/**` or `railway.json` changes.

### 2.2 Project Settings (UI)
- **Root Directory**: `apps/backend` (CRITICAL: Must set this in UI)
- **Build Command**: *Default (leave empty)*
- **Start Command**: *Default (leave empty)*

### 2.2 Environment Variables
Set these variables in the Railway "Variables" tab for your service.

| Variable | Required | Description | Example |
| :--- | :--- | :--- | :--- |
| `ENV` | **Yes** | Environment mode | `prod` |
| `PORT` | **Yes** | Injected by Railway (do not set manually) | `8000` |
| `CORS_ORIGINS` | **Yes** | Allowed frontend origins (no trailing slash) | `https://myapp.pages.dev` |
| `INFLUX_URL` | **Yes** | External InfluxDB URL (HTTPS) | `https://us-east-1-1.aws.cloud2.influxdata.com` |
| `INFLUX_TOKEN` | **Yes** | Admin/Write token | `MySecureToken...` |
| `INFLUX_ORG` | **Yes** | Organization Name | `energy-monitor` |
| `INFLUX_BUCKET_RAW` | **Yes** | Raw Data Bucket | `raw_sensor_data` |
| `INFLUX_BUCKET_PRED` | **Yes** | Predictions Bucket | `predictions` |
| `SUPABASE_URL` | **Yes** | Project URL | `https://xyz.supabase.co` |
| `SUPABASE_PUBLISHABLE_KEY`| **Yes** | Anon/Public Key | `eyJ...` |
| `AUTH_VERIFY_AUD` | **Yes** | Verify JWT Audience | `true` |
| `ADMIN_TOKEN` | **Yes** | Secret token for admin endpoints | `KeepThisSecret` |

> [!CAUTION]
> Do NOT use `*` or `localhost` in `CORS_ORIGINS` for production. The backend will refuse to start.

---

## 3. Deployment Steps

1. **New Project**: In Railway, choose "Deploy from GitHub repo".
2. **Select Repo**: Choose `MTS3-MCTE-Team-Project-Energy-G1`.
3. **Configure**: Go to `Settings` -> `Root Directory` and set to `apps/backend`.
4. **Variables**: Go to `Variables` and add all required env vars from above.
5. **Deploy**: The deployment should trigger automatically. Check `Deployments` tab.

---

## 4. Health Checks

Railway automatically detects the port. You can verify health:

- **Liveness**: `https://<your-app>.up.railway.app/live` -> `200 OK`
- **Readiness**: `https://<your-app>.up.railway.app/ready` -> `200 OK` (if InfluxDB connects)

If `/ready` returns `503`, check the "Deploy Logs" for specific errors (e.g., `InfluxDB connection failed` or `Configuration validation failed`).

---

## 5. Connecting Frontend
Once deployed, copy the **Railway Public URL** (e.g. `https://nilm-backend.up.railway.app`).

Update your Cloudflare Pages frontend configuration:
- Set `VITE_API_BASE_URL` to this Railway URL.
