# Deployment

This is the single source of truth for production deployment.

## Architecture

- **Backend API**: FastAPI on Railway (public).
- **Worker**: Redis stream consumer on Railway (private).
- **Redis**: Railway plugin or private service.
- **InfluxDB**: Railway service or external managed instance.
- **Frontend**: Vite SPA on Cloudflare Pages.

## Production Base URLs

| Service | Base URL | Notes |
|---|---|---|
| Frontend | `https://<cloudflare-pages-domain>` | Cloudflare Pages project domain. |
| API | `https://<railway-api-domain>` | Public Railway service domain. |
| InfluxDB (optional) | `https://<influx-domain>` | Only if you expose the UI. |

## Railway Services

### Service config paths

Railway config is defined via TOML files in the repo:

| Service | Config file path | Networking |
|--------|-------------------|-----------|
| API | `apps/backend/railway.api.toml` | Public |
| Worker | `apps/backend/railway.worker.toml` | Private |

Set these in **Railway → Service Settings → Source → Config File Path**.

### Datastores

- **Redis**: Railway Redis plugin (recommended).
- **InfluxDB**: Railway service (influxdb:2.8) or external managed InfluxDB.

For Railway private networking, use internal URLs such as:
- `REDIS_URL=redis://redis.railway.internal:6379`
- `INFLUX_URL=http://influxdb.railway.internal:8086`

## Environment Variable Matrix

Use Railway **Shared Environment** for common values and Service Overrides for API/Worker-specific values.

| Variable | Service(s) | Required | Where to set | Notes |
|---|---|---|---|---|
| `ENV` | API + Worker | Yes | Railway shared | Use `prod` in production. |
| `PORT` | API | Yes | Railway injected | Do not set manually. |
| `HOST` | API | No | Railway shared | Default `0.0.0.0`. |
| `CORS_ORIGINS` | API | Yes | Railway shared | Comma-separated, no spaces. |
| `INFLUX_URL` | API + Worker | Yes | Railway shared | Use private URL in Railway. |
| `INFLUX_TOKEN` | API + Worker | Yes | Railway shared (secret) | Influx admin or RW token. |
| `INFLUX_ORG` | API + Worker | Yes | Railway shared | Default `energy-monitor`. |
| `INFLUX_BUCKET_PRED` | API + Worker | Yes | Railway shared | Predictions bucket. |
| `REDIS_URL` | API + Worker | Yes | Railway shared | Redis connection string. |
| `REDIS_STREAM_KEY` | API + Worker | No | Railway shared | Default `nilm:readings`. |
| `REDIS_CONSUMER_GROUP` | Worker | No | Worker override | Default `nilm-infer`. |
| `PIPELINE_ENQUEUE_ENABLED` | API | Yes | API override | Must be `true` to enqueue. |
| `PIPELINE_WORKER_IN_API_ENABLED` | API | No | API override | Keep `false` when using a separate Worker. |
| `PIPELINE_ROLLING_WINDOW_SIZE` | API + Worker | No | Railway shared | Default `3600`. |
| `MODEL_ARTIFACT_BASE_URL` | API + Worker | Yes | Railway shared | Required for production models. See [RAILWAY_MODELS.md](./RAILWAY_MODELS.md). |
| `SUPABASE_URL` | API | Yes | Railway shared | Supabase project URL. |
| `SUPABASE_PUBLISHABLE_KEY` | API | Yes | Railway shared | Preferred over anon. |
| `SUPABASE_ANON_KEY` | API | Optional | Railway shared | Legacy fallback if publishable key is not set. |
| `SUPABASE_JWT_SECRET` | API | Optional | Railway shared (secret) | HS256 legacy auth. |
| `SUPABASE_JWKS_URL` | API | Optional | Railway shared | Derived automatically if omitted. |
| `AUTH_VERIFY_AUD` | API | Yes | Railway shared | Must be `true` in prod. |
| `ADMIN_TOKEN` | API | Recommended | Railway shared (secret) | Protects `/admin/*` endpoints. |
| `INGEST_TOKEN` | API | Optional | Railway shared (secret) | Server-to-server ingest auth. |

### Frontend (Cloudflare Pages)

Set in **Cloudflare Pages → Settings → Environment Variables**:

| Variable | Required | Notes |
|---|---|---|
| `VITE_BACKEND_URL` | Yes | Use the public Railway API domain. |
| `VITE_SUPABASE_URL` | Yes | Supabase project URL. |
| `VITE_SUPABASE_PUBLISHABLE_KEY` | Yes | Preferred key. |
| `VITE_SUPABASE_ANON_KEY` | Optional | Legacy alias. |
| `VITE_DEMO_MODE` | Optional | `false` for production. |

### Supabase key precedence

- Backend prefers `SUPABASE_PUBLISHABLE_KEY`, falling back to `SUPABASE_ANON_KEY`.
- Frontend prefers `VITE_SUPABASE_PUBLISHABLE_KEY`, falling back to `VITE_SUPABASE_ANON_KEY`.

## Cloudflare Pages build settings

| Setting | Value |
|---|---|
| Framework preset | Vite |
| Build command | `npm run build:web` |
| Output directory | `apps/web/dist` |

## Verification (Production)

### API health
```bash
curl https://<api-domain>/live
curl https://<api-domain>/ready
```

### Frontend network checks
- Requests should go to `VITE_BACKEND_URL/api/*`.
- Health checks use `/live` and `/ready` without `/api`.

### Pipeline checks
1. Send a reading to `POST /ingest/readings`.
2. Verify Redis stream length increases.
3. Verify InfluxDB has new points in the `predictions` bucket.

## Notes

- Local dev instructions live in [LOCAL_DEV.md](./LOCAL_DEV.md).
- Archived deployment docs are in [archives/deprecated-deploy](./archives/deprecated-deploy).
