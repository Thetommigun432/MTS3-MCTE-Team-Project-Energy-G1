# Deployment Guide

## Railway Backend Configuration

### Required Environment Variables

| Variable | Required | Example |
|----------|----------|---------|
| `ENV` | ✅ | `prod` |
| `PORT` | Auto | Injected by Railway |
| `INFLUX_URL` | ✅ | `http://influxdb.railway.internal:8086` |
| `INFLUX_TOKEN` | ✅ | Set in Railway secrets |
| `INFLUX_ORG` | ✅ | `energy-monitor` |
| `INFLUX_BUCKET_RAW` | ✅ | `raw_sensor_data` |
| `INFLUX_BUCKET_PRED` | ✅ | `predictions` |
| `SUPABASE_URL` | ✅ | `https://your-project.supabase.co` |
| `SUPABASE_ANON_KEY` | ✅ | Set in Railway secrets |
| `CORS_ORIGINS` | ✅ | `https://your-app.pages.dev,https://custom.domain.com` |

### CORS Configuration

**Critical:** Set `CORS_ORIGINS` to include all frontend domains:
```
CORS_ORIGINS=https://your-app.pages.dev,https://custom-domain.com
```

Do NOT use `*` wildcard in production.

### Health Check Verification

```bash
# Liveness (process running)
curl https://your-backend.up.railway.app/live

# Readiness (dependencies healthy)
curl https://your-backend.up.railway.app/ready
```

### railway.json

The project uses this configuration:
- **healthcheckPath**: `/live`
- **dockerfilePath**: `apps/backend/Dockerfile`

---

## Cloudflare Pages Frontend Configuration

### Required Environment Variables

| Variable | Required | Example |
|----------|----------|---------|
| `VITE_BACKEND_URL` | ✅ | `https://your-backend.up.railway.app` or `energy-monitor.up.railway.app` |
| `VITE_SUPABASE_URL` | ✅ | `https://your-project.supabase.co` |
| `VITE_SUPABASE_ANON_KEY` | ✅ | Supabase anon key |

**Note:** `VITE_BACKEND_URL` auto-normalizes bare hostnames to `https://`.

---

## Troubleshooting

### "API unreachable" in frontend
1. Check `VITE_BACKEND_URL` is set correctly
2. Verify backend `/live` endpoint responds
3. Ensure `CORS_ORIGINS` includes your frontend domain

### CORS errors
- Backend must have frontend origin in `CORS_ORIGINS`
- Check browser console for blocked preflight requests

### 503 on /ready
- Check `/ready` response for which dependency failed
- Common issues: InfluxDB not reachable, missing buckets
