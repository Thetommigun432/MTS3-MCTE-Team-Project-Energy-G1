# NILM Backend

Production-grade FastAPI backend for the Energy Monitor.

**Full Documentation**:
- [Backend Documentation](../../docs/backend.md)
- [Railway Deployment Guide](../../docs/deployment/railway.md)

## Railway Deployment Quick Reference

This service uses **Config as Code** (`railway.json` at repo root).

### Critical UI Settings
You must configure these in the Railway Service **Settings**:

1.  **Root Directory**: `/apps/backend`
    - *Why*: Scopes the build context to this folder.
2.  **Builder**: Dockerfile
    - *Note*: Railway will automatically find the `Dockerfile` in `/apps/backend` because of the Root Directory setting.

### Health Checks
- **Path**: `/live`
- **Success**: 200 OK
- **Timeout**: 300s (recommended)

### Environment Variables
See `.env.example`.
- `ENV=prod`
- `CORS_ORIGINS` (No wildcards)
- `INFLUX_URL` (Must be external/hosted)
