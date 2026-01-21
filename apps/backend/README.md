# NILM Backend

Production-grade FastAPI backend for the Energy Monitor.

**Full Documentation**:
- [Backend Documentation](../../docs/backend.md)
- [Railway Deployment Guide](../../docs/deployment/railway.md)

## Railway Deployment Quick Reference

This service uses **Config as Code** (`railway.json` at repo root).

### Digital Tweak: Root Directory
**NOTE**: This project uses **Root Build Context** (Strategy B).
The `railway.json` forces Railway to use `apps/backend/Dockerfile`, but builds from the repository root.
- **You do NOT need to set Root Directory in UI** (leave as `/`).
- **Local Build**: Run from repo root: `docker build -f apps/backend/Dockerfile -t backend .`

### Health Checks
- **Path**: `/live`
- **Success**: 200 OK
- **Timeout**: 300s (recommended)

### Environment Variables
See `.env.example`.
- `ENV=prod`
- `CORS_ORIGINS` (No wildcards)
- `INFLUX_URL` (Must be external/hosted)
