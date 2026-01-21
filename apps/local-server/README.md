# Local Server (DEPRECATED)

> ⚠️ **DEPRECATED**: This service has been superseded by the unified backend at `apps/backend`.

## Status

This folder is preserved for reference during migration. **Do not build or deploy this service.**

## Canonical Backend

The current production API server is located at:
- **Path**: `apps/backend`
- **Technology**: FastAPI (Python 3.12)
- **Port**: 8000
- **Analytics Endpoints**: `GET /analytics/readings`, `GET /analytics/predictions`

## Why Deprecated?

The unified backend at `apps/backend` includes:
- All InfluxDB query functionality
- Proper Flux query templating (no injection)
- JWT authentication
- Authorization with building/appliance access control

See `apps/backend/README.md` for full documentation.
