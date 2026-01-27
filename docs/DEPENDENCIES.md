# Dependencies Audit

This document summarizes key dependencies and where they are defined.

## Backend (Python)

**Source of truth**: `apps/backend/pyproject.toml`

### Core Framework
- **fastapi**: `0.128.0`
- **uvicorn[standard]**: `0.40.0`
- **pydantic**: `2.12.5`

### Data & ML
- **influxdb-client[async]**: `1.50.0`
- **numpy**: `>=1.26.0,<2.0.0`
- **torch**: `>=2.2.0`

### Integrations
- **supabase**: `2.27.2`
- **redis[hiredis]**: `7.1.0`
- **pyarrow**: `>=14.0.0`
- **pandas**: `>=2.1.0`

## Frontend (Node)

**Source of truth**: `apps/web/package.json`

### Core
- **react**, **react-dom**: `^19.x`
- **vite**: `^7.x`
- **typescript**: `^5.x`

### UI & Visualization
- **tailwindcss**: `^4.x`
- **@radix-ui/**: UI primitives
- **recharts**: Charts
- **lucide-react**: Icons

### Data & Auth
- **@supabase/supabase-js**: Supabase client
- **react-hook-form**: Forms
- **date-fns**: Date utilities

## Notes

- The backend uses `pyproject.toml` and does not rely on `requirements.txt`.
- Keep versions aligned with the repository’s lockfiles and Dockerfiles.
