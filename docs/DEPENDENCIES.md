# Dependencies Audit

This document serves as the canonical list of approved dependencies for the NILM Energy Monitor project.

## 1. Backend (Python)
**File**: `apps/backend/requirements.txt`
**Constraint**: `apps/backend/constraints.txt` (for PyTorch CPU)

### Core Framework
- **fastapi**: Web framework. Pinned to `0.115.6`.
- **uvicorn[standard]**: ASGI server. Pinned to `0.34.0`.
- **pydantic**: Validation. Pinned to `2.10.4`.

### Data & ML
- **influxdb-client[async]**: Time-series DB client. Pinned to `1.47.0`.
- **numpy**: Pinned to `<2.0.0` for compatibility.
- **torch**: PyTorch (CPU only). Pinned to `2.5.1` in constraints.

### Utils
- **supabase**: Database/Auth client. Pinned to `2.11.0`.
- **python-json-logger**: Structured logging.

## 2. Frontend (Node.js)
**File**: `apps/web/package.json`

### Core
- **react**, **react-dom**: v19.
- **vite**: v7.
- **typescript**: v5.x.

### UI Components
- **tailwindcss**: Styling.
- **lucide-react**: Icons.
- **recharts**: Data visualization.
- **@radix-ui/**: Accessible primitives.

### State & Logic
- **@supabase/supabase-js**: Auth & DB.
- **date-fns**: Date manipulation.
- **react-hook-form**: Form handling.
