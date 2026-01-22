# Frontend-Backend Integration Audit

**Date:** 2026-01-22
**Branch:** frontend (synced with backend)
**Status:** Mostly Resolved - Remaining Items Documented

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current Architecture](#current-architecture)
3. [API Call Inventory](#api-call-inventory)
4. [Environment Variables Map](#environment-variables-map)
5. [Authentication Flow Analysis](#authentication-flow-analysis)
6. [Gap Analysis](#gap-analysis)
7. [Legacy References](#legacy-references)
8. [Integration Tasks](#integration-tasks)
9. [Appendix: Technical Details](#appendix-technical-details)

---

## Executive Summary

This audit documents the integration status between the React/Vite frontend (`apps/web`) and FastAPI backend (`apps/backend`). The architecture is fundamentally sound with proper JWT authentication, CORS configuration, and a unified backend. However, several API contract mismatches and legacy references need resolution.

**Key Findings:**

✅ **Strengths:**
- Clean architecture: single unified backend at `localhost:8000`
- Vite proxy correctly configured for local dev
- Comprehensive JWT verification (JWKS + HS256 fallback)
- Generic API client with automatic Bearer token injection
- Request ID middleware on backend

✅ **Resolved (as of 2026-01-22):**
- API endpoints aligned: frontend now calls `/analytics/*` correctly
- Error handling updated with code, request_id extraction
- Request ID generation in frontend API client
- Query parameter validation in energy.ts

⚠️ **Remaining Issues:**
- Frontend test failure: `useNilmCsvData.test.ts` fails due to Supabase client init during import
- Some legacy references in documentation
- `getBuildings()` and `getAppliances()` in energy.ts are stubs (return empty arrays)

---

## Current Architecture

### Local Development

```
┌─────────────────────────────────────────────────────────────┐
│                         Developer                            │
│                    http://localhost:8080                     │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
                  ┌────────────────┐
                  │  Vite Dev      │
                  │  Server        │
                  │  (Port 8080)   │
                  └────────┬───────┘
                           │
                  /api/* → │ (proxy)
                           ▼
                  ┌────────────────┐
                  │  FastAPI       │◄──────┐
                  │  Backend       │       │ Read models
                  │  (Port 8000)   │       │
                  └────┬───────────┘       │
                       │                   │
         ┌─────────────┼───────────────┐   │
         │             │               │   │
         ▼             ▼               ▼   ▼
    ┌────────┐   ┌──────────┐   ┌─────────────┐
    │InfluxDB│   │ Supabase │   │Model Registry│
    │(Docker)│   │  (Cloud) │   │  (Local FS) │
    └────────┘   └──────────┘   └─────────────┘
```

### Production

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser Client                            │
│              https://frontend-domain.com                     │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           │ CORS-enabled requests
                           │ Authorization: Bearer {token}
                           ▼
                  ┌────────────────┐
                  │  FastAPI       │
                  │  Backend       │
                  │  (Railway)     │◄──────┐
                  └────┬───────────┘       │
                       │                   │
         ┌─────────────┼───────────────┐   │
         │             │               │   │
         ▼             ▼               ▼   ▼
    ┌────────┐   ┌──────────┐   ┌─────────────┐
    │InfluxDB│   │ Supabase │   │Model Registry│
    │(Railway│   │  (Cloud) │   │  (Railway)  │
    │ or ext)│   │          │   │             │
    └────────┘   └──────────┘   └─────────────┘
```

---

## API Call Inventory

### Frontend → Backend Call Map

| Frontend Call | Expected Backend | Actual Backend | Status | Auth |
|---------------|------------------|----------------|--------|------|
| `GET /api/energy/readings` | `/api/energy/readings` | `/analytics/readings` | ❌ Mismatch | JWT |
| `GET /api/energy/insights` | `/api/energy/insights` | (none) | ❌ Missing | JWT |
| `POST /api/energy/reports` | `/api/energy/reports` | (none) | ❌ Missing | JWT |
| `GET /api/energy/reports/{id}` | `/api/energy/reports/{id}` | (none) | ❌ Missing | JWT |
| `GET /api/energy/buildings` | `/api/energy/buildings` | (none - use Supabase) | ℹ️ Supabase | JWT |
| `GET /api/energy/appliances` | `/api/energy/appliances` | (none - use Supabase) | ℹ️ Supabase | JWT |
| `GET /api/local/predictions` | `/api/local/predictions` | (removed) | ❌ Legacy | None |
| - | - | `POST /infer` | ✅ Add to frontend | JWT |
| - | - | `GET /models` | ✅ Add to frontend | None |
| - | - | `GET /analytics/predictions` | ✅ Add to frontend | JWT |
| `GET /live` | `/live` | `/live` | ✅ Match | None |
| `GET /ready` | `/ready` | `/ready` | ✅ Match | None |

### Backend API Surface

| Endpoint | Method | Auth | Purpose | Request Params | Response |
|----------|--------|------|---------|----------------|----------|
| `/live` | GET | No | Liveness probe | None | `{status: "ok"}` |
| `/ready` | GET | No | Readiness check | None | `{status: "ok", checks: {...}}` |
| `/health` | GET | No | Health details | None | Environment info |
| `/infer` | POST | JWT | Run inference | `InferRequest` | `InferResponse` |
| `/models` | GET | No | List models | None | `ModelsListResponse` |
| `/analytics/readings` | GET | JWT | Get sensor data | Query params | `ReadingsResponse` |
| `/analytics/predictions` | GET | JWT | Get predictions | Query params | `PredictionsResponse` |
| `/admin/reload-models` | POST | JWT+Admin | Reload registry | None | Status |
| `/admin/cache/invalidate` | POST | JWT+Admin | Clear cache | Body | Result |
| `/admin/cache/stats` | GET | JWT+Admin | Cache stats | None | Stats |
| `/metrics` | GET | No | Prometheus | None | Metrics |

---

## Environment Variables Map

### Frontend (apps/web/.env.example)

**Current:**
```env
VITE_SUPABASE_URL=
VITE_SUPABASE_PUBLISHABLE_KEY=
# or fallback
VITE_SUPABASE_ANON_KEY=

VITE_DEMO_MODE=false
VITE_DEMO_EMAIL=demo@example.com
VITE_DEMO_PASSWORD=

VITE_LOCAL_MODE=false
```

**Missing:**
```env
VITE_API_BASE_URL=                    # For production: Railway backend URL
                                       # For local dev: "/api" (use proxy)
```

**Recommendation:**
```env
# Required for production
VITE_SUPABASE_URL=https://xxx.supabase.co
VITE_SUPABASE_PUBLISHABLE_KEY=eyJxxx...

# Production backend (leave empty for local dev with proxy)
VITE_API_BASE_URL=

# Optional: Demo mode
VITE_DEMO_MODE=false
VITE_DEMO_EMAIL=demo@example.com
VITE_DEMO_PASSWORD=demo123

# Optional: Local dev mode (deprecated - consider removing)
VITE_LOCAL_MODE=false
```

### Backend (apps/backend/.env.example)

**Current:**
```env
ENV=dev
DEBUG=false
PORT=8000
HOST=0.0.0.0

CORS_ORIGINS=http://localhost:3000,http://localhost:8080

INFLUX_URL=http://localhost:8086
INFLUX_TOKEN=
INFLUX_ORG=energy-monitor
INFLUX_BUCKET_RAW=raw_sensor_data
INFLUX_BUCKET_PRED=predictions

SUPABASE_URL=
SUPABASE_PUBLISHABLE_KEY=
SUPABASE_SECRET_KEY=

RATE_LIMIT_PER_USER=60/minute
RATE_LIMIT_PER_IP=120/minute

ADMIN_TOKEN=
```

**Missing Production CORS:**
```env
# Update CORS_ORIGINS to include production frontend
CORS_ORIGINS=http://localhost:8080,http://localhost:5173,https://your-frontend-domain.com
```

**Complete Recommendation:**
```env
# Environment
ENV=dev  # dev, test, prod
DEBUG=false
PORT=8000
HOST=0.0.0.0

# CORS (comma-separated, no spaces)
CORS_ORIGINS=http://localhost:8080,http://localhost:5173,https://production-frontend.com

# InfluxDB
INFLUX_URL=http://localhost:8086
INFLUX_TOKEN=your-admin-token-here
INFLUX_ORG=energy-monitor
INFLUX_BUCKET_RAW=raw_sensor_data
INFLUX_BUCKET_PRED=predictions
INFLUX_TIMEOUT_MS=10000

# Supabase
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_PUBLISHABLE_KEY=eyJxxx...
SUPABASE_SECRET_KEY=eyJyyy...  # Service role key (backend only)

# Authentication
AUTH_VERIFY_AUD=true

# Rate Limiting
RATE_LIMIT_PER_USER=60/minute
RATE_LIMIT_PER_IP=120/minute
MAX_BODY_BYTES=262144

# Admin (production only)
ADMIN_TOKEN=secure-random-token-here

# Model Registry
MODEL_REGISTRY_PATH=/app/models/registry.json
MODELS_DIR=/app/models

# Caching
AUTHZ_CACHE_TTL_SECONDS=60
IDEMPOTENCY_CACHE_TTL_SECONDS=600
JWKS_CACHE_TTL_HOURS=6
```

---

## Authentication Flow Analysis

### Token Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│ 1. User Login (Supabase)                                     │
│    supabase.auth.signInWithPassword({email, password})      │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Session Stored (LocalStorage/SessionStorage)             │
│    Key: sb-{projectRef}-auth-token                          │
│    Value: { access_token, refresh_token, expires_at, ... }  │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. API Request (api.ts)                                      │
│    - Reads localStorage                                      │
│    - Extracts access_token                                   │
│    - Adds: Authorization: Bearer {access_token}             │
│    - Adds: X-Request-ID: req_xxx                            │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Backend Verification (FastAPI)                            │
│    - Extract Bearer token from header                        │
│    - Detect algorithm (RS256 or HS256)                       │
│    - Verify signature (JWKS for RS256, secret for HS256)    │
│    - Check exp, iat, iss, aud                               │
│    - Extract user_id (sub), role                            │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Authorization Check (if resource-scoped)                  │
│    - Require building access: query Supabase for ownership  │
│    - Require appliance access: check building + appliance   │
│    - Cache result for 60 seconds                            │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Response with Request ID                                  │
│    - Success: 200 + data + X-Request-ID header              │
│    - Error: 401/403 + error object + request_id            │
└─────────────────────────────────────────────────────────────┘
```

### Token Format

**JWT Payload (from Supabase):**
```json
{
  "sub": "user-uuid-here",
  "email": "user@example.com",
  "role": "authenticated",
  "aud": "authenticated",
  "iss": "https://xxx.supabase.co/auth/v1",
  "iat": 1234567890,
  "exp": 1234571490,
  "app_metadata": {
    "provider": "email"
  },
  "user_metadata": {}
}
```

**Backend TokenPayload (extracted):**
```python
{
  "sub": "user-uuid",           # User ID (required)
  "email": "user@example.com",  # Optional
  "role": "authenticated",      # Optional (can be admin, super_admin)
  "exp": 1234571490,           # Expiration
  "iat": 1234567890,           # Issued at
  "iss": "https://...",        # Issuer
  "aud": "authenticated",      # Audience
  "raw": {...}                 # Full payload
}
```

### Authorization Matrix

| Endpoint | Anonymous | Authenticated User | Admin | Super Admin |
|----------|-----------|-------------------|-------|-------------|
| `/live` | ✅ | ✅ | ✅ | ✅ |
| `/ready` | ✅ | ✅ | ✅ | ✅ |
| `/health` | ✅ | ✅ | ✅ | ✅ |
| `/models` | ✅ | ✅ | ✅ | ✅ |
| `/infer` | ❌ | ✅ (owned building) | ✅ | ✅ |
| `/analytics/readings` | ❌ | ✅ (owned building) | ✅ | ✅ |
| `/analytics/predictions` | ❌ | ✅ (owned building) | ✅ | ✅ |
| `/admin/*` | ❌ | ❌ | ✅ (+token) | ✅ (+token) |
| `/metrics` | ✅ | ✅ | ✅ | ✅ |

**Notes:**
- "Owned building" = user must have access to the building via Supabase `building_users` or `building_appliances` table
- Admin endpoints additionally require `X-Admin-Token` header in production
- Admins have universal building/appliance access

---

## Gap Analysis

### 1. API Contract Mismatches

**High Priority:**

| Issue | Impact | Solution |
|-------|--------|----------|
| Frontend calls `/api/energy/readings` but backend has `/analytics/readings` | **Blocking** - feature broken | Update frontend `energy.ts` to call `/analytics/readings` |
| Frontend calls `/api/energy/insights` (doesn't exist) | **High** - feature unavailable | Remove frontend calls OR add backend endpoint |
| Frontend calls `/api/energy/reports` (doesn't exist) | **High** - feature unavailable | Remove frontend calls OR add backend endpoint |
| Frontend calls `/api/local/predictions` (removed) | **Medium** - local mode broken | Update to call `/analytics/predictions` OR remove feature |
| Frontend doesn't call `/infer` endpoint | **High** - inference unavailable | Add inference calls where needed |
| Frontend doesn't call `/models` endpoint | **Low** - model selection limited | Add model listing if UI needs it |

**Recommended Actions:**

1. **Update `apps/web/src/services/energy.ts`:**
   ```typescript
   // Change from:
   async getReadings() { return api.get('/api/energy/readings', ...); }

   // To:
   async getReadings() { return api.get('/analytics/readings', ...); }
   ```

2. **Add inference method:**
   ```typescript
   async runInference(request: InferRequest) {
     return api.post('/infer', request);
   }
   ```

3. **Add models method:**
   ```typescript
   async getModels() {
     return api.get('/models');
   }
   ```

4. **Remove or stub out:**
   - `getInsights()` - no backend support
   - `generateReport()` - no backend support
   - `getReport()` - no backend support

### 2. Error Handling Alignment

**Issue:** Frontend and backend use different error formats.

**Backend Format:**
```json
{
  "error": {
    "code": "AUTHZ_BUILDING_ACCESS_DENIED",
    "message": "You do not have permission to access this building",
    "details": {"building_id": "bldg-123"}
  },
  "request_id": "req_1234567890_abc123"
}
```

**Frontend ApiError Class:**
```typescript
class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public data?: any
  )
}
```

**Gap:** Frontend doesn't extract `error.code` or `request_id`.

**Solution:** Update `apps/web/src/services/api.ts`:
```typescript
class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public code?: string,
    public requestId?: string,
    public details?: any
  ) {
    super(message);
  }
}

// In error handling:
const errorData = await response.json();
throw new ApiError(
  errorData.error?.message || 'Request failed',
  response.status,
  errorData.error?.code,
  errorData.request_id,
  errorData.error?.details
);
```

### 3. Request ID Propagation

**Issue:** Frontend doesn't generate or send X-Request-ID headers.

**Impact:** Hard to trace requests across frontend/backend logs.

**Solution:** Update `apps/web/src/services/api.ts`:
```typescript
// Generate request ID
const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
headers['X-Request-ID'] = requestId;

console.log(`[API] ${method} ${url} [${requestId}]`);
```

### 4. Query Parameter Validation

**Issue:** Frontend may send invalid query params to analytics endpoints.

**Backend Requirements:**
- `building_id`: 1-64 chars, alphanumeric + dash/underscore
- `start`: ISO8601 or relative (e.g., "-7d")
- `end`: ISO8601 or relative (e.g., "now()")
- `resolution`: enum ["1s", "1m", "15m"]
- `appliance_id`: optional, 1-64 chars

**Frontend Should Validate:**
- Date ranges are valid ISO8601 or relative
- building_id is required
- resolution is one of allowed values

**Solution:** Add validation in `apps/web/src/services/energy.ts`:
```typescript
function validateQueryParams(params: AnalyticsParams): void {
  if (!params.building_id || params.building_id.length > 64) {
    throw new Error('Invalid building_id');
  }

  if (params.resolution && !['1s', '1m', '15m'].includes(params.resolution)) {
    throw new Error('Invalid resolution');
  }

  // ... more validation
}
```

---

## Legacy References

### Files with Deprecated Service References

**High Priority (Breaks functionality):**

1. **`apps/web/src/hooks/useLocalInfluxPredictions.ts` (line 87)**
   ```typescript
   // Calls non-existent endpoint
   fetch(`/api/local/predictions?buildingId=${buildingId}&start=${start}&end=${end}`)
   ```
   **Fix:** Update to call `/analytics/predictions` OR remove hook entirely

2. **`apps/web/.env.example` (line 46)**
   ```env
   # References removed script
   # Start with: npm run local:server
   ```
   **Fix:** Remove line or update to `docker compose up -d`

**Medium Priority (Misleading error messages):**

3. **`scripts/generate-predictions.ts` (lines 35, 60, 293)**
   ```typescript
   const INFERENCE_SERVICE_URL = process.env.INFERENCE_SERVICE_URL || 'http://localhost:8000';
   // Falls back to mock mode if service unavailable
   ```
   **Fix:** Remove inference service fallback, use backend directly

4. **`scripts/write-to-influx.ts` (line 63)**
   ```typescript
   console.log('Make sure the inference-service is running (docker compose up -d inference-service)');
   ```
   **Fix:** Update message to `docker compose up -d`

**Low Priority (Documentation):**

5. **`compose.yaml` (lines 139-149)**
   ```yaml
   # Legacy Inference Service (DEPRECATED - now handled by unified backend)
   # inference-service:
   #   build: ./apps/inference-service
   ```
   **Fix:** Delete commented section entirely

6. **`README.md` (lines 13-15)**
   ```markdown
   | inference-service | apps/inference-service | ⚠️ Deprecated |
   | local-server | apps/local-server | ⚠️ Deprecated |
   ```
   **Fix:** Remove deprecated services table

7. **`docs/LOCAL_DEVELOPMENT.md` (extensive)**
   - References `npm run local:dev` (doesn't exist)
   - References `npm run local:server` (doesn't exist)
   - Describes Node.js/Express at port 3001 (removed)
   **Fix:** Major rewrite needed

8. **`apps/web/docs/README.md` (lines 85, 112-113)**
   ```markdown
   npm run local:dev
   npm run local:server
   ```
   **Fix:** Remove references to non-existent scripts

---

## Integration Tasks

### Prioritized Task List

**Phase 1: Critical (Blocks functionality)**

- [ ] **T1.1** Update `apps/web/src/services/energy.ts` - Change `/api/energy/readings` → `/analytics/readings`
- [ ] **T1.2** Add `/infer` endpoint calls in `energy.ts`
- [ ] **T1.3** Add `/models` endpoint calls in `energy.ts`
- [ ] **T1.4** Update or remove `apps/web/src/hooks/useLocalInfluxPredictions.ts`
- [ ] **T1.5** Enhance `apps/web/src/services/api.ts` error handling (add code, request_id)
- [ ] **T1.6** Add X-Request-ID header generation in `api.ts`

**Phase 2: High Priority (Improves stability)**

- [ ] **T2.1** Update `apps/web/.env.example` - Add `VITE_API_BASE_URL`, remove legacy refs
- [ ] **T2.2** Update `apps/backend/.env.example` - Add production CORS origins
- [ ] **T2.3** Add query param validation in `energy.ts`
- [ ] **T2.4** Add error message mapping utility (code → user-friendly text)
- [ ] **T2.5** Update error display components to show request_id

**Phase 3: Medium Priority (Cleanup)**

- [ ] **T3.1** Remove inference-service references from `scripts/generate-predictions.ts`
- [ ] **T3.2** Remove inference-service references from `scripts/write-to-influx.ts`
- [ ] **T3.3** Delete commented inference-service section from `compose.yaml`
- [ ] **T3.4** Remove `/api/energy/insights` calls from frontend (or add backend support)
- [ ] **T3.5** Remove `/api/energy/reports` calls from frontend (or add backend support)

**Phase 4: Low Priority (Documentation)**

- [ ] **T4.1** Update `README.md` - Remove deprecated services table
- [ ] **T4.2** Rewrite `docs/LOCAL_DEVELOPMENT.md` for FastAPI setup
- [ ] **T4.3** Update `apps/web/docs/README.md` - Remove invalid npm scripts
- [ ] **T4.4** Create `docs/DEPLOYMENT.md` with production setup guide
- [ ] **T4.5** Add architecture diagram to documentation

### Verification Tasks

**Local Development:**

- [ ] **V1** Start backend: `docker compose up -d`
- [ ] **V2** Verify backend health: `curl http://localhost:8000/ready`
- [ ] **V3** Seed test data: `npm run predictions:seed`
- [ ] **V4** Start frontend: `cd apps/web && npm run dev`
- [ ] **V5** Test login flow
- [ ] **V6** Test analytics/readings endpoint
- [ ] **V7** Test analytics/predictions endpoint
- [ ] **V8** Test inference endpoint
- [ ] **V9** Verify error messages show request_id
- [ ] **V10** Check browser console for CORS errors (should be none)

**Production:**

- [ ] **P1** Verify Railway backend `/live` returns 200
- [ ] **P2** Verify Railway backend `/ready` returns 200
- [ ] **P3** Test deployed frontend can call backend
- [ ] **P4** Verify CORS headers present
- [ ] **P5** Test full auth flow (login → API call)
- [ ] **P6** Verify JWT verification works with production JWKS

**Automated:**

- [ ] **A1** Frontend: `npm run lint`
- [ ] **A2** Frontend: `npm run typecheck`
- [ ] **A3** Frontend: `npm run build`
- [ ] **A4** Backend: `pytest`
- [ ] **A5** Backend: `ruff check .`
- [ ] **A6** Backend: `mypy .`

---

## Appendix: Technical Details

### A. Vite Proxy Configuration

**File:** `apps/web/vite.config.ts`

```typescript
export default defineConfig({
  server: {
    host: "::",
    port: 8080,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
```

**Behavior:**
- Local dev: `http://localhost:8080/api/infer` → proxied to → `http://localhost:8000/api/infer`
- Production: Frontend directly calls `VITE_API_BASE_URL + /infer`

### B. Backend Middleware Stack (Execution Order)

1. **RequestSizeLimitMiddleware** (ASGI)
   - Rejects requests over 256KB
   - Returns 413 if too large

2. **RateLimitMiddleware** (Starlette)
   - 60 req/min per authenticated user
   - 120 req/min per IP (unauthenticated)
   - Returns 429 with Retry-After header

3. **MetricsMiddleware** (Starlette)
   - Records Prometheus metrics

4. **RequestIdMiddleware** (Starlette)
   - Generates or uses X-Request-ID
   - Adds to response headers

5. **CORSMiddleware** (Starlette)
   - Validates origin against CORS_ORIGINS
   - Adds CORS headers to response

### C. Error Code Reference

| Code | Status | Meaning | User Message |
|------|--------|---------|--------------|
| `AUTH_MISSING_TOKEN` | 401 | No Authorization header | Please log in to continue |
| `AUTH_INVALID_TOKEN` | 401 | Token malformed/invalid | Your session is invalid. Please log in again. |
| `AUTH_EXPIRED_TOKEN` | 401 | Token expired | Your session has expired. Please log in again. |
| `AUTHZ_BUILDING_ACCESS_DENIED` | 403 | User doesn't own building | You don't have access to this building |
| `AUTHZ_APPLIANCE_ACCESS_DENIED` | 403 | User doesn't own appliance | You don't have access to this appliance |
| `AUTHZ_INSUFFICIENT_ROLE` | 403 | User is not admin | You don't have permission for this action |
| `VALIDATION_FAILED` | 422 | Request validation failed | Invalid request parameters |
| `RATE_LIMITED` | 429 | Too many requests | Too many requests. Please wait and try again. |
| `INFLUX_WRITE_FAILED` | 503 | Can't write to InfluxDB | Unable to save data. Please try again. |
| `INFLUX_READ_FAILED` | 503 | Can't read from InfluxDB | Unable to load data. Please try again. |
| `MODEL_NOT_FOUND` | 500 | Model not in registry | Model not found |
| `INFERENCE_FAILED` | 500 | Prediction failed | Prediction failed. Please try again. |

### D. Query Parameter Formats

**Analytics Endpoints (`/analytics/readings`, `/analytics/predictions`):**

```typescript
interface AnalyticsParams {
  building_id: string;        // Required, 1-64 chars, alphanumeric + dash/underscore
  appliance_id?: string;      // Optional, 1-64 chars
  start: string;              // ISO8601 or relative (e.g., "-7d", "2024-01-01T00:00:00Z")
  end: string;                // ISO8601 or relative (e.g., "now()", "2024-01-08T00:00:00Z")
  resolution?: '1s'|'1m'|'15m'; // Default: "1m"
}
```

**Examples:**
```
Good:
  /analytics/readings?building_id=bldg-123&start=-7d&end=now()&resolution=15m
  /analytics/predictions?building_id=bldg-123&appliance_id=fridge-1&start=2024-01-01T00:00:00Z&end=2024-01-08T00:00:00Z

Bad:
  /analytics/readings?building_id=&start=invalid  (empty building_id, invalid date)
  /analytics/predictions?start=-7d&end=now()      (missing building_id)
```

### E. Request/Response Examples

**Successful Inference:**

Request:
```http
POST /infer HTTP/1.1
Host: localhost:8000
Authorization: Bearer eyJhbGc...
X-Request-ID: req_1234567890_abc123
Content-Type: application/json

{
  "building_id": "bldg-123",
  "appliance_id": "fridge-1",
  "window": [0.5, 0.6, ..., 0.7],  // 1000 floats
  "timestamp": "2024-01-15T12:00:00Z",
  "model_id": null
}
```

Response:
```http
HTTP/1.1 200 OK
X-Request-ID: req_1234567890_abc123
Content-Type: application/json

{
  "predicted_kw": 1.25,
  "confidence": 0.92,
  "model_version": "lstm-v3-2024",
  "request_id": "req_1234567890_abc123",
  "persisted": true
}
```

**Error Response:**

Request:
```http
GET /analytics/readings?building_id=bldg-999&start=-7d&end=now() HTTP/1.1
Host: localhost:8000
Authorization: Bearer eyJhbGc...
```

Response:
```http
HTTP/1.1 403 Forbidden
X-Request-ID: req_1234567891_def456
Content-Type: application/json

{
  "error": {
    "code": "AUTHZ_BUILDING_ACCESS_DENIED",
    "message": "You do not have permission to access this building",
    "details": {
      "building_id": "bldg-999",
      "user_id": "user-123"
    }
  },
  "request_id": "req_1234567891_def456"
}
```

---

**End of Audit**

This document will be updated as integration tasks are completed. Last updated: 2026-01-21
