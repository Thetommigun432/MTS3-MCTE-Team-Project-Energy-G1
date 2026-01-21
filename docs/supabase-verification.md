# Supabase Verification Guide

This guide details how to verify the hardened Supabase integration.

## 1. Environment Check
Ensure your `.env` (or `.env.local` for web, `.env` for backend) contains the new canonical keys.

**Frontend (`apps/web/.env.local`)**:
```bash
VITE_SUPABASE_URL=https://<project>.supabase.co
VITE_SUPABASE_PUBLISHABLE_KEY=pk_...
# VITE_SUPABASE_ANON_KEY=... (Optional fallback)
```

**Backend (`apps/backend/.env`)**:
```bash
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_PUBLISHABLE_KEY=pk_...
# SUPABASE_ANON_KEY=... (Optional fallback)
SUPABASE_JWT_SECRET=...     # Required for HS256
```

## 2. Frontend Verification (Manual)

1.  **Start the app**: `npm run dev` in `apps/web`.
2.  **Open Browser**: Go to `http://localhost:3000`.
3.  **Check Network**: Open DevTools > Network.
    *   Reload page.
    *   Look for requests to `<your-project>.supabase.co`.
    *   **Verify Header**: Request Headers should include `apikey: <your-publishable-key>`.
4.  **Login**:
    *   Perform a login (or Signup/Login with magic link if configured).
    *   **Verify Session**: Check Application > Local Storage > `sb-<project-id>-auth-token` exists.
5.  **Backend Call**:
    *   Navigate to a dashboard page that calls the Python backend.
    *   Check the request to `http://localhost:8000/...`.
    *   **Verify Auth**: Header should be `Authorization: Bearer <access_token>`.

## 3. Backend Verification (Smoke Test)

Use `curl` to verify backend JWT rejection rules (Fail Closed).

**Requirement**: You need a running backend (`make dev` or `python -m app.main`).

### 3.1 Verify 401 on Missing Token
```bash
curl -i http://localhost:8000/api/v1/protected-endpoint
# Expected: 401 Unauthorized "Authorization header required"
```

### 3.2 Verify 401 on Invalid Token
```bash
curl -i -H "Authorization: Bearer invalid-token" http://localhost:8000/api/v1/protected-endpoint
# Expected: 401 Unauthorized "Token verification failed"
```

### 3.3 Verify 403 on Valid Token but No Access
(Requires a valid JWT for a user)
```bash
# Get a token from frontend localStorage or Supabase dashboard
TOKEN="eyJ..."
BUILDING_ID="some-random-uuid"

curl -i -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/buildings/$BUILDING_ID/readings
# Expected: 403 Forbidden "Access denied to building..."
# (Unless you happen to own that random UUID)
```

## 4. Automated Tests
Run the backend test suite to verify auth logic.

```bash
cd apps/backend
pytest tests/test_auth.py  # (Or relevant test file)
```
*Note: Ensure `TEST_JWT_SECRET` is set in `.env` if running integration tests.*
