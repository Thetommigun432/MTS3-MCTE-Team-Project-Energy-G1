# Cloudflare Pages Migration

This document describes the frontend deployment setup using Cloudflare Pages Git integration.

## What Changed

| Before | After |
|--------|-------|
| `wrangler` devDependency present | Removed (unused) |
| Deployment method unclear | Git-integrated Cloudflare Pages |
| `VITE_SUPABASE_PUBLISHABLE_KEY` | `VITE_SUPABASE_ANON_KEY` (old still works) |
| `VITE_API_BASE_URL` | `VITE_BACKEND_URL` (old still works) |

### Files Removed/Modified
- **Removed**: `wrangler` devDependency from `apps/web/package.json`
- **Modified**: `apps/web/.env.example` - standardized env var names
- **Modified**: `apps/web/src/lib/env.ts` - backward-compatible config

### Files Kept
- `apps/web/public/_redirects` - SPA routing for React Router
- `apps/web/public/_headers` - Security headers
- `.github/workflows/deploy-gh-pages.yml` - GitHub Pages (separate deployment)

## Cloudflare Pages Project Settings

Configure these settings in the Cloudflare Pages dashboard:

| Setting | Value |
|---------|-------|
| **Framework preset** | Vite |
| **Root directory** | `apps/web` |
| **Build command** | `npm run build` |
| **Output directory** | `dist` |

## Environment Variables

Set these in **Cloudflare Pages > Settings > Environment variables**:

| Variable | Description | Required |
|----------|-------------|----------|
| `VITE_BACKEND_URL` | Railway backend URL (e.g., `https://your-app.railway.app`) | Yes |
| `VITE_SUPABASE_URL` | Supabase project URL | Yes |
| `VITE_SUPABASE_ANON_KEY` | Supabase anon/publishable key | Yes |
| `VITE_DEMO_MODE` | Set to `true` for demo deployments | No |

> **Note**: Environment variables starting with `VITE_` are embedded at build time. After changing them, trigger a new deployment.

## SPA Routing

The `apps/web/public/_redirects` file enables client-side routing for React Router:

```
/* /index.html 200
```

This tells Cloudflare Pages to serve `index.html` for all routes, allowing React Router to handle navigation. The file is automatically copied to `dist/` during build.

## Build Verification

After running `npm run build` from `apps/web/`:

```bash
# Verify _redirects is included
ls dist/_redirects

# Preview locally
npm run preview
```

## Local Development

For local development, the frontend defaults `VITE_BACKEND_URL` to `http://localhost:8000` if not set.

```bash
# Install dependencies
cd apps/web && npm install

# Run dev server
npm run dev
```

## Deployment Flow

1. Push changes to connected branch (e.g., `main` or `frontend`)
2. Cloudflare Pages automatically:
   - Clones repository
   - Navigates to `apps/web`
   - Runs `npm run build`
   - Deploys contents of `dist/`
3. Environment variables are injected at build time
