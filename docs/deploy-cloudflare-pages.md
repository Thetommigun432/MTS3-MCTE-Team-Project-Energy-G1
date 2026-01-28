# Cloudflare Pages Deployment Guide

This guide covers deploying the NILM Energy Monitor frontend to Cloudflare Pages.

## Overview

The frontend is a Vite-built React SPA that connects to the Railway backend API.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Cloudflare Pages                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Static Assets                         │   │
│  │                                                         │   │
│  │  - index.html                                           │   │
│  │  - assets/*.js (bundled React app)                      │   │
│  │  - assets/*.css                                         │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                    │
│                            │ API calls                          │
│                            ▼                                    │
│              ┌─────────────────────────┐                       │
│              │  VITE_BACKEND_URL       │                       │
│              │  (Railway API)          │                       │
│              └─────────────────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. Cloudflare account
2. Wrangler CLI: `npm install -g wrangler`
3. Railway backend deployed and accessible

## Build Configuration

### Cloudflare Pages Settings

| Setting | Value |
|---------|-------|
| Framework preset | Vite |
| Build command | `npm run build:web` |
| Build output directory | `apps/web/dist` |
| Root directory | (leave empty - use repo root) |
| Node.js version | 22.x |

### Required Environment Variables

Set these in Cloudflare Pages > Settings > Environment Variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `VITE_BACKEND_URL` | Yes | Railway API URL (e.g., `https://api.railway.app`) |
| `VITE_SUPABASE_URL` | Yes | Supabase project URL |
| `VITE_SUPABASE_PUBLISHABLE_KEY` | Yes | Supabase public key |
| `VITE_DEMO_MODE` | No | Set to `false` for production |

**Important**: Vite environment variables are injected at **build time**, not runtime. Any changes require a rebuild.

## Deployment Methods

### Method 1: Cloudflare Dashboard (Recommended)

1. Go to Cloudflare Dashboard > Pages
2. Click "Create a project"
3. Connect your Git repository (GitHub/GitLab)
4. Configure build settings:
   - Build command: `npm run build:web`
   - Build output directory: `apps/web/dist`
5. Add environment variables
6. Deploy

The site will automatically rebuild on commits to your main branch.

### Method 2: Wrangler CLI

```bash
# Login to Cloudflare
wrangler login

# Create Pages project (first time only)
wrangler pages project create nilm-energy-monitor

# Build the frontend locally
npm run build:web

# Deploy
wrangler pages deploy apps/web/dist --project-name nilm-energy-monitor
```

### Method 3: GitHub Actions

Create `.github/workflows/deploy-pages.yml`:

```yaml
name: Deploy to Cloudflare Pages

on:
  push:
    branches: [main]
    paths:
      - 'apps/web/**'
      - 'package*.json'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '22'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Build
        run: npm run build:web
        env:
          VITE_BACKEND_URL: ${{ secrets.VITE_BACKEND_URL }}
          VITE_SUPABASE_URL: ${{ secrets.VITE_SUPABASE_URL }}
          VITE_SUPABASE_PUBLISHABLE_KEY: ${{ secrets.VITE_SUPABASE_PUBLISHABLE_KEY }}
          VITE_DEMO_MODE: 'false'

      - name: Deploy to Cloudflare Pages
        uses: cloudflare/pages-action@v1
        with:
          apiToken: ${{ secrets.CLOUDFLARE_API_TOKEN }}
          accountId: ${{ secrets.CLOUDFLARE_ACCOUNT_ID }}
          projectName: nilm-energy-monitor
          directory: apps/web/dist
```

Required GitHub secrets:
- `CLOUDFLARE_API_TOKEN`: Create at Cloudflare Dashboard > My Profile > API Tokens
- `CLOUDFLARE_ACCOUNT_ID`: Found in Cloudflare Dashboard URL
- `VITE_BACKEND_URL`: Your Railway API URL
- `VITE_SUPABASE_URL`: Your Supabase project URL
- `VITE_SUPABASE_PUBLISHABLE_KEY`: Your Supabase public key

## CORS Configuration

The Railway API must allow requests from your Cloudflare Pages domain.

Update the API's `CORS_ORIGINS` environment variable:

```
CORS_ORIGINS=https://nilm-energy-monitor.pages.dev,https://your-custom-domain.com
```

Include:
- Your Cloudflare Pages default domain (`.pages.dev`)
- Any custom domains you configure
- Preview deployment URLs if needed (`*.nilm-energy-monitor.pages.dev`)

## Custom Domains

1. Go to Cloudflare Pages > Your Project > Custom domains
2. Add your domain
3. Cloudflare will automatically configure DNS if the domain is on Cloudflare
4. Update CORS_ORIGINS in Railway to include the custom domain

## Environment Differences

### Local Development

```bash
# .env file (or defaults)
VITE_BACKEND_URL="/api"  # Uses Vite proxy to localhost:8000
VITE_SUPABASE_URL="https://your-project.supabase.co"
VITE_SUPABASE_PUBLISHABLE_KEY="your-key"
```

The Vite dev server proxies `/api` requests to the local backend, avoiding CORS issues.

### Production (Cloudflare Pages)

```bash
# Cloudflare Pages environment variables
VITE_BACKEND_URL="https://api.your-railway-app.railway.app"
VITE_SUPABASE_URL="https://your-project.supabase.co"
VITE_SUPABASE_PUBLISHABLE_KEY="your-key"
VITE_DEMO_MODE="false"
```

The frontend makes direct requests to the Railway API.

## Verification

After deployment:

1. Visit your Cloudflare Pages URL
2. Open browser DevTools > Network tab
3. Verify API calls go to your Railway backend URL
4. Check for CORS errors in Console
5. Test authentication with Supabase

## Troubleshooting

### "Failed to fetch" or Network Errors

- Check `VITE_BACKEND_URL` is set correctly (include `https://`)
- Verify Railway API is running and accessible
- Check CORS_ORIGINS includes your Pages domain

### CORS Errors

```
Access to fetch at 'https://api...' from origin 'https://...' has been blocked by CORS policy
```

- Add your Pages domain to Railway API's `CORS_ORIGINS`
- Redeploy the Railway API service
- Include both `.pages.dev` and any custom domains

### Environment Variables Not Working

- Vite variables must be prefixed with `VITE_`
- Variables are baked in at build time
- After changing variables, trigger a new deployment
- Clear Cloudflare cache if needed

### Build Failures

Check that:
- Node.js version matches (22.x)
- `npm ci` runs from repo root (workspaces)
- Build command is `npm run build:web`
- Output directory is `apps/web/dist`

## Preview Deployments

Cloudflare Pages automatically creates preview deployments for pull requests. To allow API access from preview URLs:

Add a wildcard pattern to CORS_ORIGINS:
```
CORS_ORIGINS=https://nilm-energy-monitor.pages.dev,https://*.nilm-energy-monitor.pages.dev
```

Note: Railway may not support wildcard CORS. In that case, preview deployments won't be able to call the API unless you add each preview URL explicitly.
