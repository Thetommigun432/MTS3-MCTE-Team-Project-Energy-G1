# Deprecated: archived deployment doc

> This document is archived and no longer maintained.
> Use [docs/DEPLOYMENT.md](../../DEPLOYMENT.md) instead.

# Deploying to Cloudflare Pages

This frontend is a React + Vite SPA designed to run on Cloudflare Pages.

## Environment Variables

Since Cloudflare Pages builds your app similarly to a static site, environment variables are **embedded at build time**. If you change a variable, you must **redeploy/rebuild** the site.

### Required Variables

| Variable | Value (Example) | Description |
|----------|-----------------|-------------|
| `VITE_BACKEND_URL` | `https://energy-monitor.up.railway.app` | **Real** Railway backend URL. Must use HTTPS. |
| `VITE_SUPABASE_URL` | `https://xyz.supabase.co` | Your Supabase project URL. |
| `VITE_SUPABASE_ANON_KEY` | `eyJ...` | Your Supabase anonymous (public) key. |
| `VITE_DEMO_MODE` | `false` | Set to `false` for production to default to API mode. |

### Configuration in Cloudflare Dashboard

1. Go to **Cloudflare Dashboard** > **Workers & Pages** > Your Project.
2. Click **Settings** > **Environment variables**.
3. Add the variables above for the **Production** environment.
   > **Note:** Cloudflare sometimes adds quotes to variables. The app is patched to handle this, but try to enter values **without** extra surrounding quotes.

## build Settings

| Setting | Value |
|---------|-------|
| **Framework Preset** | `Vite` |
| **Build Command** | `npm run build` (or `npm run build:web` depending on repo root) |
| **Build Output Directory** | `dist` (or `apps/web/dist`) |

If deploying from the monorepo root:
- **Build Command**: `npm run build:web`
- **Output Directory**: `apps/web/dist`

## Verification

After deployment:
1. Open the Cloudflare URL.
2. Open **Settings** (Top Right) -> **Debug Info** (if available) or check Network Tab.
3. Switch to **API Mode** in the top bar.
4. If you see "Auth error" or data loads, connection is successful.
5. If you see "Network Error" or "API unreachable", check:
   - Does `VITE_BACKEND_URL` start with `https://`?
   - Is `CORS_ORIGINS` on Railway set to include your Cloudflare domain?
