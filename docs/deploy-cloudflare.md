# Cloudflare Pages Deployment

This document describes the exact Cloudflare Pages configuration for deploying the NILM Energy Monitor frontend.

## Cloudflare Pages Settings

### Git-Connected Deployment

Configure these settings in the Cloudflare Pages dashboard:

| Setting | Value |
|---------|-------|
| **Framework preset** | `None` |
| **Root directory** | *(leave empty - repo root)* |
| **Build command** | `npm run build` |
| **Build output directory** | `dist` |

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NODE_VERSION` | Yes | `22` (must match `.nvmrc`) |
| `VITE_SUPABASE_URL` | Yes | Supabase project URL |
| `VITE_SUPABASE_ANON_KEY` | Yes | Supabase anonymous key |
| `VITE_BACKEND_URL` | Yes | Railway backend URL |
| `VITE_DEMO_MODE` | No | `false` for production |

## How It Works

1. Cloudflare runs `npm ci` from repo root
2. Cloudflare runs `npm run build` which:
   - Runs `vite build` in `apps/web` → outputs to `apps/web/dist`
   - Runs `postbuild` script → copies `apps/web/dist` to `/dist`
3. Cloudflare deploys from `/dist` at repo root

## Config-as-Code

The `wrangler.toml` in `apps/web/` is **not used** for Git-connected Pages deployments. It only applies to `wrangler pages deploy` CLI commands.

For Git-connected deployments, settings must be configured in the Cloudflare dashboard.

## Troubleshooting

### "Output directory `dist` not found"

**Cause:** Build output isn't at repo root `/dist`.

**Solution:**
1. Ensure `npm run build` includes the postbuild step
2. Root `package.json` should have:
   ```json
   "build": "npm run build --workspace=apps/web && npm run postbuild",
   "postbuild": "node -e \"...(copies apps/web/dist to dist)\""
   ```
3. Verify locally: `npm run build && ls dist/`

### "npm ci: lockfile mismatch"

**Cause:** `package-lock.json` is out of sync with `package.json`.

**Solution:**
```bash
rm -rf node_modules apps/web/node_modules
npm install
git add package-lock.json
git commit -m "fix: sync lockfile"
git push
```

### Build fails with module errors

**Cause:** npm ci failed or workspace resolution failed.

**Solution:**
1. Verify `NODE_VERSION=22` is set in Cloudflare environment
2. Verify `.nvmrc` at repo root contains `22`
3. Test locally: `rm -rf node_modules && npm ci && npm run build`

## Verify Deployment

After deployment, test:

1. **Homepage loads**: `https://your-site.pages.dev/`
2. **SPA routing works**: `https://your-site.pages.dev/app/dashboard` (should NOT 404)
3. **Refresh works**: On any `/app/*` route, press F5 (should reload same page)

## Files That Control Deployment

| File | Purpose |
|------|---------|
| `/package.json` | Build scripts, npm workspaces |
| `/package-lock.json` | Deterministic installs |
| `/.nvmrc` | Node version for Cloudflare |
| `/apps/web/vite.config.ts` | Vite build settings |
| `/apps/web/public/_redirects` | SPA routing rules |
| `/apps/web/public/_headers` | Security headers |
