# Cloudflare Pages Deployment

## Build Configuration

| Setting | Value |
|---------|-------|
| **Root directory** | `/` (repo root) |
| **Build command** | `npm run build` |
| **Output directory** | `apps/web/dist` |

## Environment Variables (Required)

| Variable | Description |
|----------|-------------|
| `VITE_SUPABASE_URL` | Supabase project URL |
| `VITE_SUPABASE_ANON_KEY` | Supabase anon/publishable key |
| `VITE_BACKEND_URL` | Railway backend URL |

## How It Works

This repo uses **npm workspaces** with the frontend at `apps/web`. Cloudflare builds from the repo root:

1. `npm ci` installs all dependencies from root `package-lock.json`
2. `npm run build` triggers `npm run build --workspace=apps/web`
3. Output goes to `apps/web/dist`

## SPA Routing

`apps/web/public/_redirects` handles React Router:
```
/* /index.html 200
```

## Troubleshooting

### "npm ci requires package.json and package-lock.json to be in sync"
- Run `npm install` at repo root to regenerate `package-lock.json`
- Commit the updated lockfile
- Do NOT create a separate `apps/web/package-lock.json`

### Wrong Node version
- `.nvmrc` pins Node 20 (Cloudflare's default LTS)
- `package.json` has `"engines": { "node": ">=20" }`

### Build output not found
- Ensure output directory is `apps/web/dist` (not `dist`)
- Build command must run from repo root
