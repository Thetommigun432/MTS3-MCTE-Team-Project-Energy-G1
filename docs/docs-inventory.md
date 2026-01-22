# Documentation Inventory & Restructuring Plan

## 1. Inventory

| File Path | Status | Target Location | Notes |
| :--- | :--- | :--- | :--- |
| `README.md` | Keep | `README.md` | Clean up, link to `docs/README.md` |
| `apps/backend/README.md` | Move | `docs/backend.md` | Merge backend specific info |
| `apps/web/docs/README.md` | Move | `docs/frontend.md` | Merge frontend specific info |
| `apps/web/docs/DEPLOY_CLOUDFLARE_PAGES.md` | Move | `docs/deployment/cloudflare.md` | Definitive Cloudflare guide |
| `apps/web/docs/DEPLOYMENT_STEPS.md` | Archive | `docs/_archive/web-deployment-steps.md` | Redundant with Cloudflare guide |
| `apps/web/docs/SECURITY.md` | Move | `docs/security.md` | Or merge into architecture/supa |
| `docs/CLOUDFLARE_DEPLOYMENT.md` | Merge | `docs/deployment/cloudflare.md` | Consolidate |
| `docs/INFLUX_SCHEMA.md` | Merge | `docs/influx.md` | Consolidate |
| `docs/LOCAL_DEVELOPMENT.md` | Move | `docs/getting-started.md` | Rename to standard |
| `docs/SUPABASE_SETUP.md` | Move | `docs/supabase.md` | Rename to standard |
| `docs/railway-cloudflare-deploy-audit.md` | Split | `docs/deployment/railway.md` | Extract Railway specific info |
| `docs/influx.md` | Keep | `docs/influx.md` | Ensure it has schema info |
| `docs/auth-jwt-migration-report.md` | Archive | `docs/_archive/` | Historical report |
| `docs/influx-setup-report.md` | Archive | `docs/_archive/` | Historical report |
| `docs/modernization-report.md` | Archive | `docs/_archive/` | Historical report |
| `docs/integration-audit.md` | Archive | `docs/_archive/` | Historical report |
| `docs/repo-structure-audit.md` | Archive | `docs/_archive/` | Historical report |
| `docs/supabase-setup-report.md` | Archive | `docs/_archive/` | Historical report |
| `docs/supabase-verification.md` | Archive | `docs/_archive/` | Historical report |

## 2. Target Structure

```text
docs/
├── README.md                # Index
├── getting-started.md       # Local Dev
├── architecture.md          # System Overview (New)
├── backend.md              
├── frontend.md
├── influx.md
├── supabase.md
├── deployment/
│   ├── railway.md
│   └── cloudflare.md
├── operations/
│   ├── troubleshooting.md   # From various audits
│   └── runbooks.md          # (New)
└── _archive/
```

## 3. Action Plan
1.  Create directory structure.
2.  Move files to `_archive` first to clear clutter.
3.  Move/Rename active files to target structure.
4.  Consolidate content (merge splits).
5.  Update Internal Links.
