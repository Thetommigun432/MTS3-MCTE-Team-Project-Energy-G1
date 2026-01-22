# Repository Structure Audit

## Before: Current Tree (Top 3 Levels)

```
MTS3-MCTE-Team-Project-Energy-G1/
├── .claude/                          # IDE plugin
├── .env.local                        # ⚠️ Should not be committed
├── .env.local.example
├── .github/workflows/
├── .gitignore
├── .vscode/
├── README.md
├── apps/
│   ├── backend/                      # ✅ Active FastAPI
│   │   ├── app/
│   │   ├── models/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── local-server/                 # ❌ LEFTOVER (deleted from git but dir remains)
│   │   └── node_modules/
│   └── web/                          # ✅ Active React/Vite
│       ├── src/
│       ├── docs/
│       └── public/
├── compose.yaml                      # ✅ Canonical
├── data/                             # Datasets (77 files)
│   └── processed/15min/
├── docs/                             # Documentation
│   ├── assets/
│   ├── brief/
│   ├── diagram/
│   └── presentations/
├── influxdb-config/                  # ❌ COMMITTED DATA
├── influxdb-data/                    # ❌ COMMITTED DATA (188KB)
│   ├── engine/
│   ├── influxd.bolt
│   └── influxd.sqlite
├── infra/
│   └── influxdb/
│       ├── docker-compose.yml        # ❌ DUPLICATE compose
│       ├── influxdb-config/
│       └── influxdb-data/
├── model_exploration/                # ❌ TRAINING ONLY - 11 notebooks + 3 .pth (43MB)
├── model_highfreq/                   # ❌ TRAINING ONLY - 1 .pth (2.7MB)
├── models/                           # Empty?
├── node_modules/
├── package.json
├── package-lock.json
├── preprocessing/                    # ❌ TRAINING ONLY
├── pretraining/                      # ❌ TRAINING ONLY - notebooks
├── progress_logs/                    # ❌ ORPHAN logs
├── requirements-training.txt
├── scripts/
├── supabase/                         # Supabase config + edge functions
│   ├── functions/
│   └── migrations/
├── supabase-schema.sql               # ❌ ORPHAN - move to supabase/
└── train_model.py                    # ❌ ORPHAN - move to training/
```

---

## Structure Issues Found

### Critical (Must Fix)

| Issue | Location | Action |
|-------|----------|--------|
| Committed InfluxDB data | `influxdb-data/`, `influxdb-config/` | Delete + gitignore |
| Leftover deprecated folder | `apps/local-server/` | Delete entirely |
| Large .pth files committed | `model_exploration/*.pth` | Move to training/, gitignore |
| Duplicate compose | `infra/influxdb/docker-compose.yml` | Delete (use root) |

### Medium (Should Fix)

| Issue | Location | Action |
|-------|----------|--------|
| Training code at root | `model_exploration/`, `model_highfreq/`, `preprocessing/`, `pretraining/` | Consolidate to `training/` |
| Orphan SQL file | `supabase-schema.sql` | Move to `supabase/schema.sql` |
| Orphan Python script | `train_model.py` | Move to `training/` |
| Orphan logs | `progress_logs/` | Move to `training/logs/` or delete |
| Missing .env.example | Root, apps/web, apps/backend | Ensure all present |

### Minor (Nice to Have)

| Issue | Location | Action |
|-------|----------|--------|
| Inconsistent naming | `model_exploration` vs `model-exploration` | Standardize to kebab-case |
| Empty models/ folder | root | Delete (backend has own) |

---

## Target Structure (After Cleanup)

```
MTS3-MCTE-Team-Project-Energy-G1/
├── .env.example                      # Root env template
├── .github/workflows/
├── .gitignore                        # Updated with all ignores
├── README.md                         # Source of truth
├── compose.yaml                      # Single canonical compose
│
├── apps/
│   ├── backend/                      # FastAPI Python 3.12
│   │   ├── app/
│   │   ├── models/                   # Runtime model artifacts
│   │   ├── .env.example
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── web/                          # React 19 + Vite 7
│       ├── src/
│       ├── public/
│       ├── .env.example
│       └── package.json
│
├── docs/                             # All documentation
│   └── repo-structure-audit.md       # This file
│
├── scripts/                          # Dev tooling
│
├── supabase/                         # Supabase config
│   ├── functions/
│   ├── migrations/
│   └── schema.sql
│
└── training/                         # ML training (not runtime)
    ├── README.md
    ├── requirements.txt              # Training deps
    ├── train_model.py
    ├── notebooks/
    └── outputs/                      # .gitignore'd
```

---

## Decisions Made

### Naming Convention
- **Directories**: kebab-case (e.g., `model-exploration` → move to `training/`)
- **Files**: snake_case for Python, kebab-case for configs

### Canonical Locations
- **Frontend**: `apps/web/`
- **Backend**: `apps/backend/`
- **Compose**: `compose.yaml` (root only)
- **Training**: `training/` (consolidated)
- **Supabase**: `supabase/`

### Legacy Policy
- No legacy/ folder needed - all deprecated code deleted
- Large model files (.pth) gitignored, not committed

---

## Changes to Make

### Phase 1: Delete Committed Artifacts
1. Remove `influxdb-data/`, `influxdb-config/` from git
2. Remove `apps/local-server/` entirely
3. Remove `infra/influxdb/` (duplicate)
4. Remove `models/` (empty at root)
5. Update .gitignore

### Phase 2: Consolidate Training
1. Create `training/` directory
2. Move: `model_exploration/`, `model_highfreq/`, `preprocessing/`, `pretraining/` → `training/`
3. Move: `train_model.py`, `progress_logs/`, `requirements-training.txt` → `training/`
4. Gitignore training outputs (*.pth)

### Phase 3: Fix Orphans
1. Move `supabase-schema.sql` → `supabase/schema.sql`
2. Create missing `.env.example` files

### Phase 4: Update References
1. Update README
2. Verify compose works
3. Verify builds

---

## Verification Commands

```bash
# Validate compose
docker compose config

# Build backend
docker compose build backend

# Start and check health
docker compose up -d
curl http://localhost:8000/live

# Frontend
cd apps/web && npm run dev

# Quality checks
npm run lint && npm run typecheck && npm run build
```
