# NILM Energy Monitor - Source Code Structure

**Document Version:** 1.0  
**Project:** NILM Energy Monitor  
**Date:** January 2026  

---

## Table of Contents

1. [ZIP Archive Structure](#1-zip-archive-structure)
2. [Directory Descriptions](#2-directory-descriptions)
3. [Key Files Explained](#3-key-files-explained)
4. [File Naming Conventions](#4-file-naming-conventions)
5. [What Each Part Contains](#5-what-each-part-contains)
6. [Creating the Final ZIP](#6-creating-the-final-zip)

---

## 1. ZIP Archive Structure

The final submission ZIP should contain the following structure:

```
NILM-Energy-Monitor-Final.zip
│
├── apps/
│   ├── backend/                    # Python FastAPI Backend
│   │   ├── app/
│   │   │   ├── api/               # API route handlers
│   │   │   ├── core/              # Configuration, security
│   │   │   ├── domain/            # Business logic
│   │   │   ├── infra/             # Database clients
│   │   │   ├── schemas/           # Pydantic models
│   │   │   └── main.py            # Application entry point
│   │   ├── models/                # Trained ML models (.pth, .safetensors)
│   │   ├── scripts/               # Utility scripts
│   │   ├── tests/                 # Test files
│   │   ├── Dockerfile             # Container definition
│   │   ├── requirements.txt       # Python dependencies
│   │   └── README.md              # Backend documentation
│   │
│   └── web/                        # React/TypeScript Frontend
│       ├── src/
│       │   ├── components/        # React components
│       │   ├── contexts/          # State management
│       │   ├── hooks/             # Custom React hooks
│       │   ├── pages/             # Page components
│       │   ├── services/          # API client services
│       │   ├── lib/               # Utilities
│       │   └── types/             # TypeScript types
│       ├── public/                # Static assets
│       ├── package.json           # Node dependencies
│       ├── vite.config.ts         # Build configuration
│       ├── tailwind.config.ts     # Styling configuration
│       └── README.md              # Frontend documentation
│
├── data/
│   └── processed/
│       └── 15min/
│           └── model_ready/       # Training data (sample)
│               └── heatpump/
│                   ├── X_train.npy (sample)
│                   └── y_train.npy (sample)
│
├── docs/
│   ├── deliverables/              # Final documentation
│   │   ├── 01_PRESENTATION.md
│   │   ├── 02_FUNCTIONAL_ANALYSIS.md
│   │   ├── 03_DESIGN_DOCUMENT.md
│   │   ├── 04_USER_MANUAL.md
│   │   ├── 05_INSTALLATION_GUIDE.md
│   │   ├── 06_PROJECT_MANAGEMENT.md
│   │   ├── 07_ARCHITECTURE_OVERVIEW.md
│   │   └── 08_SOURCE_CODE_STRUCTURE.md
│   ├── API.md                     # API reference
│   ├── DEPLOYMENT.md              # Production deployment guide
│   ├── LOCAL_DEV.md               # Local development guide
│   ├── OPERATIONS.md              # Operations runbook
│   ├── PROJECT.md                 # Project overview
│   └── frontend.md                # Frontend documentation
│
├── supabase/
│   ├── schema.sql                 # Database schema
│   ├── migrations/                # Database migrations
│   └── functions/                 # Edge functions
│
├── training/
│   ├── train_model.py             # Training script
│   ├── requirements.txt           # Training dependencies
│   └── notebooks/                 # Jupyter notebooks (selected)
│
├── scripts/
│   ├── generate-predictions.ts    # Prediction generation
│   └── verify-influx.ts           # Database verification
│
├── compose.yaml                   # Docker Compose configuration
├── .env.local.example             # Environment template
├── README.md                      # Main project README
├── CLAUDE.md                      # AI assistant context
└── package.json                   # Root package configuration
```

---

## 2. Directory Descriptions

### 2.1 `/apps/backend/`

The Python FastAPI backend application containing all server-side logic.

| Subdirectory | Purpose |
|--------------|---------|
| `app/` | Main application code |
| `app/api/` | REST API endpoint definitions |
| `app/core/` | Configuration, security, settings |
| `app/domain/` | Business logic and services |
| `app/infra/` | Infrastructure (DB clients, external services) |
| `app/schemas/` | Request/response Pydantic models |
| `models/` | Trained PyTorch model files |
| `scripts/` | Utility and seeding scripts |
| `tests/` | Pytest test files |

### 2.2 `/apps/web/`

The React TypeScript frontend application.

| Subdirectory | Purpose |
|--------------|---------|
| `src/components/` | Reusable UI components |
| `src/components/ui/` | Base primitives (buttons, cards) |
| `src/components/nilm/` | Domain-specific components |
| `src/components/layout/` | Layout shells, navigation |
| `src/contexts/` | React Context providers |
| `src/hooks/` | Custom React hooks |
| `src/pages/` | Route page components |
| `src/services/` | API client functions |
| `src/lib/` | Utility functions |
| `src/types/` | TypeScript type definitions |
| `public/` | Static assets (images, icons) |

### 2.3 `/docs/`

Project documentation including deliverables.

| File/Directory | Purpose |
|----------------|---------|
| `deliverables/` | Final submission documents |
| `API.md` | Backend API reference |
| `OPERATIONS.md` | Operations runbook and troubleshooting |
| `PROJECT.md` | Architecture and project overview |
| `frontend.md` | Frontend-specific documentation |

### 2.4 `/supabase/`

Database schema and serverless functions.

| File/Directory | Purpose |
|----------------|---------|
| `schema.sql` | Complete database schema |
| `migrations/` | Incremental schema changes |
| `functions/` | Edge functions for server-side logic |

### 2.5 `/training/`

Machine learning model training code.

| File | Purpose |
|------|---------|
| `train_model.py` | Main training script |
| `requirements.txt` | Training-specific dependencies |
| `notebooks/` | Jupyter notebooks for experimentation |

### 2.6 `/data/` (Samples Only)

Processed data files. Full datasets are not included due to size.

| Directory | Purpose |
|-----------|---------|
| `processed/15min/model_ready/` | Model-ready numpy arrays |

---

## 3. Key Files Explained

### 3.1 Root Level Files

| File | Purpose |
|------|---------|
| `compose.yaml` | Docker Compose configuration for local development |
| `.env.local.example` | Template for environment variables |
| `README.md` | Project introduction and quick start |
| `CLAUDE.md` | Context file for AI assistants |
| `package.json` | Root Node.js configuration (monorepo) |

### 3.2 Backend Key Files

| File | Purpose |
|------|---------|
| `app/main.py` | FastAPI application entry point, router mounting |
| `app/api/analytics.py` | Readings and predictions endpoints |
| `app/api/inference.py` | ML inference endpoint |
| `app/core/config.py` | Environment configuration using Pydantic |
| `app/core/security.py` | JWT validation and auth utilities |
| `app/domain/nilm_service.py` | NILM business logic |
| `app/infra/influx_client.py` | InfluxDB connection and queries |
| `Dockerfile` | Container image definition |
| `requirements.txt` | Python package dependencies |

### 3.3 Frontend Key Files

| File | Purpose |
|------|---------|
| `src/main.tsx` | Application entry point, router setup |
| `src/App.tsx` | Root component with providers |
| `src/pages/app/Dashboard.tsx` | Main dashboard page |
| `src/contexts/AuthContext.tsx` | Authentication state management |
| `src/contexts/EnergyContext.tsx` | Energy data state management |
| `src/hooks/useEnergyData.ts` | Data fetching hook |
| `src/services/api.ts` | Axios API client configuration |
| `vite.config.ts` | Vite build configuration with proxy |
| `tailwind.config.ts` | Tailwind CSS theme configuration |
| `package.json` | Node.js dependencies |

### 3.4 Training Key Files

| File | Purpose |
|------|---------|
| `train_model.py` | Unified training script with CLI |
| `requirements.txt` | Training dependencies (full PyTorch) |

### 3.5 Database Key Files

| File | Purpose |
|------|---------|
| `supabase/schema.sql` | Complete PostgreSQL schema |
| `supabase/functions/` | Serverless edge functions |

---

## 4. File Naming Conventions

### 4.1 General Rules

| Type | Convention | Example |
|------|------------|---------|
| React Components | PascalCase | `Dashboard.tsx`, `ApplianceCard.tsx` |
| Hooks | camelCase with `use` prefix | `useAuth.ts`, `useEnergyData.ts` |
| Utilities | camelCase | `formatDate.ts`, `parseEnergy.ts` |
| Python Modules | snake_case | `influx_client.py`, `nilm_service.py` |
| Test Files | `test_` prefix or `.test.` | `test_analytics.py`, `Dashboard.test.tsx` |
| Configuration | lowercase with dots | `vite.config.ts`, `tailwind.config.ts` |

### 4.2 Model Files

Trained model files follow this pattern:
```
{architecture}_{appliance}_best.pth
```

Examples:
- `transformer_heatpump_best.pth`
- `cnn_dishwasher_best.pth`
- `unet_washingmachine_best.pth`

### 4.3 Data Files

```
{split}.npy
```

Examples:
- `X_train.npy`, `X_val.npy`, `X_test.npy`
- `y_train.npy`, `y_val.npy`, `y_test.npy`

---

## 5. What Each Part Contains

### 5.1 Backend Application (`apps/backend/`)

**Lines of Code:** ~3,500 Python  
**Primary Language:** Python 3.12  
**Framework:** FastAPI

**Contents:**
- REST API endpoints for analytics, inference, admin
- InfluxDB client for time-series queries
- Supabase client for auth and metadata
- PyTorch model loading and inference
- Pydantic schemas for validation
- Prometheus metrics export
- Health check endpoints

### 5.2 Frontend Application (`apps/web/`)

**Lines of Code:** ~8,000 TypeScript  
**Primary Language:** TypeScript  
**Framework:** React 19 + Vite 7

**Contents:**
- Page components (Dashboard, Appliances, Settings)
- Reusable UI component library (shadcn/ui based)
- Authentication flow with Supabase
- Data visualization with Recharts
- State management with React Context
- API service layer
- Type definitions for all data structures

### 5.3 Training Code (`training/`)

**Lines of Code:** ~1,200 Python  
**Primary Language:** Python  
**Framework:** PyTorch

**Contents:**
- Model architecture definitions (CNN, Transformer, UNet)
- Custom loss functions (WeightedNILMLoss)
- Training loop with validation
- Distributed training support
- Hyperparameter configuration
- Jupyter notebooks for experimentation

### 5.4 Database Schema (`supabase/`)

**SQL Lines:** ~300  
**Platform:** Supabase (PostgreSQL)

**Contents:**
- User profiles table
- Buildings and appliances tables
- Model registry tables
- Row-Level Security policies
- Trigger functions
- Edge functions for custom logic

---

## 6. Creating the Final ZIP

### 6.1 Files to Include

```bash
# Core application
apps/
docs/
supabase/
training/
scripts/

# Configuration
compose.yaml
.env.local.example
.env.example (if present)
README.md
CLAUDE.md
package.json

# Data (samples only, <10MB)
data/processed/15min/model_ready/heatpump/ (sample files)
```

### 6.2 Files to Exclude

```bash
# Dependencies (will be reinstalled)
node_modules/
venv/
__pycache__/
.pytest_cache/

# Build outputs
dist/
build/
*.egg-info/

# Environment files with secrets
.env
.env.local (actual, not .example)

# Large data files
data/processed/**/*.npy (except samples)
*.pth (large model files)

# IDE/Editor
.vscode/
.idea/

# Logs
logs/
*.log

# Git
.git/
```

### 6.3 Creating the ZIP

**Windows (PowerShell):**
```powershell
# Navigate to project root
cd C:\Users\Tommaso\Documents\HOWEST\TeamProject\MTS3-MCTE-Team-Project-Energy-G1

# Create ZIP excluding unwanted files
Compress-Archive -Path apps, docs, supabase, training, scripts, compose.yaml, README.md, CLAUDE.md, package.json -DestinationPath NILM-Energy-Monitor-Final.zip
```

**macOS/Linux:**
```bash
zip -r NILM-Energy-Monitor-Final.zip \
  apps/ docs/ supabase/ training/ scripts/ \
  compose.yaml README.md CLAUDE.md package.json \
  -x "*/node_modules/*" \
  -x "*/__pycache__/*" \
  -x "*/venv/*" \
  -x "*/.git/*" \
  -x "*.pyc"
```

### 6.4 Expected ZIP Size

| Component | Approximate Size |
|-----------|------------------|
| Backend code | 500 KB |
| Frontend code | 1.5 MB |
| Documentation | 500 KB |
| Training code | 200 KB |
| Configuration | 50 KB |
| Data samples | 5 MB |
| **Total** | **~8 MB** |

**Note:** Trained model files (.pth) are large (50-500MB each) and should be:
- Uploaded separately to cloud storage
- Documented with download links in README
- Or included only if specifically required

---

*Document Version: 1.0 | Last Updated: January 2026*
