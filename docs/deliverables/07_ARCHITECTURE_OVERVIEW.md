# NILM Energy Monitor - Technical Architecture Overview

**Document Version:** 1.0  
**Format:** A4 Single Page  
**Project:** NILM Energy Monitor  
**Date:** January 2026  

---

## System Architecture Overview

The NILM Energy Monitor is a **cloud-native web application** implementing a classic three-tier architecture with a specialized machine learning inference layer. The system follows a **monorepo structure** containing frontend and backend codebases, deployable independently to different hosting providers.

---

## Core Components

### 1. Frontend Application (React SPA)

**Technology:** React 19, TypeScript, Vite 7, Tailwind CSS

The frontend is a Single Page Application (SPA) responsible for all user interactions. It handles authentication via Supabase Auth SDK, fetches data from the backend REST API, and renders interactive visualizations using Recharts. The application supports multiple operational modes: Demo (static CSV data), Local (Docker/InfluxDB), and Production (cloud services).

**Key modules:** Dashboard visualization, authentication flow, building management, data export.

### 2. Backend API (FastAPI)

**Technology:** FastAPI, Python 3.12, Uvicorn, Pydantic

The backend provides a RESTful API layer that handles authentication verification, authorization (RBAC), data retrieval from InfluxDB, and orchestration of ML inference requests. It exposes endpoints for readings, predictions, model management, and administrative functions. JWT tokens from Supabase are validated on every protected request.

**Key modules:** Analytics endpoints, inference orchestration, auth middleware, caching layer.

### 3. Machine Learning Inference Engine (PyTorch)

**Technology:** PyTorch 2.5 (CPU), NumPy, Safetensors

The inference engine loads trained Seq2Point neural network models (CNN, Transformer, UNet architectures) and processes sliding windows of aggregate power data to predict individual appliance consumption. Models are loaded on startup and cached in memory for low-latency inference (<2 seconds per request). Each appliance has a dedicated trained model.

**Key modules:** Model registry, window preprocessing, confidence scoring, result aggregation.

### 4. Time-Series Database (InfluxDB)

**Technology:** InfluxDB 2.8

InfluxDB stores all high-frequency energy readings and prediction results. Two primary buckets are used: `raw_sensor_data` for ingested smart meter readings and `predictions` for disaggregated appliance predictions with confidence scores. Data is tagged by building ID and appliance name for efficient querying.

**Schema:** Measurement `appliance_prediction` with tags (building_id, appliance_name) and fields (predicted_kw, confidence).

### 5. Authentication & Metadata (Supabase)

**Technology:** Supabase (PostgreSQL + Auth)

Supabase handles user authentication (email/password, OAuth), user profiles, and relational metadata including buildings, appliances, and model registries. Row-Level Security (RLS) policies ensure users can only access their own data. The frontend uses the Supabase JS SDK; the backend validates JWTs using Supabase's public keys.

**Schema:** profiles, buildings, appliances, building_appliances, appliance_models.

---

## Data & Communication Flow

1. **User Authentication:** Browser → Supabase Auth → JWT issued → stored in frontend.

2. **Data Request:** Frontend sends JWT in Authorization header → Backend validates signature → queries InfluxDB → returns JSON response.

3. **Inference Request:** Frontend/Scheduler triggers POST to `/infer` → Backend retrieves power window from InfluxDB → forwards to PyTorch engine → predictions returned and persisted to InfluxDB.

4. **Real-time Updates:** Frontend polls backend at configurable intervals (default: 30 seconds) or on user-triggered refresh.

---

## External Dependencies

| Dependency | Purpose | Type |
|------------|---------|------|
| **Cloudflare Pages** | Frontend hosting & CDN | Deployment |
| **Railway** | Backend hosting & managed services | Deployment |
| **InfluxDB Cloud** | Managed time-series database | Database |
| **Supabase** | Managed PostgreSQL + Auth | Database/Auth |
| **Redis** | Caching layer (optional) | Performance |

---

## Diagram Description (For Manual Drawing)

**Suggested Diagram: Component Diagram (UML-style)**

Draw the following components as boxes with labeled connections:

```
[User/Browser] 
    ↓ HTTPS
[React SPA] ←--Supabase SDK--→ [Supabase Auth]
    ↓ REST API + JWT
[FastAPI Backend] ←--Supabase Client--→ [Supabase Postgres]
    ↓                    ↓
[PyTorch Engine]    [InfluxDB]
    ↓                    ↑
    └── predictions ─────┘
```

**Label the connections:**
- Browser → React: HTTPS (port 443)
- React → Supabase Auth: Auth SDK (login/session)
- React → FastAPI: REST API with JWT header
- FastAPI → Supabase Postgres: Metadata queries
- FastAPI → InfluxDB: Time-series read/write
- FastAPI → PyTorch: Inference calls (in-process)
- PyTorch → InfluxDB: Prediction storage

---

*This document provides a high-level technical overview. For detailed specifications, refer to the full documentation in the `/docs` directory.*
