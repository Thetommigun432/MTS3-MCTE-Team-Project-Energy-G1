# Project Overview & Architecture

## 1. Project Goals
The **NILM Energy Monitor** is a real-time Non-Intrusive Load Monitoring application. It uses deep learning to disaggregate total household power consumption into individual appliance usage (e.g., separating "Fridge", "Kettle", "Heat Pump" from a single smart meter reading).

**Key Capabilities:**
- **Real-time Analytics**: Visualize live energy usage and historical trends.
- **Disaggregation Inference**: Run on-demand ML predictions on power windows.
- **Multi-Mode Operation**: Supports Local (Docker/Influx), Demo (static CSV), and Production (Cloud/Supabase) modes.

## 2. System Architecture

The project is a monorepo implementing a split frontend/backend architecture:

```mermaid
graph TD
    User[User / Browser]
    
    subgraph Frontend [Cloudflare Pages]
        UI[React SPA (Vite)]
        Auth[Supabase Auth Client]
    end
    
    subgraph Backend [Railway]
        API[FastAPI Service]
        ML[PyTorch Inference Engine]
    end
    
    subgraph Data [External / Managed]
        SB[Supabase (Postgres + Auth)]
        Influx[InfluxDB (Time Series)]
    end

    User -->|HTTPS| UI
    UI -->|JWT| API
    UI -->|Auth SDK| SB
    
    API -->|Read/Write| Influx
    API -->|Metadata| SB
    API -->|Load Models| ML
```

For implementation details, see:
- [**Frontend Guide**](./frontend.md)
- [**API Reference**](./API.md)
- [**Integration Guide**](./integration.md)

### 2.1 Component Interaction
1.  **Authentication**: User logs in via Frontend using Supabase Auth. A JWT is issued.
2.  **API Requests**: Frontend sends JWT in `Authorization` header to Backend.
3.  **Authorization**: Backend validates JWT signature (RS256/HS256) and checks access against Supabase tables (RBAC).
4.  **Data Processing**:
    - **Readings**: Fetched efficiently from InfluxDB.
    - **Inference**: Power data window sent to PyTorch engine; results persisted to InfluxDB.
5.  **Data Visualization**: Frontend renders charts using Recharts.

## 3. Repository Structure

See `apps/web` and `apps/backend` for source code.

```
MTS3-MCTE-Team-Project-Energy-G1/
├── apps/
│   ├── backend/                # ✅ FastAPI backend (Python)
│   └── web/                    # ✅ React frontend (TS/Vite)
├── docs/                       # ✅ Canonical Documentation
├── checkpoints/                # Model checkpoints
├── data/                       # Processed datasets
├── scripts/                    # Seeding & Utility Scripts
└── training/                   # Training scripts + notebooks
```

## 4. Data Models

### 4.1 InfluxDB Schema (Bucket: `predictions`)
Optimized for time-series access.

- **Measurement**: `appliance_prediction`
- **Tags** (Indexed):
  - `building_id`: UUID
  - `appliance_name`: e.g., "HeatPump"
- **Fields**:
  - `predicted_kw` (float): Power value
  - `confidence` (float): Model confidence score (0-1)
- **Timestamp**: Epoch milliseconds

### 4.2 Supabase Schema (Postgres)
Handles RBAC and metadata.

- `profiles`: User data linked to Auth.
- `buildings`: Buildings owned by users.
- `org_appliances`: Appliance templates (metadata like rated power).
- `building_appliances`: Join table linking buildings to appliances.
