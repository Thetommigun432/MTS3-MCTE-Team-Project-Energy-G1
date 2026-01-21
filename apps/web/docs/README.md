# NILM Energy Monitor - Frontend

React/TypeScript dashboard for Non-Intrusive Load Monitoring (NILM) energy disaggregation.

## Features

- **Appliance Detection**: See which appliances are currently ON based on total building consumption
- **Energy Disaggregation**: View predicted kW breakdown across ~15 appliances
- **Data Modes**:
  - **Demo Mode**: Uses bundled CSV training data for testing
  - **API Mode**: Live predictions from Supabase backend
  - **Local Mode**: Predictions from local InfluxDB (for development)

## Tech Stack

- **Vite 7** - Build tool and dev server
- **React 19** - UI framework
- **TypeScript 5.9** - Type safety
- **shadcn/ui** - Component library (Radix + Tailwind)
- **Recharts** - Data visualization
- **Supabase** - Auth and database (production)
- **InfluxDB** - Time-series storage (local dev)

---

## Quick Start

### Prerequisites

- Node.js 18+ with npm
- Docker (for local InfluxDB mode)

### Installation

```bash
cd apps/web
npm install
```

### Environment Setup

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

**Required variables for production:**

```env
VITE_SUPABASE_PROJECT_ID="your-project-id"
VITE_SUPABASE_PUBLISHABLE_KEY="your-anon-key"
VITE_SUPABASE_URL="https://your-project.supabase.co"
```

**For local development with InfluxDB:**

```env
VITE_LOCAL_MODE="true"
VITE_LOCAL_API_URL="http://localhost:3001"
```

---

## Development

### Standard Dev (Demo/API Mode)

```bash
npm run dev
```

Opens at http://localhost:8080

### Local InfluxDB Mode

```bash
# 1. Start InfluxDB (from repo root)
docker compose up -d

# 2. Seed predictions
npm run predictions:seed

# 3. Start frontend + local API together
npm run local:dev
```

Access:

- **Dashboard**: http://localhost:8080
- **InfluxDB UI**: http://localhost:8086 (admin / admin12345)

### Verify Local Data

```bash
npm run predictions:verify
```

---

## Available Scripts

| Script                       | Description                       |
| ---------------------------- | --------------------------------- |
| `npm run dev`                | Start Vite dev server             |
| `npm run build`              | Production build                  |
| `npm run preview`            | Preview production build          |
| `npm run lint`               | ESLint check                      |
| `npm run typecheck`          | TypeScript type check             |
| `npm run format`             | Format with Prettier              |
| `npm run format:check`       | Check formatting                  |
| `npm run local:dev`          | Start frontend + local API server |
| `npm run local:server`       | Start local API server only       |
| `npm run predictions:seed`   | Seed InfluxDB with predictions    |
| `npm run predictions:verify` | Verify InfluxDB data              |

---

## Project Structure

```
src/
├── components/         # UI components
│   ├── ui/            # shadcn/ui primitives
│   ├── nilm/          # NILM-specific components
│   ├── layout/        # Navigation, sidebar
│   └── brand/         # Logo, illustrations
├── contexts/          # React contexts (Auth, Energy, Theme)
├── hooks/             # Custom hooks
├── pages/             # Route pages
│   ├── app/           # Protected app pages
│   └── auth/          # Auth flow pages
├── services/          # API service layer
├── types/             # TypeScript types
└── lib/               # Utilities
```

---

## Demo Mode

For presentations, enable demo mode:

```env
VITE_DEMO_MODE="true"
```

Demo credentials:

- **Email**: admin@demo.local
- **Password**: admin123

---

## Building for Production

```bash
npm run build
```

Output is in `dist/`. Deploy to any static host (Azure Static Web Apps, Vercel, Netlify).

For Azure Static Web Apps, see [DEPLOYMENT_STEPS.md](./DEPLOYMENT_STEPS.md).

---

## Documentation

- [Local Development Guide](../docs/LOCAL_DEVELOPMENT.md) - Full local setup with InfluxDB
- [Security Notes](./docs/SECURITY.md) - Auth and API security
- [Supabase Setup](../docs/SUPABASE_SETUP.md) - Backend configuration
