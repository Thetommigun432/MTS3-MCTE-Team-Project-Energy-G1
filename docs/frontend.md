# Frontend Documentation (`apps/web`)

## 1. Overview
The Frontend is a **React 19** Single Page Application (SPA) built with **Vite 7**. It serves as the primary user interface for the NILM Energy Monitor, visualizing real-time energy data and disaggregated appliance usage.

- **URL**: `http://localhost:8080` (Dev), Production URL via Cloudflare.
- **Key Features**: Real-time charts, Supabase Auth integration, Demo mode, Dark/Light theme.

## 2. Technology Stack
- **Core**: React 19, TypeScript 5.9
- **Build**: Vite 7 (ESBuild)
- **Styling**: Tailwind CSS v3, `shadcn/ui` (Radix Primitives + `class-variance-authority`)
- **State Management**: React Context (`AuthContext`, `EnergyContext`)
- **Data Visualization**: Recharts (ResponsiveContainer, AreaChart, BarChart)
- **Protocol**: REST API (via Axios) + Supabase Client

## 3. Architecture & Directory Structure

```
apps/web/src/
├── components/
│   ├── auth/          # ProtectedRoute, AuthGuard components
│   ├── brand/         # Logos and branding assets
│   ├── layout/        # LayoutShell, Sidebar, Navbar
│   ├── nilm/          # Domain visualizations (RealtimeChart, ApplianceList)
│   └── ui/            # Reusable primitives (Buttons, Cards, Inputs)
├── contexts/
│   ├── AuthContext.tsx    # Supabase Session, User, Login/Logout methods
│   ├── EnergyContext.tsx  # Global energy data, selected building, date range
│   └── ThemeContext.tsx   # Color theme management
├── hooks/
│   ├── useAuth.ts            # Consumes AuthContext
│   ├── useBuildings.ts       # Fetches user's buildings
│   ├── useEnergyData.ts      # Fetches readings/predictions
│   └── useManagedAppliances.ts # Managed appliance state
├── pages/
│   ├── app/           # Authenticated routes (Dashboard, Settings)
│   └── auth/          # Public auth routes (Login, Signup, Verify)
├── services/
│   ├── api.ts         # Base Axios instance with Interceptors
│   └── energy.ts      # Typed API methods (getReadings, getPredictions)
└── lib/               # Utilities (Dates, Formatters, Env validation)
```

## 4. State Management Strategy

### AuthContext
- **Source of Truth**: Supabase `onAuthStateChange` listener.
- **Behavior**:
  - Initializes session on mount.
  - Provides `user`, `session`, `signIn`, `signOut`.
  - Redirects to `/login` if unauthenticated (via ProtectedRoute).

### EnergyContext
- **Scope**: Data necessary for the NILM dashboard.
- **State**:
  - `dateRange`: { start, end }
  - `selectedBuilding`: Current building ID.
  - `readings`: Aggregated power data.
  - `predictions`: Disaggregated appliance data.
- **Actions**: `refresh()`, `setBuilding()`.

## 5. Environment & Configuration

Environment variables are typed and validated in `src/lib/env.ts`.

| Variable | Description |
|----------|-------------|
| `VITE_SUPABASE_URL` | Supabase Project URL. |
| `VITE_SUPABASE_ANON_KEY` | Public API Key. |
| `VITE_BACKEND_URL` | Production Backend URL. In Dev, this is empty (proxy used). |
| `VITE_DEMO_MODE` | `true` bypasses Auth and uses static CSV data. |
| `VITE_LOCAL_MODE` | *(Deprecated)* Use local InfluxDB directly. |

## 6. Development Workflow

### Proxy Configuration
In `vite.config.ts`, requests to `/api/*` are proxied to `http://localhost:8000`.
This avoids CORS issues during local development.

```typescript
// vite.config.ts
proxy: {
  '/api': {
    target: 'http://localhost:8000',
    changeOrigin: true,
    rewrite: (path) => path.replace(/^\/api/, '')
  }
}
```

### Scripts
- `npm run dev`: Start dev server (Port 8080).
- `npm run build`: Production build (Output: `dist/`).
- `npm run typecheck`: Run `tsc --noEmit`.
- `npm run test`: Run Vitest suite.

## 7. Authentication Flow
1.  **Login**: User submits credentials to Supabase via `AuthContext`.
2.  **Session**: Supabase returns a `session` object containing a JWT (`access_token`).
3.  **API Requests**: `services/api.ts` interceptor attaches `Authorization: Bearer <token>` to every request.
4.  **Expiry**: Supabase client auto-refreshes the token.

## 8. Deployment (Cloudflare Pages)
- **Build Command**: `npm ci && npm run build`
- **Output Directory**: `dist`
- **Routing**: SPA routing is handled by `public/_redirects` file containing `/* /index.html 200`.
