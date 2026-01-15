# NILM Energy Monitor

Real-time Non-Intrusive Load Monitoring (NILM) web application with deep learning-based energy disaggregation.

## 🏗️ Project Structure (Monorepo)

```
/
├── apps/
│   ├── web/              # React + Vite + TypeScript frontend
│   └── local-server/     # Node.js InfluxDB proxy (optional)
├── supabase/             # Supabase migrations + Edge Functions
├── infra/
│   └── influxdb/         # Docker Compose for local InfluxDB
├── data/                 # Training data (gitignored)
├── docs/                 # Documentation
├── preprocessing/        # Python data preparation scripts
├── pretraining/          # ML model notebooks
└── scripts/              # Build/utility scripts
```

## 🚀 Quick Start (Web App)

### Prerequisites
- Node.js 18+
- npm 9+
- Supabase project (for auth/database)

### 1. Install dependencies
```bash
npm install
```

### 2. Configure environment
```bash
# Copy example env file
cp apps/web/.env.example apps/web/.env.local

# Edit with your Supabase credentials
# VITE_SUPABASE_URL=https://your-project.supabase.co
# VITE_SUPABASE_ANON_KEY=your-anon-key
```

### 3. Run development server
```bash
npm run dev
```

The app will be available at `http://localhost:8080`

### 4. Build for production
```bash
npm run build
```

## 📜 Available Scripts

| Script | Description |
|--------|-------------|
| `npm run dev` | Start Vite dev server |
| `npm run build` | Production build |
| `npm run preview` | Preview production build |
| `npm run lint` | Run ESLint |
| `npm run typecheck` | TypeScript type checking |
| `npm run format` | Format code with Prettier |
| `npm run local:server` | Start InfluxDB proxy |
| `npm run local:dev` | Run frontend + local server |

## 🗃️ Supabase Setup

### Deploy Migrations
```bash
cd supabase
supabase link --project-ref <your-project-ref>
supabase db push
```

### Deploy Edge Functions
```bash
supabase functions deploy invite-user-to-org
supabase functions deploy admin-invite
supabase functions deploy log-login-event
# ... deploy other functions as needed
```

### Required Tables
- `profiles` - User profiles
- `organizations` - Multi-tenant orgs
- `org_members` - Org membership + roles
- `pending_org_invites` - Invites for non-registered users
- `invitations` - Invitation history

### Key Edge Functions
- `invite-user-to-org` - Invite users to organizations (admin only)
- `admin-invite` - Legacy global invites
- `log-login-event` - Audit login history

## 🔐 Security Checklist

### Supabase Dashboard Settings
1. **Enable leaked password protection**: Go to Authentication → Settings → Enable "Leaked Password Protection"
2. **Configure email templates**: Authentication → Email Templates
3. **Set up redirect URLs**: Authentication → URL Configuration

### RLS Policies
All tables have Row Level Security enabled with proper policies:
- Users can only view/edit their own data
- Org admins can manage org members
- Service role used only in Edge Functions

### Environment Variables
- **NEVER** commit `.env`, `.env.local`, or `.env.production`
- Use `.env.example` as template
- For CI/CD, use secrets managers

## 🧪 ML Pipeline (NILM Model Training)

### Setup Python Environment
```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Training Workflow
1. **Data Cleaning**: `python clean_excel.py`
2. **Data Exploration**: `python explore_data.py`
3. **Data Preparation**: `python prepare_data.py`
4. **Model Training**: `python train_nilm.py`
5. **Evaluation**: `python predict_nilm.py`

### Model Architecture
- **Approach**: Sequence-to-Point LSTM
- **Input**: 60-timestep consumption sequences
- **Output**: Per-appliance power predictions
- **Strategy**: One model per appliance

## 📖 Additional Documentation

- [Local Development Guide](docs/LOCAL_DEVELOPMENT.md)
- [Supabase Setup Details](docs/SUPABASE_SETUP.md)
- [InfluxDB Schema](docs/INFLUX_SCHEMA.md)
- [Deployment Guide](apps/web/docs/DEPLOYMENT_STEPS.md)
- Dropout for regularization
- Linear output (regression)

## Notes

- Data must be in `data/influxdb_query_20251020_074134_cleaned.xlsx`
- The model requires at least 1000 samples per appliance
- Sequence length can be modified in `prepare_data.py` (default: 60)