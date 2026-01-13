-- ============================================
-- SUPABASE DATABASE SCHEMA FOR ENERGY MONITORING APP
-- Execute this in Supabase Dashboard → SQL Editor
-- ============================================

-- ============================================
-- PROFILES (extends auth.users)
-- ============================================
CREATE TABLE IF NOT EXISTS public.profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  username TEXT UNIQUE,
  full_name TEXT,
  display_name TEXT,
  avatar_url TEXT,
  role TEXT DEFAULT 'user' CHECK (role IN ('user', 'admin', 'viewer')),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Auto-create profile on signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (id, username, full_name, display_name)
  VALUES (
    NEW.id,
    NEW.raw_user_meta_data->>'username',
    NEW.raw_user_meta_data->>'full_name',
    NEW.raw_user_meta_data->>'display_name'
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- ============================================
-- BUILDINGS
-- ============================================
CREATE TABLE IF NOT EXISTS public.buildings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  address TEXT,
  timezone TEXT DEFAULT 'UTC',
  owner_id UUID REFERENCES public.profiles(id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- APPLIANCES (reference data)
-- ============================================
CREATE TABLE IF NOT EXISTS public.appliances (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  category TEXT CHECK (category IN ('kitchen', 'laundry', 'hvac', 'lighting', 'electronics', 'other')),
  typical_power_kw NUMERIC(6,3),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Seed common appliances
INSERT INTO public.appliances (name, category, typical_power_kw) VALUES
  ('Dishwasher', 'kitchen', 1.8),
  ('Washing Machine', 'laundry', 0.5),
  ('Dryer', 'laundry', 3.0),
  ('Refrigerator', 'kitchen', 0.15),
  ('HVAC', 'hvac', 3.5),
  ('Water Heater', 'other', 4.5),
  ('Oven', 'kitchen', 2.5),
  ('Microwave', 'kitchen', 1.2)
ON CONFLICT DO NOTHING;

-- ============================================
-- BUILDING_APPLIANCES (which appliances in which building)
-- ============================================
CREATE TABLE IF NOT EXISTS public.building_appliances (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  building_id UUID NOT NULL REFERENCES public.buildings(id) ON DELETE CASCADE,
  appliance_id UUID NOT NULL REFERENCES public.appliances(id) ON DELETE CASCADE,
  alias TEXT,  -- e.g., "Kitchen Dishwasher"
  is_active BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(building_id, appliance_id)
);

-- ============================================
-- ML MODELS
-- ============================================
CREATE TABLE IF NOT EXISTS public.appliance_models (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  appliance_id UUID NOT NULL REFERENCES public.appliances(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  description TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.model_versions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_id UUID NOT NULL REFERENCES public.appliance_models(id) ON DELETE CASCADE,
  version TEXT NOT NULL,
  artifact_path TEXT,  -- Storage bucket path
  metrics JSONB,       -- {"accuracy": 0.95, "f1": 0.92}
  is_active BOOLEAN DEFAULT FALSE,
  trained_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(model_id, version)
);

-- ============================================
-- PREDICTIONS (from NILM model)
-- ============================================
CREATE TABLE IF NOT EXISTS public.disaggregation_predictions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  building_id UUID NOT NULL REFERENCES public.buildings(id) ON DELETE CASCADE,
  appliance_id UUID NOT NULL REFERENCES public.appliances(id) ON DELETE CASCADE,
  model_version_id UUID REFERENCES public.model_versions(id) ON DELETE SET NULL,
  ts TIMESTAMPTZ NOT NULL,
  predicted_kw NUMERIC(8,4) NOT NULL,
  confidence NUMERIC(4,3) CHECK (confidence >= 0 AND confidence <= 1),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for time-series queries
CREATE INDEX IF NOT EXISTS idx_predictions_building_time
  ON public.disaggregation_predictions(building_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_building_appliance_time
  ON public.disaggregation_predictions(building_id, appliance_id, ts DESC);

-- Prevent duplicate predictions for same timestamp
CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_unique
  ON public.disaggregation_predictions(building_id, appliance_id, ts, model_version_id);

-- ============================================
-- INFERENCE RUNS (batch job tracking)
-- ============================================
CREATE TABLE IF NOT EXISTS public.inference_runs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  building_id UUID NOT NULL REFERENCES public.buildings(id) ON DELETE CASCADE,
  status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ,
  error_message TEXT,
  predictions_count INT DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- ENABLE ROW LEVEL SECURITY (RLS)
-- ============================================
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.buildings ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.appliances ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.building_appliances ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.appliance_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.model_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.disaggregation_predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.inference_runs ENABLE ROW LEVEL SECURITY;

-- ============================================
-- RLS POLICIES
-- ============================================

-- Profiles: all authenticated users can view profiles (for team page)
DROP POLICY IF EXISTS "Users can view own profile" ON public.profiles;
DROP POLICY IF EXISTS "All authenticated users can view profiles" ON public.profiles;
CREATE POLICY "All authenticated users can view profiles"
  ON public.profiles FOR SELECT
  TO authenticated
  USING (true);

-- Profiles: users can only update their own profile
DROP POLICY IF EXISTS "Users can update own profile" ON public.profiles;
CREATE POLICY "Users can update own profile"
  ON public.profiles FOR UPDATE
  USING (auth.uid() = id);

-- Buildings: owner can CRUD
DROP POLICY IF EXISTS "Owners can manage their buildings" ON public.buildings;
CREATE POLICY "Owners can manage their buildings"
  ON public.buildings FOR ALL
  USING (auth.uid() = owner_id);

-- Appliances: everyone can read (reference data)
DROP POLICY IF EXISTS "Anyone can read appliances" ON public.appliances;
CREATE POLICY "Anyone can read appliances"
  ON public.appliances FOR SELECT
  TO authenticated
  USING (true);

-- Building Appliances: based on building ownership
DROP POLICY IF EXISTS "Building owners can manage appliances" ON public.building_appliances;
CREATE POLICY "Building owners can manage appliances"
  ON public.building_appliances FOR ALL
  USING (
    EXISTS (
      SELECT 1 FROM public.buildings
      WHERE id = building_appliances.building_id
      AND owner_id = auth.uid()
    )
  );

-- Predictions: based on building ownership
DROP POLICY IF EXISTS "Building owners can read predictions" ON public.disaggregation_predictions;
CREATE POLICY "Building owners can read predictions"
  ON public.disaggregation_predictions FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.buildings
      WHERE id = disaggregation_predictions.building_id
      AND owner_id = auth.uid()
    )
  );

-- Models: authenticated users can read
DROP POLICY IF EXISTS "Authenticated can read models" ON public.appliance_models;
CREATE POLICY "Authenticated can read models"
  ON public.appliance_models FOR SELECT
  TO authenticated
  USING (true);

DROP POLICY IF EXISTS "Authenticated can read model versions" ON public.model_versions;
CREATE POLICY "Authenticated can read model versions"
  ON public.model_versions FOR SELECT
  TO authenticated
  USING (true);

-- Inference runs: based on building ownership
DROP POLICY IF EXISTS "Building owners can manage inference runs" ON public.inference_runs;
CREATE POLICY "Building owners can manage inference runs"
  ON public.inference_runs FOR ALL
  USING (
    EXISTS (
      SELECT 1 FROM public.buildings
      WHERE id = inference_runs.building_id
      AND owner_id = auth.uid()
    )
  );

-- ============================================
-- VERIFICATION QUERIES
-- ============================================

-- Check all tables exist
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;

-- Check RLS is enabled
SELECT tablename, rowsecurity FROM pg_tables
WHERE schemaname = 'public';
