-- ============================================
-- COMPLETE SCHEMA MIGRATION: Organizations + Missing Tables
-- Adds organization layer, login_history, and comprehensive indexes
-- ============================================

-- ============================================
-- 1. ORGANIZATIONS (multi-tenant support)
-- ============================================
CREATE TABLE IF NOT EXISTS public.organizations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  slug TEXT UNIQUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Organization members with roles
CREATE TABLE IF NOT EXISTS public.org_members (
  org_id UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  role TEXT NOT NULL DEFAULT 'member' CHECK (role IN ('admin', 'member', 'viewer')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (org_id, user_id)
);

-- Add org_id to buildings (optional, for org-based access)
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.tables 
    WHERE table_schema = 'public' 
    AND table_name = 'buildings'
  ) AND NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_schema = 'public' 
    AND table_name = 'buildings' 
    AND column_name = 'org_id'
  ) THEN
    ALTER TABLE public.buildings ADD COLUMN org_id UUID REFERENCES public.organizations(id) ON DELETE SET NULL;
  END IF;
END $$;

-- Add org_id to org_appliances if not exists
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.tables 
    WHERE table_schema = 'public' 
    AND table_name = 'org_appliances'
  ) AND NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_schema = 'public' 
    AND table_name = 'org_appliances' 
    AND column_name = 'org_id'
  ) THEN
    ALTER TABLE public.org_appliances ADD COLUMN org_id UUID REFERENCES public.organizations(id) ON DELETE SET NULL;
  END IF;
END $$;

-- ============================================
-- 2. LOGIN HISTORY (rename from login_events if needed)
-- ============================================
-- Keep login_events as-is if it exists, but add login_history for service_role inserts
CREATE TABLE IF NOT EXISTS public.login_history (
  id BIGSERIAL PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  event TEXT NOT NULL DEFAULT 'login',
  user_agent TEXT,
  ip_address TEXT,
  device_label TEXT,
  success BOOLEAN DEFAULT true,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================
-- 3. USER SETTINGS ENHANCEMENTS
-- ============================================
-- Add remember_me column if table exists and column doesn't
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.tables 
    WHERE table_schema = 'public' AND table_name = 'user_settings'
  ) AND NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_schema = 'public' 
    AND table_name = 'user_settings' 
    AND column_name = 'remember_me'
  ) THEN
    ALTER TABLE public.user_settings ADD COLUMN remember_me BOOLEAN DEFAULT true;
  END IF;
END $$;

-- Add theme column if table exists and column doesn't
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.tables 
    WHERE table_schema = 'public' AND table_name = 'user_settings'
  ) AND NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_schema = 'public' 
    AND table_name = 'user_settings' 
    AND column_name = 'theme'
  ) THEN
    ALTER TABLE public.user_settings ADD COLUMN theme TEXT DEFAULT 'dark';
  END IF;
END $$;

-- ============================================
-- 4. PROFILE ENHANCEMENTS
-- ============================================
-- Add role column if table exists and column doesn't
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.tables 
    WHERE table_schema = 'public' AND table_name = 'profiles'
  ) AND NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_schema = 'public' 
    AND table_name = 'profiles' 
    AND column_name = 'role'
  ) THEN
    ALTER TABLE public.profiles ADD COLUMN role TEXT DEFAULT 'user' CHECK (role IN ('user', 'admin', 'viewer'));
  END IF;
END $$;

-- ============================================
-- 5. INFERENCE RUNS TABLE (only if buildings exists)
-- ============================================
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.tables 
    WHERE table_schema = 'public' AND table_name = 'buildings'
  ) AND NOT EXISTS (
    SELECT 1 FROM information_schema.tables 
    WHERE table_schema = 'public' AND table_name = 'inference_runs'
  ) THEN
    CREATE TABLE public.inference_runs (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      building_id UUID NOT NULL REFERENCES public.buildings(id) ON DELETE CASCADE,
      status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
      started_at TIMESTAMPTZ,
      finished_at TIMESTAMPTZ,
      notes TEXT,
      error_message TEXT,
      predictions_count INT DEFAULT 0,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
  END IF;
END $$;

-- Add run_id to predictions if both tables exist
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.tables 
    WHERE table_schema = 'public' AND table_name = 'predictions'
  ) AND EXISTS (
    SELECT 1 FROM information_schema.tables 
    WHERE table_schema = 'public' AND table_name = 'inference_runs'
  ) AND NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_schema = 'public' 
    AND table_name = 'predictions' 
    AND column_name = 'run_id'
  ) THEN
    ALTER TABLE public.predictions ADD COLUMN run_id UUID REFERENCES public.inference_runs(id) ON DELETE SET NULL;
  END IF;
END $$;

-- ============================================
-- 6. DISAGGREGATION_PREDICTIONS (only if dependencies exist)
-- ============================================
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.tables 
    WHERE table_schema = 'public' AND table_name = 'buildings'
  ) AND EXISTS (
    SELECT 1 FROM information_schema.tables 
    WHERE table_schema = 'public' AND table_name = 'org_appliances'
  ) AND NOT EXISTS (
    SELECT 1 FROM information_schema.tables 
    WHERE table_schema = 'public' AND table_name = 'disaggregation_predictions'
  ) THEN
    CREATE TABLE public.disaggregation_predictions (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      building_id UUID NOT NULL REFERENCES public.buildings(id) ON DELETE CASCADE,
      appliance_id UUID NOT NULL REFERENCES public.org_appliances(id) ON DELETE CASCADE,
      ts TIMESTAMPTZ NOT NULL,
      predicted_kw NUMERIC(8,4) NOT NULL,
      confidence NUMERIC(4,3) CHECK (confidence >= 0 AND confidence <= 1),
      model_version_id UUID,
      run_id UUID,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE UNIQUE INDEX idx_disagg_unique ON public.disaggregation_predictions(building_id, appliance_id, ts, model_version_id);
  END IF;
END $$;

-- ============================================
-- 7. INDEXES (only create if tables exist)
-- ============================================
-- Organization indexes
CREATE INDEX IF NOT EXISTS idx_org_members_user_id ON public.org_members(user_id);
CREATE INDEX IF NOT EXISTS idx_org_members_org_id ON public.org_members(org_id);

-- Buildings org_id index (only if column exists)
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_schema = 'public' AND table_name = 'buildings' AND column_name = 'org_id'
  ) THEN
    CREATE INDEX IF NOT EXISTS idx_buildings_org_id ON public.buildings(org_id);
  END IF;
END $$;

-- Predictions indexes (only if table exists)
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.tables 
    WHERE table_schema = 'public' AND table_name = 'disaggregation_predictions'
  ) THEN
    CREATE INDEX IF NOT EXISTS idx_disagg_predictions_building_ts 
      ON public.disaggregation_predictions(building_id, ts DESC);
    CREATE INDEX IF NOT EXISTS idx_disagg_predictions_building_appliance_ts 
      ON public.disaggregation_predictions(building_id, appliance_id, ts DESC);
  END IF;
END $$;

-- Login history indexes
CREATE INDEX IF NOT EXISTS idx_login_history_user_id ON public.login_history(user_id);
CREATE INDEX IF NOT EXISTS idx_login_history_created_at ON public.login_history(created_at DESC);

-- ============================================
-- 8. TRIGGERS FOR UPDATED_AT (create function if not exists, then trigger)
-- ============================================
-- First create the function if it doesn't exist
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER 
LANGUAGE plpgsql
SET search_path = public
AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$;

-- Then create the trigger
DROP TRIGGER IF EXISTS update_organizations_updated_at ON public.organizations;
CREATE TRIGGER update_organizations_updated_at
  BEFORE UPDATE ON public.organizations
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();

-- ============================================
-- 9. HELPER FUNCTIONS
-- ============================================

-- Function to check if user is member of an organization
CREATE OR REPLACE FUNCTION public.is_org_member(org_uuid UUID)
RETURNS BOOLEAN AS $$
BEGIN
  RETURN EXISTS (
    SELECT 1 FROM public.org_members
    WHERE org_id = org_uuid AND user_id = auth.uid()
  );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to check if user is admin of an organization
CREATE OR REPLACE FUNCTION public.is_org_admin(org_uuid UUID)
RETURNS BOOLEAN AS $$
BEGIN
  RETURN EXISTS (
    SELECT 1 FROM public.org_members
    WHERE org_id = org_uuid AND user_id = auth.uid() AND role = 'admin'
  );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get user's organizations
CREATE OR REPLACE FUNCTION public.get_user_org_ids()
RETURNS SETOF UUID AS $$
BEGIN
  RETURN QUERY SELECT org_id FROM public.org_members WHERE user_id = auth.uid();
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
