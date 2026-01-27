-- ============================================
-- MIGRATION: Restore Buildings Table
-- Date: 2026-01-27
-- Purpose: Re-create buildings table that was dropped in schema_cleanup
--          Add demo building support and building_members for access control
-- ============================================

-- ============================================
-- 1. BUILDINGS TABLE
-- Using TEXT id to support existing 'building-1' style IDs from InfluxDB
-- ============================================
CREATE TABLE IF NOT EXISTS public.buildings (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  address TEXT,
  description TEXT,
  timezone TEXT DEFAULT 'UTC',
  status TEXT DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'maintenance')),
  is_demo BOOLEAN DEFAULT FALSE,
  user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  org_id UUID REFERENCES public.organizations(id) ON DELETE SET NULL,
  stream_key TEXT UNIQUE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for common queries
CREATE INDEX IF NOT EXISTS idx_buildings_user_id ON public.buildings(user_id);
CREATE INDEX IF NOT EXISTS idx_buildings_org_id ON public.buildings(org_id);
CREATE INDEX IF NOT EXISTS idx_buildings_is_demo ON public.buildings(is_demo) WHERE is_demo = TRUE;
CREATE INDEX IF NOT EXISTS idx_buildings_stream_key ON public.buildings(stream_key) WHERE stream_key IS NOT NULL;

-- ============================================
-- 2. BUILDING_MEMBERS TABLE
-- For shared access to buildings (beyond owner)
-- ============================================
CREATE TABLE IF NOT EXISTS public.building_members (
  building_id TEXT NOT NULL REFERENCES public.buildings(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  role TEXT DEFAULT 'viewer' CHECK (role IN ('viewer', 'editor', 'admin')),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (building_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_building_members_user_id ON public.building_members(user_id);

-- ============================================
-- 3. ENABLE ROW LEVEL SECURITY
-- ============================================
ALTER TABLE public.buildings ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.building_members ENABLE ROW LEVEL SECURITY;

-- ============================================
-- 4. RLS POLICIES FOR BUILDINGS
-- ============================================

-- Drop existing policies if any
DROP POLICY IF EXISTS "buildings_select" ON public.buildings;
DROP POLICY IF EXISTS "buildings_insert" ON public.buildings;
DROP POLICY IF EXISTS "buildings_update" ON public.buildings;
DROP POLICY IF EXISTS "buildings_delete" ON public.buildings;
DROP POLICY IF EXISTS "Owners can manage their buildings" ON public.buildings;

-- SELECT: Users can see buildings they own, are members of, or demo buildings
CREATE POLICY "buildings_select" ON public.buildings
  FOR SELECT TO authenticated
  USING (
    is_demo = TRUE
    OR user_id = (SELECT auth.uid())
    OR id IN (SELECT building_id FROM public.building_members WHERE user_id = (SELECT auth.uid()))
    OR org_id IN (SELECT org_id FROM public.org_members WHERE user_id = (SELECT auth.uid()))
  );

-- INSERT: Authenticated users can create buildings
CREATE POLICY "buildings_insert" ON public.buildings
  FOR INSERT TO authenticated
  WITH CHECK (
    user_id IS NULL OR user_id = (SELECT auth.uid())
  );

-- UPDATE: Only owner can update (or org admin)
CREATE POLICY "buildings_update" ON public.buildings
  FOR UPDATE TO authenticated
  USING (
    user_id = (SELECT auth.uid())
    OR id IN (SELECT building_id FROM public.building_members WHERE user_id = (SELECT auth.uid()) AND role = 'admin')
    OR org_id IN (SELECT org_id FROM public.org_members WHERE user_id = (SELECT auth.uid()) AND role = 'admin')
  )
  WITH CHECK (
    user_id = (SELECT auth.uid())
    OR id IN (SELECT building_id FROM public.building_members WHERE user_id = (SELECT auth.uid()) AND role = 'admin')
    OR org_id IN (SELECT org_id FROM public.org_members WHERE user_id = (SELECT auth.uid()) AND role = 'admin')
  );

-- DELETE: Only owner can delete
CREATE POLICY "buildings_delete" ON public.buildings
  FOR DELETE TO authenticated
  USING (
    user_id = (SELECT auth.uid())
    OR org_id IN (SELECT org_id FROM public.org_members WHERE user_id = (SELECT auth.uid()) AND role = 'admin')
  );

-- ============================================
-- 5. RLS POLICIES FOR BUILDING_MEMBERS
-- ============================================

DROP POLICY IF EXISTS "building_members_select" ON public.building_members;
DROP POLICY IF EXISTS "building_members_insert" ON public.building_members;
DROP POLICY IF EXISTS "building_members_delete" ON public.building_members;

-- SELECT: See own memberships or memberships for buildings you own
CREATE POLICY "building_members_select" ON public.building_members
  FOR SELECT TO authenticated
  USING (
    user_id = (SELECT auth.uid())
    OR building_id IN (SELECT id FROM public.buildings WHERE user_id = (SELECT auth.uid()))
  );

-- INSERT: Building owners can add members
CREATE POLICY "building_members_insert" ON public.building_members
  FOR INSERT TO authenticated
  WITH CHECK (
    building_id IN (SELECT id FROM public.buildings WHERE user_id = (SELECT auth.uid()))
  );

-- DELETE: Building owners or self can remove membership
CREATE POLICY "building_members_delete" ON public.building_members
  FOR DELETE TO authenticated
  USING (
    user_id = (SELECT auth.uid())
    OR building_id IN (SELECT id FROM public.buildings WHERE user_id = (SELECT auth.uid()))
  );

-- ============================================
-- 6. SEED DEMO AND DEVELOPMENT DATA
-- ============================================

-- Demo building (accessible to all authenticated users via is_demo=TRUE policy)
INSERT INTO public.buildings (id, name, address, description, status, is_demo, stream_key)
VALUES 
  ('demo-residential-001', 'Demo Residential Building', '123 Demo Street', 'Sample building for demonstration purposes', 'active', TRUE, 'demo-residential-001'),
  ('building-1', 'Building 1', 'Main Facility', 'Primary monitored building from InfluxDB data', 'active', FALSE, 'building-1')
ON CONFLICT (id) DO UPDATE SET
  name = EXCLUDED.name,
  is_demo = EXCLUDED.is_demo,
  stream_key = EXCLUDED.stream_key;

-- ============================================
-- 7. UPDATE TRIGGER FOR updated_at
-- ============================================
CREATE OR REPLACE FUNCTION public.update_buildings_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS buildings_updated_at ON public.buildings;
CREATE TRIGGER buildings_updated_at
  BEFORE UPDATE ON public.buildings
  FOR EACH ROW EXECUTE FUNCTION public.update_buildings_updated_at();

-- ============================================
-- 8. NOTIFY POSTGREST TO RELOAD SCHEMA
-- ============================================
NOTIFY pgrst, 'reload schema';

-- ============================================
-- 9. GRANT PERMISSIONS
-- ============================================
GRANT SELECT, INSERT, UPDATE, DELETE ON public.buildings TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.building_members TO authenticated;
GRANT SELECT ON public.buildings TO anon;
