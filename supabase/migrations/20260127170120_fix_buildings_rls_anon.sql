-- ============================================
-- MIGRATION: Fix Buildings RLS for Anon Access
-- Date: 2026-01-27
-- Purpose: Allow anonymous users to see demo buildings
--          Fix potential RLS recursion issues
-- ============================================

-- ============================================
-- 1. DROP EXISTING POLICIES (clean slate)
-- ============================================
DROP POLICY IF EXISTS "buildings_select" ON public.buildings;
DROP POLICY IF EXISTS "buildings_select_anon" ON public.buildings;
DROP POLICY IF EXISTS "buildings_select_demo" ON public.buildings;
DROP POLICY IF EXISTS "buildings_insert" ON public.buildings;
DROP POLICY IF EXISTS "buildings_update" ON public.buildings;
DROP POLICY IF EXISTS "buildings_delete" ON public.buildings;

DROP POLICY IF EXISTS "building_members_select" ON public.building_members;
DROP POLICY IF EXISTS "building_members_insert" ON public.building_members;
DROP POLICY IF EXISTS "building_members_delete" ON public.building_members;

-- ============================================
-- 2. BUILDING_MEMBERS POLICIES (NON-RECURSIVE)
-- Must be simple to avoid circular dependency with buildings
-- ============================================

-- SELECT: Users can see their own memberships only
CREATE POLICY "building_members_select" ON public.building_members
  FOR SELECT TO authenticated
  USING (user_id = auth.uid());

-- INSERT: Building owners can add members (checked via buildings table)
CREATE POLICY "building_members_insert" ON public.building_members
  FOR INSERT TO authenticated
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.buildings b
      WHERE b.id = building_id AND b.user_id = auth.uid()
    )
  );

-- DELETE: Building owners or self can remove membership
CREATE POLICY "building_members_delete" ON public.building_members
  FOR DELETE TO authenticated
  USING (
    user_id = auth.uid()
    OR EXISTS (
      SELECT 1 FROM public.buildings b
      WHERE b.id = building_id AND b.user_id = auth.uid()
    )
  );

-- ============================================
-- 3. BUILDINGS POLICIES (ANON + AUTHENTICATED)
-- ============================================

-- SELECT for ANON: Only demo buildings visible to anonymous users
CREATE POLICY "buildings_select_anon" ON public.buildings
  FOR SELECT TO anon
  USING (is_demo = TRUE);

-- SELECT for AUTHENTICATED: Demo buildings OR owned OR member access
-- Using a simple EXISTS that doesn't recurse back to buildings
CREATE POLICY "buildings_select_auth" ON public.buildings
  FOR SELECT TO authenticated
  USING (
    is_demo = TRUE
    OR user_id = auth.uid()
    OR EXISTS (
      SELECT 1 FROM public.building_members bm
      WHERE bm.building_id = id AND bm.user_id = auth.uid()
    )
  );

-- INSERT: Authenticated users can create buildings (owned by them or unowned)
CREATE POLICY "buildings_insert" ON public.buildings
  FOR INSERT TO authenticated
  WITH CHECK (user_id IS NULL OR user_id = auth.uid());

-- UPDATE: Only building owner can update
CREATE POLICY "buildings_update" ON public.buildings
  FOR UPDATE TO authenticated
  USING (user_id = auth.uid())
  WITH CHECK (user_id = auth.uid());

-- DELETE: Only building owner can delete
CREATE POLICY "buildings_delete" ON public.buildings
  FOR DELETE TO authenticated
  USING (user_id = auth.uid());

-- ============================================
-- 4. ENSURE DEMO BUILDINGS EXIST (IDEMPOTENT)
-- ============================================

-- Ensure demo building exists
INSERT INTO public.buildings (id, name, address, description, status, is_demo, stream_key, created_at)
VALUES 
  ('demo-residential-001', 'Demo Residential', '123 Demo Street', 'Demo building for testing', 'active', TRUE, 'demo-residential-001', NOW())
ON CONFLICT (id) DO UPDATE SET
  is_demo = TRUE,
  name = EXCLUDED.name;

-- Ensure building-1 exists (used by InfluxDB data) - NOT a demo building
INSERT INTO public.buildings (id, name, address, description, status, is_demo, stream_key, created_at)
VALUES 
  ('building-1', 'Building 1', 'Main Facility', 'Primary monitored building', 'active', FALSE, 'building-1', NOW())
ON CONFLICT (id) DO UPDATE SET
  is_demo = FALSE,
  name = EXCLUDED.name;

-- ============================================
-- 5. NOTIFY POSTGREST TO RELOAD SCHEMA
-- ============================================
NOTIFY pgrst, 'reload schema';
