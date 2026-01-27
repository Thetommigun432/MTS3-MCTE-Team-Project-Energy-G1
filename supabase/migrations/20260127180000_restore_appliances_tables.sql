-- ============================================
-- MIGRATION: Restore Appliances Tables
-- Date: 2026-01-27
-- Purpose: Restore appliances and building_appliances tables
--          that were dropped in schema_cleanup migration
-- ============================================

-- ============================================
-- 1. CREATE APPLIANCES TABLE (reference data)
-- ============================================
CREATE TABLE IF NOT EXISTS public.appliances (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL UNIQUE,
  category TEXT CHECK (category IN ('kitchen', 'laundry', 'hvac', 'lighting', 'electronics', 'ev', 'other')),
  typical_power_kw NUMERIC(6,3),
  is_enabled BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.appliances ENABLE ROW LEVEL SECURITY;

-- ============================================
-- 2. SEED APPLIANCES DATA
-- ============================================
INSERT INTO public.appliances (name, category, typical_power_kw, is_enabled) VALUES
  ('Dishwasher', 'kitchen', 1.8, TRUE),
  ('Washing Machine', 'laundry', 0.5, TRUE),
  ('Dryer', 'laundry', 3.0, TRUE),
  ('Refrigerator', 'kitchen', 0.15, TRUE),
  ('Heat Pump', 'hvac', 3.5, TRUE),
  ('Water Heater', 'other', 4.5, TRUE),
  ('Oven', 'kitchen', 2.5, TRUE),
  ('Microwave', 'kitchen', 1.2, TRUE),
  ('EV Charger', 'ev', 7.2, TRUE),
  ('Stove', 'kitchen', 2.0, TRUE)
ON CONFLICT (name) DO NOTHING;

-- ============================================
-- 3. CREATE BUILDING_APPLIANCES TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS public.building_appliances (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  building_id TEXT NOT NULL REFERENCES public.buildings(id) ON DELETE CASCADE,
  appliance_id UUID NOT NULL REFERENCES public.appliances(id) ON DELETE CASCADE,
  alias TEXT,  -- e.g., "Kitchen Dishwasher"
  is_active BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(building_id, appliance_id)
);

-- Enable RLS
ALTER TABLE public.building_appliances ENABLE ROW LEVEL SECURITY;

-- Index for lookups
CREATE INDEX IF NOT EXISTS idx_building_appliances_building 
  ON public.building_appliances(building_id);

-- ============================================
-- 4. RLS POLICIES FOR APPLIANCES
-- Appliances is reference data - allow read for all (anon + authenticated)
-- ============================================
DROP POLICY IF EXISTS "appliances_select_all" ON public.appliances;
CREATE POLICY "appliances_select_all"
  ON public.appliances FOR SELECT
  TO anon, authenticated
  USING (true);

-- Only authenticated users can modify appliances
DROP POLICY IF EXISTS "appliances_insert_auth" ON public.appliances;
CREATE POLICY "appliances_insert_auth"
  ON public.appliances FOR INSERT
  TO authenticated
  WITH CHECK (true);

DROP POLICY IF EXISTS "appliances_update_auth" ON public.appliances;
CREATE POLICY "appliances_update_auth"
  ON public.appliances FOR UPDATE
  TO authenticated
  USING (true);

-- ============================================
-- 5. RLS POLICIES FOR BUILDING_APPLIANCES
-- Allow anon to read demo building appliances
-- Allow authenticated to manage their own building appliances
-- ============================================

-- Anon can read appliances for demo buildings
DROP POLICY IF EXISTS "building_appliances_select_anon" ON public.building_appliances;
CREATE POLICY "building_appliances_select_anon"
  ON public.building_appliances FOR SELECT
  TO anon
  USING (
    EXISTS (
      SELECT 1 FROM public.buildings b
      WHERE b.id = building_appliances.building_id
      AND b.is_demo = TRUE
    )
  );

-- Authenticated can read appliances for their buildings OR demo buildings
DROP POLICY IF EXISTS "building_appliances_select_auth" ON public.building_appliances;
CREATE POLICY "building_appliances_select_auth"
  ON public.building_appliances FOR SELECT
  TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM public.buildings b
      WHERE b.id = building_appliances.building_id
      AND (b.is_demo = TRUE OR b.user_id = auth.uid())
    )
  );

-- Authenticated can insert/update/delete for their own buildings
DROP POLICY IF EXISTS "building_appliances_insert_auth" ON public.building_appliances;
CREATE POLICY "building_appliances_insert_auth"
  ON public.building_appliances FOR INSERT
  TO authenticated
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.buildings b
      WHERE b.id = building_appliances.building_id
      AND (b.is_demo = TRUE OR b.user_id = auth.uid())
    )
  );

DROP POLICY IF EXISTS "building_appliances_update_auth" ON public.building_appliances;
CREATE POLICY "building_appliances_update_auth"
  ON public.building_appliances FOR UPDATE
  TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM public.buildings b
      WHERE b.id = building_appliances.building_id
      AND b.user_id = auth.uid()
    )
  );

DROP POLICY IF EXISTS "building_appliances_delete_auth" ON public.building_appliances;
CREATE POLICY "building_appliances_delete_auth"
  ON public.building_appliances FOR DELETE
  TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM public.buildings b
      WHERE b.id = building_appliances.building_id
      AND b.user_id = auth.uid()
    )
  );

-- ============================================
-- 6. SEED DEMO BUILDING APPLIANCES
-- Link appliances to demo buildings
-- ============================================
INSERT INTO public.building_appliances (building_id, appliance_id, alias, is_active)
SELECT 
  'demo-residential-001',
  a.id,
  a.name,
  TRUE
FROM public.appliances a
WHERE a.name IN ('Dishwasher', 'Washing Machine', 'Heat Pump', 'EV Charger', 'Oven')
ON CONFLICT (building_id, appliance_id) DO NOTHING;

-- Also add to building-1 for testing
INSERT INTO public.building_appliances (building_id, appliance_id, alias, is_active)
SELECT 
  'building-1',
  a.id,
  a.name,
  TRUE
FROM public.appliances a
WHERE a.name IN ('Dishwasher', 'Washing Machine', 'Heat Pump')
ON CONFLICT (building_id, appliance_id) DO NOTHING;
