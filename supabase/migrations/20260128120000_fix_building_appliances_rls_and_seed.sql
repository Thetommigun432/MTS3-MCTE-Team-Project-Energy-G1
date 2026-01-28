-- Fix RLS policies for building_appliances to allow access to public buildings (user_id IS NULL)

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

-- Authenticated can read appliances for their buildings, demo buildings, OR public buildings
DROP POLICY IF EXISTS "building_appliances_select_auth" ON public.building_appliances;
CREATE POLICY "building_appliances_select_auth"
  ON public.building_appliances FOR SELECT
  TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM public.buildings b
      WHERE b.id = building_appliances.building_id
      AND (b.is_demo = TRUE OR b.user_id IS NULL OR b.user_id = auth.uid())
    )
  );

-- Authenticated can insert for their own buildings or public buildings
DROP POLICY IF EXISTS "building_appliances_insert_auth" ON public.building_appliances;
CREATE POLICY "building_appliances_insert_auth"
  ON public.building_appliances FOR INSERT
  TO authenticated
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.buildings b
      WHERE b.id = building_appliances.building_id
      AND (b.is_demo = TRUE OR b.user_id IS NULL OR b.user_id = auth.uid())
    )
  );

-- Seed appliances for Building 1 (TCN models use these appliance IDs)
-- First ensure the appliances exist in the appliances table
INSERT INTO public.appliances (name, category, typical_power_kw)
VALUES 
  ('Heat Pump', 'hvac', 3.5),
  ('Dishwasher', 'kitchen', 1.8),
  ('Washing Machine', 'laundry', 0.5),
  ('Dryer', 'laundry', 3.0),
  ('Oven', 'kitchen', 2.5),
  ('Stove', 'kitchen', 3.0),
  ('Range Hood', 'kitchen', 0.2),
  ('EV Charger', 'ev', 7.2),
  ('EV Socket', 'ev', 7.2),
  ('Rainwater Pump', 'other', 0.8)
ON CONFLICT (name) DO UPDATE SET
  category = EXCLUDED.category,
  typical_power_kw = EXCLUDED.typical_power_kw;

-- Link appliances to Building 1 using a CTE to get appliance IDs
INSERT INTO public.building_appliances (building_id, appliance_id, alias, is_active)
SELECT 'building-1', a.id, a.name, true
FROM public.appliances a
WHERE a.name IN (
  'Heat Pump', 'Dishwasher', 'Washing Machine', 'Dryer', 'Oven', 
  'Stove', 'Range Hood', 'EV Charger', 'EV Socket', 'Rainwater Pump'
)
ON CONFLICT (building_id, appliance_id) DO UPDATE SET
  alias = EXCLUDED.alias,
  is_active = EXCLUDED.is_active;
