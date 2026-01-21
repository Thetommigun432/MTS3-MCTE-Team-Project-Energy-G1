-- ============================================
-- MIGRATION: Create Demo User for Presentations
-- Date: 2026-01-15
-- Purpose: Create a demo account that teachers can use to view the dashboard
-- ============================================

-- Create demo user using Supabase's auth.users
-- Note: This uses the admin API approach via a function
-- The actual user creation happens via the ensure-demo-user edge function
-- This migration just ensures the profile exists if the user is created manually

-- First, let's create a function that can be called to set up demo data
CREATE OR REPLACE FUNCTION public.setup_demo_user(
  demo_user_id UUID,
  demo_email TEXT DEFAULT 'demo@energy-monitor.app'
)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  demo_building_id UUID;
  appliance_record RECORD;
BEGIN
  -- Ensure profile exists
  INSERT INTO public.profiles (id, email, display_name, role)
  VALUES (demo_user_id, demo_email, 'Demo User', 'admin')
  ON CONFLICT (id) DO UPDATE SET
    email = EXCLUDED.email,
    display_name = COALESCE(profiles.display_name, EXCLUDED.display_name),
    role = 'admin';

  -- Check if demo building exists for this user
  SELECT id INTO demo_building_id
  FROM public.buildings
  WHERE owner_id = demo_user_id
  LIMIT 1;

  -- Create demo building if it doesn't exist
  IF demo_building_id IS NULL THEN
    INSERT INTO public.buildings (name, address, timezone, owner_id)
    VALUES ('Demo Building', '123 Energy Street, Amsterdam', 'Europe/Amsterdam', demo_user_id)
    RETURNING id INTO demo_building_id;
  END IF;

  -- Add appliances to the demo building (if not already added)
  FOR appliance_record IN SELECT id FROM public.appliances LOOP
    INSERT INTO public.building_appliances (building_id, appliance_id, is_active)
    VALUES (demo_building_id, appliance_record.id, true)
    ON CONFLICT (building_id, appliance_id) DO NOTHING;
  END LOOP;

  -- Generate some sample predictions for the demo building (last 7 days)
  -- Only if no predictions exist
  IF NOT EXISTS (
    SELECT 1 FROM public.disaggregation_predictions 
    WHERE building_id = demo_building_id
    LIMIT 1
  ) THEN
    INSERT INTO public.disaggregation_predictions (building_id, appliance_id, ts, predicted_kw, confidence)
    SELECT 
      demo_building_id,
      a.id,
      generate_series(
        now() - interval '7 days',
        now(),
        interval '15 minutes'
      ) as ts,
      -- Random power values based on typical appliance power
      ROUND((a.typical_power_kw * (0.5 + random() * 0.5))::numeric, 4),
      ROUND((0.7 + random() * 0.3)::numeric, 3) as confidence
    FROM public.appliances a
    ON CONFLICT DO NOTHING;
  END IF;

END;
$$;

-- Comment for documentation
COMMENT ON FUNCTION public.setup_demo_user(UUID, TEXT) IS 
  'Sets up demo data for a user including profile, building, appliances, and sample predictions. Call after creating demo user via auth.';
