-- Fix RLS policy to allow access to buildings with no owner (public buildings)
-- Building 1 has is_demo=FALSE but user_id=NULL, so it needs to be accessible

DROP POLICY IF EXISTS "buildings_select_auth" ON public.buildings;

CREATE POLICY "buildings_select_auth" ON public.buildings
  FOR SELECT TO authenticated
  USING (
    is_demo = TRUE
    OR user_id IS NULL  -- Public buildings (no owner)
    OR user_id = auth.uid()
    OR EXISTS (
      SELECT 1 FROM public.building_members bm
      WHERE bm.building_id = id AND bm.user_id = auth.uid()
    )
  );
