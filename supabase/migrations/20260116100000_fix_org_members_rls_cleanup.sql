-- ============================================
-- FIX: Clean up duplicate RLS policies on org_members
-- and fix auth.uid() to use (select auth.uid()) pattern
-- ============================================

-- 1. Drop the duplicate policies we created
DROP POLICY IF EXISTS "Users can add themselves to org" ON public.org_members;
DROP POLICY IF EXISTS "Users can view their own membership" ON public.org_members;

-- 2. Check if optimized policies already exist, if not create them
-- The existing policies are:
-- - org_members_select_authenticated (for SELECT)
-- - org_members_admin_insert (for INSERT by admins)

-- We need to ensure users can INSERT themselves when creating their first org
-- Drop the admin-only insert policy and create a combined one
DROP POLICY IF EXISTS "org_members_admin_insert" ON public.org_members;

-- Create a single INSERT policy that allows:
-- 1. Users to add themselves (for first org creation)
-- 2. Admins to add anyone to their org
CREATE POLICY "org_members_insert_policy"
  ON public.org_members
  FOR INSERT
  TO authenticated
  WITH CHECK (
    -- Users can always add themselves
    user_id = (SELECT auth.uid())
    OR
    -- Admins can add anyone to their org
    org_id IN (
      SELECT om.org_id 
      FROM public.org_members om 
      WHERE om.user_id = (SELECT auth.uid()) 
      AND om.role = 'admin'
    )
  );

-- Ensure the SELECT policy uses proper pattern (should already exist but recreate to be safe)
DROP POLICY IF EXISTS "org_members_select_authenticated" ON public.org_members;
CREATE POLICY "org_members_select_authenticated"
  ON public.org_members
  FOR SELECT
  TO authenticated
  USING (
    -- Users can see their own memberships
    user_id = (SELECT auth.uid())
    OR
    -- Users can see members of orgs they belong to
    org_id IN (
      SELECT om.org_id 
      FROM public.org_members om 
      WHERE om.user_id = (SELECT auth.uid())
    )
  );
