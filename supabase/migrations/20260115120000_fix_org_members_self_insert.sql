-- ============================================
-- FIX: Allow users to add themselves to org_members after creating an org
-- This fixes the chicken-and-egg RLS issue where:
-- 1. User creates an organization (allowed by policy)
-- 2. User cannot add themselves as member (blocked by RLS)
-- ============================================

-- Allow authenticated users to insert themselves as members
-- This is needed when a user creates a new organization
DROP POLICY IF EXISTS "Users can add themselves to org" ON public.org_members;
CREATE POLICY "Users can add themselves to org"
  ON public.org_members
  FOR INSERT
  TO authenticated
  WITH CHECK (user_id = auth.uid());

-- Also ensure the SELECT policy works for first-time users
-- (they need to check if they have any memberships)
DROP POLICY IF EXISTS "Users can view their own membership" ON public.org_members;
CREATE POLICY "Users can view their own membership"
  ON public.org_members
  FOR SELECT
  TO authenticated
  USING (user_id = auth.uid());
