-- ============================================
-- RLS PERFORMANCE OPTIMIZATION MIGRATION
-- Fixes: auth_rls_initplan, multiple_permissive_policies
-- Date: 2026-01-14
-- ============================================

-- ============================================
-- BACKUP: Current policies (for reference)
-- ============================================
-- Run this SELECT to see current policies before migration:
/*
SELECT 
  schemaname,
  tablename,
  policyname,
  permissive,
  roles,
  cmd,
  qual,
  with_check
FROM pg_policies
WHERE schemaname = 'public'
  AND tablename IN ('profiles', 'org_members', 'organizations', 'login_history')
ORDER BY tablename, policyname;
*/

-- ============================================
-- INDEXES: Optimize RLS policy join performance
-- ============================================

-- org_members: Composite indexes for common access patterns
-- Pattern 1: Filter by org_id first, then user_id (for listing org members)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_org_members_org_user 
  ON public.org_members(org_id, user_id);

-- Pattern 2: Filter by user_id first, then org_id (for user's orgs lookup)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_org_members_user_org 
  ON public.org_members(user_id, org_id);

-- Pattern 3: Role-based queries (admin lookups)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_org_members_user_role 
  ON public.org_members(user_id, role);

-- login_history: User lookup optimization
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_login_history_user_created 
  ON public.login_history(user_id, created_at DESC);

-- ============================================
-- PROFILES: Fix auth_rls_initplan + consolidate policies
-- ============================================

-- Drop all existing SELECT policies on profiles
DROP POLICY IF EXISTS "All authenticated users can view profiles" ON public.profiles;
DROP POLICY IF EXISTS "Users can view own profile" ON public.profiles;
DROP POLICY IF EXISTS "Authenticated users can view profiles" ON public.profiles;

-- Single consolidated SELECT policy with (select ...) wrapper
-- This prevents the auth_rls_initplan warning by ensuring auth.uid() is evaluated once
CREATE POLICY "profiles_select_authenticated"
  ON public.profiles
  FOR SELECT
  TO authenticated
  USING (true);  -- All authenticated users can view all profiles

-- Fix UPDATE policy with (select ...) wrapper for auth.uid()
DROP POLICY IF EXISTS "Users can update own profile" ON public.profiles;
CREATE POLICY "profiles_update_own"
  ON public.profiles
  FOR UPDATE
  TO authenticated
  USING (id = (SELECT auth.uid()))
  WITH CHECK (id = (SELECT auth.uid()));

-- ============================================
-- ORG_MEMBERS: Fix multiple_permissive_policies
-- ============================================

-- Drop the multiple SELECT policies that cause the warning
DROP POLICY IF EXISTS "Users can view their org memberships" ON public.org_members;
DROP POLICY IF EXISTS "Users can view fellow org members" ON public.org_members;

-- Single consolidated SELECT policy combining both conditions
-- User can see: 1) their own membership OR 2) any membership in orgs they belong to
CREATE POLICY "org_members_select_authenticated"
  ON public.org_members
  FOR SELECT
  TO authenticated
  USING (
    -- User's own membership
    user_id = (SELECT auth.uid())
    OR
    -- Fellow members in user's orgs (subquery wrapped for initplan fix)
    org_id IN (
      SELECT om.org_id 
      FROM public.org_members om 
      WHERE om.user_id = (SELECT auth.uid())
    )
  );

-- Fix admin management policy with (select ...) wrapper
DROP POLICY IF EXISTS "Org admins can manage members" ON public.org_members;
CREATE POLICY "org_members_admin_manage"
  ON public.org_members
  FOR ALL
  TO authenticated
  USING (
    org_id IN (
      SELECT om.org_id 
      FROM public.org_members om 
      WHERE om.user_id = (SELECT auth.uid()) 
        AND om.role = 'admin'
    )
  )
  WITH CHECK (
    org_id IN (
      SELECT om.org_id 
      FROM public.org_members om 
      WHERE om.user_id = (SELECT auth.uid()) 
        AND om.role = 'admin'
    )
  );

-- ============================================
-- ORGANIZATIONS: Fix auth_rls_initplan
-- ============================================

DROP POLICY IF EXISTS "Users can view their organizations" ON public.organizations;
CREATE POLICY "organizations_select_member"
  ON public.organizations
  FOR SELECT
  TO authenticated
  USING (
    id IN (
      SELECT om.org_id 
      FROM public.org_members om 
      WHERE om.user_id = (SELECT auth.uid())
    )
  );

DROP POLICY IF EXISTS "Org admins can update their organization" ON public.organizations;
CREATE POLICY "organizations_update_admin"
  ON public.organizations
  FOR UPDATE
  TO authenticated
  USING (
    id IN (
      SELECT om.org_id 
      FROM public.org_members om 
      WHERE om.user_id = (SELECT auth.uid()) 
        AND om.role = 'admin'
    )
  )
  WITH CHECK (
    id IN (
      SELECT om.org_id 
      FROM public.org_members om 
      WHERE om.user_id = (SELECT auth.uid()) 
        AND om.role = 'admin'
    )
  );

DROP POLICY IF EXISTS "Authenticated users can create organizations" ON public.organizations;
CREATE POLICY "organizations_insert_authenticated"
  ON public.organizations
  FOR INSERT
  TO authenticated
  WITH CHECK (true);

-- ============================================
-- LOGIN_HISTORY: Fix auth_rls_initplan
-- ============================================

DROP POLICY IF EXISTS "Users can view own login history" ON public.login_history;
CREATE POLICY "login_history_select_own"
  ON public.login_history
  FOR SELECT
  TO authenticated
  USING (user_id = (SELECT auth.uid()));

-- ============================================
-- VERIFY: Check policies after migration
-- ============================================
-- Run this to verify changes:
/*
SELECT 
  tablename,
  policyname,
  permissive,
  cmd
FROM pg_policies
WHERE schemaname = 'public'
  AND tablename IN ('profiles', 'org_members', 'organizations', 'login_history')
ORDER BY tablename, cmd;

-- Expected result:
-- profiles: 1 SELECT policy, 1 UPDATE policy
-- org_members: 1 SELECT policy, 1 ALL policy (for admin)
-- organizations: 1 SELECT, 1 UPDATE, 1 INSERT policy
-- login_history: 1 SELECT policy
*/
