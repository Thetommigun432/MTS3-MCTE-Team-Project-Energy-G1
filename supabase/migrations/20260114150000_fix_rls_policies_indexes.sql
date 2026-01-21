-- ============================================
-- MIGRATION: Fix RLS Performance + Foreign Key Indexes
-- Date: 2026-01-14
-- Fixes:
--   - unindexed_foreign_keys (PERFORMANCE)
--   - multiple_permissive_policies (PERFORMANCE)
--   - rls_policy_always_true on invitations (SECURITY)
-- ============================================

-- ============================================
-- 1. ADD MISSING INDEXES FOR FOREIGN KEYS
-- ============================================

-- invitations.invited_by foreign key
CREATE INDEX IF NOT EXISTS idx_invitations_invited_by 
  ON public.invitations(invited_by);

-- invitations.org_id foreign key
CREATE INDEX IF NOT EXISTS idx_invitations_org_id 
  ON public.invitations(org_id);

-- organizations.created_by foreign key
CREATE INDEX IF NOT EXISTS idx_organizations_created_by 
  ON public.organizations(created_by);

-- pending_org_invites.invited_by foreign key
CREATE INDEX IF NOT EXISTS idx_pending_org_invites_invited_by 
  ON public.pending_org_invites(invited_by);

-- ============================================
-- 2. FIX MULTIPLE PERMISSIVE POLICIES - invitations
-- Consolidate into single SELECT policy
-- ============================================

DROP POLICY IF EXISTS "invitations_admin_select" ON public.invitations;
DROP POLICY IF EXISTS "invitations_admin_manage" ON public.invitations;

-- Single SELECT policy
CREATE POLICY "invitations_select"
  ON public.invitations
  FOR SELECT
  TO authenticated
  USING (
    invited_by = (SELECT auth.uid())
    OR
    org_id IN (
      SELECT om.org_id 
      FROM public.org_members om 
      WHERE om.user_id = (SELECT auth.uid()) 
        AND om.role = 'admin'
    )
  );

-- Separate INSERT policy (org admins only)
CREATE POLICY "invitations_insert"
  ON public.invitations
  FOR INSERT
  TO authenticated
  WITH CHECK (
    invited_by = (SELECT auth.uid())
    AND (
      org_id IS NULL 
      OR org_id IN (
        SELECT om.org_id 
        FROM public.org_members om 
        WHERE om.user_id = (SELECT auth.uid()) 
          AND om.role = 'admin'
      )
    )
  );

-- Separate UPDATE policy (org admins or inviter)
CREATE POLICY "invitations_update"
  ON public.invitations
  FOR UPDATE
  TO authenticated
  USING (
    invited_by = (SELECT auth.uid())
    OR
    org_id IN (
      SELECT om.org_id 
      FROM public.org_members om 
      WHERE om.user_id = (SELECT auth.uid()) 
        AND om.role = 'admin'
    )
  )
  WITH CHECK (
    invited_by = (SELECT auth.uid())
    OR
    org_id IN (
      SELECT om.org_id 
      FROM public.org_members om 
      WHERE om.user_id = (SELECT auth.uid()) 
        AND om.role = 'admin'
    )
  );

-- Separate DELETE policy
CREATE POLICY "invitations_delete"
  ON public.invitations
  FOR DELETE
  TO authenticated
  USING (
    invited_by = (SELECT auth.uid())
    OR
    org_id IN (
      SELECT om.org_id 
      FROM public.org_members om 
      WHERE om.user_id = (SELECT auth.uid()) 
        AND om.role = 'admin'
    )
  );

-- ============================================
-- 3. FIX MULTIPLE PERMISSIVE POLICIES - org_members
-- The org_members_admin_manage policy uses FOR ALL which includes SELECT
-- Change it to only cover INSERT/UPDATE/DELETE
-- ============================================

DROP POLICY IF EXISTS "org_members_admin_manage" ON public.org_members;

-- Admin INSERT policy
CREATE POLICY "org_members_admin_insert"
  ON public.org_members
  FOR INSERT
  TO authenticated
  WITH CHECK (
    org_id IN (
      SELECT om.org_id 
      FROM public.org_members om 
      WHERE om.user_id = (SELECT auth.uid()) 
        AND om.role = 'admin'
    )
    OR
    -- Allow users to join org if they created it
    user_id = (SELECT auth.uid())
  );

-- Admin UPDATE policy
CREATE POLICY "org_members_admin_update"
  ON public.org_members
  FOR UPDATE
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

-- Admin DELETE policy
CREATE POLICY "org_members_admin_delete"
  ON public.org_members
  FOR DELETE
  TO authenticated
  USING (
    org_id IN (
      SELECT om.org_id 
      FROM public.org_members om 
      WHERE om.user_id = (SELECT auth.uid()) 
        AND om.role = 'admin'
    )
  );

-- ============================================
-- 4. FIX MULTIPLE PERMISSIVE POLICIES - pending_org_invites
-- Consolidate into single SELECT policy
-- ============================================

DROP POLICY IF EXISTS "pending_invites_admin_select" ON public.pending_org_invites;
DROP POLICY IF EXISTS "pending_invites_admin_manage" ON public.pending_org_invites;

-- Single SELECT policy
CREATE POLICY "pending_invites_select"
  ON public.pending_org_invites
  FOR SELECT
  TO authenticated
  USING (
    org_id IN (
      SELECT om.org_id 
      FROM public.org_members om 
      WHERE om.user_id = (SELECT auth.uid()) 
        AND om.role = 'admin'
    )
  );

-- INSERT policy
CREATE POLICY "pending_invites_insert"
  ON public.pending_org_invites
  FOR INSERT
  TO authenticated
  WITH CHECK (
    invited_by = (SELECT auth.uid())
    AND org_id IN (
      SELECT om.org_id 
      FROM public.org_members om 
      WHERE om.user_id = (SELECT auth.uid()) 
        AND om.role = 'admin'
    )
  );

-- UPDATE policy
CREATE POLICY "pending_invites_update"
  ON public.pending_org_invites
  FOR UPDATE
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

-- DELETE policy
CREATE POLICY "pending_invites_delete"
  ON public.pending_org_invites
  FOR DELETE
  TO authenticated
  USING (
    org_id IN (
      SELECT om.org_id 
      FROM public.org_members om 
      WHERE om.user_id = (SELECT auth.uid()) 
        AND om.role = 'admin'
    )
  );

-- ============================================
-- NOTE: Unused Indexes
-- ============================================
-- The unused_index warnings are INFO level only.
-- These indexes may become used as the app gets traffic.
-- Do NOT drop them based solely on the linter warning.
-- Monitor pg_stat_user_indexes after traffic accumulates.

-- ============================================
-- NOTE: Leaked Password Protection
-- ============================================
-- This setting must be enabled in the Supabase Dashboard:
-- Authentication → Settings → Enable "Leaked Password Protection"
-- Cannot be changed via SQL migration.
