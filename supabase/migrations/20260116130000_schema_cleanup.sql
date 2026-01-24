-- ============================================
-- MIGRATION: Schema Cleanup & Consolidation
-- Date: 2026-01-16
-- Purpose: Remove unused tables, consolidate RLS policies, clean up functions
-- ============================================

-- ============================================
-- 1. DROP UNUSED TABLES
-- These tables were created but never used in the application
-- ============================================

-- Drop tables in correct order (respecting foreign keys)
DROP TABLE IF EXISTS public.disaggregation_predictions CASCADE;
DROP TABLE IF EXISTS public.inference_runs CASCADE;
DROP TABLE IF EXISTS public.predictions CASCADE;
DROP TABLE IF EXISTS public.readings CASCADE;
DROP TABLE IF EXISTS public.building_appliances CASCADE;
DROP TABLE IF EXISTS public.user_settings CASCADE;
DROP TABLE IF EXISTS public.appliances CASCADE;
DROP TABLE IF EXISTS public.buildings CASCADE;
DROP TABLE IF EXISTS public.login_events CASCADE;

-- ============================================
-- 2. DROP UNUSED FUNCTIONS
-- ============================================

DROP FUNCTION IF EXISTS public.setup_demo_user(UUID, TEXT);
DROP FUNCTION IF EXISTS public.is_org_member(UUID);
DROP FUNCTION IF EXISTS public.is_org_admin(UUID);
DROP FUNCTION IF EXISTS public.get_user_org_ids();

-- ============================================
-- 3. CONSOLIDATE PROFILES RLS POLICIES
-- Remove all duplicates and create single clean policies
-- ============================================

-- Drop all existing profile policies
DROP POLICY IF EXISTS "Users can view own profile" ON public.profiles;
DROP POLICY IF EXISTS "Users can update own profile" ON public.profiles;
DROP POLICY IF EXISTS "Users can insert own profile" ON public.profiles;
DROP POLICY IF EXISTS "Users can delete own profile" ON public.profiles;
DROP POLICY IF EXISTS "All authenticated users can view profiles" ON public.profiles;
DROP POLICY IF EXISTS "profiles_select_authenticated" ON public.profiles;
DROP POLICY IF EXISTS "profiles_update_own" ON public.profiles;
DROP POLICY IF EXISTS "Authenticated users can view profiles" ON public.profiles;

-- Create clean profile policies
CREATE POLICY "profiles_select" ON public.profiles
  FOR SELECT TO authenticated USING (true);

CREATE POLICY "profiles_insert" ON public.profiles
  FOR INSERT TO authenticated 
  WITH CHECK (id = (SELECT auth.uid()));

CREATE POLICY "profiles_update" ON public.profiles
  FOR UPDATE TO authenticated 
  USING (id = (SELECT auth.uid()))
  WITH CHECK (id = (SELECT auth.uid()));

CREATE POLICY "profiles_delete" ON public.profiles
  FOR DELETE TO authenticated 
  USING (id = (SELECT auth.uid()));

-- ============================================
-- 4. CONSOLIDATE ORGANIZATIONS RLS POLICIES
-- ============================================

DROP POLICY IF EXISTS "Users can view their organizations" ON public.organizations;
DROP POLICY IF EXISTS "Org admins can update their organization" ON public.organizations;
DROP POLICY IF EXISTS "Authenticated users can create organizations" ON public.organizations;
DROP POLICY IF EXISTS "organizations_select_member" ON public.organizations;
DROP POLICY IF EXISTS "organizations_update_admin" ON public.organizations;
DROP POLICY IF EXISTS "organizations_insert_creator" ON public.organizations;
DROP POLICY IF EXISTS "organizations_insert_authenticated" ON public.organizations;

CREATE POLICY "organizations_select" ON public.organizations
  FOR SELECT TO authenticated
  USING (
    id IN (SELECT org_id FROM public.org_members WHERE user_id = (SELECT auth.uid()))
  );

CREATE POLICY "organizations_insert" ON public.organizations
  FOR INSERT TO authenticated
  WITH CHECK (created_by IS NULL OR created_by = (SELECT auth.uid()));

CREATE POLICY "organizations_update" ON public.organizations
  FOR UPDATE TO authenticated
  USING (
    id IN (
      SELECT org_id FROM public.org_members 
      WHERE user_id = (SELECT auth.uid()) AND role = 'admin'
    )
  );

CREATE POLICY "organizations_delete" ON public.organizations
  FOR DELETE TO authenticated
  USING (
    id IN (
      SELECT org_id FROM public.org_members 
      WHERE user_id = (SELECT auth.uid()) AND role = 'admin'
    )
  );

-- ============================================
-- 5. CONSOLIDATE ORG_MEMBERS RLS POLICIES
-- ============================================

DROP POLICY IF EXISTS "Users can view their org memberships" ON public.org_members;
DROP POLICY IF EXISTS "Users can view fellow org members" ON public.org_members;
DROP POLICY IF EXISTS "Org admins can manage members" ON public.org_members;
DROP POLICY IF EXISTS "org_members_select_authenticated" ON public.org_members;
DROP POLICY IF EXISTS "org_members_admin_manage" ON public.org_members;
DROP POLICY IF EXISTS "org_members_admin_insert" ON public.org_members;
DROP POLICY IF EXISTS "org_members_admin_update" ON public.org_members;
DROP POLICY IF EXISTS "org_members_admin_delete" ON public.org_members;
DROP POLICY IF EXISTS "Users can add themselves to org" ON public.org_members;
DROP POLICY IF EXISTS "Users can view their own membership" ON public.org_members;
DROP POLICY IF EXISTS "org_members_insert_policy" ON public.org_members;

CREATE POLICY "org_members_select" ON public.org_members
  FOR SELECT TO authenticated
  USING (
    user_id = (SELECT auth.uid())
    OR org_id IN (SELECT org_id FROM public.org_members WHERE user_id = (SELECT auth.uid()))
  );

CREATE POLICY "org_members_insert" ON public.org_members
  FOR INSERT TO authenticated
  WITH CHECK (
    user_id = (SELECT auth.uid())
    OR org_id IN (
      SELECT org_id FROM public.org_members 
      WHERE user_id = (SELECT auth.uid()) AND role = 'admin'
    )
  );

CREATE POLICY "org_members_update" ON public.org_members
  FOR UPDATE TO authenticated
  USING (
    org_id IN (
      SELECT org_id FROM public.org_members 
      WHERE user_id = (SELECT auth.uid()) AND role = 'admin'
    )
  );

CREATE POLICY "org_members_delete" ON public.org_members
  FOR DELETE TO authenticated
  USING (
    user_id = (SELECT auth.uid())
    OR org_id IN (
      SELECT org_id FROM public.org_members 
      WHERE user_id = (SELECT auth.uid()) AND role = 'admin'
    )
  );

-- ============================================
-- 6. CONSOLIDATE INVITATIONS RLS POLICIES
-- ============================================

DROP POLICY IF EXISTS "Users can view invitations they created" ON public.invitations;
DROP POLICY IF EXISTS "Users can create invitations" ON public.invitations;
DROP POLICY IF EXISTS "Users can cancel invitations they created" ON public.invitations;
DROP POLICY IF EXISTS "Users can view their own invitations" ON public.invitations;
DROP POLICY IF EXISTS "invitations_select" ON public.invitations;
DROP POLICY IF EXISTS "invitations_insert" ON public.invitations;
DROP POLICY IF EXISTS "invitations_update" ON public.invitations;
DROP POLICY IF EXISTS "invitations_delete" ON public.invitations;
DROP POLICY IF EXISTS "invitations_admin_select" ON public.invitations;
DROP POLICY IF EXISTS "invitations_admin_manage" ON public.invitations;

CREATE POLICY "invitations_select" ON public.invitations
  FOR SELECT TO authenticated
  USING (
    invited_by = (SELECT auth.uid())
    OR email = (SELECT email FROM auth.users WHERE id = (SELECT auth.uid()))
    OR org_id IN (
      SELECT org_id FROM public.org_members 
      WHERE user_id = (SELECT auth.uid()) AND role = 'admin'
    )
  );

CREATE POLICY "invitations_insert" ON public.invitations
  FOR INSERT TO authenticated
  WITH CHECK (invited_by = (SELECT auth.uid()));

CREATE POLICY "invitations_update" ON public.invitations
  FOR UPDATE TO authenticated
  USING (
    invited_by = (SELECT auth.uid())
    OR org_id IN (
      SELECT org_id FROM public.org_members 
      WHERE user_id = (SELECT auth.uid()) AND role = 'admin'
    )
  );

CREATE POLICY "invitations_delete" ON public.invitations
  FOR DELETE TO authenticated
  USING (
    invited_by = (SELECT auth.uid())
    OR org_id IN (
      SELECT org_id FROM public.org_members 
      WHERE user_id = (SELECT auth.uid()) AND role = 'admin'
    )
  );

-- ============================================
-- 7. CONSOLIDATE LOGIN_HISTORY RLS POLICIES
-- ============================================

DROP POLICY IF EXISTS "Users can view own login history" ON public.login_history;
DROP POLICY IF EXISTS "login_history_select" ON public.login_history;

CREATE POLICY "login_history_select" ON public.login_history
  FOR SELECT TO authenticated
  USING (user_id = (SELECT auth.uid()));

-- ============================================
-- 8. CONSOLIDATE PENDING_ORG_INVITES RLS POLICIES
-- ============================================

DROP POLICY IF EXISTS "pending_invites_admin_select" ON public.pending_org_invites;
DROP POLICY IF EXISTS "pending_invites_admin_manage" ON public.pending_org_invites;
DROP POLICY IF EXISTS "pending_invites_select" ON public.pending_org_invites;
DROP POLICY IF EXISTS "pending_invites_insert" ON public.pending_org_invites;
DROP POLICY IF EXISTS "pending_invites_update" ON public.pending_org_invites;
DROP POLICY IF EXISTS "pending_invites_delete" ON public.pending_org_invites;

CREATE POLICY "pending_org_invites_select" ON public.pending_org_invites
  FOR SELECT TO authenticated
  USING (
    org_id IN (
      SELECT org_id FROM public.org_members 
      WHERE user_id = (SELECT auth.uid()) AND role = 'admin'
    )
  );

CREATE POLICY "pending_org_invites_insert" ON public.pending_org_invites
  FOR INSERT TO authenticated
  WITH CHECK (
    invited_by = (SELECT auth.uid())
    AND org_id IN (
      SELECT org_id FROM public.org_members 
      WHERE user_id = (SELECT auth.uid()) AND role = 'admin'
    )
  );

CREATE POLICY "pending_org_invites_update" ON public.pending_org_invites
  FOR UPDATE TO authenticated
  USING (
    org_id IN (
      SELECT org_id FROM public.org_members 
      WHERE user_id = (SELECT auth.uid()) AND role = 'admin'
    )
  );

CREATE POLICY "pending_org_invites_delete" ON public.pending_org_invites
  FOR DELETE TO authenticated
  USING (
    org_id IN (
      SELECT org_id FROM public.org_members 
      WHERE user_id = (SELECT auth.uid()) AND role = 'admin'
    )
  );

-- ============================================
-- 9. DROP UNUSED INDEXES
-- ============================================

DROP INDEX IF EXISTS idx_appliances_building_id;
DROP INDEX IF EXISTS idx_buildings_org_id;
DROP INDEX IF EXISTS idx_disagg_predictions_building_ts;
DROP INDEX IF EXISTS idx_disagg_predictions_building_appliance_ts;
DROP INDEX IF EXISTS idx_predictions_building_ts;
DROP INDEX IF EXISTS idx_predictions_appliance_ts;
DROP INDEX IF EXISTS idx_predictions_user_ts;
DROP INDEX IF EXISTS idx_readings_building_ts;

-- ============================================
-- 10. DROP UNUSED TRIGGERS (only if tables exist)
-- ============================================

DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'buildings') THEN
    DROP TRIGGER IF EXISTS update_buildings_updated_at ON public.buildings;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'appliances') THEN
    DROP TRIGGER IF EXISTS update_appliances_updated_at ON public.appliances;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'user_settings') THEN
    DROP TRIGGER IF EXISTS update_user_settings_updated_at ON public.user_settings;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'building_appliances') THEN
    DROP TRIGGER IF EXISTS update_building_appliances_updated_at ON public.building_appliances;
  END IF;
END $$;

-- ============================================
-- 11. CLEAN UP STORAGE - Remove unused policies
-- ============================================

-- Drop potentially duplicate avatar policies
DROP POLICY IF EXISTS "Public read access for avatars" ON storage.objects;
DROP POLICY IF EXISTS "Public avatar access" ON storage.objects;

-- Keep only authenticated access for avatars
DROP POLICY IF EXISTS "Authenticated users can view avatars" ON storage.objects;
CREATE POLICY "avatars_select" ON storage.objects
  FOR SELECT TO authenticated
  USING (bucket_id = 'avatars');

-- ============================================
-- 12. VERIFY ESSENTIAL INDEXES EXIST
-- ============================================

CREATE INDEX IF NOT EXISTS idx_org_members_user_id ON public.org_members(user_id);
CREATE INDEX IF NOT EXISTS idx_org_members_org_id ON public.org_members(org_id);
CREATE INDEX IF NOT EXISTS idx_org_members_user_org ON public.org_members(user_id, org_id);
CREATE INDEX IF NOT EXISTS idx_login_history_user_id ON public.login_history(user_id);
CREATE INDEX IF NOT EXISTS idx_login_history_created_at ON public.login_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_invitations_email ON public.invitations(email);
CREATE INDEX IF NOT EXISTS idx_invitations_invited_by ON public.invitations(invited_by);
CREATE INDEX IF NOT EXISTS idx_pending_org_invites_email ON public.pending_org_invites(email);

-- ============================================
-- SUMMARY OF CHANGES
-- ============================================
-- Dropped tables: buildings, appliances, building_appliances, predictions,
--                 readings, user_settings, inference_runs, disaggregation_predictions,
--                 login_events
-- Dropped functions: setup_demo_user, is_org_member, is_org_admin, get_user_org_ids
-- Consolidated RLS policies: profiles (4), organizations (4), org_members (4),
--                           invitations (4), login_history (1), pending_org_invites (4)
-- Kept tables: profiles, organizations, org_members, invitations, login_history,
--              pending_org_invites, org_appliances, models, model_versions

SELECT 'Schema cleanup complete. Removed unused tables and consolidated RLS policies.' AS status;
