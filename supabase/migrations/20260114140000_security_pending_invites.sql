-- ============================================
-- MIGRATION: Fix RLS Security & Performance + Pending Invites + Function Security
-- Date: 2026-01-14
-- Fixes:
--   - function_search_path_mutable (SECURITY)
--   - rls_policy_always_true for organizations INSERT (SECURITY)
--   - Add pending_org_invites table for invite reconciliation
--   - Update handle_new_user to reconcile pending invites
-- ============================================

-- ============================================
-- 1. CREATE PENDING_ORG_INVITES TABLE
-- For storing invites before user exists
-- ============================================
CREATE TABLE IF NOT EXISTS public.pending_org_invites (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  org_id UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
  email TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'member' CHECK (role IN ('admin', 'member', 'viewer')),
  invited_by UUID NOT NULL REFERENCES auth.users(id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at TIMESTAMPTZ NOT NULL DEFAULT (now() + interval '7 days'),
  UNIQUE(org_id, email)
);

-- Enable RLS on pending_org_invites
ALTER TABLE public.pending_org_invites ENABLE ROW LEVEL SECURITY;

-- Only org admins can view/manage pending invites for their org
CREATE POLICY "pending_invites_admin_select"
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

CREATE POLICY "pending_invites_admin_manage"
  ON public.pending_org_invites
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

-- Index for quick email lookup during reconciliation
CREATE INDEX IF NOT EXISTS idx_pending_org_invites_email ON public.pending_org_invites(email);

-- ============================================
-- 2. ADD created_by TO ORGANIZATIONS TABLE
-- For secure INSERT policy
-- ============================================
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_schema = 'public' 
    AND table_name = 'organizations' 
    AND column_name = 'created_by'
  ) THEN
    ALTER TABLE public.organizations 
    ADD COLUMN created_by UUID REFERENCES auth.users(id) ON DELETE SET NULL;
  END IF;
END $$;

-- ============================================
-- 3. FIX ORGANIZATIONS INSERT POLICY (rls_policy_always_true)
-- Replace WITH CHECK (true) with proper ownership check
-- ============================================
DROP POLICY IF EXISTS "organizations_insert_authenticated" ON public.organizations;
DROP POLICY IF EXISTS "Authenticated users can create organizations" ON public.organizations;

CREATE POLICY "organizations_insert_creator"
  ON public.organizations
  FOR INSERT
  TO authenticated
  WITH CHECK (
    created_by IS NULL OR created_by = (SELECT auth.uid())
  );

-- ============================================
-- 4. FIX HELPER FUNCTIONS (function_search_path_mutable)
-- Recreate with SET search_path = public
-- ============================================

-- Fix is_org_member
CREATE OR REPLACE FUNCTION public.is_org_member(org_uuid UUID)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  RETURN EXISTS (
    SELECT 1 FROM public.org_members
    WHERE org_id = org_uuid AND user_id = (SELECT auth.uid())
  );
END;
$$;

-- Fix is_org_admin
CREATE OR REPLACE FUNCTION public.is_org_admin(org_uuid UUID)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  RETURN EXISTS (
    SELECT 1 FROM public.org_members
    WHERE org_id = org_uuid AND user_id = (SELECT auth.uid()) AND role = 'admin'
  );
END;
$$;

-- Fix get_user_org_ids
CREATE OR REPLACE FUNCTION public.get_user_org_ids()
RETURNS SETOF UUID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  RETURN QUERY SELECT org_id FROM public.org_members WHERE user_id = (SELECT auth.uid());
END;
$$;

-- ============================================
-- 5. UPDATE handle_new_user TO RECONCILE PENDING INVITES
-- When a new user signs up, check if they have pending org invites
-- ============================================
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  pending_invite RECORD;
BEGIN
  -- Create profile for new user
  INSERT INTO public.profiles (id, email, display_name)
  VALUES (
    NEW.id,
    NEW.email,
    COALESCE(NEW.raw_user_meta_data ->> 'display_name', split_part(NEW.email, '@', 1))
  )
  ON CONFLICT (id) DO NOTHING;

  -- Reconcile any pending org invites for this email
  FOR pending_invite IN
    SELECT * FROM public.pending_org_invites
    WHERE email = NEW.email
    AND expires_at > now()
  LOOP
    -- Add user as member of the org
    INSERT INTO public.org_members (org_id, user_id, role)
    VALUES (pending_invite.org_id, NEW.id, pending_invite.role)
    ON CONFLICT (org_id, user_id) DO UPDATE SET role = EXCLUDED.role;

    -- Delete the pending invite
    DELETE FROM public.pending_org_invites WHERE id = pending_invite.id;
  END LOOP;

  RETURN NEW;
END;
$$;

-- ============================================
-- 6. INVITATIONS TABLE (if not exists)
-- For tracking invitation status
-- ============================================
CREATE TABLE IF NOT EXISTS public.invitations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'member' CHECK (role IN ('admin', 'member', 'viewer')),
  org_id UUID REFERENCES public.organizations(id) ON DELETE CASCADE,
  invited_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'accepted', 'expired', 'cancelled')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at TIMESTAMPTZ NOT NULL DEFAULT (now() + interval '7 days')
);

-- Enable RLS
ALTER TABLE public.invitations ENABLE ROW LEVEL SECURITY;

-- Policies for invitations
DROP POLICY IF EXISTS "invitations_admin_select" ON public.invitations;
CREATE POLICY "invitations_admin_select"
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

DROP POLICY IF EXISTS "invitations_admin_manage" ON public.invitations;
CREATE POLICY "invitations_admin_manage"
  ON public.invitations
  FOR ALL
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
  WITH CHECK (true);

-- Index for email lookup
CREATE INDEX IF NOT EXISTS idx_invitations_email ON public.invitations(email);
CREATE INDEX IF NOT EXISTS idx_invitations_status ON public.invitations(status);

-- ============================================
-- 7. DOCUMENTATION COMMENTS
-- ============================================
COMMENT ON TABLE public.pending_org_invites IS 
  'Stores organization invites for users who have not yet signed up. Reconciled by handle_new_user trigger.';

COMMENT ON TABLE public.invitations IS 
  'Tracks invitation history and status for the UI.';

COMMENT ON FUNCTION public.handle_new_user() IS 
  'Creates profile for new users and reconciles pending org invites.';

-- ============================================
-- NOTE: Unused indexes
-- ============================================
-- The Supabase linter may flag some indexes as unused.
-- This is expected until query statistics accumulate.
-- Do NOT drop indexes based solely on the linter warning.
-- Monitor pg_stat_user_indexes after significant traffic.
