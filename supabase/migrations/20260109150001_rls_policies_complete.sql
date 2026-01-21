-- ============================================
-- RLS POLICIES MIGRATION (SAFE VERSION)
-- Comprehensive Row Level Security for all tables
-- Only applies to tables that exist
-- ============================================

-- ============================================
-- ENABLE RLS ON CORE NEW TABLES
-- ============================================
ALTER TABLE IF EXISTS public.organizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.org_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.login_history ENABLE ROW LEVEL SECURITY;

-- ============================================
-- ORGANIZATIONS POLICIES
-- ============================================
DROP POLICY IF EXISTS "Users can view their organizations" ON public.organizations;
CREATE POLICY "Users can view their organizations"
  ON public.organizations
  FOR SELECT
  TO authenticated
  USING (
    id IN (SELECT org_id FROM public.org_members WHERE user_id = auth.uid())
  );

DROP POLICY IF EXISTS "Org admins can update their organization" ON public.organizations;
CREATE POLICY "Org admins can update their organization"
  ON public.organizations
  FOR UPDATE
  TO authenticated
  USING (
    id IN (
      SELECT org_id FROM public.org_members 
      WHERE user_id = auth.uid() AND role = 'admin'
    )
  );

DROP POLICY IF EXISTS "Authenticated users can create organizations" ON public.organizations;
CREATE POLICY "Authenticated users can create organizations"
  ON public.organizations
  FOR INSERT
  TO authenticated
  WITH CHECK (true);

-- ============================================
-- ORG_MEMBERS POLICIES
-- ============================================
DROP POLICY IF EXISTS "Users can view their org memberships" ON public.org_members;
CREATE POLICY "Users can view their org memberships"
  ON public.org_members
  FOR SELECT
  TO authenticated
  USING (user_id = auth.uid());

DROP POLICY IF EXISTS "Users can view fellow org members" ON public.org_members;
CREATE POLICY "Users can view fellow org members"
  ON public.org_members
  FOR SELECT
  TO authenticated
  USING (
    org_id IN (SELECT org_id FROM public.org_members WHERE user_id = auth.uid())
  );

DROP POLICY IF EXISTS "Org admins can manage members" ON public.org_members;
CREATE POLICY "Org admins can manage members"
  ON public.org_members
  FOR ALL
  TO authenticated
  USING (
    org_id IN (
      SELECT org_id FROM public.org_members 
      WHERE user_id = auth.uid() AND role = 'admin'
    )
  );

-- ============================================
-- LOGIN_HISTORY POLICIES
-- ============================================
DROP POLICY IF EXISTS "Users can view own login history" ON public.login_history;
CREATE POLICY "Users can view own login history"
  ON public.login_history
  FOR SELECT
  TO authenticated
  USING (user_id = auth.uid());

-- Note: INSERT is done via Edge Function with service_role key
