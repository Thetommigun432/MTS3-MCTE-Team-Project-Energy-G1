-- ============================================
-- MIGRATION: Add remaining foreign key indexes
-- Date: 2026-01-16
-- Note: "Unused index" warnings are INFO level only and will resolve
--       once the app runs queries. Foreign key indexes are important
--       for JOIN performance and cascading DELETE operations.
-- ============================================

-- Add missing foreign key indexes
CREATE INDEX IF NOT EXISTS idx_organizations_created_by 
  ON public.organizations(created_by);

CREATE INDEX IF NOT EXISTS idx_pending_org_invites_invited_by 
  ON public.pending_org_invites(invited_by);

-- Note: The following indexes show as "unused" but are REQUIRED:
-- - idx_models_user_id: FK to auth.users, used by RLS policies
-- - idx_models_org_appliance_id: FK to org_appliances, used for JOINs
-- - idx_model_versions_user_id: FK to auth.users, used by RLS policies
-- - idx_model_versions_model_id: FK to models, used for JOINs
-- - idx_org_appliances_user_id: FK to auth.users, used by RLS policies
-- - idx_invitations_invited_by: FK to auth.users
-- - idx_invitations_org_id: FK to organizations
-- - idx_invitations_email: Used for email lookups in RLS
-- - idx_org_members_user_org: Used by RLS policies for member lookups
-- - idx_login_history_user_created: Used for user's login history queries
--
-- These indexes will show as "used" once the app generates traffic.

SELECT 'Added remaining FK indexes' AS status;
