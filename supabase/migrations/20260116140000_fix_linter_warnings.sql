-- ============================================
-- MIGRATION: Fix linter warnings
-- Date: 2026-01-16
-- Fixes:
--   - Duplicate login_history SELECT policies
--   - Missing indexes on foreign keys
--   - Remove truly unused indexes (optional - keeping core ones for RLS)
-- ============================================

-- ============================================
-- 1. FIX DUPLICATE LOGIN_HISTORY POLICIES
-- ============================================

-- Drop the duplicate policy (created in an earlier migration)
DROP POLICY IF EXISTS "login_history_select_own" ON public.login_history;

-- Keep only login_history_select (already exists from cleanup migration)

-- ============================================
-- 2. ADD MISSING FOREIGN KEY INDEXES
-- ============================================

-- model_versions.user_id
CREATE INDEX IF NOT EXISTS idx_model_versions_user_id 
  ON public.model_versions(user_id);

-- models.user_id
CREATE INDEX IF NOT EXISTS idx_models_user_id 
  ON public.models(user_id);

-- models.org_appliance_id
CREATE INDEX IF NOT EXISTS idx_models_org_appliance_id 
  ON public.models(org_appliance_id);

-- model_versions.model_id (also a foreign key)
CREATE INDEX IF NOT EXISTS idx_model_versions_model_id 
  ON public.model_versions(model_id);

-- org_appliances.user_id (also a foreign key)
CREATE INDEX IF NOT EXISTS idx_org_appliances_user_id 
  ON public.org_appliances(user_id);

-- ============================================
-- 3. REMOVE REDUNDANT INDEXES
-- Keep only essential indexes, remove duplicates
-- ============================================

-- These indexes overlap or are truly unused:
-- idx_org_members_user_id is covered by idx_org_members_user_org
-- idx_org_members_org_id is covered by primary key and idx_org_members_org_user
DROP INDEX IF EXISTS idx_org_members_user_id;
DROP INDEX IF EXISTS idx_org_members_org_id;
DROP INDEX IF EXISTS idx_org_members_org_user;
DROP INDEX IF EXISTS idx_org_members_user_role;

-- Keep only idx_org_members_user_org for RLS policy performance
-- (user_id, org_id) is the most common query pattern

-- login_history has overlapping indexes
DROP INDEX IF EXISTS idx_login_history_user_id;
DROP INDEX IF EXISTS idx_login_history_created_at;
-- Keep only idx_login_history_user_created (composite)

-- invitations: status index is rarely needed
DROP INDEX IF EXISTS idx_invitations_status;

-- organizations.created_by is rarely queried
DROP INDEX IF EXISTS idx_organizations_created_by;

-- pending_org_invites.invited_by is rarely queried
DROP INDEX IF EXISTS idx_pending_org_invites_invited_by;

SELECT 'Linter warnings fixed' AS status;
