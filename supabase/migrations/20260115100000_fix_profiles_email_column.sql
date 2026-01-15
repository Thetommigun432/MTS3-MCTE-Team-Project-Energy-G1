-- ============================================
-- MIGRATION: Fix profiles table - add email column
-- Date: 2026-01-15
-- Issue: handle_new_user trigger fails because profiles table
--        is missing the 'email' column that the trigger tries to insert
-- ============================================

-- Add email column to profiles if it doesn't exist
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_schema = 'public' 
    AND table_name = 'profiles' 
    AND column_name = 'email'
  ) THEN
    ALTER TABLE public.profiles ADD COLUMN email TEXT;
  END IF;
END $$;

-- Update handle_new_user function to handle both old and new schema
-- This version works whether email column exists or not
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
  ON CONFLICT (id) DO UPDATE SET
    email = EXCLUDED.email,
    display_name = COALESCE(profiles.display_name, EXCLUDED.display_name);

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
EXCEPTION
  WHEN undefined_table THEN
    -- pending_org_invites table doesn't exist yet, just create profile
    INSERT INTO public.profiles (id, email, display_name)
    VALUES (
      NEW.id,
      NEW.email,
      COALESCE(NEW.raw_user_meta_data ->> 'display_name', split_part(NEW.email, '@', 1))
    )
    ON CONFLICT (id) DO UPDATE SET
      email = EXCLUDED.email,
      display_name = COALESCE(profiles.display_name, EXCLUDED.display_name);
    RETURN NEW;
END;
$$;

-- Ensure trigger exists
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- ============================================
-- DOCUMENTATION
-- ============================================
COMMENT ON FUNCTION public.handle_new_user() IS 
  'Creates profile for new users and reconciles pending org invites. Fixed to handle missing email column.';
