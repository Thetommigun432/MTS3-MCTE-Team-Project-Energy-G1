-- Drop the existing SELECT policy
DROP POLICY IF EXISTS "Users can view invitations they created" ON public.invitations;

-- Create a more restrictive SELECT policy that allows:
-- 1. The inviter to see invitations they created
-- 2. The invited user to see invitations sent to their email (for accepting)
CREATE POLICY "Users can view their own invitations"
ON public.invitations
FOR SELECT
TO authenticated
USING (
  auth.uid() = invited_by 
  OR email = (SELECT email FROM auth.users WHERE id = auth.uid())
);

-- Add an index on email for faster lookups
CREATE INDEX IF NOT EXISTS idx_invitations_email ON public.invitations(email);