-- Allow all authenticated users to view all profiles (for team page)
-- This enables the Team Members page to display all users in the organization

CREATE POLICY "All authenticated users can view profiles"
  ON public.profiles FOR SELECT
  TO authenticated
  USING (true);
