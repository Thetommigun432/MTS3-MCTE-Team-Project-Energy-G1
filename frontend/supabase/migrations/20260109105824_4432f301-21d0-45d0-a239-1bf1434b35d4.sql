-- Fix 1: Make avatars bucket private and update policies
UPDATE storage.buckets SET public = false WHERE id = 'avatars';

-- Remove the old public SELECT policy (if it exists with this name)
DROP POLICY IF EXISTS "Public avatar access" ON storage.objects;
DROP POLICY IF EXISTS "Authenticated users can view avatars" ON storage.objects;

-- Create new authenticated-only SELECT policy
CREATE POLICY "Authenticated users can view avatars"
ON storage.objects
FOR SELECT
TO authenticated
USING (bucket_id = 'avatars');

-- Fix 2: Add database constraints for input validation
-- Buildings table constraints
ALTER TABLE public.buildings 
ADD CONSTRAINT buildings_name_length CHECK (length(name) <= 200);

ALTER TABLE public.buildings 
ADD CONSTRAINT buildings_name_not_empty CHECK (trim(name) != '');

ALTER TABLE public.buildings 
ADD CONSTRAINT buildings_address_length CHECK (address IS NULL OR length(address) <= 500);

ALTER TABLE public.buildings 
ADD CONSTRAINT buildings_description_length CHECK (description IS NULL OR length(description) <= 1000);

-- Appliances table constraints
ALTER TABLE public.appliances 
ADD CONSTRAINT appliances_name_length CHECK (length(name) <= 200);

ALTER TABLE public.appliances 
ADD CONSTRAINT appliances_name_not_empty CHECK (trim(name) != '');

ALTER TABLE public.appliances 
ADD CONSTRAINT appliances_rated_power_range CHECK (rated_power_kw IS NULL OR (rated_power_kw > 0 AND rated_power_kw <= 10000));

ALTER TABLE public.appliances 
ADD CONSTRAINT appliances_notes_length CHECK (notes IS NULL OR length(notes) <= 1000);

-- Profiles table constraints
ALTER TABLE public.profiles 
ADD CONSTRAINT profiles_display_name_length CHECK (display_name IS NULL OR length(display_name) <= 100);

ALTER TABLE public.profiles 
ADD CONSTRAINT profiles_avatar_url_length CHECK (avatar_url IS NULL OR length(avatar_url) <= 2000);