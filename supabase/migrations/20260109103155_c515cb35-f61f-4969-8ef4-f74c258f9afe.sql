-- Create invitations table for tracking pending invites
CREATE TABLE public.invitations (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  email TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'viewer',
  invited_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  expires_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT (now() + interval '7 days'),
  accepted_at TIMESTAMP WITH TIME ZONE,
  CONSTRAINT valid_role CHECK (role IN ('admin', 'member', 'viewer')),
  CONSTRAINT valid_status CHECK (status IN ('pending', 'accepted', 'expired', 'cancelled'))
);

-- Create login_events table for tracking login history
CREATE TABLE public.login_events (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  ip_address TEXT,
  user_agent TEXT,
  device_label TEXT,
  success BOOLEAN NOT NULL DEFAULT true
);

-- Create index for faster queries
CREATE INDEX idx_login_events_user_id ON public.login_events(user_id);
CREATE INDEX idx_login_events_created_at ON public.login_events(created_at DESC);
CREATE INDEX idx_invitations_email ON public.invitations(email);
CREATE INDEX idx_invitations_status ON public.invitations(status);

-- Enable RLS on invitations
ALTER TABLE public.invitations ENABLE ROW LEVEL SECURITY;

-- Invitations policies: only admins can manage invitations (for now, any authenticated user can invite)
CREATE POLICY "Users can view invitations they created"
  ON public.invitations
  FOR SELECT
  USING (auth.uid() = invited_by);

CREATE POLICY "Users can create invitations"
  ON public.invitations
  FOR INSERT
  WITH CHECK (auth.uid() = invited_by);

CREATE POLICY "Users can cancel invitations they created"
  ON public.invitations
  FOR UPDATE
  USING (auth.uid() = invited_by);

-- Enable RLS on login_events
ALTER TABLE public.login_events ENABLE ROW LEVEL SECURITY;

-- Login events policies: users can only see their own login events
CREATE POLICY "Users can view their own login events"
  ON public.login_events
  FOR SELECT
  USING (auth.uid() = user_id);

-- Allow inserts from authenticated users for their own events
CREATE POLICY "Users can insert their own login events"
  ON public.login_events
  FOR INSERT
  WITH CHECK (auth.uid() = user_id);