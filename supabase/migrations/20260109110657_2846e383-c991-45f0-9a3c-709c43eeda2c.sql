-- =====================================================
-- PHASE 1: Schema for "One Model Per Appliance" Workflow
-- =====================================================

-- 1. Org-level appliance catalog (slug is unique per org, used by models)
CREATE TABLE IF NOT EXISTS public.org_appliances (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    slug TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'other',
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT org_appliances_slug_unique UNIQUE (user_id, slug),
    CONSTRAINT org_appliances_name_length CHECK (length(name) <= 100),
    CONSTRAINT org_appliances_slug_format CHECK (slug ~ '^[a-z0-9_]+$')
);

-- Enable RLS
ALTER TABLE public.org_appliances ENABLE ROW LEVEL SECURITY;

-- RLS policies for org_appliances
CREATE POLICY "Users can view their own org appliances"
ON public.org_appliances FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own org appliances"
ON public.org_appliances FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own org appliances"
ON public.org_appliances FOR UPDATE
USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own org appliances"
ON public.org_appliances FOR DELETE
USING (auth.uid() = user_id);

-- Trigger for updated_at
CREATE TRIGGER update_org_appliances_updated_at
BEFORE UPDATE ON public.org_appliances
FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

-- 2. Building-appliance assignments (assign org appliances to buildings)
CREATE TABLE IF NOT EXISTS public.building_appliances (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id UUID NOT NULL REFERENCES public.buildings(id) ON DELETE CASCADE,
    org_appliance_id UUID NOT NULL REFERENCES public.org_appliances(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    alias TEXT, -- Optional per-building alias
    is_enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT building_appliances_unique UNIQUE (building_id, org_appliance_id)
);

-- Enable RLS
ALTER TABLE public.building_appliances ENABLE ROW LEVEL SECURITY;

-- RLS policies
CREATE POLICY "Users can view their own building appliances"
ON public.building_appliances FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own building appliances"
ON public.building_appliances FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own building appliances"
ON public.building_appliances FOR UPDATE
USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own building appliances"
ON public.building_appliances FOR DELETE
USING (auth.uid() = user_id);

-- Trigger for updated_at
CREATE TRIGGER update_building_appliances_updated_at
BEFORE UPDATE ON public.building_appliances
FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

-- 3. Models table (one model per appliance)
CREATE TABLE IF NOT EXISTS public.models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    org_appliance_id UUID NOT NULL REFERENCES public.org_appliances(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    architecture TEXT DEFAULT 'seq2point',
    is_active BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT models_name_length CHECK (length(name) <= 100)
);

-- Enable RLS
ALTER TABLE public.models ENABLE ROW LEVEL SECURITY;

-- RLS policies
CREATE POLICY "Users can view their own models"
ON public.models FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own models"
ON public.models FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own models"
ON public.models FOR UPDATE
USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own models"
ON public.models FOR DELETE
USING (auth.uid() = user_id);

-- Trigger for updated_at
CREATE TRIGGER update_models_updated_at
BEFORE UPDATE ON public.models
FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

-- 4. Model versions table
CREATE TABLE IF NOT EXISTS public.model_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL REFERENCES public.models(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    version TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'uploading', 'ready', 'failed', 'archived')),
    model_artifact_path TEXT,
    scaler_artifact_path TEXT,
    metrics JSONB DEFAULT '{}',
    training_config JSONB DEFAULT '{}',
    is_active BOOLEAN NOT NULL DEFAULT false,
    trained_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT model_versions_unique UNIQUE (model_id, version)
);

-- Enable RLS
ALTER TABLE public.model_versions ENABLE ROW LEVEL SECURITY;

-- RLS policies
CREATE POLICY "Users can view their own model versions"
ON public.model_versions FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own model versions"
ON public.model_versions FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own model versions"
ON public.model_versions FOR UPDATE
USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own model versions"
ON public.model_versions FOR DELETE
USING (auth.uid() = user_id);

-- Trigger for updated_at
CREATE TRIGGER update_model_versions_updated_at
BEFORE UPDATE ON public.model_versions
FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

-- 5. Predictions table (stores inference results)
CREATE TABLE IF NOT EXISTS public.predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    building_id UUID NOT NULL REFERENCES public.buildings(id) ON DELETE CASCADE,
    org_appliance_id UUID NOT NULL REFERENCES public.org_appliances(id) ON DELETE CASCADE,
    model_version_id UUID REFERENCES public.model_versions(id) ON DELETE SET NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    power_kw NUMERIC NOT NULL,
    confidence NUMERIC,
    is_on BOOLEAN,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT predictions_power_range CHECK (power_kw >= 0)
);

-- Enable RLS
ALTER TABLE public.predictions ENABLE ROW LEVEL SECURITY;

-- RLS policies
CREATE POLICY "Users can view their own predictions"
ON public.predictions FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own predictions"
ON public.predictions FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own predictions"
ON public.predictions FOR DELETE
USING (auth.uid() = user_id);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_predictions_building_ts 
ON public.predictions(building_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_predictions_appliance_ts 
ON public.predictions(org_appliance_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_predictions_user_ts 
ON public.predictions(user_id, timestamp DESC);

-- 6. Aggregate readings table (smart meter data)
CREATE TABLE IF NOT EXISTS public.readings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    building_id UUID NOT NULL REFERENCES public.buildings(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL,
    aggregate_kw NUMERIC NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT readings_power_range CHECK (aggregate_kw >= 0)
);

-- Enable RLS
ALTER TABLE public.readings ENABLE ROW LEVEL SECURITY;

-- RLS policies
CREATE POLICY "Users can view their own readings"
ON public.readings FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own readings"
ON public.readings FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own readings"
ON public.readings FOR DELETE
USING (auth.uid() = user_id);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_readings_building_ts 
ON public.readings(building_id, timestamp DESC);

-- 7. User settings table (for high contrast, etc.)
CREATE TABLE IF NOT EXISTS public.user_settings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL UNIQUE REFERENCES auth.users(id) ON DELETE CASCADE,
    high_contrast BOOLEAN NOT NULL DEFAULT false,
    compact_mode BOOLEAN NOT NULL DEFAULT false,
    reduce_motion BOOLEAN NOT NULL DEFAULT false,
    selected_building_id UUID REFERENCES public.buildings(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Enable RLS
ALTER TABLE public.user_settings ENABLE ROW LEVEL SECURITY;

-- RLS policies
CREATE POLICY "Users can view their own settings"
ON public.user_settings FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own settings"
ON public.user_settings FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own settings"
ON public.user_settings FOR UPDATE
USING (auth.uid() = user_id);

-- Trigger for updated_at
CREATE TRIGGER update_user_settings_updated_at
BEFORE UPDATE ON public.user_settings
FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

-- 8. Create storage bucket for model artifacts
INSERT INTO storage.buckets (id, name, public)
VALUES ('models', 'models', false)
ON CONFLICT (id) DO NOTHING;

-- Storage policies for models bucket
CREATE POLICY "Users can view their own model artifacts"
ON storage.objects FOR SELECT
TO authenticated
USING (bucket_id = 'models' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Users can upload their own model artifacts"
ON storage.objects FOR INSERT
TO authenticated
WITH CHECK (bucket_id = 'models' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Users can update their own model artifacts"
ON storage.objects FOR UPDATE
TO authenticated
USING (bucket_id = 'models' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Users can delete their own model artifacts"
ON storage.objects FOR DELETE
TO authenticated
USING (bucket_id = 'models' AND auth.uid()::text = (storage.foldername(name))[1]);