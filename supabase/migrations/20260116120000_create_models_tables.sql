-- ============================================
-- MIGRATION: Create models tables and seed demo data
-- Date: 2026-01-16
-- Purpose: Create org_appliances, models, model_versions tables
--          with proper RLS and seed demo data
-- ============================================

-- ============================================
-- 1. Create org_appliances table
-- ============================================
CREATE TABLE IF NOT EXISTS public.org_appliances (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    slug TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'other',
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Add unique constraint separately to handle IF NOT EXISTS
DO $$ BEGIN
  ALTER TABLE public.org_appliances 
    ADD CONSTRAINT org_appliances_slug_unique UNIQUE (user_id, slug);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Enable RLS
ALTER TABLE public.org_appliances ENABLE ROW LEVEL SECURITY;

-- RLS policies for org_appliances
CREATE POLICY "org_appliances_select_own" ON public.org_appliances 
  FOR SELECT USING ((SELECT auth.uid()) = user_id);

CREATE POLICY "org_appliances_insert_own" ON public.org_appliances 
  FOR INSERT WITH CHECK ((SELECT auth.uid()) = user_id);

CREATE POLICY "org_appliances_update_own" ON public.org_appliances 
  FOR UPDATE USING ((SELECT auth.uid()) = user_id);

CREATE POLICY "org_appliances_delete_own" ON public.org_appliances 
  FOR DELETE USING ((SELECT auth.uid()) = user_id);

-- ============================================
-- 2. Create models table
-- ============================================
CREATE TABLE IF NOT EXISTS public.models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    org_appliance_id UUID NOT NULL REFERENCES public.org_appliances(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    architecture TEXT DEFAULT 'seq2point',
    is_active BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Enable RLS
ALTER TABLE public.models ENABLE ROW LEVEL SECURITY;

-- RLS policies
CREATE POLICY "models_select_own" ON public.models 
  FOR SELECT USING ((SELECT auth.uid()) = user_id);

CREATE POLICY "models_insert_own" ON public.models 
  FOR INSERT WITH CHECK ((SELECT auth.uid()) = user_id);

CREATE POLICY "models_update_own" ON public.models 
  FOR UPDATE USING ((SELECT auth.uid()) = user_id);

CREATE POLICY "models_delete_own" ON public.models 
  FOR DELETE USING ((SELECT auth.uid()) = user_id);

-- ============================================
-- 3. Create model_versions table
-- ============================================
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
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Add unique constraint
DO $$ BEGIN
  ALTER TABLE public.model_versions 
    ADD CONSTRAINT model_versions_unique UNIQUE (model_id, version);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Enable RLS
ALTER TABLE public.model_versions ENABLE ROW LEVEL SECURITY;

-- RLS policies
CREATE POLICY "model_versions_select_own" ON public.model_versions 
  FOR SELECT USING ((SELECT auth.uid()) = user_id);

CREATE POLICY "model_versions_insert_own" ON public.model_versions 
  FOR INSERT WITH CHECK ((SELECT auth.uid()) = user_id);

CREATE POLICY "model_versions_update_own" ON public.model_versions 
  FOR UPDATE USING ((SELECT auth.uid()) = user_id);

CREATE POLICY "model_versions_delete_own" ON public.model_versions 
  FOR DELETE USING ((SELECT auth.uid()) = user_id);

-- ============================================
-- 4. Seed demo data
-- ============================================
DO $$
DECLARE
  v_user UUID;
  v_appliance UUID;
  v_model UUID;
BEGIN
  SELECT id INTO v_user FROM public.profiles WHERE email = 'demo@energy-monitor.app' LIMIT 1;
  
  IF v_user IS NULL THEN
    RAISE NOTICE 'No demo user found';
    RETURN;
  END IF;

  RAISE NOTICE 'Seeding models for user %', v_user;

  -- Heat Pump
  INSERT INTO public.org_appliances (user_id, name, slug, category, description)
  VALUES (v_user, 'Heat Pump', 'heatpump', 'hvac', 'Main HVAC heat pump')
  ON CONFLICT (user_id, slug) DO UPDATE SET name = EXCLUDED.name
  RETURNING id INTO v_appliance;

  INSERT INTO public.models (user_id, org_appliance_id, name, architecture, is_active)
  VALUES (v_user, v_appliance, 'Heat Pump NILM Model', 'seq2point', true)
  RETURNING id INTO v_model;

  INSERT INTO public.model_versions (model_id, user_id, version, status, is_active, trained_at, metrics)
  VALUES (v_model, v_user, 'v1.0.0', 'ready', true, now() - interval '30 days',
    '{"accuracy": 0.92, "f1_score": 0.89, "mae": 0.045}'::jsonb);

  -- Washing Machine
  INSERT INTO public.org_appliances (user_id, name, slug, category, description)
  VALUES (v_user, 'Washing Machine', 'washing_machine', 'laundry', 'Front-load washing machine')
  ON CONFLICT (user_id, slug) DO UPDATE SET name = EXCLUDED.name
  RETURNING id INTO v_appliance;

  INSERT INTO public.models (user_id, org_appliance_id, name, architecture, is_active)
  VALUES (v_user, v_appliance, 'Washing Machine Model', 'seq2point', true)
  RETURNING id INTO v_model;

  INSERT INTO public.model_versions (model_id, user_id, version, status, is_active, trained_at, metrics)
  VALUES (v_model, v_user, 'v1.0.0', 'ready', true, now() - interval '25 days',
    '{"accuracy": 0.88, "f1_score": 0.85, "mae": 0.052}'::jsonb);

  -- Dishwasher
  INSERT INTO public.org_appliances (user_id, name, slug, category, description)
  VALUES (v_user, 'Dishwasher', 'dishwasher', 'kitchen', 'Kitchen dishwasher')
  ON CONFLICT (user_id, slug) DO UPDATE SET name = EXCLUDED.name
  RETURNING id INTO v_appliance;

  INSERT INTO public.models (user_id, org_appliance_id, name, architecture, is_active)
  VALUES (v_user, v_appliance, 'Dishwasher Model', 'cnn', false)
  RETURNING id INTO v_model;

  INSERT INTO public.model_versions (model_id, user_id, version, status, is_active, trained_at, metrics)
  VALUES (v_model, v_user, 'v0.9.0', 'ready', true, now() - interval '45 days',
    '{"accuracy": 0.78, "f1_score": 0.72, "mae": 0.068}'::jsonb);

  -- Dryer
  INSERT INTO public.org_appliances (user_id, name, slug, category, description)
  VALUES (v_user, 'Dryer', 'dryer', 'laundry', 'Electric clothes dryer')
  ON CONFLICT (user_id, slug) DO UPDATE SET name = EXCLUDED.name
  RETURNING id INTO v_appliance;

  INSERT INTO public.models (user_id, org_appliance_id, name, architecture, is_active)
  VALUES (v_user, v_appliance, 'Dryer Model', 'transformer', true)
  RETURNING id INTO v_model;

  INSERT INTO public.model_versions (model_id, user_id, version, status, is_active, trained_at, metrics)
  VALUES (v_model, v_user, 'v1.2.0', 'ready', true, now() - interval '10 days',
    '{"accuracy": 0.94, "f1_score": 0.91, "mae": 0.038}'::jsonb);

  RAISE NOTICE 'Demo models seeded successfully';
END;
$$;
