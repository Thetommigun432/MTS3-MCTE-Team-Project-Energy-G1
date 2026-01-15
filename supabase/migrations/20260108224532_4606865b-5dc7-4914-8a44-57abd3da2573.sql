-- Create appliances table linked to buildings
CREATE TABLE public.appliances (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  building_id UUID NOT NULL REFERENCES public.buildings(id) ON DELETE CASCADE,
  user_id UUID NOT NULL,
  name TEXT NOT NULL,
  type TEXT NOT NULL DEFAULT 'other',
  rated_power_kw NUMERIC(10,3),
  status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'unknown')),
  notes TEXT,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable Row Level Security
ALTER TABLE public.appliances ENABLE ROW LEVEL SECURITY;

-- Create policies for user access
CREATE POLICY "Users can view their own appliances" 
ON public.appliances 
FOR SELECT 
USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own appliances" 
ON public.appliances 
FOR INSERT 
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own appliances" 
ON public.appliances 
FOR UPDATE 
USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own appliances" 
ON public.appliances 
FOR DELETE 
USING (auth.uid() = user_id);

-- Create trigger for automatic timestamp updates
CREATE TRIGGER update_appliances_updated_at
BEFORE UPDATE ON public.appliances
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();

-- Create index for faster building lookups
CREATE INDEX idx_appliances_building_id ON public.appliances(building_id);