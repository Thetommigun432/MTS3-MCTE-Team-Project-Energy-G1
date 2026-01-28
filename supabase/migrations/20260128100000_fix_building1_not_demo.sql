-- Fix Building 1 to NOT be a demo building
-- Building 1 is the real monitored building with InfluxDB data
UPDATE public.buildings 
SET is_demo = FALSE 
WHERE id = 'building-1';
