-- Add stream_key column to buildings table
-- This allows mapping between Supabase building UUIDs and legacy stream identifiers (e.g., building_1)
-- Used for querying Redis streams and InfluxDB series that use non-UUID identifiers

ALTER TABLE buildings ADD COLUMN IF NOT EXISTS stream_key text UNIQUE;

-- Index for efficient lookups when resolving stream_key for queries
CREATE INDEX IF NOT EXISTS idx_buildings_stream_key ON buildings(stream_key) WHERE stream_key IS NOT NULL;

-- Add comment explaining the column purpose
COMMENT ON COLUMN buildings.stream_key IS 'Optional identifier used for Redis streams and InfluxDB when different from UUID (e.g., building_1). Must match data stream source.';
