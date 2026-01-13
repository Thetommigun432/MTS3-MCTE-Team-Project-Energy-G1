import { useState, useEffect, useCallback } from 'react';
import { supabase } from '@/integrations/supabase/client';
import { useAuth } from '@/contexts/AuthContext';

export interface ManagedAppliance {
  id: string;
  building_id: string;
  building_name: string;
  name: string;
  type: string;
  rated_power_kw: number | null;
  status: string;
  notes: string | null;
}

export interface ManagedAppliancesData {
  appliances: ManagedAppliance[];
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

/**
 * Hook to fetch all managed appliances for the current user across all buildings
 */
export function useManagedAppliances(): ManagedAppliancesData {
  const { user } = useAuth();
  const [appliances, setAppliances] = useState<ManagedAppliance[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchAppliances = useCallback(async () => {
    if (!user) {
      setAppliances([]);
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      // Fetch appliances with their building names
      const { data, error: fetchError } = await supabase
        .from('appliances')
        .select(`
          id,
          building_id,
          name,
          type,
          rated_power_kw,
          status,
          notes,
          buildings!inner(name)
        `)
        .eq('status', 'active')
        .order('name');

      if (fetchError) throw fetchError;

      // Transform the data to flatten building name
      const transformed: ManagedAppliance[] = (data || []).map((item) => ({
        id: item.id,
        building_id: item.building_id,
        building_name: item.buildings?.name || 'Unknown Building',
        name: item.name,
        type: item.type,
        rated_power_kw: item.rated_power_kw,
        status: item.status,
        notes: item.notes,
      }));

      setAppliances(transformed);
    } catch (err) {
      console.error('Error fetching managed appliances:', err);
      setError(err instanceof Error ? err.message : 'Failed to load appliances');
    } finally {
      setLoading(false);
    }
  }, [user]);

  useEffect(() => {
    fetchAppliances();
  }, [fetchAppliances]);

  return {
    appliances,
    loading,
    error,
    refetch: fetchAppliances,
  };
}

/**
 * Get a mapping of appliance names to their rated power
 */
export function getAppliancePowerMap(appliances: ManagedAppliance[]): Record<string, number | null> {
  const map: Record<string, number | null> = {};
  appliances.forEach((a) => {
    map[a.name] = a.rated_power_kw;
  });
  return map;
}

/**
 * Get unique appliance names from managed appliances
 */
export function getApplianceNames(appliances: ManagedAppliance[]): string[] {
  return [...new Set(appliances.map((a) => a.name))];
}
