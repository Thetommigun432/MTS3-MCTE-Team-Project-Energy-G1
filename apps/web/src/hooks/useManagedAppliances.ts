import { useState, useEffect, useCallback } from "react";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/contexts/AuthContext";

export interface ManagedAppliance {
  id: string; // building_appliance_id
  building_id: string;
  building_name: string;
  org_appliance_id: string;
  name: string;
  type: string;
  rated_power_kw: number | null;
  status: "active" | "inactive"; // derived from is_enabled
  notes: null; // Schema doesn't have notes on building_appliances, keeping for interface compat if needed or null
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

      // Fetch enabled appliances via building_appliances join
      const { data, error: fetchError } = await supabase
        .from("building_appliances")
        .select(
          `
          id,
          building_id,
          is_enabled,
          buildings!inner(name),
          org_appliances!inner(
            id,
            name,
            rated_power_kw
          )
        `,
        )
        .eq("is_enabled", true);

      if (fetchError) throw fetchError;

      // Transform the data
      const transformed: ManagedAppliance[] = (data || []).map((item) => ({
        id: item.id,
        building_id: item.building_id,
        building_name: (item.buildings as { name?: string })?.name || "Unknown Building",
        org_appliance_id: (item.org_appliances as { id?: string })?.id || "",
        name: (item.org_appliances as { name?: string })?.name || "Unknown Appliance",
        type: "appliance", // Default type since column doesn't exist
        rated_power_kw: (item.org_appliances as { rated_power_kw?: number | null })?.rated_power_kw ?? null,
        status: item.is_enabled ? "active" : "inactive",
        notes: null as null,
      }));


      setAppliances(transformed);
    } catch (err) {
      console.error("Error fetching managed appliances:", err);
      setError(
        err instanceof Error ? err.message : "Failed to load appliances",
      );
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
export function getAppliancePowerMap(
  appliances: ManagedAppliance[],
): Record<string, number | null> {
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
