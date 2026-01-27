import { useState, useEffect, useCallback } from "react";
import { supabase } from "@/integrations/supabase/client";
import { isSupabaseEnabled } from "@/lib/env";
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
    if (!user || !isSupabaseEnabled()) {
      setAppliances([]);
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      // Fetch org_appliances directly since building_appliances table is missing
      const { data, error: fetchError } = await supabase
        .from("org_appliances")
        .select("id, name, rated_power_kw")
        .order("name");

      if (fetchError) throw fetchError;

      // Transform the data
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const transformed: ManagedAppliance[] = (data || []).map((item: any) => ({
        id: item.id, // Using org_appliance id as the main id for now
        building_id: "unassigned",
        building_name: "Unassigned",
        org_appliance_id: item.id,
        name: item.name || "Unknown Appliance",
        type: "appliance",
        rated_power_kw: item.rated_power_kw ?? null,
        status: "active", // Default to active since we don't have is_enabled
        notes: null,
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

/**
 * Sync appliances from backend models to Supabase org_appliances
 */
/**
 * Sync appliances from backend models to Supabase org_appliances
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function syncAppliancesFromModels(models: any[]) {
  if (!models || models.length === 0) return;

  // Get current user for user_id field
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) {
    console.error("Cannot sync appliances: No authenticated user");
    return;
  }

  // Extract unique appliances from models
  const uniqueAppliances = new Map<string, string>(); // slug -> name

  models.forEach((model) => {
    if (!model.is_active) return;

    // Handle multi-head models
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    if (model.heads && model.heads.length > 0) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      model.heads.forEach((h: any) => {
        if (h.appliance_id)
          uniqueAppliances.set(
            h.appliance_id,
            formatApplianceName(h.appliance_id),
          );
      });
    }
    // Handle single-head models
    else if (model.appliance_id && model.appliance_id !== "multi") {
      uniqueAppliances.set(
        model.appliance_id,
        formatApplianceName(model.appliance_id),
      );
    }
  });

  if (uniqueAppliances.size === 0) return;

  // Upsert into org_appliances
  const upsertData = Array.from(uniqueAppliances.entries()).map(
    ([slug, name]) => ({
      name: name,
      slug: slug,
      rated_power_kw: 0,
      user_id: user.id,
    }),
  );

  // NOTE: We rely on the Supabase policy to allow upserts if user owns the record or is admin
  const { error } = await supabase
    .from("org_appliances")
    .upsert(upsertData, { onConflict: "slug", ignoreDuplicates: true });

  if (error) {
    console.error("Failed to sync appliances:", error);
  } else {
    console.log(`Synced ${upsertData.length} appliances to Supabase`);
  }
}

function formatApplianceName(slug: string): string {
  return slug
    .split(/[-_]/)
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}
