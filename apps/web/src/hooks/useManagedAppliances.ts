import { useState, useEffect, useCallback, useRef } from "react";
import { supabase } from "@/integrations/supabase/client";
import { isSupabaseEnabled } from "@/lib/env";
import { isDemoMode, onModeChange, DataSource } from "@/lib/dataSource";
import { useAuth } from "@/contexts/AuthContext";

// Track if schema warning has been logged (session-level)
let schemaWarningLogged = false;

/**
 * Check if error is a Supabase schema/table missing error
 */
function isSchemaError(err: unknown): boolean {
  if (!err || typeof err !== 'object') return false;
  const error = err as { code?: string; message?: string };
  // PGRST205: table not found, 42703: column not found, 42P01: relation not found
  return ['PGRST205', '42703', '42P01', 'PGRST200'].includes(error.code || '') ||
    (error.message?.includes('does not exist') ?? false);
}

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
  // Track if schema is unavailable (don't retry)
  const schemaUnavailable = useRef(false);

  const fetchAppliances = useCallback(async () => {
    // Skip Supabase calls in demo mode or if not configured
    if (!user || !isSupabaseEnabled() || isDemoMode() || schemaUnavailable.current) {
      setAppliances([]);
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      // Fetch from 'appliances' table (the actual table name in the schema)
      const { data, error: fetchError } = await supabase
        .from("appliances")
        .select("id, name, typical_power_kw, category, is_enabled")
        .eq("is_enabled", true)
        .order("name");

      if (fetchError) {
        // Check if it's a schema error (table/column doesn't exist)
        if (isSchemaError(fetchError)) {
          schemaUnavailable.current = true;
          if (!schemaWarningLogged) {
            console.warn("[useManagedAppliances] Supabase schema not available (table missing). Using empty state.");
            schemaWarningLogged = true;
          }
          setAppliances([]);
          setLoading(false);
          return;
        }
        throw fetchError;
      }

      // Transform the data
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const transformed: ManagedAppliance[] = (data || []).map((item: any) => ({
        id: item.id,
        building_id: "unassigned",
        building_name: "Unassigned",
        org_appliance_id: item.id,
        name: item.name || "Unknown Appliance",
        type: item.category || "appliance",
        rated_power_kw: item.typical_power_kw ?? null,
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

  // Subscribe to mode changes and re-fetch
  useEffect(() => {
    const unsubscribe = onModeChange((_newMode: DataSource) => {
      // Reset schema check on mode change
      schemaUnavailable.current = false;
      fetchAppliances();
    });
    return unsubscribe;
  }, [fetchAppliances]);

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
 * Sync appliances from backend models to Supabase appliances table
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function syncAppliancesFromModels(models: any[]) {
  if (!models || models.length === 0) return;

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

  // Upsert into appliances table (actual schema)
  const upsertData = Array.from(uniqueAppliances.entries()).map(
    ([_slug, name]) => ({
      name: name,
      category: "appliance",
      typical_power_kw: null,
      is_enabled: true,
    }),
  );

  // NOTE: We rely on the Supabase policy to allow upserts
  const { error } = await supabase
    .from("appliances")
    .upsert(upsertData, { onConflict: "name", ignoreDuplicates: true });

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
