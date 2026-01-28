/**
 * Hook for managing organization-level appliances
 */

import { useState, useEffect, useCallback, useRef } from "react";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import { isSupabaseEnabled } from "@/lib/env";
import { toast } from "sonner";

// Track if schema warning has been logged (session-level)
let schemaWarningLogged = false;

/**
 * Check if error is a Supabase schema/table missing error
 */
function isSchemaError(err: unknown): boolean {
  if (!err || typeof err !== 'object') return false;
  const error = err as { code?: string; message?: string };
  return ['PGRST205', '42703', '42P01', 'PGRST200'].includes(error.code || '') ||
    (error.message?.includes('does not exist') ?? false);
}

export interface OrgAppliance {
  id: string;
  name: string;
  slug: string;
  category: string;
  description: string | null;
  created_at: string;
  has_model: boolean;
}

export interface UseOrgAppliancesResult {
  appliances: OrgAppliance[];
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
  createAppliance: (
    name: string,
    slug: string,
    category: string,
    description?: string,
  ) => Promise<string | null>;
  updateAppliance: (
    id: string,
    updates: Partial<Pick<OrgAppliance, "name" | "category" | "description">>,
  ) => Promise<boolean>;
  deleteAppliance: (id: string) => Promise<boolean>;
}

export function useOrgAppliances(): UseOrgAppliancesResult {
  const { user, isAuthenticated } = useAuth();
  const [appliances, setAppliances] = useState<OrgAppliance[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // Track if schema is unavailable (don't retry)
  const schemaUnavailable = useRef(false);

  const fetchAppliances = useCallback(async () => {
    if (!isAuthenticated || !user || !isSupabaseEnabled() || schemaUnavailable.current) {
      setAppliances([]);
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      // Fetch from 'appliances' table (actual schema)
      const { data, error: fetchError } = await supabase
        .from("appliances")
        .select(
          `
          id,
          name,
          category,
          typical_power_kw,
          is_enabled,
          created_at
        `,
        )
        .eq("is_enabled", true)
        .order("name");

      if (fetchError) {
        // Check if it's a schema error (table/column doesn't exist)
        if (isSchemaError(fetchError)) {
          schemaUnavailable.current = true;
          if (!schemaWarningLogged) {
            console.warn("[useOrgAppliances] Supabase schema not available. Using empty state.");
            schemaWarningLogged = true;
          }
          setAppliances([]);
          setLoading(false);
          return;
        }
        throw fetchError;
      }

      const transformed: OrgAppliance[] = (data || []).map((a) => ({
        id: a.id,
        name: a.name,
        slug: a.name.toLowerCase().replace(/\s+/g, '_'), // Generate slug from name
        category: a.category,
        description: null, // Not in actual schema
        created_at: a.created_at,
        has_model: false, // Models table relation not available in actual schema
      }));

      setAppliances(transformed);
    } catch (err) {
      console.error("Error fetching org appliances:", err);
      setError(
        err instanceof Error ? err.message : "Failed to load appliances",
      );
    } finally {
      setLoading(false);
    }
  }, [user, isAuthenticated]);

  useEffect(() => {
    fetchAppliances();
  }, [fetchAppliances]);

  const createAppliance = useCallback(
    async (
      name: string,
      slug: string,
      category: string,
      description?: string,
    ): Promise<string | null> => {
      if (!user) return null;

      try {
        // Using 'appliances' table with actual schema columns
        const { data, error } = await supabase
          .from("appliances")
          .insert({
            name,
            category,
            typical_power_kw: null, // Default, can be updated later
            is_enabled: true,
          })
          .select("id")
          .single();

        if (error) throw error;

        toast.success("Appliance created");
        await fetchAppliances();
        return data.id;
      } catch (err) {
        console.error("Error creating appliance:", err);
        toast.error("Failed to create appliance");
        return null;
      }
    },
    [user, fetchAppliances],
  );

  const updateAppliance = useCallback(
    async (
      id: string,
      updates: Partial<Pick<OrgAppliance, "name" | "category" | "description">>,
    ): Promise<boolean> => {
      try {
        // Map to actual schema columns
        const { error } = await supabase
          .from("appliances")
          .update({
            name: updates.name,
            category: updates.category,
          })
          .eq("id", id);

        if (error) throw error;

        toast.success("Appliance updated");
        await fetchAppliances();
        return true;
      } catch (err) {
        console.error("Error updating appliance:", err);
        toast.error("Failed to update appliance");
        return false;
      }
    },
    [fetchAppliances],
  );

  const deleteAppliance = useCallback(
    async (id: string): Promise<boolean> => {
      try {
        const { error } = await supabase
          .from("appliances")
          .delete()
          .eq("id", id);

        if (error) throw error;

        toast.success("Appliance deleted");
        await fetchAppliances();
        return true;
      } catch (err) {
        console.error("Error deleting appliance:", err);
        toast.error("Failed to delete appliance");
        return false;
      }
    },
    [fetchAppliances],
  );

  return {
    appliances,
    loading,
    error,
    refetch: fetchAppliances,
    createAppliance,
    updateAppliance,
    deleteAppliance,
  };
}
