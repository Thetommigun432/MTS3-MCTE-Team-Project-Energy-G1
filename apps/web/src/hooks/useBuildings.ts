/**
 * Hook for managing buildings
 */

import { useState, useEffect, useCallback } from "react";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";
import { isSupabaseEnabled } from "@/lib/env";

export interface Building {
  id: string;
  name: string;
  address: string | null;
  description: string | null;
  status: string;
  total_appliances: number;
  created_at: string;
}

export interface UseBuildingsResult {
  buildings: Building[];
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
  createBuilding: (
    name: string,
    address?: string,
    description?: string,
  ) => Promise<string | null>;
  updateBuilding: (
    id: string,
    updates: Partial<
      Pick<Building, "name" | "address" | "description" | "status">
    >,
  ) => Promise<boolean>;
  deleteBuilding: (id: string) => Promise<boolean>;
}

// Demo building for when Supabase is not configured
const DEMO_BUILDING: Building = {
  id: "demo-building-1",
  name: "Demo Building",
  address: "123 Demo Street",
  description: "Sample building for demonstration",
  status: "active",
  total_appliances: 5,
  created_at: new Date().toISOString(),
};

export function useBuildings(): UseBuildingsResult {
  const { user, isAuthenticated } = useAuth();
  const supabaseEnabled = isSupabaseEnabled();
  const [buildings, setBuildings] = useState<Building[]>(
    supabaseEnabled ? [] : [DEMO_BUILDING]
  );
  const [loading, setLoading] = useState(supabaseEnabled);
  const [error, setError] = useState<string | null>(null);

  const fetchBuildings = useCallback(async () => {
    // Return demo data when Supabase is not enabled
    if (!supabaseEnabled) {
      setBuildings([DEMO_BUILDING]);
      setLoading(false);
      return;
    }

    if (!isAuthenticated || !user) {
      setBuildings([]);
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      // Fetch buildings with appliance count in a single query using Supabase's
      // relation counting feature to avoid N+1 queries
      const { data, error: fetchError } = await supabase
        .from("buildings")
        .select(`
          *,
          building_appliances!left(id, is_enabled)
        `)
        .order("name");

      if (fetchError) throw fetchError;

      // Process the joined data to count enabled appliances
      const buildingsWithCounts = (data || []).map((building) => {
        const appliances = building.building_appliances || [];
        const enabledCount = appliances.filter(
          (a: { is_enabled: boolean }) => a.is_enabled
        ).length;

        // Remove the nested appliances data, keep only the count
        const { building_appliances: _, ...buildingData } = building;
        return {
          ...buildingData,
          total_appliances: enabledCount,
        };
      });

      setBuildings(buildingsWithCounts);
    } catch (err) {
      console.error("Error fetching buildings:", err);
      setError(err instanceof Error ? err.message : "Failed to load buildings");
    } finally {
      setLoading(false);
    }
  }, [user, isAuthenticated, supabaseEnabled]);

  useEffect(() => {
    fetchBuildings();
  }, [fetchBuildings]);

  const createBuilding = useCallback(
    async (
      name: string,
      address?: string,
      description?: string,
    ): Promise<string | null> => {
      if (!supabaseEnabled) {
        toast.info("Building management requires Supabase connection");
        return null;
      }
      if (!user) return null;

      try {
        const { data, error } = await supabase
          .from("buildings")
          .insert({
            user_id: user.id,
            name,
            address: address || null,
            description: description || null,
          })
          .select("id")
          .single();

        if (error) throw error;

        toast.success("Building created");
        await fetchBuildings();
        return data.id;
      } catch (err) {
        console.error("Error creating building:", err);
        toast.error("Failed to create building");
        return null;
      }
    },
    [user, fetchBuildings, supabaseEnabled],
  );

  const updateBuilding = useCallback(
    async (
      id: string,
      updates: Partial<
        Pick<Building, "name" | "address" | "description" | "status">
      >,
    ): Promise<boolean> => {
      if (!supabaseEnabled) {
        toast.info("Building management requires Supabase connection");
        return false;
      }

      try {
        const { error } = await supabase
          .from("buildings")
          .update(updates)
          .eq("id", id);

        if (error) throw error;

        toast.success("Building updated");
        await fetchBuildings();
        return true;
      } catch (err) {
        console.error("Error updating building:", err);
        toast.error("Failed to update building");
        return false;
      }
    },
    [fetchBuildings, supabaseEnabled],
  );

  const deleteBuilding = useCallback(
    async (id: string): Promise<boolean> => {
      if (!supabaseEnabled) {
        toast.info("Building management requires Supabase connection");
        return false;
      }

      try {
        const { error } = await supabase
          .from("buildings")
          .delete()
          .eq("id", id);

        if (error) throw error;

        toast.success("Building deleted");
        await fetchBuildings();
        return true;
      } catch (err) {
        console.error("Error deleting building:", err);
        toast.error("Failed to delete building");
        return false;
      }
    },
    [fetchBuildings, supabaseEnabled],
  );

  return {
    buildings,
    loading,
    error,
    refetch: fetchBuildings,
    createBuilding,
    updateBuilding,
    deleteBuilding,
  };
}
