/**
 * Hook for managing buildings
 */

import { useState, useEffect, useCallback } from "react";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";

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

export function useBuildings(): UseBuildingsResult {
  const { user, isAuthenticated } = useAuth();
  const [buildings, setBuildings] = useState<Building[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchBuildings = useCallback(async () => {
    if (!isAuthenticated || !user) {
      setBuildings([]);
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const { data, error: fetchError } = await supabase
        .from("buildings")
        .select("*")
        .order("name");

      if (fetchError) throw fetchError;

      // Count appliances for each building
      const buildingsWithCounts = await Promise.all(
        (data || []).map(async (building) => {
          const { count } = await supabase
            .from("building_appliances")
            .select("*", { count: "exact", head: true })
            .eq("building_id", building.id)
            .eq("is_enabled", true);

          return {
            ...building,
            total_appliances: count || 0,
          };
        }),
      );

      setBuildings(buildingsWithCounts);
    } catch (err) {
      console.error("Error fetching buildings:", err);
      setError(err instanceof Error ? err.message : "Failed to load buildings");
    } finally {
      setLoading(false);
    }
  }, [user, isAuthenticated]);

  useEffect(() => {
    fetchBuildings();
  }, [fetchBuildings]);

  const createBuilding = useCallback(
    async (
      name: string,
      address?: string,
      description?: string,
    ): Promise<string | null> => {
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
    [user, fetchBuildings],
  );

  const updateBuilding = useCallback(
    async (
      id: string,
      updates: Partial<
        Pick<Building, "name" | "address" | "description" | "status">
      >,
    ): Promise<boolean> => {
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
    [fetchBuildings],
  );

  const deleteBuilding = useCallback(
    async (id: string): Promise<boolean> => {
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
    [fetchBuildings],
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
