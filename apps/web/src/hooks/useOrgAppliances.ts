/**
 * Hook for managing organization-level appliances
 */

import { useState, useEffect, useCallback } from "react";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";

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

  const fetchAppliances = useCallback(async () => {
    if (!isAuthenticated || !user) {
      setAppliances([]);
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      // Fetch org appliances with model info
      const { data, error: fetchError } = await supabase
        .from("org_appliances")
        .select(
          `
          id,
          name,
          slug,
          category,
          description,
          created_at,
          models(id)
        `,
        )
        .order("name");

      if (fetchError) throw fetchError;

      const transformed: OrgAppliance[] = (data || []).map((a) => ({
        id: a.id,
        name: a.name,
        slug: a.slug,
        category: a.category,
        description: a.description,
        created_at: a.created_at,
        has_model: (a.models?.length || 0) > 0,
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
        const { data, error } = await supabase
          .from("org_appliances")
          .insert({
            user_id: user.id,
            name,
            slug,
            category,
            description: description || null,
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
        const { error } = await supabase
          .from("org_appliances")
          .update(updates)
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
          .from("org_appliances")
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
