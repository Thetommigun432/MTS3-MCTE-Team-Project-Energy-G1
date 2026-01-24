/**
 * Hook for managing NILM models and versions
 */

import { useState, useEffect, useCallback } from "react";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import { edgeFunctions } from "@/lib/supabaseHelpers";
import { toast } from "sonner";

export interface ModelVersion {
  id: string;
  version: string;
  status: "pending" | "uploading" | "ready" | "failed";
  is_active: boolean;
  trained_at: string | null;
  metrics: Record<string, number> | null;
  training_config: Record<string, unknown> | null;
  created_at: string;
}

export interface Model {
  id: string;
  name: string;
  architecture: string | null;
  is_active: boolean;
  org_appliance_id: string;
  org_appliance_name?: string;
  org_appliance_slug?: string;
  versions: ModelVersion[];
  active_version?: ModelVersion | null;
  created_at: string;
}

export interface UseModelsResult {
  models: Model[];
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
  registerModel: (
    orgApplianceId: string,
    name: string,
    architecture?: string,
  ) => Promise<string | null>;
  uploadModelVersion: (
    modelId: string,
    version: string,
    modelFile: File,
    scalerFile?: File,
  ) => Promise<boolean>;
  setActiveVersion: (versionId: string) => Promise<boolean>;
}

export function useModels(buildingId?: string | null): UseModelsResult {
  const { user, isAuthenticated } = useAuth();
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchModels = useCallback(async () => {
    if (!isAuthenticated || !user) {
      setModels([]);
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      // Fetch models with their org_appliance info and versions
      const { data: modelsData, error: modelsError } = await supabase
        .from("models")
        .select(
          `
          id,
          name,
          architecture,
          is_active,
          org_appliance_id,
          created_at,
          org_appliances!inner(name, slug),
          model_versions(
            id,
            version,
            status,
            is_active,
            trained_at,
            metrics,
            training_config,
            created_at
          )
        `,
        )
        .order("created_at", { ascending: false });

      if (modelsError) throw modelsError;

      // If buildingId is provided, filter to models for appliances assigned to that building
      let filteredModels = modelsData || [];

      if (buildingId) {
        const { data: buildingAppliances } = await supabase
          .from("building_appliances")
          .select("org_appliance_id")
          .eq("building_id", buildingId)
          .eq("is_enabled", true);

        const enabledApplianceIds = new Set(
          buildingAppliances?.map((ba) => ba.org_appliance_id) || [],
        );
        filteredModels = filteredModels.filter((m) =>
          enabledApplianceIds.has(m.org_appliance_id),
        );
      }

      const transformed: Model[] = filteredModels.map((m) => {
        const versions: ModelVersion[] = (m.model_versions || [])
          .map((v) => ({
            id: v.id,
            version: v.version,
            status: v.status as ModelVersion["status"],
            is_active: v.is_active,
            trained_at: v.trained_at,
            metrics: v.metrics as Record<string, number> | null,
            training_config: v.training_config as Record<string, unknown> | null,
            created_at: v.created_at,
          }))
          .sort(
            (a: ModelVersion, b: ModelVersion) =>
              new Date(b.created_at).getTime() -
              new Date(a.created_at).getTime(),
          );

        const activeVersion = versions.find((v) => v.is_active) || null;

        return {
          id: m.id,
          name: m.name,
          architecture: m.architecture,
          is_active: m.is_active,
          org_appliance_id: m.org_appliance_id,
          org_appliance_name: m.org_appliances?.name,
          org_appliance_slug: m.org_appliances?.slug,
          versions,
          active_version: activeVersion,
          created_at: m.created_at,
        };
      });

      setModels(transformed);
    } catch (err) {
      console.error("Error fetching models:", err);
      setError(err instanceof Error ? err.message : "Failed to load models");
    } finally {
      setLoading(false);
    }
  }, [user, isAuthenticated, buildingId]);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  const registerModel = useCallback(
    async (
      orgApplianceId: string,
      name: string,
      architecture?: string,
    ): Promise<string | null> => {
      try {
        const { data, error } = await edgeFunctions.registerModel({
          org_appliance_id: orgApplianceId,
          name,
          architecture,
        });

        if (error || !data?.success) {
          toast.error(
            error?.message || data?.message || "Failed to register model",
          );
          return null;
        }

        toast.success("Model registered successfully");
        await fetchModels();
        return data.model_id || null;
      } catch (err) {
        console.error("Failed to register model:", err);
        toast.error("Failed to register model");
        return null;
      }
    },
    [fetchModels],
  );

  const uploadModelVersion = useCallback(
    async (
      modelId: string,
      version: string,
      modelFile: File,
      scalerFile?: File,
    ): Promise<boolean> => {
      try {
        // Step 1: Create version and get upload URLs
        const { data: uploadData, error: uploadError } =
          await edgeFunctions.createModelVersionUpload({
            model_id: modelId,
            version,
            has_scaler: !!scalerFile,
          });

        if (uploadError || !uploadData?.success) {
          toast.error(
            uploadError?.message ||
            uploadData?.message ||
            "Failed to create upload",
          );
          return false;
        }

        // Step 2: Upload model file
        if (uploadData.model_upload_url) {
          const modelResponse = await fetch(uploadData.model_upload_url, {
            method: "PUT",
            body: modelFile,
            headers: { "Content-Type": "application/octet-stream" },
          });
          if (!modelResponse.ok) {
            toast.error("Failed to upload model file");
            return false;
          }
        }

        // Step 3: Upload scaler file if provided
        if (scalerFile && uploadData.scaler_upload_url) {
          const scalerResponse = await fetch(uploadData.scaler_upload_url, {
            method: "PUT",
            body: scalerFile,
            headers: { "Content-Type": "application/octet-stream" },
          });
          if (!scalerResponse.ok) {
            toast.error("Failed to upload scaler file");
            return false;
          }
        }

        // Step 4: Finalize the version
        const { error: finalizeError } =
          await edgeFunctions.finalizeModelVersion({
            version_id: uploadData.version_id!,
          });

        if (finalizeError) {
          console.error("Finalize error:", finalizeError);
          toast.error("Failed to finalize upload. Please try again.");
          return false;
        }

        toast.success(`Model version ${version} uploaded successfully`);
        await fetchModels();
        return true;
      } catch (err) {
        console.error("Failed to upload model version:", err);
        toast.error("Failed to upload model version");
        return false;
      }
    },
    [fetchModels],
  );

  const setActiveVersion = useCallback(
    async (versionId: string): Promise<boolean> => {
      try {
        const { error } = await edgeFunctions.setActiveVersion({
          version_id: versionId,
        });

        if (error) {
          console.error("Set active version error:", error);
          toast.error("Failed to set active version. Please try again.");
          return false;
        }

        toast.success("Active version updated");
        await fetchModels();
        return true;
      } catch (err) {
        console.error("Failed to set active version:", err);
        toast.error("Failed to set active version");
        return false;
      }
    },
    [fetchModels],
  );

  return {
    models,
    loading,
    error,
    refetch: fetchModels,
    registerModel,
    uploadModelVersion,
    setActiveVersion,
  };
}
