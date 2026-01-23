/**
 * Hook for managing NILM models
 * Uses Backend API source of truth.
 */
import { useState, useEffect, useCallback } from "react";
import { energyApi, Model } from "@/services/energy";
import { toast } from "sonner";
import { getDataSource } from "@/lib/dataSource";

export interface UseModelsResult {
  models: Model[]; // Using backend Model type
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
  // Management actions disabled in API mode until backend adds them
  registerModel: () => Promise<null>;
  uploadModelVersion: () => Promise<boolean>;
  setActiveVersion: () => Promise<boolean>;
}

export function useModels(_buildingId?: string | null): UseModelsResult {
  const [models, setModels] = useState<Model[]>([]);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchModels = useCallback(async () => {
    // In Demo mode, we might want simulated models, but for now we settle for empty or mocked
    if (getDataSource() === 'demo') {
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const response = await energyApi.getModels();
      // Backend returns { models: [...], count: ... }
      setModels(response.models || []);

    } catch (err) {
      console.error("Error fetching models:", err);
      setError(err instanceof Error ? err.message : "Failed to load models");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  // Stubs for actions not yet supported by Backend API
  const registerModel = async () => {
    toast.info("Model registration is managed via backend configuration.");
    return null;
  };

  const uploadModelVersion = async () => {
    toast.info("Model uploading is managed via backend configuration.");
    return false;
  };

  const setActiveVersion = async () => {
    toast.info("Active version is managed via backend configuration.");
    return false;
  };

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
