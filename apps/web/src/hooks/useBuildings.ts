/**
 * Hook for managing buildings
 * Uses Backend API for discovery.
 */

import { useState, useEffect, useCallback } from "react";
import { energyApi } from "@/services/energy";
import { toast } from "sonner";
import { getDataSource } from "@/lib/dataSource";

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
  createBuilding: (name: string) => Promise<string | null>;
  updateBuilding: (id: string, updates: Partial<Building>) => Promise<boolean>;
  deleteBuilding: (id: string) => Promise<boolean>;
}

// Demo building
const DEMO_BUILDING: Building = {
  id: "demo-residential-001",
  name: "Residential Demo Building",
  address: "Demo Location",
  description: "Sample building for demonstration",
  status: "active",
  total_appliances: 5,
  created_at: new Date().toISOString(),
};

export function useBuildings(): UseBuildingsResult {
  const [buildings, setBuildings] = useState<Building[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchBuildings = useCallback(async () => {
    const dataSource = getDataSource();

    if (dataSource === 'demo') {
      setBuildings([DEMO_BUILDING]);
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      // Discovery via Backend API
      const response = await energyApi.getBuildings();
      const buildingIds = response.buildings || [];

      // Backend only returns IDs, so we map them to Building objects
      const mappedBuildings: Building[] = buildingIds.map(id => ({
        id,
        name: id, // Fallback name
        address: "Discovered from Data",
        description: "Active building in InfluxDB",
        status: "active",
        total_appliances: 0, // Unknown without analytics query
        created_at: new Date().toISOString()
      }));

      // If no buildings found in API mode, ensure empty list (or maybe show demo if configured?)
      setBuildings(mappedBuildings);

    } catch (err) {
      console.error("Error fetching buildings:", err);
      setError(err instanceof Error ? err.message : "Failed to load buildings");
      // Fallback to empty
      setBuildings([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchBuildings();
  }, [fetchBuildings]);

  // Mutations disabled in API Discovery mode
  // Mutations disabled in API Discovery mode
  const createBuilding = async (_name: string): Promise<string | null> => {
    toast.info("Building creation disabled in API Discovery Mode");
    return null;
  };

  const updateBuilding = async (_id: string, _updates: Partial<Building>): Promise<boolean> => {
    toast.info("Building updates disabled in API Discovery Mode");
    return false;
  };

  const deleteBuilding = async (_id: string): Promise<boolean> => {
    toast.info("Building deletion disabled in API Discovery Mode");
    return false;
  };

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

