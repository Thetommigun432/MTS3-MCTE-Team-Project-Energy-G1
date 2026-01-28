import {
  createContext,
  useContext,
  useState,
  useMemo,
  useEffect,
  useCallback,
  ReactNode,
} from "react";
import {
  DataMode,
  DateRange,
  InsightData,
  ApplianceStatus,
  Building,
} from "@/types/energy";
import {
  useNilmCsvData,
  NilmDataRow,
  computeEnergyKwh,
  getTopAppliancesByEnergy,
  isApplianceOn,
} from "@/hooks/useNilmCsvData";
import {
  useManagedAppliances,
  ManagedAppliance,
} from "@/hooks/useManagedAppliances";
import { useBuildings } from "@/hooks/useBuildings";
import { energyApi, isEnergyApiAvailable, ReadingDataPoint, ApiError } from "@/services/energy";
import { startOfDayLocal, endOfDayLocal } from "@/lib/dateUtils";
import { useAuth } from "@/contexts/AuthContext";
import { getDataSource, setDataSource } from "@/lib/dataSource";


interface EnergyContextType {
  mode: DataMode;
  setMode: (mode: DataMode) => void;
  selectedBuilding: string;
  setSelectedBuilding: (building: string) => void;
  selectedBuildingId: string | null;
  setSelectedBuildingId: (id: string | null) => void;
  selectedAppliance: string;
  setSelectedAppliance: (appliance: string) => void;
  dateRange: DateRange;
  setDateRange: (range: DateRange) => void;
  loading: boolean;
  error: string | null;
  filteredRows: NilmDataRow[];
  buildings: Building[];
  appliances: string[];
  insights: InsightData;
  currentApplianceStatus: ApplianceStatus[];
  topAppliances: { name: string; totalKwh: number }[];
  dataDateRange: { min: Date; max: Date } | null;
  isRefreshing: boolean;
  lastRefreshed: Date | null;
  refresh: () => Promise<void>;
  apiError: string | null;
  isApiAvailable: boolean;
  // Managed appliances from Supabase
  managedAppliances: ManagedAppliance[];
  managedAppliancesLoading: boolean;
  refetchManagedAppliances: () => Promise<void>;
}

const EnergyContext = createContext<EnergyContextType | null>(null);

export function EnergyProvider({ children }: { children: ReactNode }) {
  const { isAuthenticated } = useAuth();
  const {
    rows: demoRows,
    appliances: demoAppliances,
    loading: demoLoading,
    error: demoError,
    refetch: demoRefetch,
  } = useNilmCsvData();
  const {
    appliances: managedAppliances,
    loading: managedAppliancesLoading,
    refetch: refetchManagedAppliances,
  } = useManagedAppliances();
  const {
    buildings: supabaseBuildings,
  } = useBuildings();
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [apiRows, setApiRows] = useState<NilmDataRow[]>([]);
  const [apiLoading, setApiLoading] = useState(false);

  // Determine initial mode from centralized dataSource (which reads localStorage)
  const [mode, setModeInternal] = useState<DataMode>(() => {
    return getDataSource();
  });

  const [selectedBuilding, setSelectedBuilding] = useState("Demo Building");
  const [selectedBuildingId, setSelectedBuildingId] = useState<string | null>(
    null,
  );
  const [selectedAppliance, setSelectedAppliance] = useState("All");

  // Date range for local influx queries (initialized below)
  const [dateRange, setDateRangeInternal] = useState<DateRange>(() => {
    const now = new Date();
    const end = endOfDayLocal(now);
    // Default to 30 days to capture more historical data
    const start = startOfDayLocal(
      new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000),
    );
    return { start, end };
  });

  // Check if API is available (user must be authenticated for API mode)
  const isApiAvailable = useMemo(
    () => isEnergyApiAvailable() || Boolean(isAuthenticated && selectedBuildingId),
    [isAuthenticated, selectedBuildingId],
  );

  // STRICT mode switching: Each mode shows ONLY its own data (empty if none)
  const rows = useMemo(() => {
    if (mode === "api") {
      // API mode: return API rows only, even if empty
      return apiRows;
    }
    // Demo mode: return demo rows only
    return demoRows;
  }, [mode, apiRows, demoRows]);

  const loading = mode === "api" ? apiLoading : demoLoading;
  const error =
    mode === "api" && apiError
      ? apiError
      : mode === "demo"
        ? demoError
        : null;

  // Mode setter that clears stale data when switching
  const setMode = useCallback(
    (newMode: DataMode) => {
      if (newMode === mode) return;

      // Update centralized data source (persists to localStorage + notifies hooks)
      setDataSource(newMode);

      // Clear API data when switching away from API mode
      if (mode === "api") {
        setApiRows([]);
        setApiError(null);
      }

      // Clear any errors when switching modes
      setApiError(null);

      setModeInternal(newMode);
      setLastRefreshed(null);
    },
    [mode],
  );

  // Fetch readings from unified backend API
  const fetchApiReadings = useCallback(async () => {
    if (!isAuthenticated || !selectedBuildingId) {
      return [];
    }

    try {
      // Use the unified backend API instead of Edge Functions
      // Query all historical data (up to 1 year back) to show complete InfluxDB data
      const response = await energyApi.getReadings({
        building_id: selectedBuildingId,
        start: startOfDayLocal(
          new Date(Date.now() - 365 * 24 * 60 * 60 * 1000),
        ).toISOString(),
        end: endOfDayLocal(new Date()).toISOString(),
        resolution: "15m" // Request downsampled data for performance
      });

      if (!response.data) {
        throw new Error("Failed to fetch readings: No data in response");
      }

      // Transform Backend API response (ReadingDataPoint[]) to NilmDataRow format
      // Access response.data because response is ReadingsResponse (which has .data field)
      // api.ts already unwraps the fetch response body.
      const transformed: NilmDataRow[] = response.data.map((r: ReadingDataPoint) => {
        // Use type assertion for flexible data handling from backend
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const anyRecord = r as any;

        return {
          time: new Date(r.time),
          aggregate: r.value,
          appliances: anyRecord.appliances || {},
          confidence: anyRecord.confidence || {},  // Per-appliance confidence from backend
        };
      });

      return transformed.sort((a, b) => a.time.getTime() - b.time.getTime());
    } catch (err) {
      console.warn("Failed to fetch API readings:", err);
      throw err;
    }
  }, [isAuthenticated, selectedBuildingId]);

  // Refresh function - only triggered by user action
  const refresh = useCallback(async () => {
    if (isRefreshing || loading) return;
    setIsRefreshing(true);
    setApiError(null);

    if (mode === "api") {
      // API mode: fetch from API only, do NOT fall back to demo
      if (isAuthenticated && selectedBuildingId) {
        // Supabase edge function mode
        try {
          setApiLoading(true);
          const readings = await fetchApiReadings();

          if (readings.length > 0) {
            setApiRows(readings);
            setApiError(null);
          } else {
            // No data but request succeeded - show warning, keep empty
            setApiError(
              "No readings available from API for selected building/date range",
            );
          }
        } catch (err) {
          console.warn("API fetch failed:", err);
          // Use specific error message if available
          if (err instanceof ApiError) {
            setApiError(`${err.getUserMessage()} (${err.errorType})`);
          } else {
            setApiError(
              "API unreachable — click Refresh to retry or switch to Demo mode",
            );
          }
        } finally {
          setApiLoading(false);
        }
      } else if (isApiAvailable) {
        // Legacy API mode removed - use Supabase Auth to access data
        setApiError("Legacy API access unavailable. Please log in.");
        setApiLoading(false);
      } else {
        // API mode but missing configuration or building selection
        if (isAuthenticated && !selectedBuildingId) {
          setApiError("Select a building to load data in API mode.");
        } else {
          setApiError(
            "API mode requires configuration. Set VITE_BACKEND_URL or switch to Demo mode.",
          );
        }
      }
    } else {
      // Demo mode - just refetch CSV
      await demoRefetch();
    }

    setLastRefreshed(new Date());
    setIsRefreshing(false);
  }, [
    mode,
    isApiAvailable,
    isAuthenticated,
    selectedBuildingId,
    isRefreshing,
    loading,
    demoRefetch,
    fetchApiReadings,
  ]);

  // Set initial lastRefreshed when data first loads
  useEffect(() => {
    if (!demoLoading && demoRows.length > 0 && !lastRefreshed) {
      setLastRefreshed(new Date());
    }
  }, [demoLoading, demoRows.length, lastRefreshed]);

  // Compute data date range
  const dataDateRange = useMemo(() => {
    if (rows.length === 0) return null;
    return {
      min: rows[0].time,
      max: rows[rows.length - 1].time,
    };
  }, [rows]);

  // Wrapper to ensure timezone-safe date range
  const setDateRange = useCallback((range: DateRange) => {
    setDateRangeInternal({
      start: startOfDayLocal(range.start),
      end: endOfDayLocal(range.end),
    });
  }, []);

  // Update date range when data loads
  useEffect(() => {
    if (dataDateRange) {
      const end = endOfDayLocal(dataDateRange.max);
      const start = startOfDayLocal(
        new Date(dataDateRange.max.getTime() - 7 * 24 * 60 * 60 * 1000),
      );
      if (start < dataDateRange.min) {
        setDateRangeInternal({
          start: startOfDayLocal(dataDateRange.min),
          end,
        });
      } else {
        setDateRangeInternal({ start, end });
      }
    }
  }, [dataDateRange]);

  // Auto-select first building when entering API mode or demo mode
  useEffect(() => {
    if (mode === "api" && !selectedBuildingId && supabaseBuildings.length > 0) {
      // Prefer building-1 if it exists (matches simulator default)
      const building1 = supabaseBuildings.find(b => b.id === "building-1");
      setSelectedBuildingId(building1?.id ?? supabaseBuildings[0].id);
    } else if (mode === "demo" && !selectedBuildingId) {
      // Auto-select demo building
      setSelectedBuildingId("demo-residential-001");
    }
  }, [mode, selectedBuildingId, supabaseBuildings]);

  // Auto-fetch data when mode is API and building is selected (initial load)
  useEffect(() => {
    if (mode === "api" && isAuthenticated && selectedBuildingId && apiRows.length === 0 && !apiLoading && !apiError) {
      // Trigger initial data fetch
      refresh();
    }
  }, [mode, isAuthenticated, selectedBuildingId, apiRows.length, apiLoading, apiError, refresh]);

  // Silent background update - fetches new data without showing loading states
  // This prevents the "page refresh" feeling during live updates
  const silentUpdate = useCallback(async () => {
    if (!isAuthenticated || !selectedBuildingId || isRefreshing || apiLoading) {
      return;
    }
    
    try {
      const readings = await fetchApiReadings();
      if (readings.length > 0) {
        // Silently update data without triggering loading states
        setApiRows(readings);
        setLastRefreshed(new Date());
      }
    } catch (err) {
      // Silently fail - don't show error for background updates
      console.debug("Silent update failed:", err);
    }
  }, [isAuthenticated, selectedBuildingId, isRefreshing, apiLoading, fetchApiReadings]);

  // Periodic polling for live updates in API mode (every 30 seconds)
  // Uses silent update to avoid visual "refresh" effect
  useEffect(() => {
    if (mode !== "api" || !isAuthenticated || !selectedBuildingId || apiRows.length === 0) return;
    
    const POLL_INTERVAL = 30000; // 30 seconds - longer interval for smoother UX
    
    const pollInterval = setInterval(() => {
      silentUpdate();
    }, POLL_INTERVAL);
    
    return () => clearInterval(pollInterval);
  }, [mode, isAuthenticated, selectedBuildingId, apiRows.length, silentUpdate]);

  // Build a lookup map for managed appliance metadata
  const managedApplianceMap = useMemo(() => {
    const map: Record<string, ManagedAppliance> = {};
    managedAppliances.forEach((a) => {
      map[a.name] = a;
    });
    return map;
  }, [managedAppliances]);

  // Extract appliance keys from API rows (the actual keys in row.appliances)
  const apiApplianceKeys = useMemo(() => {
    if (mode !== "api" || apiRows.length === 0) return [];
    
    // Collect all unique appliance keys from all rows
    const keySet = new Set<string>();
    for (const row of apiRows) {
      if (row.appliances) {
        Object.keys(row.appliances).forEach(key => keySet.add(key));
      }
    }
    return Array.from(keySet).sort();
  }, [mode, apiRows]);

  // Appliances list: In API mode, prefer keys from actual API data
  const appliances = useMemo(() => {
    if (mode === "api") {
      // Priority 1: Use actual keys from API response data (most accurate)
      if (apiApplianceKeys.length > 0) {
        return apiApplianceKeys;
      }
      // Priority 2: Fall back to managed appliances from Supabase
      if (managedAppliances.length > 0) {
        return managedAppliances.map(a => a.name).sort();
      }
      // Priority 3: Return empty (will populate when data loads)
      return [];
    }

    // In demo mode, fallback to managed or demo
    const managedNames = managedAppliances.map((a) => a.name);
    if (managedNames.length > 0) {
      return managedNames;
    }
    return demoAppliances;
  }, [mode, apiApplianceKeys, managedAppliances, demoAppliances]);

  // Buildings from Supabase for API mode, demo building for demo mode
  // STRICT separation: API mode only shows real buildings, demo mode only shows demo building
  const buildings = useMemo((): Building[] => {
    if (mode === "api") {
      // API mode: Only show real buildings from Supabase (exclude demo buildings)
      if (supabaseBuildings.length > 0) {
        return supabaseBuildings
          .filter(b => !b.id.includes('demo')) // Filter out any demo buildings
          .map(b => ({
            id: b.id,
            name: b.name,
            address: b.address,
            status: b.status as 'active' | 'inactive' | 'maintenance' | undefined
          }));
      }
      // No buildings yet - return empty (user should see "loading" or "no buildings")
      return [];
    }
    // Demo mode: Only return demo building
    return [{
      id: "demo-residential-001",
      name: "Residential Demo Building",
      address: "Demo Location - Training Dataset",
      status: 'active'
    }];
  }, [mode, supabaseBuildings]);

  // Filter rows by date range (with proper timezone handling)
  const filteredRows = useMemo(() => {
    const dateFiltered = rows.filter((row) => {
      return row.time >= dateRange.start && row.time <= dateRange.end;
    });

    // If a specific appliance is selected, filter the appliances data in each row
    if (selectedAppliance !== "All") {
      return dateFiltered.map((row) => ({
        ...row,
        appliances: {
          [selectedAppliance]: row.appliances[selectedAppliance] || 0,
        },
        // Recalculate aggregate to show only the selected appliance's consumption
        aggregate: row.appliances[selectedAppliance] || 0,
      }));
    }

    return dateFiltered;
  }, [rows, dateRange, selectedAppliance]);

  // Determine which appliances to show based on filter
  const activeAppliances = useMemo(() => {
    return selectedAppliance === "All" ? appliances : [selectedAppliance];
  }, [selectedAppliance, appliances]);

  // Get top 5 appliances by energy in the filtered range
  const topAppliances = useMemo(() => {
    return getTopAppliancesByEnergy(filteredRows, activeAppliances, 5);
  }, [filteredRows, activeAppliances]);

  // Compute insights
  const insights = useMemo((): InsightData => {
    if (filteredRows.length === 0) {
      return {
        peakLoad: { kW: 0, timestamp: "" },
        totalEnergy: 0,
        topAppliance: { name: "-", confidence: 0 },
        overallConfidence: { level: "Low", percentage: 0 },
      };
    }

    // Peak load
    const peak = filteredRows.reduce(
      (max, row) => (row.aggregate > max.aggregate ? row : max),
      filteredRows[0],
    );

    // Total energy (aggregate)
    const totalEnergy = filteredRows.reduce(
      (sum, row) => sum + computeEnergyKwh(row.aggregate),
      0,
    );

    // Top appliance by kWh
    const topApp = topAppliances[0] || { name: "-", totalKwh: 0 };

    // Average confidence across all appliances at latest timestamp
    // Use real confidence values from backend (InfluxDB)
    const latestRow = filteredRows[filteredRows.length - 1];
    let totalConfidence = 0;
    let count = 0;
    const confidenceRecord = latestRow.confidence || {};
    appliances.forEach((name) => {
      // Use backend confidence directly (0 if not available)
      const backendConfidence = typeof confidenceRecord === 'object' 
        ? (confidenceRecord[name] ?? 0)
        : 0;
      totalConfidence += backendConfidence;
      count++;
    });
    const avgConfidence = count > 0 ? totalConfidence / count : 0;
    // Confidence thresholds: High (>=0.75), Medium (>=0.50), Low (<0.50)
    const level: "Good" | "Medium" | "Low" =
      avgConfidence >= 0.75 ? "Good" : avgConfidence >= 0.50 ? "Medium" : "Low";

    return {
      peakLoad: { kW: peak.aggregate, timestamp: peak.time.toISOString() },
      totalEnergy,
      topAppliance: { name: topApp.name, confidence: avgConfidence },
      overallConfidence: { level, percentage: avgConfidence * 100 },
    };
  }, [filteredRows, topAppliances, appliances]);

  // Current appliance status (from latest row in range) with managed appliance metadata
  const currentApplianceStatus = useMemo((): ApplianceStatus[] => {
    if (filteredRows.length === 0) return [];

    const latestRow = filteredRows[filteredRows.length - 1];
    const confidenceRecord = latestRow.confidence || {};

    // Use the unified 'appliances' list which:
    // - In API mode: uses API data keys first, then Supabase fallback
    // - In demo mode: uses demo appliances
    const applianceNames = appliances.length > 0 ? appliances : demoAppliances;

    return applianceNames
      .map((name) => {
        const estKw = latestRow.appliances[name] || 0;
        // Try to find managed appliance metadata (may not match if name format differs)
        const managed = managedApplianceMap[name];
        const ratedKw = managed?.rated_power_kw ?? null;
        
        // Use backend confidence directly from InfluxDB (0 if not available)
        const backendConfidence = typeof confidenceRecord === 'object' 
          ? (confidenceRecord[name] ?? 0)
          : 0;
        const confidence = backendConfidence;

        return {
          name,
          on: isApplianceOn(estKw, ratedKw),  // Dynamic threshold based on rated power
          confidence,
          est_kW: estKw,
          rated_kW: ratedKw,
          type: managed?.type,
          building_name: managed?.building_name,
        };
      })
      .sort((a, b) => b.est_kW - a.est_kW);
  }, [filteredRows, appliances, demoAppliances, managedApplianceMap]);

  return (
    <EnergyContext.Provider
      value={{
        mode,
        setMode,
        selectedBuilding,
        setSelectedBuilding,
        selectedBuildingId,
        setSelectedBuildingId,
        selectedAppliance,
        setSelectedAppliance,
        dateRange,
        setDateRange,
        loading,
        error,
        filteredRows,
        buildings,
        appliances,
        insights,
        currentApplianceStatus,
        topAppliances,
        dataDateRange,
        isRefreshing,
        lastRefreshed,
        refresh,
        apiError,
        isApiAvailable,
        managedAppliances,
        managedAppliancesLoading,
        refetchManagedAppliances,
      }}
    >
      {children}
    </EnergyContext.Provider>
  );
}

export function useEnergy() {
  const context = useContext(EnergyContext);
  if (!context) throw new Error("useEnergy must be used within EnergyProvider");
  return context;
}
