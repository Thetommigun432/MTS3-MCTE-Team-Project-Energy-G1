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
  ON_THRESHOLD,
  computeConfidence,
  computeEnergyKwh,
  getTopAppliancesByEnergy,
} from "@/hooks/useNilmCsvData";
import {
  useManagedAppliances,
  ManagedAppliance,
} from "@/hooks/useManagedAppliances";
import { useBuildings } from "@/hooks/useBuildings";
import { energyApi, isEnergyApiAvailable, ReadingDataPoint, ApiError } from "@/services/energy";
import { startOfDayLocal, endOfDayLocal } from "@/lib/dateUtils";
import { useAuth } from "@/contexts/AuthContext";


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

  // Determine initial mode from environment variables
  // CRITICAL: Production defaults to API mode; Demo mode requires explicit opt-in
  const [mode, setModeInternal] = useState<DataMode>(() => {
    if (import.meta.env.VITE_DEMO_MODE === "true") return "demo";
    return "api"; // Default to API mode in production
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
    const start = startOfDayLocal(
      new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000),
    );
    return { start, end };
  });

  // Local mode removed - unified backend handles all data via API mode

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
      const response = await energyApi.getReadings({
        building_id: selectedBuildingId,
        start: startOfDayLocal(
          new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
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
        // Backend returns "value", we map to "aggregate"
        // Backend returns additional fields, we check for appliance breakdowns if available
        // Note: The backend /readings endpoint currently returns aggregate data.
        // If appliance breakdowns are needed, we might need to fetch predictions or 
        // rely on the backend to provide disaggregated data in the same call.
        // For now, we map the aggregate value.

        // Use type assertion for flexible data handling from backend
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const anyRecord = r as any;

        return {
          time: new Date(r.time),
          aggregate: r.value,
          // If the API returns appliance estimates in the extra fields, use them
          // Otherwise default to empty structure (will be filled by predictions if fetched separately)
          appliances: anyRecord.appliances || {},
          confidence: anyRecord.confidence || 0,
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
      setSelectedBuildingId(supabaseBuildings[0].id);
    } else if (mode === "demo" && !selectedBuildingId) {
      // Auto-select demo building
      setSelectedBuildingId("demo-residential-001");
    }
  }, [mode, selectedBuildingId, supabaseBuildings]);

  // Fetch API appliances when building changes
  const [apiAppliances, setApiAppliances] = useState<string[]>([]);

  useEffect(() => {
    if (mode === "api" && selectedBuildingId && isAuthenticated) {
      energyApi.getAppliances(selectedBuildingId)
        .then(res => setApiAppliances(res.appliances || []))
        .catch(err => console.error("Failed to fetch appliances:", err));
    } else {
      setApiAppliances([]);
    }
  }, [mode, selectedBuildingId, isAuthenticated]);

  // Build a lookup map for managed appliance metadata
  const managedApplianceMap = useMemo(() => {
    const map: Record<string, ManagedAppliance> = {};
    managedAppliances.forEach((a) => {
      map[a.name] = a;
    });
    return map;
  }, [managedAppliances]);

  // Appliances list: API usage vs Demo/Managed fallback
  const appliances = useMemo(() => {
    if (mode === "api") {
      return apiAppliances;
    }

    // In demo mode, fallback to managed or demo
    const managedNames = managedAppliances.map((a) => a.name);
    if (managedNames.length > 0) {
      return managedNames;
    }
    return demoAppliances;
  }, [mode, apiAppliances, managedAppliances, demoAppliances]);

  // Buildings from Supabase for API mode, demo building for demo mode
  const buildings = useMemo((): Building[] => {
    if (mode === "api" && supabaseBuildings.length > 0) {
      return supabaseBuildings.map(b => ({
        id: b.id,
        name: b.name,
        address: b.address,
        status: b.status as 'active' | 'inactive' | 'maintenance' | undefined
      }));
    }
    // For demo mode, return demo building with realistic details
    // Based on the NILM dataset which contains appliances from a residential building
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
    const latestRow = filteredRows[filteredRows.length - 1];
    let totalConfidence = 0;
    let count = 0;
    appliances.forEach((name) => {
      const kw = latestRow.appliances[name] || 0;
      totalConfidence += computeConfidence(kw);
      count++;
    });
    const avgConfidence = count > 0 ? totalConfidence / count : 0;
    const level: "Good" | "Medium" | "Low" =
      avgConfidence >= 0.5 ? "Good" : avgConfidence >= 0.3 ? "Medium" : "Low";

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

    // If we have managed appliances, use their names; otherwise use demo appliances
    const applianceNames =
      managedAppliances.length > 0
        ? managedAppliances.map((a) => a.name)
        : demoAppliances;

    return applianceNames
      .map((name) => {
        const estKw = latestRow.appliances[name] || 0;
        const managed = managedApplianceMap[name];

        return {
          name,
          on: estKw >= ON_THRESHOLD,
          confidence: computeConfidence(estKw),
          est_kW: estKw,
          rated_kW: managed?.rated_power_kw ?? null,
          type: managed?.type,
          building_name: managed?.building_name,
        };
      })
      .sort((a, b) => b.est_kW - a.est_kW);
  }, [filteredRows, managedAppliances, demoAppliances, managedApplianceMap]);

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
