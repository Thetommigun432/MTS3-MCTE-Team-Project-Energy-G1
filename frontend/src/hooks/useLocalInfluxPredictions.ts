import { useState, useEffect } from "react";
import type { NilmDataRow } from "@/types/energy";

interface UseLocalInfluxPredictionsOptions {
  buildingId?: string;
  startDate?: Date;
  endDate?: Date;
  enabled?: boolean;
}

interface UseLocalInfluxPredictionsResult {
  data: NilmDataRow[];
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

interface LocalPredictionRow {
  _time: string;
  building_id?: string;
  appliance_name: string;
  predicted_kw?: number;
  confidence?: number;
}

interface LocalPredictionsResponse {
  success: boolean;
  data: LocalPredictionRow[];
  error?: string;
}

/**
 * React hook to fetch NILM predictions from local InfluxDB via the local API server
 *
 * @param options Configuration options
 * @returns Prediction data, loading state, error state, and refetch function
 */
export function useLocalInfluxPredictions({
  buildingId = "local",
  startDate,
  endDate,
  enabled = true,
}: UseLocalInfluxPredictionsOptions = {}): UseLocalInfluxPredictionsResult {
  const [data, setData] = useState<NilmDataRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refetchTrigger, setRefetchTrigger] = useState(0);

  const refetch = () => {
    setRefetchTrigger((prev) => prev + 1);
  };

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      return;
    }

    let isMounted = true;

    async function fetchPredictions() {
      try {
        setLoading(true);
        setError(null);

        // Build query parameters
        const params = new URLSearchParams({
          buildingId,
        });

        // Add time range if specified
        if (startDate) {
          params.append("start", startDate.toISOString());
        } else {
          params.append("start", "-7d"); // Default to last 7 days
        }

        if (endDate) {
          params.append("end", endDate.toISOString());
        } else {
          params.append("end", "now()");
        }

        // Fetch from local API server (proxied by Vite dev server)
        const response = await fetch(`/api/local/predictions?${params}`);

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = (await response.json()) as LocalPredictionsResponse;

        if (!result.success) {
          throw new Error(result.error || "Unknown error from local API");
        }

        if (!isMounted) return;

        // Transform InfluxDB rows to NilmDataRow format
        const transformedData = transformInfluxToNilm(result.data ?? []);
        setData(transformedData);
      } catch (err) {
        if (!isMounted) return;

        const errorMessage =
          err instanceof Error ? err.message : "Failed to fetch predictions";
        setError(errorMessage);
        console.error("Error fetching local predictions:", err);

        // Provide helpful error messages
        if (
          errorMessage.includes("Failed to fetch") ||
          errorMessage.includes("ECONNREFUSED")
        ) {
          setError(
            "Cannot connect to local API server. Make sure it is running with: npm run local:server",
          );
        } else if (errorMessage.includes("404")) {
          setError(
            "Local API endpoint not found. Check Vite proxy configuration.",
          );
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    }

    fetchPredictions();

    // Cleanup function
    return () => {
      isMounted = false;
    };
  }, [buildingId, startDate, endDate, enabled, refetchTrigger]);

  return { data, loading, error, refetch };
}

/**
 * Transform InfluxDB query results to NilmDataRow format
 * Groups predictions by timestamp and aggregates appliance data
 *
 * @param influxRows Raw rows from InfluxDB query
 * @returns Array of NilmDataRow objects suitable for charts and UI
 */
function transformInfluxToNilm(
  influxRows: LocalPredictionRow[],
): Array<
  NilmDataRow & {
    confidence: number;
    metadata: { building_id?: string; source: string };
  }
> {
  // Group predictions by timestamp
  const grouped = new Map<
    string,
    NilmDataRow & {
      confidence: number;
      metadata: { building_id?: string; source: string };
    }
  >();

  influxRows.forEach((row) => {
    const timeKey = new Date(row._time).toISOString();

    if (!grouped.has(timeKey)) {
      grouped.set(timeKey, {
        time: new Date(row._time),
        aggregate: 0,
        appliances: {},
        confidence: 0,
        metadata: {
          building_id: row.building_id,
          source: "local_influx",
        },
      });
    }

    const entry = grouped.get(timeKey)!;

    // Add appliance prediction
    const applianceName = row.appliance_name;
    const predictedKw = row.predicted_kw || 0;
    const confidenceScore = row.confidence || 0;

    entry.appliances[applianceName] = predictedKw;
    entry.aggregate += predictedKw;

    // Track maximum confidence score across all appliances
    entry.confidence = Math.max(entry.confidence, confidenceScore);
  });

  // Convert grouped data to array and sort by time
  const result = Array.from(grouped.values()).sort(
    (a, b) => a.time.getTime() - b.time.getTime(),
  );

  return result;
}
