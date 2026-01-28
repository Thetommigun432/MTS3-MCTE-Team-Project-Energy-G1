import { useState, useEffect, useCallback } from "react";
import Papa from "papaparse";
import { supabase } from "@/integrations/supabase/client";
import { getEnv } from "@/lib/env";

export interface NilmDataRow {
  time: Date;
  aggregate: number;
  appliances: Record<string, number>;
  confidence?: Record<string, number>;  // Per-appliance confidence values
  inferenceType?: 'ml' | 'mock' | 'demo';
  modelVersion?: string;
}

export interface NilmData {
  rows: NilmDataRow[];
  appliances: string[];
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

const APPLIANCE_COLUMNS = [
  "RangeHood",
  "Dryer",
  "Stove",
  "GarageCabinet",
  "ChargingStation_Socket",
  "Oven",
  "RainwaterPump",
  "SmappeeCharger",
  "Dishwasher",
  "HeatPump",
  "HeatPump_Controller",
  "WashingMachine",
];

// ON/OFF threshold configuration
// MIN_ON_THRESHOLD: absolute minimum (20W) to reject sensor noise
// ON_RATIO: percentage of rated power (5%) for scaled threshold
// Formula: on = estKw >= max(MIN_ON_THRESHOLD, ratedKw * ON_RATIO)
export const MIN_ON_THRESHOLD = 0.02;  // 20W minimum
export const ON_RATIO = 0.05;          // 5% of rated power
export const ON_THRESHOLD = 0.05;      // Legacy fallback (50W) when rated power unknown

/**
 * Compute dynamic ON threshold for an appliance based on its rated power.
 * - If ratedKw is provided: max(20W, 5% of rated power)
 * - If ratedKw is unknown: fallback to 50W
 * 
 * Examples:
 * - RangeHood (0.5kW rated): max(0.02, 0.5*0.05) = max(0.02, 0.025) = 0.025 kW (25W)
 * - HeatPump (5.0kW rated): max(0.02, 5.0*0.05) = max(0.02, 0.25) = 0.25 kW (250W)
 * - EVCharger (7.5kW rated): max(0.02, 7.5*0.05) = max(0.02, 0.375) = 0.375 kW (375W)
 */
export function computeOnThreshold(ratedKw: number | null | undefined): number {
  if (ratedKw != null && ratedKw > 0) {
    return Math.max(MIN_ON_THRESHOLD, ratedKw * ON_RATIO);
  }
  return ON_THRESHOLD; // Legacy fallback
}

/**
 * Determine if an appliance is ON based on estimated power and rated power.
 */
export function isApplianceOn(estKw: number, ratedKw: number | null | undefined): boolean {
  return estKw >= computeOnThreshold(ratedKw);
}

/**
 * Compute fallback confidence from power reading when backend doesn't provide confidence.
 * This is a FALLBACK only - prefer using real confidence from API.
 * 
 * Logic:
 * - Very low power (<50W): High confidence (0.80-0.95) - clearly OFF
 * - Very high power (>500W): High confidence (0.75-0.95) - clearly ON
 * - Middle region: Lower confidence (0.45-0.70) - uncertain state
 */
export function computeConfidence(estKw: number): number {
  const absKw = Math.abs(estKw);
  
  if (absKw < 0.05) {
    // Clearly OFF - high confidence
    return 0.80 + 0.15 * (1 - absKw / 0.05);
  } else if (absKw > 0.5) {
    // Clearly ON - confidence increases with power (capped at 2kW)
    const normPower = Math.min(absKw / 2.0, 1.0);
    return 0.75 + 0.20 * normPower;
  } else {
    // Uncertain middle region
    const midpoint = 0.275;
    const distanceFromMid = Math.abs(absKw - midpoint) / midpoint;
    return 0.45 + 0.25 * Math.tanh(distanceFromMid * 2);
  }
}

// Compute energy in kWh (15 min intervals = 0.25h)
export function computeEnergyKwh(kw: number): number {
  return kw * 0.25;
}

// Compute total energy for an array of kW readings
export function computeTotalEnergy(kwReadings: number[]): number {
  return kwReadings.reduce((sum, kw) => sum + computeEnergyKwh(kw), 0);
}

// Get top N appliances by total kWh in the data
export function getTopAppliancesByEnergy(
  rows: NilmDataRow[],
  appliances: string[],
  topN: number = 5,
): { name: string; totalKwh: number }[] {
  const energyByAppliance: Record<string, number> = {};

  appliances.forEach((name) => {
    energyByAppliance[name] = 0;
  });

  rows.forEach((row) => {
    appliances.forEach((name) => {
      const kw = row.appliances[name] || 0;
      energyByAppliance[name] += computeEnergyKwh(kw);
    });
  });

  return Object.entries(energyByAppliance)
    .map(([name, totalKwh]) => ({ name, totalKwh }))
    .sort((a, b) => b.totalKwh - a.totalKwh)
    .slice(0, topN);
}

/**
 * Parse CSV client-side using PapaParse
 * Fallback when Supabase edge function is not available
 */
function parseCSVClientSide(csvText: string): NilmDataRow[] {
  const parsed: NilmDataRow[] = [];

  Papa.parse<Record<string, string>>(csvText, {
    header: true,
    skipEmptyLines: true,
    complete: (results) => {
      results.data.forEach((row) => {
        // Parse time
        const timeStr = row["Time"];
        if (!timeStr) return;

        const time = new Date(timeStr);
        if (isNaN(time.getTime())) return;

        // Parse aggregate - clamp negatives to 0
        let aggregate = parseFloat(row["Aggregate"] || "0");
        if (!isFinite(aggregate)) aggregate = 0;
        aggregate = Math.max(0, aggregate);

        // Parse appliance values - clamp negatives to 0
        const appliances: Record<string, number> = {};
        APPLIANCE_COLUMNS.forEach((colName) => {
          let value = parseFloat(row[colName] || "0");
          if (!isFinite(value)) value = 0;
          appliances[colName] = Math.max(0, value);
        });

        parsed.push({ time, aggregate, appliances });
      });
    },
  });

  // Sort by timestamp ascending
  parsed.sort((a, b) => a.time.getTime() - b.time.getTime());

  return parsed;
}

/**
 * Check if Supabase is configured and available
 */
function isSupabaseAvailable(): boolean {
  const { supabaseEnabled } = getEnv();
  return supabaseEnabled;
}

export function useNilmCsvData(): NilmData {
  const [rows, setRows] = useState<NilmDataRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchCsv = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch the raw CSV file
      const response = await fetch("/data/nilm_ready_dataset.csv", {
        cache: "no-cache",
      });

      if (!response.ok) {
        throw new Error("Failed to load CSV data");
      }

      const csvText = await response.text();

      // Try Supabase edge function first if available
      if (isSupabaseAvailable()) {
        try {
          const { data, error: fnError } = await supabase.functions.invoke(
            "parse-nilm-csv",
            {
              body: { csvContent: csvText },
            },
          );

          if (!fnError && data?.rows && !data?.error) {
            // Successfully parsed via edge function
            const parsed: NilmDataRow[] = data.rows.map(
              (row: {
                time: string;
                aggregate: number;
                appliances: Record<string, number>;
              }) => ({
                time: new Date(row.time),
                aggregate: row.aggregate,
                appliances: row.appliances,
              }),
            );

            setRows(parsed);
            setLoading(false);
            return;
          }

          // Edge function failed, fall through to client-side parsing
          console.warn(
            "Edge function failed, using client-side parsing:",
            fnError?.message || data?.error,
          );
        } catch (edgeErr) {
          console.warn(
            "Edge function unavailable, using client-side parsing:",
            edgeErr,
          );
        }
      }

      // Client-side parsing fallback (always works without Supabase)
      const parsed = parseCSVClientSide(csvText);

      if (parsed.length === 0) {
        throw new Error("No valid data rows found in CSV");
      }

      setRows(parsed);
      setLoading(false);
    } catch (err) {
      console.error("Error loading NILM data:", err);
      setError(err instanceof Error ? err.message : "Unknown error");
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchCsv();
  }, [fetchCsv]);

  return {
    rows,
    appliances: APPLIANCE_COLUMNS,
    loading,
    error,
    refetch: fetchCsv,
  };
}
