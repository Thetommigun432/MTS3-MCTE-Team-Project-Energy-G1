import { useState, useEffect, useCallback } from "react";
import Papa from "papaparse";
import { supabase } from "@/integrations/supabase/client";

export interface NilmDataRow {
  time: Date;
  aggregate: number;
  appliances: Record<string, number>;
  confidence?: number;
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

// ON/OFF threshold
export const ON_THRESHOLD = 0.05;

// Compute confidence as clamped est_kW / 1.0
export function computeConfidence(estKw: number): number {
  return Math.min(Math.max(estKw / 1.0, 0), 1);
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
  const url = import.meta.env.VITE_SUPABASE_URL;
  const key = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY;
  return Boolean(url && key);
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
