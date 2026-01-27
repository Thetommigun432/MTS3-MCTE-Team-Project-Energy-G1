import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { parse } from "https://deno.land/std@0.208.0/csv/mod.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform",
};

// Validation constants
const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10MB
const MAX_ROW_COUNT = 50000;
const MAX_KW_VALUE = 100; // 100kW max reasonable value
const MIN_KW_VALUE = -1; // Allow small negative for calibration errors

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

interface NilmDataRow {
  time: string;
  aggregate: number;
  appliances: Record<string, number>;
}

interface ParseResult {
  rows: NilmDataRow[];
  appliances: string[];
  rowCount: number;
}

function validateAndSanitizeNumber(
  value: string | undefined,
  min: number,
  max: number,
): number {
  if (!value || value.trim() === "") return 0;

  const num = parseFloat(value);

  // Reject non-finite numbers
  if (!Number.isFinite(num)) return 0;

  // Clamp to valid range
  return Math.max(min, Math.min(max, num));
}

function validateDate(dateStr: string | undefined): string | null {
  if (!dateStr || dateStr.trim() === "") return null;

  const date = new Date(dateStr);
  if (isNaN(date.getTime())) return null;

  // Validate reasonable date range (2000-2100)
  const year = date.getFullYear();
  if (year < 2000 || year > 2100) return null;

  return date.toISOString();
}

serve(async (req) => {
  // Handle CORS preflight
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { csvContent } = await req.json();

    if (!csvContent || typeof csvContent !== "string") {
      console.error("Invalid request: missing or invalid csvContent");
      return new Response(
        JSON.stringify({ error: "Missing or invalid CSV content" }),
        {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    // Validate file size
    const contentSize = new Blob([csvContent]).size;
    if (contentSize > MAX_FILE_SIZE_BYTES) {
      console.error(
        `File too large: ${contentSize} bytes exceeds ${MAX_FILE_SIZE_BYTES}`,
      );
      return new Response(
        JSON.stringify({
          error: `File size exceeds ${MAX_FILE_SIZE_BYTES / 1024 / 1024}MB limit`,
        }),
        {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    console.log(`Parsing CSV: ${contentSize} bytes`);

    // Parse CSV
    let rawRows: Record<string, string>[];
    try {
      rawRows = parse(csvContent, {
        skipFirstRow: true,
        columns: ["Time", "Aggregate", ...APPLIANCE_COLUMNS],
      }) as Record<string, string>[];
    } catch (parseError) {
      console.error("CSV parse error:", parseError);
      return new Response(JSON.stringify({ error: "Invalid CSV format" }), {
        status: 400,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    // Validate row count
    if (rawRows.length > MAX_ROW_COUNT) {
      console.error(
        `Too many rows: ${rawRows.length} exceeds ${MAX_ROW_COUNT}`,
      );
      return new Response(
        JSON.stringify({ error: `Row count exceeds ${MAX_ROW_COUNT} limit` }),
        {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    console.log(`Processing ${rawRows.length} rows`);

    // Validate and sanitize each row
    const validatedRows: NilmDataRow[] = [];
    let skippedRows = 0;

    for (const row of rawRows) {
      const time = validateDate(row["Time"]);
      if (!time) {
        skippedRows++;
        continue;
      }

      const aggregate = validateAndSanitizeNumber(
        row["Aggregate"],
        0,
        MAX_KW_VALUE,
      );

      const appliances: Record<string, number> = {};
      for (const col of APPLIANCE_COLUMNS) {
        appliances[col] = validateAndSanitizeNumber(
          row[col],
          MIN_KW_VALUE,
          MAX_KW_VALUE,
        );
        // Clamp negative values to 0 for appliances
        if (appliances[col] < 0) appliances[col] = 0;
      }

      validatedRows.push({ time, aggregate, appliances });
    }

    // Sort by time
    validatedRows.sort(
      (a, b) => new Date(a.time).getTime() - new Date(b.time).getTime(),
    );

    console.log(
      `Validated ${validatedRows.length} rows, skipped ${skippedRows} invalid rows`,
    );

    const result: ParseResult = {
      rows: validatedRows,
      appliances: APPLIANCE_COLUMNS,
      rowCount: validatedRows.length,
    };

    return new Response(JSON.stringify(result), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (error: unknown) {
    const errorMessage =
      error instanceof Error ? error.message : "Internal server error";
    console.error("Error in parse-nilm-csv function:", errorMessage);
    return new Response(JSON.stringify({ error: errorMessage }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
