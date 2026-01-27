import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform",
};

interface GetReadingsRequest {
  building_id: string;
  start_date?: string;
  end_date?: string;
}

interface ReadingEntry {
  ts: string;
  aggregate_kw: number;
  appliance_estimates: Record<string, number>;
  confidence?: Record<string, number>;
}

Deno.serve(async (req) => {
  // Handle CORS preflight
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    // Get authorization header
    const authHeader = req.headers.get("Authorization");
    if (!authHeader) {
      return new Response(
        JSON.stringify({ error: "Missing authorization header" }),
        {
          status: 401,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    // Create Supabase client with user's token
    const supabaseClient = createClient(
      Deno.env.get("SUPABASE_URL") ?? "",
      Deno.env.get("SUPABASE_ANON_KEY") ?? "",
      {
        global: { headers: { Authorization: authHeader } },
      },
    );

    // Get user from token
    const {
      data: { user },
      error: userError,
    } = await supabaseClient.auth.getUser();
    if (userError || !user) {
      return new Response(
        JSON.stringify({ error: "Invalid or expired token" }),
        {
          status: 401,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    // Parse request body
    const body: GetReadingsRequest = await req.json();
    const { building_id, start_date, end_date } = body;

    if (!building_id) {
      return new Response(
        JSON.stringify({ error: "building_id is required" }),
        {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    // Default date range: last 7 days
    const endDate = end_date ? new Date(end_date) : new Date();
    const startDate = start_date
      ? new Date(start_date)
      : new Date(endDate.getTime() - 7 * 24 * 60 * 60 * 1000);

    // Fetch aggregate readings
    const { data: aggregateData, error: aggregateError } = await supabaseClient
      .from("readings")
      .select("timestamp, aggregate_kw")
      .eq("building_id", building_id)
      .gte("timestamp", startDate.toISOString())
      .lte("timestamp", endDate.toISOString())
      .order("timestamp", { ascending: true });

    if (aggregateError) {
      console.error("Error fetching readings:", aggregateError);
      return new Response(
        JSON.stringify({
          error: "Failed to fetch readings",
          details: aggregateError.message,
        }),
        {
          status: 500,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    // Fetch predictions for the building
    const { data: predictionsData, error: predictionsError } =
      await supabaseClient
        .from("predictions")
        .select(
          `
        timestamp,
        power_kw,
        confidence,
        org_appliances!inner(slug, name)
      `,
        )
        .eq("building_id", building_id)
        .gte("timestamp", startDate.toISOString())
        .lte("timestamp", endDate.toISOString())
        .order("timestamp", { ascending: true });

    if (predictionsError) {
      console.error("Error fetching predictions:", predictionsError);
      // Continue without predictions if they fail
    }

    // Build a map of timestamps to readings
    const readingsMap = new Map<string, ReadingEntry>();

    // Add aggregate readings
    for (const reading of aggregateData || []) {
      readingsMap.set(reading.timestamp, {
        ts: reading.timestamp,
        aggregate_kw: Number(reading.aggregate_kw),
        appliance_estimates: {},
        confidence: {},
      });
    }

    // Add predictions to the readings
    for (const prediction of predictionsData || []) {
      const ts = prediction.timestamp;
      const slug =
        (prediction as { org_appliances?: { slug?: string } }).org_appliances
          ?.slug || "unknown";

      if (!readingsMap.has(ts)) {
        // Create entry if we have prediction but no aggregate
        readingsMap.set(ts, {
          ts,
          aggregate_kw: 0,
          appliance_estimates: {},
          confidence: {},
        });
      }

      const entry = readingsMap.get(ts)!;
      entry.appliance_estimates[slug] = Number(prediction.power_kw);
      if (prediction.confidence !== null) {
        entry.confidence![slug] = Number(prediction.confidence);
      }
    }

    // Convert map to sorted array
    const readings = Array.from(readingsMap.values()).sort(
      (a, b) => new Date(a.ts).getTime() - new Date(b.ts).getTime(),
    );

    console.log(
      `Fetched ${readings.length} readings for building ${building_id}`,
    );

    return new Response(JSON.stringify({ success: true, readings }), {
      status: 200,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (err) {
    console.error("Unexpected error:", err);
    return new Response(JSON.stringify({ error: "Internal server error" }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
