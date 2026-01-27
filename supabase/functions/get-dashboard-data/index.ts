import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform",
};

/**
 * Get Dashboard Data Edge Function
 *
 * Fetches real data from Postgres for the dashboard:
 * - Aggregate energy series
 * - Per-appliance predicted series
 * - "What's ON now" list
 * - Insights (peak load, daily usage)
 *
 * Request body:
 * {
 *   building_id: string;
 *   start?: string;  // ISO date
 *   end?: string;    // ISO date
 * }
 */

interface DashboardRequest {
  building_id: string;
  start?: string;
  end?: string;
}

interface ApplianceStatus {
  appliance_id: string;
  name: string;
  slug: string;
  current_power_kw: number;
  is_on: boolean;
  confidence: number;
}

interface Insight {
  type: "peak_load" | "daily_usage" | "top_consumer" | "efficiency";
  label: string;
  value: string;
  trend?: "up" | "down" | "stable";
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

    // Create Supabase client with user's token (RLS enforced)
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
    const body: DashboardRequest = await req.json();
    const { building_id, start, end } = body;

    if (!building_id) {
      return new Response(
        JSON.stringify({ error: "building_id is required" }),
        {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    // Default date range: last 24 hours
    const endDate = end ? new Date(end) : new Date();
    const startDate = start
      ? new Date(start)
      : new Date(endDate.getTime() - 24 * 60 * 60 * 1000);

    // Fetch aggregate readings (RLS will filter by building access)
    const { data: readings, error: readingsError } = await supabaseClient
      .from("readings")
      .select("timestamp, aggregate_kw")
      .eq("building_id", building_id)
      .gte("timestamp", startDate.toISOString())
      .lte("timestamp", endDate.toISOString())
      .order("timestamp", { ascending: true });

    if (readingsError) {
      console.error("Error fetching readings:", readingsError);
    }

    // Fetch predictions with appliance info
    const { data: predictions, error: predictionsError } = await supabaseClient
      .from("predictions")
      .select(
        `
        timestamp,
        power_kw,
        confidence,
        is_on,
        org_appliance_id,
        org_appliances!inner(id, name, slug)
      `,
      )
      .eq("building_id", building_id)
      .gte("timestamp", startDate.toISOString())
      .lte("timestamp", endDate.toISOString())
      .order("timestamp", { ascending: true });

    if (predictionsError) {
      console.error("Error fetching predictions:", predictionsError);
    }

    // Build aggregate series
    const aggregateSeries = (readings || []).map((r) => ({
      ts: r.timestamp,
      power_kw: Number(r.aggregate_kw),
    }));

    // Build per-appliance series
    const applianceSeries: Record<
      string,
      { ts: string; power_kw: number; confidence: number }[]
    > = {};
    const latestPredictions: Record<
      string,
      {
        power_kw: number;
        is_on: boolean;
        confidence: number;
        timestamp: string;
      }
    > = {};

    for (const pred of predictions || []) {
      const appliance = (
        pred as unknown as {
          org_appliances: { id: string; name: string; slug: string };
        }
      ).org_appliances;
      const slug = appliance?.slug || "unknown";

      if (!applianceSeries[slug]) {
        applianceSeries[slug] = [];
      }

      applianceSeries[slug].push({
        ts: pred.timestamp,
        power_kw: Number(pred.power_kw),
        confidence: Number(pred.confidence) || 0.5,
      });

      // Track latest prediction for each appliance
      if (
        !latestPredictions[slug] ||
        pred.timestamp > latestPredictions[slug].timestamp
      ) {
        latestPredictions[slug] = {
          power_kw: Number(pred.power_kw),
          is_on: pred.is_on ?? Number(pred.power_kw) > 0.05,
          confidence: Number(pred.confidence) || 0.5,
          timestamp: pred.timestamp,
        };
      }
    }

    // Build "What's ON now" list
    const { data: appliances } = await supabaseClient
      .from("org_appliances")
      .select("id, name, slug");

    const applianceMap = new Map((appliances || []).map((a) => [a.slug, a]));

    const whatsOnNow: ApplianceStatus[] = Object.entries(latestPredictions)
      .filter(([, data]) => data.is_on)
      .map(([slug, data]) => ({
        appliance_id: applianceMap.get(slug)?.id || "",
        name: applianceMap.get(slug)?.name || slug,
        slug,
        current_power_kw: data.power_kw,
        is_on: data.is_on,
        confidence: data.confidence,
      }))
      .sort((a, b) => b.current_power_kw - a.current_power_kw);

    // Calculate insights
    const insights: Insight[] = [];

    // Peak load
    if (aggregateSeries.length > 0) {
      const peak = aggregateSeries.reduce(
        (max, r) => (r.power_kw > max.power_kw ? r : max),
        aggregateSeries[0],
      );
      insights.push({
        type: "peak_load",
        label: "Peak Load",
        value: `${peak.power_kw.toFixed(2)} kW`,
      });
    }

    // Total energy (assuming 15-min intervals)
    const totalEnergy = aggregateSeries.reduce(
      (sum, r) => sum + r.power_kw * 0.25,
      0,
    );
    insights.push({
      type: "daily_usage",
      label: "Total Energy",
      value: `${totalEnergy.toFixed(2)} kWh`,
    });

    // Top consumer
    const topConsumer = Object.entries(applianceSeries)
      .map(([slug, series]) => ({
        slug,
        energy: series.reduce((sum, r) => sum + r.power_kw * 0.25, 0),
      }))
      .sort((a, b) => b.energy - a.energy)[0];

    if (topConsumer) {
      insights.push({
        type: "top_consumer",
        label: "Top Consumer",
        value: `${applianceMap.get(topConsumer.slug)?.name || topConsumer.slug} (${topConsumer.energy.toFixed(2)} kWh)`,
      });
    }

    return new Response(
      JSON.stringify({
        building_id,
        period: {
          start: startDate.toISOString(),
          end: endDate.toISOString(),
        },
        aggregate_series: aggregateSeries,
        appliance_series: applianceSeries,
        whats_on_now: whatsOnNow,
        insights,
        data_points: aggregateSeries.length,
      }),
      {
        status: 200,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      },
    );
  } catch (err) {
    console.error("Unexpected error:", err);
    return new Response(JSON.stringify({ error: "Internal server error" }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
