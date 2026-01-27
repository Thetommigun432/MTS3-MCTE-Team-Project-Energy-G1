import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform",
};

interface GenerateReportRequest {
  building_id: string;
  start_date: string;
  end_date: string;
  appliance_ids?: string[];
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
    const body: GenerateReportRequest = await req.json();
    const { building_id, start_date, end_date, appliance_ids } = body;

    if (!building_id || !start_date || !end_date) {
      return new Response(
        JSON.stringify({
          error: "building_id, start_date, and end_date are required",
        }),
        {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    const startDate = new Date(start_date);
    const endDate = new Date(end_date);

    // Fetch predictions for the building and date range
    let predictionsQuery = supabaseClient
      .from("predictions")
      .select(
        `
        timestamp,
        power_kw,
        confidence,
        org_appliance_id,
        org_appliances!inner(slug, name)
      `,
      )
      .eq("building_id", building_id)
      .gte("timestamp", startDate.toISOString())
      .lte("timestamp", endDate.toISOString())
      .order("timestamp", { ascending: true });

    if (appliance_ids && appliance_ids.length > 0) {
      predictionsQuery = predictionsQuery.in("org_appliance_id", appliance_ids);
    }

    const { data: predictions, error: predictionsError } =
      await predictionsQuery;

    if (predictionsError) {
      console.error("Error fetching predictions:", predictionsError);
      return new Response(
        JSON.stringify({ error: "Failed to fetch predictions" }),
        {
          status: 500,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    // Fetch aggregate readings for the period
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

    // Calculate summary
    const dataPointsAnalyzed = predictions?.length || 0;

    // Group predictions by appliance
    const applianceEnergy: Record<
      string,
      {
        name: string;
        slug: string;
        totalKwh: number;
        totalConfidence: number;
        count: number;
      }
    > = {};

    for (const pred of predictions || []) {
      const appliance = (
        pred as { org_appliances?: { slug?: string; name?: string } }
      ).org_appliances;
      const slug = appliance?.slug || "unknown";
      const name = appliance?.name || "Unknown";

      if (!applianceEnergy[slug]) {
        applianceEnergy[slug] = {
          name,
          slug,
          totalKwh: 0,
          totalConfidence: 0,
          count: 0,
        };
      }

      // Assume 15-minute intervals (0.25 hours)
      applianceEnergy[slug].totalKwh += Number(pred.power_kw) * 0.25;
      applianceEnergy[slug].totalConfidence += Number(pred.confidence) || 0.5;
      applianceEnergy[slug].count++;
    }

    // Calculate total energy and find peak
    let totalEnergy = 0;
    let peakPower = 0;
    let peakTimestamp = "";

    // Use aggregate readings for total/peak if available
    if (readings && readings.length > 0) {
      for (const reading of readings) {
        const power = Number(reading.aggregate_kw);
        totalEnergy += power * 0.25; // 15-minute intervals
        if (power > peakPower) {
          peakPower = power;
          peakTimestamp = reading.timestamp;
        }
      }
    } else {
      // Fall back to sum of predictions
      totalEnergy = Object.values(applianceEnergy).reduce(
        (sum, a) => sum + a.totalKwh,
        0,
      );
    }

    // Calculate average power
    const hoursInRange =
      (endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60);
    const averagePower = hoursInRange > 0 ? totalEnergy / hoursInRange : 0;

    // Build breakdown
    const totalFromAppliances = Object.values(applianceEnergy).reduce(
      (sum, a) => sum + a.totalKwh,
      0,
    );
    const breakdown = Object.values(applianceEnergy)
      .map((a) => ({
        appliance_name: a.name,
        appliance_slug: a.slug,
        energy_kwh: a.totalKwh,
        percentage:
          totalFromAppliances > 0
            ? (a.totalKwh / totalFromAppliances) * 100
            : 0,
        avg_confidence: a.count > 0 ? a.totalConfidence / a.count : 0,
      }))
      .sort((a, b) => b.energy_kwh - a.energy_kwh);

    // Calculate hourly pattern
    const hourlyPower: Record<number, { total: number; count: number }> = {};
    for (let h = 0; h < 24; h++) {
      hourlyPower[h] = { total: 0, count: 0 };
    }

    for (const reading of readings || []) {
      const hour = new Date(reading.timestamp).getHours();
      hourlyPower[hour].total += Number(reading.aggregate_kw);
      hourlyPower[hour].count++;
    }

    const hourlyPattern = Object.entries(hourlyPower).map(([hour, data]) => ({
      hour: parseInt(hour),
      avg_power_kw: data.count > 0 ? data.total / data.count : 0,
    }));

    const summary = {
      total_energy_kwh: totalEnergy,
      average_power_kw: averagePower,
      peak_power_kw: peakPower,
      peak_timestamp: peakTimestamp,
      data_points_analyzed: dataPointsAnalyzed,
    };

    console.log(
      `Report generated for building ${building_id}: ${dataPointsAnalyzed} predictions analyzed`,
    );

    return new Response(
      JSON.stringify({
        success: true,
        summary,
        breakdown,
        hourly_pattern: hourlyPattern,
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
