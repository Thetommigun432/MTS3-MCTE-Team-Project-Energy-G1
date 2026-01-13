import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
};

interface RunInferenceRequest {
  building_id: string;
  org_appliance_id: string;
  start_date: string;
  end_date: string;
}

/**
 * Run NILM inference for a specific appliance on a building's aggregate readings.
 *
 * This function:
 * 1. Fetches the active model version for the specified org_appliance
 * 2. Retrieves aggregate readings for the specified date range
 * 3. Generates predictions (currently uses a simulation since ML model execution
 *    requires a Python runtime or external ML service)
 * 4. Stores predictions in the predictions table
 *
 * Note: For production ML inference, this would integrate with an external
 * inference service (e.g., AWS SageMaker, Google Vertex AI, or a custom
 * PyTorch/ONNX inference endpoint).
 */
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
    const body: RunInferenceRequest = await req.json();
    const { building_id, org_appliance_id, start_date, end_date } = body;

    // Validate required fields
    if (!building_id || !org_appliance_id || !start_date || !end_date) {
      return new Response(
        JSON.stringify({
          error: "Missing required fields",
          required: [
            "building_id",
            "org_appliance_id",
            "start_date",
            "end_date",
          ],
        }),
        {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    // Verify building belongs to user
    const { data: building, error: buildingError } = await supabaseClient
      .from("buildings")
      .select("id, name")
      .eq("id", building_id)
      .eq("user_id", user.id)
      .single();

    if (buildingError || !building) {
      return new Response(
        JSON.stringify({ error: "Building not found or access denied" }),
        {
          status: 404,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    // Verify org_appliance belongs to user
    const { data: orgAppliance, error: applianceError } = await supabaseClient
      .from("org_appliances")
      .select("id, name, slug")
      .eq("id", org_appliance_id)
      .eq("user_id", user.id)
      .single();

    if (applianceError || !orgAppliance) {
      return new Response(
        JSON.stringify({ error: "Appliance not found or access denied" }),
        {
          status: 404,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    // Get the active model version for this appliance
    const { data: model, error: modelError } = await supabaseClient
      .from("models")
      .select(
        `
        id,
        name,
        is_active,
        model_versions!inner(
          id,
          version,
          status,
          is_active,
          model_artifact_path,
          metrics
        )
      `,
      )
      .eq("org_appliance_id", org_appliance_id)
      .eq("is_active", true)
      .single();

    if (modelError || !model) {
      return new Response(
        JSON.stringify({
          error: "No active model found for this appliance",
          suggestion: "Please register and activate a model first",
        }),
        {
          status: 404,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    // Find the active version
    const activeVersion = (
      model.model_versions as Array<{ is_active: boolean; status: string }>
    ).find((v) => v.is_active && v.status === "ready");
    if (!activeVersion) {
      return new Response(
        JSON.stringify({
          error: "No ready model version is active",
          suggestion: 'Please activate a model version with status "ready"',
        }),
        {
          status: 404,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    // Fetch aggregate readings for the date range
    const startDateTime = new Date(start_date);
    const endDateTime = new Date(end_date);

    const { data: readings, error: readingsError } = await supabaseClient
      .from("readings")
      .select("id, timestamp, aggregate_kw")
      .eq("building_id", building_id)
      .gte("timestamp", startDateTime.toISOString())
      .lte("timestamp", endDateTime.toISOString())
      .order("timestamp", { ascending: true });

    if (readingsError) {
      console.error("Error fetching readings:", readingsError);
      return new Response(
        JSON.stringify({ error: "Failed to fetch readings" }),
        {
          status: 500,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    if (!readings || readings.length === 0) {
      return new Response(
        JSON.stringify({
          success: true,
          predictions_count: 0,
          message: "No readings found in the specified date range",
        }),
        {
          status: 200,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    console.log(
      `Processing ${readings.length} readings for appliance ${orgAppliance.slug}`,
    );

    // Generate predictions
    // NOTE: In a production environment, this would call an external ML inference service
    // For now, we simulate predictions based on the aggregate readings
    const predictions = readings.map((reading) => {
      // Simulated inference: estimate appliance power as a fraction of aggregate
      // In production, this would be replaced with actual model inference
      const aggregateKw = Number(reading.aggregate_kw);

      // Simple heuristic: appliance uses between 5-25% of aggregate, with some noise
      const baseRatio = 0.15;
      const noise = (Math.random() - 0.5) * 0.1;
      const estimatedPower = Math.max(0, aggregateKw * (baseRatio + noise));

      // Confidence based on model metrics if available
      const modelAccuracy = activeVersion.metrics?.accuracy || 0.85;
      const confidence = Math.min(
        1,
        Math.max(0, modelAccuracy + (Math.random() - 0.5) * 0.1),
      );

      return {
        user_id: user.id,
        building_id: building_id,
        org_appliance_id: org_appliance_id,
        model_version_id: activeVersion.id,
        timestamp: reading.timestamp,
        power_kw: Number(estimatedPower.toFixed(4)),
        confidence: Number(confidence.toFixed(3)),
        is_on: estimatedPower > 0.05, // ON if > 50W
      };
    });

    // Delete existing predictions for this time range and appliance (to avoid duplicates)
    const { error: deleteError } = await supabaseClient
      .from("predictions")
      .delete()
      .eq("building_id", building_id)
      .eq("org_appliance_id", org_appliance_id)
      .gte("timestamp", startDateTime.toISOString())
      .lte("timestamp", endDateTime.toISOString());

    if (deleteError) {
      console.error("Error deleting old predictions:", deleteError);
      // Continue anyway - new predictions will be inserted
    }

    // Insert predictions in batches
    const batchSize = 500;
    let insertedCount = 0;

    for (let i = 0; i < predictions.length; i += batchSize) {
      const batch = predictions.slice(i, i + batchSize);
      const { error: insertError } = await supabaseClient
        .from("predictions")
        .insert(batch);

      if (insertError) {
        console.error("Error inserting predictions batch:", insertError);
        return new Response(
          JSON.stringify({
            error: "Failed to store predictions",
            partial_count: insertedCount,
            details: insertError.message,
          }),
          {
            status: 500,
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          },
        );
      }

      insertedCount += batch.length;
    }

    console.log(
      `Successfully generated ${insertedCount} predictions for ${orgAppliance.name}`,
    );

    return new Response(
      JSON.stringify({
        success: true,
        predictions_count: insertedCount,
        message: `Generated ${insertedCount} predictions for ${orgAppliance.name}`,
        model_version: activeVersion.version,
        date_range: {
          start: startDateTime.toISOString(),
          end: endDateTime.toISOString(),
        },
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
