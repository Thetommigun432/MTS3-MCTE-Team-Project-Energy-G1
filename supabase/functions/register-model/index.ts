import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform",
};

interface RegisterModelRequest {
  org_appliance_id: string;
  name: string;
  architecture?: string;
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
    const body: RegisterModelRequest = await req.json();
    const { org_appliance_id, name, architecture = "seq2point" } = body;

    if (!org_appliance_id) {
      return new Response(
        JSON.stringify({ error: "org_appliance_id is required" }),
        {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    if (!name || name.trim().length === 0) {
      return new Response(JSON.stringify({ error: "Model name is required" }), {
        status: 400,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    // Verify the org_appliance belongs to the user
    const { data: appliance, error: applianceError } = await supabaseClient
      .from("org_appliances")
      .select("id, name")
      .eq("id", org_appliance_id)
      .single();

    if (applianceError || !appliance) {
      return new Response(
        JSON.stringify({ error: "Appliance not found or access denied" }),
        {
          status: 404,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    // Check if a model already exists for this appliance
    const { data: existingModel } = await supabaseClient
      .from("models")
      .select("id")
      .eq("org_appliance_id", org_appliance_id)
      .single();

    if (existingModel) {
      return new Response(
        JSON.stringify({
          error: "A model already exists for this appliance",
          model_id: existingModel.id,
        }),
        {
          status: 409,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    // Create the model
    const { data: model, error: createError } = await supabaseClient
      .from("models")
      .insert({
        user_id: user.id,
        org_appliance_id,
        name: name.trim(),
        architecture,
        is_active: false,
      })
      .select("id")
      .single();

    if (createError) {
      console.error("Error creating model:", createError);
      return new Response(
        JSON.stringify({
          error: "Failed to create model",
          details: createError.message,
        }),
        {
          status: 500,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    console.log(
      `Model ${model.id} created for appliance ${appliance.name} by user ${user.id}`,
    );

    return new Response(
      JSON.stringify({
        success: true,
        model_id: model.id,
        message: `Model "${name}" created for ${appliance.name}`,
      }),
      {
        status: 201,
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
