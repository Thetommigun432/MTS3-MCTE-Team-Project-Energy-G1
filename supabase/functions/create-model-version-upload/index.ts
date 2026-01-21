import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
};

interface CreateModelVersionUploadRequest {
  model_id: string;
  version: string;
  has_scaler?: boolean;
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
    const body: CreateModelVersionUploadRequest = await req.json();
    const { model_id, version, has_scaler = false } = body;

    if (!model_id) {
      return new Response(JSON.stringify({ error: "model_id is required" }), {
        status: 400,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    if (!version || version.trim().length === 0) {
      return new Response(JSON.stringify({ error: "version is required" }), {
        status: 400,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    // Verify the model belongs to the user
    const { data: model, error: modelError } = await supabaseClient
      .from("models")
      .select("id, org_appliance_id")
      .eq("id", model_id)
      .single();

    if (modelError || !model) {
      return new Response(
        JSON.stringify({ error: "Model not found or access denied" }),
        {
          status: 404,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    // Check if version already exists
    const { data: existingVersion } = await supabaseClient
      .from("model_versions")
      .select("id")
      .eq("model_id", model_id)
      .eq("version", version.trim())
      .single();

    if (existingVersion) {
      return new Response(
        JSON.stringify({
          error: `Version ${version} already exists for this model`,
        }),
        {
          status: 409,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    // Create the model version record with 'uploading' status
    const modelPath = `${user.id}/${model_id}/${version}/model.pt`;
    const scalerPath = has_scaler
      ? `${user.id}/${model_id}/${version}/scaler.pkl`
      : null;

    const { data: versionRecord, error: versionError } = await supabaseClient
      .from("model_versions")
      .insert({
        model_id,
        user_id: user.id,
        version: version.trim(),
        status: "uploading",
        model_artifact_path: modelPath,
        scaler_artifact_path: scalerPath,
      })
      .select("id")
      .single();

    if (versionError) {
      console.error("Error creating model version:", versionError);
      return new Response(
        JSON.stringify({
          error: "Failed to create model version",
          details: versionError.message,
        }),
        {
          status: 500,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    // Generate signed upload URLs
    const { data: modelUploadData, error: modelUploadError } =
      await supabaseClient.storage
        .from("models")
        .createSignedUploadUrl(modelPath);

    if (modelUploadError) {
      console.error("Error creating model upload URL:", modelUploadError);
      return new Response(
        JSON.stringify({
          error: "Failed to create upload URL for model artifact",
        }),
        {
          status: 500,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    let scalerUploadUrl = null;
    if (has_scaler && scalerPath) {
      const { data: scalerUploadData, error: scalerUploadError } =
        await supabaseClient.storage
          .from("models")
          .createSignedUploadUrl(scalerPath);

      if (scalerUploadError) {
        console.error("Error creating scaler upload URL:", scalerUploadError);
        // Non-fatal, continue without scaler URL
      } else {
        scalerUploadUrl = scalerUploadData.signedUrl;
      }
    }

    console.log(
      `Model version ${version} created for model ${model_id} by user ${user.id}`,
    );

    return new Response(
      JSON.stringify({
        success: true,
        version_id: versionRecord.id,
        model_upload_url: modelUploadData.signedUrl,
        scaler_upload_url: scalerUploadUrl,
        message: `Upload URLs generated for version ${version}`,
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
