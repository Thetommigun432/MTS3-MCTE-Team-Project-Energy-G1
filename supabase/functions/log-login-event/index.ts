import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform",
};

interface LogLoginEventRequest {
  user_agent?: string;
  success?: boolean;
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
    const body: LogLoginEventRequest = await req.json().catch(() => ({}));
    const userAgent = body.user_agent || req.headers.get("user-agent") || null;
    const success = body.success !== false; // Default to true

    // Parse user agent to get device label
    let deviceLabel = "Unknown Device";
    if (userAgent) {
      if (userAgent.includes("Mobile")) {
        deviceLabel = "Mobile Browser";
      } else if (userAgent.includes("Chrome")) {
        deviceLabel = "Chrome Browser";
      } else if (userAgent.includes("Firefox")) {
        deviceLabel = "Firefox Browser";
      } else if (userAgent.includes("Safari")) {
        deviceLabel = "Safari Browser";
      } else if (userAgent.includes("Edge")) {
        deviceLabel = "Edge Browser";
      } else {
        deviceLabel = "Web Browser";
      }
    }

    // Insert login event
    const { data, error } = await supabaseClient
      .from("login_events")
      .insert({
        user_id: user.id,
        user_agent: userAgent,
        device_label: deviceLabel,
        success,
        ip_address: null, // Cannot reliably get IP from edge function
      })
      .select("id")
      .single();

    if (error) {
      console.error("Error inserting login event:", error);
      return new Response(
        JSON.stringify({
          error: "Failed to log login event",
          details: error.message,
        }),
        {
          status: 500,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    console.log(`Login event logged for user ${user.id}: ${deviceLabel}`);

    return new Response(JSON.stringify({ success: true, event_id: data.id }), {
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
