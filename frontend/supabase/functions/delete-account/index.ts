import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.49.1";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

interface DeleteRequest {
  confirmation: string; // Must be "DELETE" to confirm
}

serve(async (req) => {
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
        { status: 401, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Create Supabase clients
    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseAnonKey = Deno.env.get("SUPABASE_ANON_KEY")!;
    const supabaseServiceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

    // User client for auth verification
    const userClient = createClient(supabaseUrl, supabaseAnonKey, {
      global: { headers: { Authorization: authHeader } },
    });

    // Verify user is authenticated
    const { data: { user }, error: userError } = await userClient.auth.getUser();
    if (userError || !user) {
      console.error("Auth error:", userError);
      return new Response(
        JSON.stringify({ error: "Unauthorized" }),
        { status: 401, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Parse request body
    const { confirmation }: DeleteRequest = await req.json();

    // Require confirmation
    if (confirmation !== "DELETE") {
      return new Response(
        JSON.stringify({ error: "Please type DELETE to confirm account deletion" }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Admin client for privileged operations
    const adminClient = createClient(supabaseUrl, supabaseServiceKey);

    // Delete avatar files from storage if they exist
    const { data: avatarFiles } = await adminClient.storage
      .from("avatars")
      .list(`avatars`, {
        search: user.id,
      });

    if (avatarFiles && avatarFiles.length > 0) {
      const filesToDelete = avatarFiles.map(f => `avatars/${f.name}`);
      await adminClient.storage.from("avatars").remove(filesToDelete);
      console.log(`Deleted ${filesToDelete.length} avatar files for user ${user.id}`);
    }

    // Delete user's data from tables (cascades should handle most, but be explicit)
    // Delete buildings (will cascade to appliances due to FK)
    await adminClient.from("buildings").delete().eq("user_id", user.id);
    
    // Delete appliances explicitly in case no FK cascade
    await adminClient.from("appliances").delete().eq("user_id", user.id);
    
    // Delete login events
    await adminClient.from("login_events").delete().eq("user_id", user.id);
    
    // Delete invitations created by this user
    await adminClient.from("invitations").delete().eq("invited_by", user.id);

    // Delete profile (should cascade from auth.users, but be explicit)
    await adminClient.from("profiles").delete().eq("id", user.id);

    // Finally, delete the auth user
    const { error: deleteError } = await adminClient.auth.admin.deleteUser(user.id);
    
    if (deleteError) {
      console.error("Error deleting auth user:", deleteError);
      return new Response(
        JSON.stringify({ error: "Failed to delete account. Please contact support." }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    console.log(`Account deleted for user ${user.id} (${user.email})`);

    return new Response(
      JSON.stringify({ 
        success: true, 
        message: "Account deleted successfully",
      }),
      { status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );

  } catch (error: unknown) {
    console.error("Error in delete-account function:", error);
    const message = error instanceof Error ? error.message : "Internal server error";
    return new Response(
      JSON.stringify({ error: message }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
