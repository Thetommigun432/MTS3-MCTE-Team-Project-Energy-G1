import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.49.1";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
};

/**
 * Invite User Edge Function
 *
 * Uses service_role key to:
 * 1. Send admin invite email via Supabase Auth
 * 2. Create org_members row for the invited user
 *
 * Request body:
 * {
 *   org_id: string;        // Organization to add user to
 *   email: string;         // Email address to invite
 *   role: 'admin' | 'member' | 'viewer';
 *   redirect_to?: string;  // URL to redirect after signup
 * }
 */

interface InviteRequest {
  org_id: string;
  email: string;
  role: "admin" | "member" | "viewer";
  redirect_to?: string;
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
        {
          status: 401,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
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
    const {
      data: { user },
      error: userError,
    } = await userClient.auth.getUser();
    if (userError || !user) {
      console.error("Auth error:", userError);
      return new Response(JSON.stringify({ error: "Unauthorized" }), {
        status: 401,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    // Parse request body
    const { org_id, email, role, redirect_to }: InviteRequest =
      await req.json();

    // Validate required fields
    if (!org_id || !email || !role) {
      return new Response(
        JSON.stringify({ error: "org_id, email, and role are required" }),
        {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    // Validate email
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return new Response(JSON.stringify({ error: "Invalid email address" }), {
        status: 400,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    // Validate role
    if (!["admin", "member", "viewer"].includes(role)) {
      return new Response(
        JSON.stringify({
          error: "Invalid role. Must be admin, member, or viewer",
        }),
        {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    // Admin client for privileged operations (uses service_role key)
    const adminClient = createClient(supabaseUrl, supabaseServiceKey);

    // Check if inviting user is an admin of the organization
    const { data: membership, error: memberError } = await adminClient
      .from("org_members")
      .select("role")
      .eq("org_id", org_id)
      .eq("user_id", user.id)
      .maybeSingle();

    if (memberError || !membership || membership.role !== "admin") {
      return new Response(
        JSON.stringify({ error: "Only organization admins can invite users" }),
        {
          status: 403,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    // Check if user already exists
    const { data: existingUsers } = await adminClient.auth.admin.listUsers();
    const existingUser = existingUsers?.users.find((u) => u.email === email);

    if (existingUser) {
      // User exists - check if already a member
      const { data: existingMember } = await adminClient
        .from("org_members")
        .select("*")
        .eq("org_id", org_id)
        .eq("user_id", existingUser.id)
        .maybeSingle();

      if (existingMember) {
        return new Response(
          JSON.stringify({
            error: "User is already a member of this organization",
          }),
          {
            status: 400,
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          },
        );
      }

      // Add existing user to organization
      const { error: insertError } = await adminClient
        .from("org_members")
        .insert({
          org_id,
          user_id: existingUser.id,
          role,
        });

      if (insertError) {
        console.error("Error adding member:", insertError);
        return new Response(
          JSON.stringify({ error: "Failed to add user to organization" }),
          {
            status: 500,
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          },
        );
      }

      return new Response(
        JSON.stringify({
          success: true,
          message: "User added to organization",
          user_id: existingUser.id,
        }),
        {
          status: 200,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    // User doesn't exist - send invite email
    const siteUrl =
      Deno.env.get("SITE_URL") || req.headers.get("origin") || supabaseUrl;
    const redirectUrl = redirect_to || `${siteUrl}/app/dashboard`;

    const { data: inviteData, error: inviteError } =
      await adminClient.auth.admin.inviteUserByEmail(email, {
        redirectTo: redirectUrl,
        data: {
          invited_to_org: org_id,
          invited_role: role,
        },
      });

    if (inviteError) {
      console.error("Invite error:", inviteError);
      return new Response(
        JSON.stringify({
          error: "Failed to send invitation email",
          details: inviteError.message,
        }),
        {
          status: 500,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        },
      );
    }

    // Store pending invitation in invitations table
    const { error: invitationError } = await adminClient
      .from("invitations")
      .insert({
        email,
        role,
        invited_by: user.id,
        status: "pending",
        expires_at: new Date(
          Date.now() + 7 * 24 * 60 * 60 * 1000,
        ).toISOString(), // 7 days
      });

    if (invitationError) {
      console.warn("Failed to record invitation:", invitationError);
    }

    console.log(
      `Invitation sent to ${email} for org ${org_id} with role ${role}`,
    );

    return new Response(
      JSON.stringify({
        success: true,
        message: "Invitation sent successfully",
        invited_user_id: inviteData?.user?.id,
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
