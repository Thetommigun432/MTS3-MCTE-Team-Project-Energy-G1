import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.49.1";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
};

interface InviteRequest {
  org_id: string;
  email: string;
  role: "admin" | "member" | "viewer";
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
        }
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
    const { org_id, email, role }: InviteRequest = await req.json();

    // Validate inputs
    if (!org_id) {
      return new Response(JSON.stringify({ error: "org_id is required" }), {
        status: 400,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!email || !emailRegex.test(email)) {
      return new Response(JSON.stringify({ error: "Invalid email address" }), {
        status: 400,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    if (!["admin", "member", "viewer"].includes(role)) {
      return new Response(JSON.stringify({ error: "Invalid role. Must be admin, member, or viewer" }), {
        status: 400,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    // Admin client for privileged operations
    const adminClient = createClient(supabaseUrl, supabaseServiceKey);

    // Verify caller is admin of the organization
    const { data: callerMembership, error: membershipError } = await adminClient
      .from("org_members")
      .select("role")
      .eq("org_id", org_id)
      .eq("user_id", user.id)
      .single();

    if (membershipError || !callerMembership) {
      console.error("Membership check error:", membershipError);
      return new Response(
        JSON.stringify({ error: "You are not a member of this organization" }),
        {
          status: 403,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        }
      );
    }

    if (callerMembership.role !== "admin") {
      return new Response(
        JSON.stringify({ error: "Only organization admins can invite users" }),
        {
          status: 403,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        }
      );
    }

    // Check if the email is already a member of this org
    const { data: existingMember } = await adminClient
      .from("org_members")
      .select(`
        user_id,
        profiles:user_id (email)
      `)
      .eq("org_id", org_id);

    // Check profiles to find if email already exists as member
    const { data: existingProfile } = await adminClient
      .from("profiles")
      .select("id")
      .eq("email", email)
      .maybeSingle();

    if (existingProfile) {
      // Check if already a member
      const isMember = existingMember?.some((m: { user_id: string }) => m.user_id === existingProfile.id);
      if (isMember) {
        return new Response(
          JSON.stringify({ error: "This user is already a member of this organization" }),
          {
            status: 400,
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          }
        );
      }
    }

    // Check if there's already a pending invite for this org+email
    const { data: existingInvite } = await adminClient
      .from("pending_org_invites")
      .select("*")
      .eq("org_id", org_id)
      .eq("email", email)
      .maybeSingle();

    if (existingInvite) {
      return new Response(
        JSON.stringify({ error: "An invitation is already pending for this email" }),
        {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        }
      );
    }

    // Get the app URL for redirect
    const appUrl = req.headers.get("origin") || Deno.env.get("APP_URL") || supabaseUrl;

    // Check if user already exists in auth system
    const { data: authUsers } = await adminClient.auth.admin.listUsers();
    const existingUser = authUsers?.users.find((u) => u.email === email);

    if (existingUser) {
      // User exists - add them directly to the org
      const { error: insertMemberError } = await adminClient
        .from("org_members")
        .insert({
          org_id: org_id,
          user_id: existingUser.id,
          role: role,
        });

      if (insertMemberError) {
        console.error("Insert member error:", insertMemberError);
        return new Response(
          JSON.stringify({ error: "Failed to add user to organization" }),
          {
            status: 500,
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          }
        );
      }

      // Record in invitations table for history
      await adminClient.from("invitations").insert({
        email: email,
        role: role,
        org_id: org_id,
        invited_by: user.id,
        status: "accepted",
      });

      console.log(`Added existing user ${email} to org ${org_id}`);

      return new Response(
        JSON.stringify({
          success: true,
          message: `${email} has been added to the organization`,
          user_added: true,
        }),
        {
          status: 200,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        }
      );
    }

    // User doesn't exist - send invite email and store pending invite
    const { data: inviteData, error: inviteError } =
      await adminClient.auth.admin.inviteUserByEmail(email, {
        redirectTo: `${appUrl}/app/dashboard`,
        data: {
          role: role,
          org_id: org_id,
          invited_by: user.id,
        },
      });

    if (inviteError) {
      console.error("Invite error:", inviteError);
      return new Response(
        JSON.stringify({ error: inviteError.message || "Failed to send invitation" }),
        {
          status: 500,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        }
      );
    }

    // Store pending org invite (will be reconciled on signup via handle_new_user)
    const { error: pendingError } = await adminClient
      .from("pending_org_invites")
      .insert({
        org_id: org_id,
        email: email,
        role: role,
        invited_by: user.id,
      });

    if (pendingError) {
      console.error("Pending invite insert error:", pendingError);
      // Don't fail - the auth invite was sent
    }

    // Record in invitations table for UI tracking
    await adminClient.from("invitations").insert({
      email: email,
      role: role,
      org_id: org_id,
      invited_by: user.id,
      status: "pending",
    });

    console.log(`Invitation sent to ${email} for org ${org_id} by ${user.email}`);

    return new Response(
      JSON.stringify({
        success: true,
        message: `Invitation sent to ${email}`,
        invitation_id: inviteData?.user?.id,
      }),
      {
        status: 200,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      }
    );
  } catch (error: unknown) {
    console.error("Error in invite-user-to-org function:", error);
    const message = error instanceof Error ? error.message : "Internal server error";
    return new Response(JSON.stringify({ error: message }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
