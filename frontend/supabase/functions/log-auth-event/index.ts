import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

/**
 * Log Auth Event Edge Function
 * 
 * Inserts login/logout events into login_history table.
 * Uses service_role key to bypass RLS for secure event logging.
 * 
 * Request body:
 * {
 *   event: 'login' | 'logout' | 'signup' | 'password_reset';
 *   user_agent?: string;
 *   success?: boolean;
 * }
 */

interface LogAuthEventRequest {
  event: 'login' | 'logout' | 'signup' | 'password_reset';
  user_agent?: string;
  success?: boolean;
}

Deno.serve(async (req) => {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    // Get authorization header
    const authHeader = req.headers.get('Authorization');
    if (!authHeader) {
      return new Response(
        JSON.stringify({ error: 'Missing authorization header' }),
        { status: 401, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Create Supabase clients
    const supabaseUrl = Deno.env.get('SUPABASE_URL') ?? '';
    const supabaseAnonKey = Deno.env.get('SUPABASE_ANON_KEY') ?? '';
    const supabaseServiceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? '';

    // User client for auth verification
    const userClient = createClient(supabaseUrl, supabaseAnonKey, {
      global: { headers: { Authorization: authHeader } },
    });

    // Get user from token
    const { data: { user }, error: userError } = await userClient.auth.getUser();
    if (userError || !user) {
      return new Response(
        JSON.stringify({ error: 'Invalid or expired token' }),
        { status: 401, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Parse request body
    const body: LogAuthEventRequest = await req.json().catch(() => ({ event: 'login' }));
    const userAgent = body.user_agent || req.headers.get('user-agent') || null;
    const success = body.success !== false; // Default to true

    // Parse user agent to get device label
    let deviceLabel = 'Unknown Device';
    if (userAgent) {
      if (userAgent.includes('Mobile')) {
        deviceLabel = 'Mobile Browser';
      } else if (userAgent.includes('Chrome')) {
        deviceLabel = 'Chrome Browser';
      } else if (userAgent.includes('Firefox')) {
        deviceLabel = 'Firefox Browser';
      } else if (userAgent.includes('Safari')) {
        deviceLabel = 'Safari Browser';
      } else if (userAgent.includes('Edge')) {
        deviceLabel = 'Edge Browser';
      } else {
        deviceLabel = 'Web Browser';
      }
    }

    // Use service_role client to insert into login_history (bypasses RLS)
    const adminClient = createClient(supabaseUrl, supabaseServiceKey);

    // Insert into login_history table
    const { data, error } = await adminClient
      .from('login_history')
      .insert({
        user_id: user.id,
        event: body.event || 'login',
        user_agent: userAgent,
        device_label: deviceLabel,
        success,
        ip_address: null, // Cannot reliably get IP from edge function
      })
      .select('id')
      .single();

    if (error) {
      console.error('Error inserting login_history:', error);
      // Try falling back to login_events table if login_history doesn't exist
      const { data: fallbackData, error: fallbackError } = await adminClient
        .from('login_events')
        .insert({
          user_id: user.id,
          user_agent: userAgent,
          device_label: deviceLabel,
          success,
          ip_address: null,
        })
        .select('id')
        .single();

      if (fallbackError) {
        console.error('Error inserting login_events:', fallbackError);
        return new Response(
          JSON.stringify({ error: 'Failed to log auth event', details: error.message }),
          { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }

      return new Response(
        JSON.stringify({ success: true, event_id: fallbackData?.id }),
        { status: 200, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    console.log(`Auth event logged for user ${user.id}: ${body.event}`);

    return new Response(
      JSON.stringify({ success: true, event_id: data?.id }),
      { status: 200, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );

  } catch (err) {
    console.error('Unexpected error:', err);
    return new Response(
      JSON.stringify({ error: 'Internal server error' }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});
