import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

const DEMO_EMAIL = 'admin@demo.local';
const DEMO_PASSWORD = 'admin123';
const DEMO_DISPLAY_NAME = 'Demo Admin';

// Rate limiting: simple in-memory store (resets on function cold start)
const rateLimitStore = new Map<string, { count: number; resetAt: number }>();
const RATE_LIMIT_MAX = 10; // Max calls per window
const RATE_LIMIT_WINDOW_MS = 60 * 1000; // 1 minute

function checkRateLimit(ip: string): boolean {
  const now = Date.now();
  const entry = rateLimitStore.get(ip);
  
  if (!entry || now > entry.resetAt) {
    rateLimitStore.set(ip, { count: 1, resetAt: now + RATE_LIMIT_WINDOW_MS });
    return true;
  }
  
  if (entry.count >= RATE_LIMIT_MAX) {
    return false;
  }
  
  entry.count++;
  return true;
}

Deno.serve(async (req) => {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    // Get client IP for rate limiting (best effort)
    const clientIp = req.headers.get('x-forwarded-for')?.split(',')[0]?.trim() || 
                     req.headers.get('cf-connecting-ip') || 
                     'unknown';
    
    // Check rate limit
    if (!checkRateLimit(clientIp)) {
      return new Response(
        JSON.stringify({ error: 'Rate limit exceeded. Please try again later.' }),
        { status: 429, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Check if demo mode is enabled via environment variable
    const demoModeEnabled = Deno.env.get('DEMO_MODE_ENABLED') === 'true';
    if (!demoModeEnabled) {
      return new Response(
        JSON.stringify({ error: 'Demo mode is not enabled' }),
        { status: 403, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Create admin client with service role key
    const supabaseUrl = Deno.env.get('SUPABASE_URL');
    const supabaseServiceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');

    if (!supabaseUrl || !supabaseServiceKey) {
      console.error('Missing required environment variables');
      return new Response(
        JSON.stringify({ error: 'Server configuration error' }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    const adminClient = createClient(supabaseUrl, supabaseServiceKey, {
      auth: {
        autoRefreshToken: false,
        persistSession: false,
      },
    });

    // Check if demo user already exists
    const { data: existingUsers, error: listError } = await adminClient.auth.admin.listUsers();
    
    if (listError) {
      console.error('Error listing users:', listError);
      return new Response(
        JSON.stringify({ error: 'Failed to check existing users' }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    const existingDemoUser = existingUsers?.users?.find(u => u.email === DEMO_EMAIL);

    if (existingDemoUser) {
      // User exists, ensure profile is set up correctly
      const { error: profileError } = await adminClient
        .from('profiles')
        .upsert({
          id: existingDemoUser.id,
          email: DEMO_EMAIL,
          display_name: DEMO_DISPLAY_NAME,
        }, {
          onConflict: 'id',
        });

      if (profileError) {
        console.warn('Profile upsert warning:', profileError);
      }

      return new Response(
        JSON.stringify({ 
          ok: true, 
          message: 'Demo user already exists',
          email: DEMO_EMAIL,
        }),
        { status: 200, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Create the demo user
    const { data: newUser, error: createError } = await adminClient.auth.admin.createUser({
      email: DEMO_EMAIL,
      password: DEMO_PASSWORD,
      email_confirm: true, // Auto-confirm the email
      user_metadata: {
        display_name: DEMO_DISPLAY_NAME,
      },
    });

    if (createError) {
      console.error('Error creating demo user:', createError);
      return new Response(
        JSON.stringify({ error: 'Failed to create demo user: ' + createError.message }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Create/update profile for the new user
    if (newUser?.user) {
      const { error: profileError } = await adminClient
        .from('profiles')
        .upsert({
          id: newUser.user.id,
          email: DEMO_EMAIL,
          display_name: DEMO_DISPLAY_NAME,
        }, {
          onConflict: 'id',
        });

      if (profileError) {
        console.warn('Profile creation warning:', profileError);
      }
    }

    return new Response(
      JSON.stringify({ 
        ok: true, 
        message: 'Demo user created successfully',
        email: DEMO_EMAIL,
      }),
      { status: 201, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );

  } catch (error) {
    console.error('Unexpected error:', error);
    return new Response(
      JSON.stringify({ error: 'An unexpected error occurred' }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});
