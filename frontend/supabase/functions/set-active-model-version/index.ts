import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

interface SetActiveVersionRequest {
  version_id: string;
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

    // Create Supabase client with user's token
    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? '',
      {
        global: { headers: { Authorization: authHeader } },
      }
    );

    // Get user from token
    const { data: { user }, error: userError } = await supabaseClient.auth.getUser();
    if (userError || !user) {
      return new Response(
        JSON.stringify({ error: 'Invalid or expired token' }),
        { status: 401, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Parse request body
    const body: SetActiveVersionRequest = await req.json();
    const { version_id } = body;

    if (!version_id) {
      return new Response(
        JSON.stringify({ error: 'version_id is required' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Verify the version belongs to the user and is in 'ready' status
    const { data: version, error: versionError } = await supabaseClient
      .from('model_versions')
      .select('id, model_id, status, version')
      .eq('id', version_id)
      .single();

    if (versionError || !version) {
      return new Response(
        JSON.stringify({ error: 'Model version not found or access denied' }),
        { status: 404, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    if (version.status !== 'ready') {
      return new Response(
        JSON.stringify({ error: `Cannot activate version in status: ${version.status}` }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Deactivate all other versions for this model
    await supabaseClient
      .from('model_versions')
      .update({ is_active: false })
      .eq('model_id', version.model_id);

    // Activate the selected version
    const { error: activateError } = await supabaseClient
      .from('model_versions')
      .update({ is_active: true })
      .eq('id', version_id);

    if (activateError) {
      console.error('Error activating version:', activateError);
      return new Response(
        JSON.stringify({ error: 'Failed to activate version', details: activateError.message }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Also mark the model as active
    await supabaseClient
      .from('models')
      .update({ is_active: true })
      .eq('id', version.model_id);

    console.log(`Model version ${version.version} activated for model ${version.model_id} by user ${user.id}`);

    return new Response(
      JSON.stringify({
        success: true,
        message: `Version ${version.version} is now active`,
      }),
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
