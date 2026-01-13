import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

interface FinalizeModelVersionRequest {
  version_id: string;
  metrics?: Record<string, number>;
  training_config?: Record<string, unknown>;
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
    const body: FinalizeModelVersionRequest = await req.json();
    const { version_id, metrics = {}, training_config = {} } = body;

    if (!version_id) {
      return new Response(
        JSON.stringify({ error: 'version_id is required' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Verify the version belongs to the user and is in 'uploading' status
    const { data: version, error: versionError } = await supabaseClient
      .from('model_versions')
      .select('id, status, model_artifact_path')
      .eq('id', version_id)
      .single();

    if (versionError || !version) {
      return new Response(
        JSON.stringify({ error: 'Model version not found or access denied' }),
        { status: 404, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    if (version.status !== 'uploading') {
      return new Response(
        JSON.stringify({ error: `Cannot finalize version in status: ${version.status}` }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Verify the model artifact was uploaded
    const { data: fileData, error: fileError } = await supabaseClient.storage
      .from('models')
      .list(version.model_artifact_path.split('/').slice(0, -1).join('/'));

    const hasModelFile = fileData?.some(f => f.name === 'model.pt');

    if (!hasModelFile) {
      // Mark as failed if model file not found
      await supabaseClient
        .from('model_versions')
        .update({ status: 'failed' })
        .eq('id', version_id);

      return new Response(
        JSON.stringify({ error: 'Model artifact not found. Upload may have failed.' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Update version to 'ready' status
    const { error: updateError } = await supabaseClient
      .from('model_versions')
      .update({
        status: 'ready',
        metrics,
        training_config,
        trained_at: new Date().toISOString(),
      })
      .eq('id', version_id);

    if (updateError) {
      console.error('Error finalizing version:', updateError);
      return new Response(
        JSON.stringify({ error: 'Failed to finalize version', details: updateError.message }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    console.log(`Model version ${version_id} finalized by user ${user.id}`);

    return new Response(
      JSON.stringify({
        success: true,
        message: 'Model version finalized and ready for deployment',
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
