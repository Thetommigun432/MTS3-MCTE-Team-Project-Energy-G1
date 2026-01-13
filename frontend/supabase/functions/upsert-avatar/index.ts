import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

/**
 * Upsert Avatar Edge Function
 * 
 * Uploads user avatar to storage bucket and updates profile.
 * Handles file upload via multipart/form-data or base64.
 * 
 * Request:
 * - FormData with 'avatar' file field, OR
 * - JSON body with { avatar_base64: string, filename: string }
 */

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

    let fileData: Uint8Array;
    let fileName: string;
    let contentType: string;

    const reqContentType = req.headers.get('content-type') || '';

    if (reqContentType.includes('multipart/form-data')) {
      // Handle FormData upload
      const formData = await req.formData();
      const file = formData.get('avatar') as File | null;

      if (!file) {
        return new Response(
          JSON.stringify({ error: 'No avatar file provided' }),
          { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }

      // Validate file type
      const allowedTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
      if (!allowedTypes.includes(file.type)) {
        return new Response(
          JSON.stringify({ error: 'Invalid file type. Allowed: JPEG, PNG, GIF, WebP' }),
          { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }

      // Validate file size (max 5MB)
      if (file.size > 5 * 1024 * 1024) {
        return new Response(
          JSON.stringify({ error: 'File too large. Maximum size is 5MB' }),
          { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }

      fileData = new Uint8Array(await file.arrayBuffer());
      fileName = file.name;
      contentType = file.type;

    } else if (reqContentType.includes('application/json')) {
      // Handle base64 upload
      const body = await req.json();
      const { avatar_base64, filename } = body;

      if (!avatar_base64 || !filename) {
        return new Response(
          JSON.stringify({ error: 'avatar_base64 and filename are required' }),
          { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }

      // Decode base64
      const base64Data = avatar_base64.replace(/^data:image\/\w+;base64,/, '');
      fileData = Uint8Array.from(atob(base64Data), c => c.charCodeAt(0));
      fileName = filename;
      
      // Detect content type from extension
      const ext = filename.split('.').pop()?.toLowerCase();
      const typeMap: Record<string, string> = {
        jpg: 'image/jpeg',
        jpeg: 'image/jpeg',
        png: 'image/png',
        gif: 'image/gif',
        webp: 'image/webp',
      };
      contentType = typeMap[ext || ''] || 'image/jpeg';

    } else {
      return new Response(
        JSON.stringify({ error: 'Unsupported content type' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Generate storage path: avatars/{user_id}/avatar.{ext}
    const ext = fileName.split('.').pop() || 'jpg';
    const storagePath = `${user.id}/avatar.${ext}`;

    // Delete existing avatar first (ignore errors)
    const { data: existingFiles } = await supabaseClient.storage
      .from('avatars')
      .list(user.id);

    if (existingFiles && existingFiles.length > 0) {
      const filesToDelete = existingFiles.map(f => `${user.id}/${f.name}`);
      await supabaseClient.storage.from('avatars').remove(filesToDelete);
    }

    // Upload new avatar
    const { data: uploadData, error: uploadError } = await supabaseClient.storage
      .from('avatars')
      .upload(storagePath, fileData, {
        contentType,
        upsert: true,
      });

    if (uploadError) {
      console.error('Upload error:', uploadError);
      return new Response(
        JSON.stringify({ error: 'Failed to upload avatar', details: uploadError.message }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Get public URL
    const { data: { publicUrl } } = supabaseClient.storage
      .from('avatars')
      .getPublicUrl(storagePath);

    // Add cache-busting timestamp
    const avatarUrl = `${publicUrl}?t=${Date.now()}`;

    // Update profile with new avatar URL
    const { error: updateError } = await supabaseClient
      .from('profiles')
      .update({ avatar_url: avatarUrl, updated_at: new Date().toISOString() })
      .eq('id', user.id);

    if (updateError) {
      console.error('Profile update error:', updateError);
      return new Response(
        JSON.stringify({ error: 'Failed to update profile', details: updateError.message }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    console.log(`Avatar uploaded for user ${user.id}: ${storagePath}`);

    return new Response(
      JSON.stringify({
        success: true,
        avatar_url: avatarUrl,
        path: uploadData?.path,
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
