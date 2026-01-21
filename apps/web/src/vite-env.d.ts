/// <reference types="vite/client" />

interface ImportMetaEnv {
  // Standardized env vars
  readonly VITE_SUPABASE_URL: string;
  readonly VITE_SUPABASE_ANON_KEY: string;
  readonly VITE_BACKEND_URL?: string;
  readonly VITE_DEMO_MODE?: string;
  readonly VITE_LOCAL_MODE?: string;
  readonly VITE_DEMO_EMAIL?: string;
  readonly VITE_DEMO_PASSWORD?: string;
  readonly VITE_DEMO_USERNAME?: string;
  // Deprecated (backward compatibility)
  readonly VITE_SUPABASE_PUBLISHABLE_KEY?: string;
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
