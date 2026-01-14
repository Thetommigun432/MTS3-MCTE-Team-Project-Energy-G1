/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_SUPABASE_PUBLISHABLE_KEY: string;
  readonly VITE_SUPABASE_URL: string;
  readonly VITE_API_BASE_URL?: string;
  readonly VITE_DEMO_MODE?: string;
  readonly VITE_LOCAL_MODE?: string;
  readonly VITE_DEMO_EMAIL?: string;
  readonly VITE_DEMO_PASSWORD?: string;
  readonly VITE_DEMO_USERNAME?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
