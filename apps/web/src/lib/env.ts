type RequiredEnv = {
  VITE_SUPABASE_URL?: string;
  VITE_SUPABASE_PUBLISHABLE_KEY?: string;
};

export type AppEnv = {
  supabaseUrl: string;
  supabaseAnonKey: string;
  apiBaseUrl: string;
  demoMode: boolean;
  localMode: boolean;
  supabaseEnabled: boolean;
};

let cachedEnv: AppEnv | null = null;

function readEnv(): AppEnv {
  const env = import.meta.env as RequiredEnv & Record<string, string | undefined>;
  const supabaseUrl = env.VITE_SUPABASE_URL?.trim() || "";
  const supabaseAnonKey = env.VITE_SUPABASE_PUBLISHABLE_KEY?.trim() || "";
  const demoMode = env.VITE_DEMO_MODE === "true";
  const localMode = env.VITE_LOCAL_MODE === "true";

  // Supabase is optional in demo/local mode
  const supabaseEnabled = Boolean(supabaseUrl && supabaseAnonKey);

  // Only require Supabase credentials if not in demo/local mode
  if (!supabaseEnabled && !demoMode && !localMode) {
    const missing: string[] = [];
    if (!supabaseUrl) missing.push("VITE_SUPABASE_URL");
    if (!supabaseAnonKey) missing.push("VITE_SUPABASE_PUBLISHABLE_KEY");
    const message = `Missing required environment variables: ${missing.join(", ")}. Set VITE_DEMO_MODE=true or VITE_LOCAL_MODE=true to run without Supabase.`;
    console.error(message);
    throw new Error(message);
  }

  return {
    supabaseUrl: supabaseUrl || "https://placeholder.supabase.co",
    supabaseAnonKey: supabaseAnonKey || "placeholder-key",
    apiBaseUrl: env.VITE_API_BASE_URL?.trim() || "",
    demoMode,
    localMode,
    supabaseEnabled,
  };
}

export function getEnv(): AppEnv {
  if (!cachedEnv) {
    cachedEnv = readEnv();
  }
  return cachedEnv;
}

export function hasApiBaseUrl(): boolean {
  return Boolean(getEnv().apiBaseUrl);
}

export function isSupabaseEnabled(): boolean {
  return getEnv().supabaseEnabled;
}
