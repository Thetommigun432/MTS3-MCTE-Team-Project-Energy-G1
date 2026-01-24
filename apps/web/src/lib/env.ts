type RequiredEnv = {
  // New standardized names
  VITE_SUPABASE_URL?: string;
  VITE_SUPABASE_ANON_KEY?: string;
  VITE_BACKEND_URL?: string;
  // Deprecated names (backward compatibility)
  VITE_SUPABASE_PUBLISHABLE_KEY?: string;
  VITE_API_BASE_URL?: string;
};

export type AppEnv = {
  supabaseUrl: string;
  supabaseAnonKey: string;
  backendBaseUrl: string;
  demoMode: boolean;
  localMode: boolean;
  supabaseEnabled: boolean;
};

let cachedEnv: AppEnv | null = null;

/**
 * Helper to read env var with fallback to deprecated name
 */
function getEnvVar(
  env: Record<string, string | undefined>,
  newName: string,
  deprecatedName?: string
): string {
  const newValue = env[newName]?.trim();
  if (newValue) return newValue;

  if (deprecatedName) {
    const deprecatedValue = env[deprecatedName]?.trim();
    if (deprecatedValue) {
      console.warn(
        `[env] ${deprecatedName} is deprecated, use ${newName} instead`
      );
      return deprecatedValue;
    }
  }

  return "";
}

function readEnv(): AppEnv {
  const env = import.meta.env as RequiredEnv & Record<string, string | undefined>;

  const supabaseUrl = getEnvVar(env, "VITE_SUPABASE_URL");
  const supabaseAnonKey = getEnvVar(
    env,
    "VITE_SUPABASE_ANON_KEY",
    "VITE_SUPABASE_PUBLISHABLE_KEY"
  );
  const backendBaseUrl = getEnvVar(
    env,
    "VITE_BACKEND_URL",
    "VITE_API_BASE_URL"
  ) || (import.meta.env.DEV ? "/api" : "");

  const demoMode = env.VITE_DEMO_MODE === "true";
  const localMode = env.VITE_LOCAL_MODE === "true";

  // Supabase is optional in demo/local mode
  const supabaseEnabled = Boolean(supabaseUrl && supabaseAnonKey);

  // Only require Supabase credentials if not in demo/local mode
  if (!supabaseEnabled && !demoMode && !localMode) {
    const missing: string[] = [];
    if (!supabaseUrl) missing.push("VITE_SUPABASE_URL");
    if (!supabaseAnonKey) missing.push("VITE_SUPABASE_ANON_KEY");

    console.warn(
      `Missing required environment variables: ${missing.join(", ")}. Defaulting to DEMO MODE to prevent crash.`
    );
    // Fallback to demo mode to ensure UI renders
    return {
      supabaseUrl: "https://placeholder.supabase.co",
      supabaseAnonKey: "placeholder-key",
      backendBaseUrl,
      demoMode: true,
      localMode,
      supabaseEnabled: false,
    };
  }

  return {
    supabaseUrl: supabaseUrl || "https://placeholder.supabase.co",
    supabaseAnonKey: supabaseAnonKey || "placeholder-key",
    backendBaseUrl,
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

export function hasBackendUrl(): boolean {
  return Boolean(getEnv().backendBaseUrl);
}

export function isSupabaseEnabled(): boolean {
  return getEnv().supabaseEnabled;
}

// Deprecated - use hasBackendUrl instead
export function hasApiBaseUrl(): boolean {
  return hasBackendUrl();
}
