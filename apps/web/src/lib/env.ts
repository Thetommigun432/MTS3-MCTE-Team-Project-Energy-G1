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

/**
 * Normalize backend URL to ensure it's a valid fetch target.
 * 
 * Rules:
 * 1. If empty or undefined, return empty string (will use defaults later)
 * 2. If starts with http:// or https://, use as-is (full URL)
 * 3. If starts with /, use as-is (relative path for proxy setups)
 * 4. Otherwise, assume it's a bare hostname and prepend https://
 * 
 * @example
 * normalizeBackendUrl("https://api.example.com") => "https://api.example.com"
 * normalizeBackendUrl("energy-monitor.up.railway.app") => "https://energy-monitor.up.railway.app"
 * normalizeBackendUrl("/api") => "/api"
 * normalizeBackendUrl("") => ""
 */
export function normalizeBackendUrl(url: string | undefined): string {
  if (!url) return "";

  const trimmed = url.trim();
  if (!trimmed) return "";

  // Full URL - use as-is
  if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
    // Remove trailing slash for consistency
    return trimmed.replace(/\/$/, "");
  }

  // Relative path (e.g., /api for reverse proxy) - use as-is
  if (trimmed.startsWith("/")) {
    return trimmed.replace(/\/$/, "");
  }

  // Bare hostname (e.g., energy-monitor.up.railway.app) - add https://
  // This is the common misconfiguration case
  console.info(`[env] VITE_BACKEND_URL "${trimmed}" appears to be a bare hostname, normalizing to https://${trimmed}`);
  return `https://${trimmed}`;
}

function readEnv(): AppEnv {
  const env = import.meta.env as RequiredEnv & Record<string, string | undefined>;

  const supabaseUrl = getEnvVar(env, "VITE_SUPABASE_URL");
  const supabaseAnonKey = getEnvVar(
    env,
    "VITE_SUPABASE_ANON_KEY",
    "VITE_SUPABASE_PUBLISHABLE_KEY"
  );

  // Get raw value and normalize it
  const rawBackendUrl = getEnvVar(
    env,
    "VITE_BACKEND_URL",
    "VITE_API_BASE_URL"
  );
  const backendBaseUrl = normalizeBackendUrl(rawBackendUrl) || (import.meta.env.DEV ? "/api" : "");


  const demoMode = env.VITE_DEMO_MODE === "true";
  const localMode = env.VITE_LOCAL_MODE === "true";

  // Supabase is optional in demo/local mode
  const supabaseEnabled = Boolean(supabaseUrl && supabaseAnonKey);

  // IMPORTANT: In production, missing credentials is a configuration error.
  // We allow operation without Supabase, but log clear warnings.
  if (!supabaseEnabled && !demoMode && !localMode) {
    const missing: string[] = [];
    if (!supabaseUrl) missing.push("VITE_SUPABASE_URL");
    if (!supabaseAnonKey) missing.push("VITE_SUPABASE_ANON_KEY");

    // In production builds, this is likely a deployment configuration error
    if (import.meta.env.PROD) {
      console.error(
        `[PRODUCTION ERROR] Missing required environment variables: ${missing.join(", ")}. ` +
        `Authentication will not work. Set these in Cloudflare Pages environment settings ` +
        `or explicitly enable demo mode with VITE_DEMO_MODE=true.`
      );
    } else {
      console.warn(
        `[DEV] Missing environment variables: ${missing.join(", ")}. ` +
        `Set VITE_DEMO_MODE=true to run in demo mode, or configure Supabase credentials.`
      );
    }

    // Don't silently enable demo mode - let the app run with disabled auth
    // This makes configuration issues visible rather than masking them
    return {
      supabaseUrl: "https://placeholder.supabase.co",
      supabaseAnonKey: "placeholder-key",
      backendBaseUrl,
      demoMode: false, // Don't auto-enable demo mode - let auth fail visibly
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
