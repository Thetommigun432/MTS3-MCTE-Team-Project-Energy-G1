/**
 * Environment variable handling
 * Robustly parses and sanitizes environment variables.
 */

/**
 * Get environment variables with robust parsing.
 * Handles quirks like Cloudflare adding extra quotes.
 */
export function getEnv() {
  const env = import.meta.env;

  // Backend URL: Strip quotes, trim whitespace, ensure protocol
  // Default to "/api" (Vite proxy) if not set, enabling zero-config local dev
  const backendBaseUrl = parseUrl(env.VITE_BACKEND_URL || env.VITE_API_BASE_URL) || "/api";

  // Supabase
  let supabaseUrl = stripQuotes(env.VITE_SUPABASE_URL);
  const supabaseProjectId = stripQuotes(env.VITE_SUPABASE_PROJECT_ID);

  // If URL missing but Project ID exists, construct it
  if (!supabaseUrl && supabaseProjectId) {
    supabaseUrl = `https://${supabaseProjectId}.supabase.co`;
  }

  // Prefer Publishable Key (safe), fallback to Anon Key (legacy)
  const supabasePublishableKey = stripQuotes(env.VITE_SUPABASE_PUBLISHABLE_KEY || env.VITE_SUPABASE_ANON_KEY);

  return {
    backendBaseUrl,
    supabaseUrl,
    supabasePublishableKey,
    // Alias for backward compatibility
    supabaseAnonKey: supabasePublishableKey,
    // Enabled only if both URL and Key are present
    supabaseEnabled: !!supabaseUrl && !!supabasePublishableKey,
  };
}

export type AppEnv = ReturnType<typeof getEnv>;

export const isSupabaseEnabled = () => getEnv().supabaseEnabled;

export const hasBackendUrl = () => !!getEnv().backendBaseUrl;
export const hasApiBaseUrl = hasBackendUrl; // Alias if needed

/**
 * Helper to strip surrounding quotes if present (common in some CI/CD envs)
 */
function stripQuotes(value?: string): string {
  if (!value) return "";
  const trimmed = value.trim();
  if (
    (trimmed.startsWith('"') && trimmed.endsWith('"')) ||
    (trimmed.startsWith("'") && trimmed.endsWith("'"))
  ) {
    return trimmed.slice(1, -1);
  }
  return trimmed;
}

/**
 * Parse and validate URL
 * - Strips quotes
 * - Adds https:// if protocol missing and not localhost
 */
function parseUrl(value?: string): string {
  let url = stripQuotes(value);

  if (!url) return "";

  // If it's just a path (e.g. "/api"), return as is (useful for proxying)
  if (url.startsWith("/")) {
    return url;
  }

  // If no protocol matches (http:// or https://)
  if (!/^https?:\/\//i.test(url)) {
    // If localhost, default to http
    if (url.includes("localhost") || url.includes("127.0.0.1")) {
      url = `http://${url}`;
    } else {
      // Otherwise default to https
      url = `https://${url}`;
    }
  }

  // Remove trailing slash
  return url.replace(/\/$/, "");
}
