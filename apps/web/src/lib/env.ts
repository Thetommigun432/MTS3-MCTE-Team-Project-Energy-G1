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

  return {
    // Backend URL: Strip quotes, trim whitespace, ensure protocol
    backendBaseUrl: parseUrl(env.VITE_BACKEND_URL),

    // Supabase
    // Supabase
    // Fallback to hardcoded values for "zero-config" deployment if env vars missing
    supabaseUrl: stripQuotes(env.VITE_SUPABASE_URL) || "https://bhdcbvruzvhmcogxfkil.supabase.co",
    supabaseAnonKey: stripQuotes(env.VITE_SUPABASE_ANON_KEY) || "sb_publishable_I3L-MhSwOX8vfWt91ppHOg_uu18TByg",

    // Legacy support (check actual values)
    supabaseEnabled: true, // Always enabled since we have fallbacks

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
