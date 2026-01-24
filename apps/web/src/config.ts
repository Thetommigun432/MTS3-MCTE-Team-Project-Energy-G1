/**
 * Application Configuration
 * 
 * Single source of truth for all environment-based configuration.
 * Re-exports from lib/env.ts for convenience.
 */

export { getEnv, hasBackendUrl, isSupabaseEnabled } from "@/lib/env";
export type { AppEnv } from "@/lib/env";
