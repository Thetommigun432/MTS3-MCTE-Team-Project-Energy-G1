/**
 * Data Source Configuration
 *
 * Defines the source of truth for application data mode (API vs Demo).
 * This replaces ad-hoc env var checks throughout the codebase.
 *
 * ARCHITECTURE:
 * - Production builds default to 'api' mode
 * - Demo mode requires explicit VITE_DEMO_MODE=true
 * - Demo mode shows a visible banner to prevent confusion
 */

// Only expose what's in Vite env (Vite requires VITE_ prefix)
const ENV_DEMO_MODE = import.meta.env.VITE_DEMO_MODE === 'true';

export type DataSource = 'api' | 'demo';

/**
 * Get the current data source mode.
 *
 * Priority:
 * 1. If VITE_DEMO_MODE=true → 'demo' (in any environment)
 * 2. Otherwise → 'api'
 *
 * This ensures explicit opt-in for demo mode in all environments.
 */
export function getDataSource(): DataSource {
    // Demo mode requires explicit opt-in via environment variable
    if (ENV_DEMO_MODE) {
        return 'demo';
    }

    // Default: API mode for both dev and production
    return 'api';
}

/**
 * Check if running in Demo mode.
 * Helper for UI conditionals (e.g., showing demo banner).
 */
export function isDemoMode(): boolean {
    return getDataSource() === 'demo';
}

/**
 * Check if running in API mode.
 * Helper for data fetching conditionals.
 */
export function isApiMode(): boolean {
    return getDataSource() === 'api';
}
