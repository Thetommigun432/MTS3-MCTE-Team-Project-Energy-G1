/**
 * Data Source Configuration
 *
 * Defines the source of truth for application data mode (API vs Demo).
 * This replaces ad-hoc env var checks throughout the codebase.
 */

// Only expose what's in Vite env
const ENV_DEMO_MODE = import.meta.env.VITE_DEMO_MODE === 'true';
const IS_PROD = import.meta.env.PROD;

export type DataSource = 'api' | 'demo';

/**
 * Get the current data source mode.
 *
 * Logic:
 * 1. If explicitly set to demo via Env, allow it (even in Prod previews)
 * 2. If IS_PROD, default to 'api'
 * 3. Default to 'api'
 */
export function getDataSource(): DataSource {
    if (ENV_DEMO_MODE) {
        return 'demo';
    }

    if (IS_PROD) {
        return 'api';
    }

    return 'api';
}

/**
 * Check if running in Demo mode.
 * Helper for UI conditionals.
 */
export function isDemoMode(): boolean {
    return getDataSource() === 'demo';
}
