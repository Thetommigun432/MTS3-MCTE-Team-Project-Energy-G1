/**
 * Data Source Configuration
 *
 * Defines the source of truth for application data mode (API vs Demo).
 * This replaces ad-hoc env var checks throughout the codebase.
 *
 * ARCHITECTURE:
 * - Mode is RUNTIME switchable via localStorage
 * - VITE_DEMO_MODE=true sets the DEFAULT mode (not a lock)
 * - User can toggle between api/demo at runtime
 * - Demo mode shows a visible banner to prevent confusion
 */

// Storage key for persisted mode
const MODE_STORAGE_KEY = 'energy-monitor-mode';

// Only expose what's in Vite env (Vite requires VITE_ prefix)
const ENV_DEMO_MODE = import.meta.env.VITE_DEMO_MODE === 'true';

export type DataSource = 'api' | 'demo';

// In-memory cache for current mode (avoids repeated localStorage reads)
let cachedMode: DataSource | null = null;

// Event listeners for mode changes
type ModeChangeListener = (mode: DataSource) => void;
const modeChangeListeners: Set<ModeChangeListener> = new Set();

/**
 * Get the current data source mode.
 *
 * Priority:
 * 1. In-memory cache (fastest)
 * 2. localStorage override (user preference)
 * 3. VITE_DEMO_MODE env var (build-time default)
 * 4. 'api' (final fallback)
 */
export function getDataSource(): DataSource {
    // Return cached value if available
    if (cachedMode !== null) {
        return cachedMode;
    }

    // Check localStorage for user preference
    if (typeof window !== 'undefined') {
        const stored = localStorage.getItem(MODE_STORAGE_KEY);
        if (stored === 'api' || stored === 'demo') {
            cachedMode = stored;
            return stored;
        }
    }

    // Fall back to env var default
    const defaultMode = ENV_DEMO_MODE ? 'demo' : 'api';
    cachedMode = defaultMode;
    return defaultMode;
}

/**
 * Set the data source mode (persisted to localStorage).
 * This is the ONLY way to change the mode at runtime.
 */
export function setDataSource(mode: DataSource): void {
    if (mode !== 'api' && mode !== 'demo') {
        console.warn(`Invalid data source mode: ${mode}`);
        return;
    }

    // Update cache and storage
    cachedMode = mode;
    if (typeof window !== 'undefined') {
        localStorage.setItem(MODE_STORAGE_KEY, mode);
    }

    // Notify listeners
    modeChangeListeners.forEach(listener => {
        try {
            listener(mode);
        } catch (e) {
            console.error('Error in mode change listener:', e);
        }
    });
}

/**
 * Subscribe to mode changes.
 * Returns an unsubscribe function.
 */
export function onModeChange(listener: ModeChangeListener): () => void {
    modeChangeListeners.add(listener);
    return () => modeChangeListeners.delete(listener);
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
