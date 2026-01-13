/**
 * Custom storage adapter for Supabase auth that supports
 * "Remember me" functionality by switching between localStorage and sessionStorage
 */

const REMEMBER_ME_KEY = "nilm-remember-me";

// Check if user opted for "remember me"
export function getRememberMe(): boolean {
  try {
    return localStorage.getItem(REMEMBER_ME_KEY) === "true";
  } catch {
    return true; // Default to remember
  }
}

export function setRememberMe(value: boolean): void {
  try {
    localStorage.setItem(REMEMBER_ME_KEY, value ? "true" : "false");
  } catch {
    // Ignore storage errors
  }
}

// Custom storage that routes to the appropriate storage based on remember me setting
export const customAuthStorage = {
  getItem: (key: string): string | null => {
    try {
      // Always try localStorage first (for remembered sessions)
      const localValue = localStorage.getItem(key);
      if (localValue) return localValue;

      // Fall back to sessionStorage
      return sessionStorage.getItem(key);
    } catch {
      return null;
    }
  },

  setItem: (key: string, value: string): void => {
    try {
      const rememberMe = getRememberMe();
      if (rememberMe) {
        localStorage.setItem(key, value);
        // Clean up sessionStorage if exists
        sessionStorage.removeItem(key);
      } else {
        sessionStorage.setItem(key, value);
        // Clean up localStorage if exists
        localStorage.removeItem(key);
      }
    } catch {
      // Ignore storage errors
    }
  },

  removeItem: (key: string): void => {
    try {
      localStorage.removeItem(key);
      sessionStorage.removeItem(key);
    } catch {
      // Ignore storage errors
    }
  },
};
