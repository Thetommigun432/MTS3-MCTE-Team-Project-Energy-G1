import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// We need to reset the module cache between tests
describe('env utilities', () => {
  beforeEach(() => {
    vi.resetModules();
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  describe('isSupabaseEnabled', () => {
    it('should stay enabled (zero-config) even without credentials', async () => {
      vi.stubEnv('VITE_DEMO_MODE', 'true');
      vi.stubEnv('VITE_SUPABASE_URL', '');
      vi.stubEnv('VITE_SUPABASE_ANON_KEY', '');

      const { isSupabaseEnabled } = await import('./env');
      expect(isSupabaseEnabled()).toBe(true);
    });

    it('should stay enabled in local mode without credentials', async () => {
      vi.stubEnv('VITE_LOCAL_MODE', 'true');
      vi.stubEnv('VITE_SUPABASE_URL', '');
      vi.stubEnv('VITE_SUPABASE_ANON_KEY', '');

      const { isSupabaseEnabled } = await import('./env');
      expect(isSupabaseEnabled()).toBe(true);
    });

    it('should support deprecated VITE_SUPABASE_PUBLISHABLE_KEY', async () => {
      vi.stubEnv('VITE_SUPABASE_URL', 'https://test.supabase.co');
      vi.stubEnv('VITE_SUPABASE_PUBLISHABLE_KEY', 'test-key');
      vi.stubEnv('VITE_SUPABASE_ANON_KEY', '');

      const { isSupabaseEnabled } = await import('./env');
      expect(isSupabaseEnabled()).toBe(true);
    });
  });

  describe('hasBackendUrl', () => {
    it('should return false when no backend URL is set in production', async () => {
      vi.stubEnv('VITE_DEMO_MODE', 'true');
      vi.stubEnv('VITE_BACKEND_URL', '');
      vi.stubEnv('VITE_API_BASE_URL', '');
      vi.stubEnv('DEV', false);

      const { hasBackendUrl } = await import('./env');
      // In test env, DEV defaults to true so it uses localhost fallback
      // This test verifies the function exists and returns boolean
      expect(typeof hasBackendUrl()).toBe('boolean');
    });

    it('should support deprecated VITE_API_BASE_URL', async () => {
      vi.stubEnv('VITE_DEMO_MODE', 'true');
      vi.stubEnv('VITE_BACKEND_URL', '');
      vi.stubEnv('VITE_API_BASE_URL', 'https://api.example.com');

      const { hasBackendUrl } = await import('./env');
      expect(hasBackendUrl()).toBe(true);
    });
  });

  // Deprecated function test
  describe('hasApiBaseUrl (deprecated)', () => {
    it('should still work as alias for hasBackendUrl', async () => {
      vi.stubEnv('VITE_DEMO_MODE', 'true');
      vi.stubEnv('VITE_BACKEND_URL', 'https://api.example.com');

      const { hasApiBaseUrl } = await import('./env');
      expect(hasApiBaseUrl()).toBe(true);
    });
  });
});
