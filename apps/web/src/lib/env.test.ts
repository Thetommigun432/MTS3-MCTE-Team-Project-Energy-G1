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
    it('should return false when in demo mode without Supabase credentials', async () => {
      vi.stubEnv('VITE_DEMO_MODE', 'true');
      vi.stubEnv('VITE_SUPABASE_URL', '');
      vi.stubEnv('VITE_SUPABASE_PUBLISHABLE_KEY', '');

      const { isSupabaseEnabled } = await import('./env');
      expect(isSupabaseEnabled()).toBe(false);
    });

    it('should return false when in local mode without Supabase credentials', async () => {
      vi.stubEnv('VITE_LOCAL_MODE', 'true');
      vi.stubEnv('VITE_SUPABASE_URL', '');
      vi.stubEnv('VITE_SUPABASE_PUBLISHABLE_KEY', '');

      const { isSupabaseEnabled } = await import('./env');
      expect(isSupabaseEnabled()).toBe(false);
    });
  });

  describe('hasApiBaseUrl', () => {
    it('should return false when no API base URL is set', async () => {
      vi.stubEnv('VITE_DEMO_MODE', 'true');
      vi.stubEnv('VITE_API_BASE_URL', '');

      const { hasApiBaseUrl } = await import('./env');
      expect(hasApiBaseUrl()).toBe(false);
    });
  });
});
