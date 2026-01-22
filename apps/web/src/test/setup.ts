import '@testing-library/jest-dom';
import { vi } from 'vitest';

// Mock environment variables for tests via import.meta.env
// Vitest uses import.meta.env, not process.env for Vite projects
vi.stubEnv('VITE_DEMO_MODE', 'true');
vi.stubEnv('VITE_LOCAL_MODE', 'false');
vi.stubEnv('VITE_SUPABASE_URL', 'https://test.supabase.co');
vi.stubEnv('VITE_SUPABASE_ANON_KEY', 'test-anon-key-placeholder');

// Mock ResizeObserver which is not available in jsdom
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};

// Mock matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  }),
});
