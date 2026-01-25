import '@testing-library/jest-dom';
import { vi, beforeAll, afterAll, afterEach } from 'vitest';
import { server } from './mocks/server';

// ============================================================================
// MSW Server Lifecycle
// ============================================================================

// Start server before all tests
beforeAll(() => {
  server.listen({ onUnhandledRequest: 'bypass' });
});

// Reset handlers after each test (for test isolation)
afterEach(() => {
  server.resetHandlers();
});

// Clean up after all tests
afterAll(() => {
  server.close();
});

// ============================================================================
// Environment Variable Stubs
// ============================================================================

// Mock environment variables for tests via import.meta.env
// Vitest uses import.meta.env, not process.env for Vite projects
vi.stubEnv('VITE_DEMO_MODE', 'true');
vi.stubEnv('VITE_LOCAL_MODE', 'false');
vi.stubEnv('VITE_SUPABASE_URL', 'https://test.supabase.co');
vi.stubEnv('VITE_SUPABASE_ANON_KEY', 'test-anon-key-placeholder');
vi.stubEnv('VITE_BACKEND_URL', 'http://localhost:8000');

// ============================================================================
// DOM Mocks for jsdom
// ============================================================================

// Mock ResizeObserver which is not available in jsdom
global.ResizeObserver = class ResizeObserver {
  observe() { }
  unobserve() { }
  disconnect() { }
};

// Mock matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: (query: string): MediaQueryList => ({
    matches: false,
    media: query,
    onchange: null as ((this: MediaQueryList, ev: MediaQueryListEvent) => unknown) | null,
    addListener: () => { },
    removeListener: () => { },
    addEventListener: () => { },
    removeEventListener: () => { },
    dispatchEvent: () => false,
  }),
});
