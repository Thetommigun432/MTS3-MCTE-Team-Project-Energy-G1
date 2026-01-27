
import { describe, it, expect, vi } from 'vitest';
import { api } from './api';

// Mock getEnv to control API_BASE_URL
vi.mock('@/lib/env', () => ({
    getEnv: vi.fn(() => ({
        backendBaseUrl: 'http://localhost:3000',
        supabaseUrl: 'https://placeholder.supabase.co',
        supabaseAnonKey: 'placeholder',
        supabasePublishableKey: 'placeholder',
        supabaseEnabled: false
    })),
    isApiConfigured: () => true,
    getApiBaseUrl: () => 'http://localhost:3000'
}));

import { getEnv } from '@/lib/env';

describe('API Service URL Building', () => {
    it('should handle absolute URL base', async () => {
        vi.mocked(getEnv).mockReturnValue({
            backendBaseUrl: 'https://api.example.com',
            supabaseUrl: '',
            supabaseAnonKey: '',
            supabasePublishableKey: '',
            supabaseEnabled: false
        });

        // We can't easily test private buildUrl directly without exporting it,
        // but we can spy on fetch to see the URL it gets.
        global.fetch = vi.fn().mockResolvedValue({
            ok: true,
            headers: new Headers(),
            json: () => Promise.resolve({})
        });

        await api.get('/test');

        expect(global.fetch).toHaveBeenCalledWith(
            expect.stringContaining('https://api.example.com/test'),
            expect.any(Object)
        );
    });

    it('should handle relative API base (proxy)', async () => {
        vi.mocked(getEnv).mockReturnValue({
            backendBaseUrl: '/api',
            supabaseUrl: '',
            supabaseAnonKey: '',
            supabasePublishableKey: '',
            supabaseEnabled: false
        });

        global.fetch = vi.fn().mockResolvedValue({
            ok: true,
            headers: new Headers(),
            json: () => Promise.resolve({})
        });

        await api.get('/users');

        // Should construct /api/users
        expect(global.fetch).toHaveBeenCalledWith(
            '/api/users',
            expect.any(Object)
        );
    });

    it('should correctly append query parameters', async () => {
        vi.mocked(getEnv).mockReturnValue({
            backendBaseUrl: 'https://api.example.com',
            supabaseUrl: '',
            supabaseAnonKey: '',
            supabasePublishableKey: '',
            supabaseEnabled: false
        });

        global.fetch = vi.fn().mockResolvedValue({
            ok: true,
            headers: new Headers(),
            json: () => Promise.resolve({})
        });

        await api.get('/search', { params: { q: 'test', page: 1 } });

        expect(global.fetch).toHaveBeenCalledWith(
            'https://api.example.com/search?q=test&page=1',
            expect.any(Object)
        );
    });
});
