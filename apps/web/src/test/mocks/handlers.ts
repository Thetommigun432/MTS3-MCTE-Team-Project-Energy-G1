/**
 * MSW request handlers for API mocking in tests.
 * 
 * These handlers mock the backend API responses for frontend tests.
 */
import { http, HttpResponse } from 'msw';

// Mock data for models endpoint
const mockModels = [
    {
        model_id: 'heatpump-v3-sota',
        model_version: 'v3-sota',
        appliance_id: 'HeatPump',
        architecture: 'wavenilm_v3',
        is_active: true,
        input_window_size: 1536,
    },
    {
        model_id: 'evcharger-v3-sota',
        model_version: 'v3-sota',
        appliance_id: 'EVCharger',
        architecture: 'wavenilm_v3',
        is_active: true,
        input_window_size: 1536,
    },
    {
        model_id: 'washingmachine-v3-sota',
        model_version: 'v3-sota',
        appliance_id: 'WashingMachine',
        architecture: 'wavenilm_v3',
        is_active: false,
        input_window_size: 1536,
    },
];

// Default backend URL for tests (can be overridden via env)
const BACKEND_URL = 'http://localhost:8000';

export const handlers = [
    // Health check - /live
    http.get(`${BACKEND_URL}/live`, () => {
        return HttpResponse.json({
            status: 'ok',
            timestamp: new Date().toISOString(),
        });
    }),

    // Readiness check - /ready
    http.get(`${BACKEND_URL}/ready`, () => {
        return HttpResponse.json({
            status: 'ready',
            checks: {
                influxdb_connected: true,
                influx_bucket_raw: true,
                influx_bucket_pred: true,
                registry_loaded: true,
                models_count: mockModels.filter(m => m.is_active).length,
                redis_available: true,
            },
        });
    }),

    // Models list - /models
    http.get(`${BACKEND_URL}/models`, () => {
        return HttpResponse.json({
            models: mockModels,
            count: mockModels.length,
            active_count: mockModels.filter(m => m.is_active).length,
        });
    }),

    // Single model - /models/:id
    http.get(`${BACKEND_URL}/models/:modelId`, ({ params }) => {
        const model = mockModels.find(m => m.model_id === params.modelId);
        if (model) {
            return HttpResponse.json(model);
        }
        return HttpResponse.json(
            { detail: 'Model not found' },
            { status: 404 }
        );
    }),
];

// Error scenario handlers (for testing error states)
export const errorHandlers = {
    networkError: http.get(`${BACKEND_URL}/live`, () => {
        return HttpResponse.error();
    }),

    serverError: http.get(`${BACKEND_URL}/live`, () => {
        return HttpResponse.json(
            { detail: 'Internal server error' },
            { status: 500 }
        );
    }),

    unreadyHealth: http.get(`${BACKEND_URL}/ready`, () => {
        return HttpResponse.json(
            {
                status: 'unavailable',
                checks: {
                    influxdb_connected: false,
                    influx_bucket_raw: false,
                    influx_bucket_pred: false,
                    registry_loaded: true,
                    models_count: 0,
                    redis_available: false,
                },
            },
            { status: 503 }
        );
    }),
};
