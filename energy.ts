/**
 * NILM Energy Monitor - API Service
 * 
 * Service layer for communicating with the FastAPI backend.
 * Handles all API calls for the per-appliance model architecture.
 */

import type {
  ModelInfo,
  BatchPredictionResponse,
  AppliancePrediction,
  InferenceRequest,
  BatchSensorReadingRequest,
  IngestionResponse,
  ReadingsResponse,
  PredictionsHistoryResponse,
  HealthResponse,
  DetailedHealthResponse,
  QueueStats,
  ApiError,
} from '../types/api.types';

// ============================================================================
// Configuration
// ============================================================================

const getBaseUrl = (): string => {
  // In production, use the configured backend URL
  if (import.meta.env.VITE_BACKEND_URL) {
    return import.meta.env.VITE_BACKEND_URL;
  }
  // In development, use the Vite proxy (requests to /api/* are proxied)
  return '/api';
};

const BASE_URL = getBaseUrl();

// ============================================================================
// HTTP Client
// ============================================================================

interface RequestOptions extends RequestInit {
  params?: Record<string, string | number | boolean | undefined>;
}

class ApiClient {
  private baseUrl: string;
  private authToken: string | null = null;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  setAuthToken(token: string | null) {
    this.authToken = token;
  }

  private buildUrl(endpoint: string, params?: Record<string, string | number | boolean | undefined>): string {
    const url = new URL(endpoint, this.baseUrl.startsWith('http') ? this.baseUrl : window.location.origin + this.baseUrl);
    
    // Handle relative URLs
    if (!this.baseUrl.startsWith('http')) {
      url.pathname = this.baseUrl + endpoint;
    }
    
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          url.searchParams.append(key, String(value));
        }
      });
    }
    return url.toString();
  }

  private async request<T>(endpoint: string, options: RequestOptions = {}): Promise<T> {
    const { params, ...fetchOptions } = options;
    const url = this.buildUrl(endpoint, params);

    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    if (this.authToken) {
      (headers as Record<string, string>)['Authorization'] = `Bearer ${this.authToken}`;
    }

    const response = await fetch(url, {
      ...fetchOptions,
      headers,
    });

    if (!response.ok) {
      const error: ApiError = await response.json().catch(() => ({ 
        detail: `HTTP ${response.status}: ${response.statusText}` 
      }));
      throw new Error(error.detail);
    }

    return response.json();
  }

  async get<T>(endpoint: string, params?: Record<string, string | number | boolean | undefined>): Promise<T> {
    return this.request<T>(endpoint, { method: 'GET', params });
  }

  async post<T>(endpoint: string, body?: unknown, options?: RequestOptions): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: body ? JSON.stringify(body) : undefined,
      ...options,
    });
  }
}

const apiClient = new ApiClient(BASE_URL);

// Export for auth integration
export const setApiAuthToken = (token: string | null) => apiClient.setAuthToken(token);

// ============================================================================
// Health & Status Endpoints
// ============================================================================

/**
 * Check if the API is alive
 */
export async function checkLiveness(): Promise<{ status: string }> {
  return apiClient.get('/live');
}

/**
 * Check API readiness with component status
 */
export async function checkReadiness(): Promise<HealthResponse> {
  return apiClient.get('/ready');
}

/**
 * Get detailed health information
 */
export async function getDetailedHealth(): Promise<DetailedHealthResponse> {
  return apiClient.get('/health');
}

// ============================================================================
// Model Registry Endpoints
// ============================================================================

/**
 * List all registered per-appliance models
 */
export async function listModels(): Promise<ModelInfo[]> {
  return apiClient.get('/models');
}

/**
 * Get information about a specific model
 */
export async function getModelInfo(applianceId: string): Promise<ModelInfo> {
  return apiClient.get(`/models/${applianceId}`);
}

// ============================================================================
// Inference Endpoints
// ============================================================================

/**
 * Run inference for all appliances on a power window
 * 
 * @param buildingId - Building identifier
 * @param window - Array of 599 power readings
 * @param options - Optional settings
 */
export async function runInference(
  buildingId: string,
  window: number[],
  options: { timestamp?: Date; persist?: boolean } = {}
): Promise<BatchPredictionResponse> {
  const request: InferenceRequest = {
    building_id: buildingId,
    window,
    timestamp: options.timestamp?.toISOString(),
    persist: options.persist ?? true,
  };

  return apiClient.post('/infer', request);
}

/**
 * Run inference for a single specific appliance
 */
export async function runSingleInference(
  buildingId: string,
  applianceId: string,
  window: number[],
  options: { timestamp?: Date; persist?: boolean } = {}
): Promise<AppliancePrediction> {
  const request: InferenceRequest = {
    building_id: buildingId,
    window,
    timestamp: options.timestamp?.toISOString(),
    persist: options.persist ?? true,
  };

  return apiClient.post(`/infer/${applianceId}`, request);
}

// ============================================================================
// Data Ingestion Endpoints
// ============================================================================

/**
 * Ingest a batch of sensor readings for processing
 */
export async function ingestReadings(
  buildingId: string,
  readings: Array<{ power_watts: number; timestamp?: string }>
): Promise<IngestionResponse> {
  const request: BatchSensorReadingRequest = {
    building_id: buildingId,
    readings,
  };

  return apiClient.post('/ingest/batch', request);
}

// ============================================================================
// Analytics / Historical Data Endpoints
// ============================================================================

/**
 * Query historical sensor readings
 */
export async function getReadings(
  buildingId: string,
  start: string | Date,
  end: string | Date,
  resolution: string = '1m'
): Promise<ReadingsResponse> {
  return apiClient.get('/analytics/readings', {
    building_id: buildingId,
    start: start instanceof Date ? start.toISOString() : start,
    end: end instanceof Date ? end.toISOString() : end,
    resolution,
  });
}

/**
 * Query historical prediction data
 */
export async function getPredictions(
  buildingId: string,
  start: string | Date,
  end: string | Date,
  applianceId?: string
): Promise<PredictionsHistoryResponse> {
  return apiClient.get('/analytics/predictions', {
    building_id: buildingId,
    start: start instanceof Date ? start.toISOString() : start,
    end: end instanceof Date ? end.toISOString() : end,
    appliance_id: applianceId,
  });
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Format a prediction for display
 */
export function formatPrediction(prediction: AppliancePrediction): string {
  const state = prediction.is_on ? 'ON' : 'OFF';
  const power = prediction.predicted_watts.toFixed(1);
  const confidence = (prediction.confidence * 100).toFixed(0);
  return `${prediction.appliance_name}: ${power}W (${state}, ${confidence}% confidence)`;
}

/**
 * Calculate total power from predictions
 */
export function calculateTotalPower(predictions: AppliancePrediction[]): number {
  return predictions.reduce((sum, p) => sum + p.predicted_watts, 0);
}

/**
 * Group predictions by category (for UI organization)
 */
export function groupPredictionsByCategory(
  predictions: AppliancePrediction[]
): Record<string, AppliancePrediction[]> {
  // Category mapping (would come from backend in production)
  const categoryMap: Record<string, string> = {
    fridge: 'always_on',
    kettle: 'high_power',
    microwave: 'high_power',
    heat_pump: 'hvac',
    washing_machine: 'cyclic',
  };

  const grouped: Record<string, AppliancePrediction[]> = {};

  predictions.forEach(prediction => {
    const category = categoryMap[prediction.appliance_id] || 'other';
    if (!grouped[category]) {
      grouped[category] = [];
    }
    grouped[category].push(prediction);
  });

  return grouped;
}

/**
 * Check if backend is in demo mode (for UI display)
 */
export function isDemoMode(): boolean {
  return import.meta.env.VITE_DEMO_MODE === 'true';
}

// ============================================================================
// Polling Helper
// ============================================================================

/**
 * Create a polling function for real-time updates
 */
export function createPoller(
  buildingId: string,
  callback: (predictions: BatchPredictionResponse) => void,
  errorCallback: (error: Error) => void,
  intervalMs: number = 5000
): { start: () => void; stop: () => void } {
  let intervalId: number | null = null;
  let isPolling = false;

  // Mock window data generator (replace with actual data source)
  const generateMockWindow = (): number[] => {
    return Array.from({ length: 599 }, () => Math.random() * 200 + 400);
  };

  const poll = async () => {
    if (!isPolling) return;

    try {
      const window = generateMockWindow(); // TODO: Get actual sensor data
      const predictions = await runInference(buildingId, window, { persist: true });
      callback(predictions);
    } catch (error) {
      errorCallback(error instanceof Error ? error : new Error(String(error)));
    }
  };

  return {
    start: () => {
      if (isPolling) return;
      isPolling = true;
      poll(); // Initial poll
      intervalId = window.setInterval(poll, intervalMs);
    },
    stop: () => {
      isPolling = false;
      if (intervalId !== null) {
        window.clearInterval(intervalId);
        intervalId = null;
      }
    },
  };
}

// ============================================================================
// Default Export
// ============================================================================

export default {
  // Health
  checkLiveness,
  checkReadiness,
  getDetailedHealth,
  
  // Models
  listModels,
  getModelInfo,
  
  // Inference
  runInference,
  runSingleInference,
  
  // Ingestion
  ingestReadings,
  
  // Analytics
  getReadings,
  getPredictions,
  
  // Utilities
  formatPrediction,
  calculateTotalPower,
  groupPredictionsByCategory,
  isDemoMode,
  createPoller,
  setApiAuthToken,
};
