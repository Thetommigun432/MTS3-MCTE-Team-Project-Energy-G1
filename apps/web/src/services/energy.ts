/**
 * Energy API Service
 * Typed interface for energy data endpoints.
 *
 * This service interfaces with the FastAPI backend for:
 * - Analytics: readings and predictions from InfluxDB
 * - Inference: running ML predictions
 * - Models: listing available ML models
 */

import { api, isApiConfigured } from "./api";

// ============================================================================
// Analytics Types (readings and predictions from InfluxDB)
// ============================================================================

export interface AnalyticsParams {
  building_id: string;
  start: string; // ISO8601 or relative (e.g., "-7d", "2024-01-01T00:00:00Z")
  end: string; // ISO8601 or relative (e.g., "now()", "2024-01-08T00:00:00Z")
  appliance_id?: string;
  resolution?: "1s" | "1m" | "15m";
}

export interface ReadingDataPoint {
  time: string; // ISO8601
  value: number;
  [key: string]: unknown; // Additional fields allowed
}

export interface ReadingsResponse {
  building_id: string;
  appliance_id: string | null;
  start: string;
  end: string;
  resolution: string;
  data: ReadingDataPoint[];
  count: number;
}

export interface PredictionDataPoint {
  time: string; // ISO8601
  predicted_kw: number;
  confidence?: number;
  model_version?: string;
  [key: string]: unknown; // Additional fields allowed
}

export interface PredictionsResponse {
  building_id: string;
  appliance_id: string | null;
  start: string;
  end: string;
  resolution: string;
  data: PredictionDataPoint[];
  count: number;
}

// ============================================================================
// Inference Types (ML prediction)
// ============================================================================

export interface InferRequest {
  building_id: string;
  appliance_id: string;
  window: number[]; // Array of 1000 floats (power readings)
  timestamp?: string; // ISO8601, optional
  model_id?: string; // Optional, defaults to active model
}

export interface InferResponse {
  predicted_kw: number;
  confidence: number;
  model_version: string;
  request_id: string;
  persisted: boolean;
}

// ============================================================================
// Model Registry Types
// ============================================================================

export interface Model {
  model_id: string;
  model_version: string;
  appliance_id: string;
  architecture: string;
  input_window_size: number;
  is_active: boolean;
  cached: boolean;
}

export interface ModelsListResponse {
  models: Model[];
  count: number;
}

// ============================================================================
// Legacy Types (kept for backward compatibility during migration)
// ============================================================================

export interface EnergyReading {
  timestamp: string;
  building: string;
  appliance: string;
  power_kw: number;
  energy_kwh: number;
  status: "on" | "off";
}

// ============================================================================
// Energy API
// ============================================================================

/**
 * Energy data API endpoints
 */
export const energyApi = {
  /**
   * Fetch sensor readings from InfluxDB
   * Endpoint: GET /analytics/readings
   */
  getReadings: (params: AnalyticsParams) =>
    api.get<ReadingsResponse>("/analytics/readings", { params }),

  /**
   * Fetch predictions from InfluxDB
   * Endpoint: GET /analytics/predictions
   */
  getPredictions: (params: AnalyticsParams) =>
    api.get<PredictionsResponse>("/analytics/predictions", { params }),

  /**
   * Run inference on a power window and persist the prediction
   * Endpoint: POST /infer
   */
  runInference: (request: InferRequest) =>
    api.post<InferResponse>("/infer", request),

  /**
   * List available ML models
   * Endpoint: GET /models
   */
  getModels: () => api.get<ModelsListResponse>("/models"),

  /**
   * Get available buildings (from Supabase via frontend logic)
   * Note: This should query Supabase directly, not via backend
   */
  getBuildings: () => {
    // TODO: Implement Supabase query for buildings
    // For now, return empty array to prevent breaking existing code
    return Promise.resolve([]);
  },

  /**
   * Get available appliances for a building (from Supabase via frontend logic)
   * Note: This should query Supabase directly, not via backend
   */
  getAppliances: (building?: string) => {
    // TODO: Implement Supabase query for appliances
    // For now, return empty array to prevent breaking existing code
    return Promise.resolve([]);
  },
};

/**
 * Check if the energy API is available
 */
export function isEnergyApiAvailable(): boolean {
  return isApiConfigured();
}

/**
 * Validate analytics query parameters
 */
export function validateAnalyticsParams(params: AnalyticsParams): void {
  if (!params.building_id || params.building_id.length > 64) {
    throw new Error("Invalid building_id: must be 1-64 characters");
  }

  if (params.appliance_id && params.appliance_id.length > 64) {
    throw new Error("Invalid appliance_id: must be 1-64 characters");
  }

  if (
    params.resolution &&
    !["1s", "1m", "15m"].includes(params.resolution)
  ) {
    throw new Error('Invalid resolution: must be "1s", "1m", or "15m"');
  }

  if (!params.start || !params.end) {
    throw new Error("start and end parameters are required");
  }
}

/**
 * Validate inference request
 */
export function validateInferRequest(request: InferRequest): void {
  if (!request.building_id || request.building_id.length > 64) {
    throw new Error("Invalid building_id: must be 1-64 characters");
  }

  if (!request.appliance_id || request.appliance_id.length > 64) {
    throw new Error("Invalid appliance_id: must be 1-64 characters");
  }

  if (!Array.isArray(request.window)) {
    throw new Error("window must be an array of numbers");
  }

  if (request.window.length < 1 || request.window.length > 10000) {
    throw new Error("window must contain 1-10000 elements");
  }

  if (request.window.some((v) => typeof v !== "number" || !isFinite(v))) {
    throw new Error("window must contain only finite numbers (no NaN/Inf)");
  }
}
