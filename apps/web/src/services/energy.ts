/**
 * Energy API Service
 * Typed interface for energy data endpoints.
 *
 * This service interfaces with the FastAPI backend for:
 * - Analytics: readings and predictions from InfluxDB
 * - Inference: running ML predictions
 * - Models: listing available ML models
 */

import { api, isApiConfigured, ApiError, ApiErrorType } from "./api";

// Re-export ApiError for consumers
export { ApiError, ApiErrorType };

// ============================================================================
// Analytics Types (readings and predictions from InfluxDB)
// ============================================================================

export interface AnalyticsParams {
  building_id: string;
  start: string; // ISO8601 or relative (e.g., "-7d", "2024-01-01T00:00:00Z")
  end: string; // ISO8601 or relative (e.g., "now()", "2024-01-08T00:00:00Z")
  appliance_id?: string;
  resolution?: "1s" | "1m" | "15m";
  include_disaggregation?: boolean;
}

export interface ReadingDataPoint {
  time: string; // ISO8601
  value: number;
  appliances?: Record<string, number>;
  confidence?: Record<string, number>;
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
  appliance_id?: string; // Optional for multi-head models
  window: number[]; // Array of power readings (e.g., 1024 floats for transformer)
  timestamp?: string; // ISO8601, optional
  model_id?: string; // Optional, defaults to active model
}

/**
 * Inference response from the backend.
 *
 * NOTE: Multi-head models return `predicted_kw` and `confidence` as objects
 * mapping appliance keys to values. Single-head models return scalar values.
 */
export interface InferResponse {
  // Multi-head: { "heatpump": 0.5, "dishwasher": 0.0 }
  // Single-head (legacy): number
  predicted_kw: Record<string, number> | number;
  confidence: Record<string, number> | number;
  model_version: string;
  request_id: string;
  persisted: boolean;
}

// ============================================================================
// Model Registry Types
// ============================================================================

/**
 * Output head configuration for multi-head models.
 * Each head predicts power for one appliance.
 */
export interface ModelHead {
  appliance_id: string;
  field_key: string;
}

/**
 * Model performance metrics (optional).
 */
export interface ModelMetrics {
  mae?: number | null;
  rmse?: number | null;
  f1_score?: number | null;
  accuracy?: number | null;
}

/**
 * Model metadata from the backend registry.
 *
 * NOTE: Modern NILM models are multi-head (one model predicts all appliances).
 * The `appliance_id` field is for backward compatibility with single-head models.
 * For multi-head models, use the `heads` array to get the list of appliances.
 */
export interface Model {
  model_id: string;
  model_version: string;
  appliance_id: string; // "multi" for multi-head models, specific appliance for single-head
  architecture: string;
  input_window_size: number;
  is_active: boolean;
  cached: boolean;
  // Multi-head support
  heads?: ModelHead[]; // List of appliance heads (empty for single-head models)
  metrics?: ModelMetrics | null; // Performance metrics (if available)
}


export interface ModelsListResponse {
  models: Model[];
  count: number;
}

/**
 * Get the list of appliances supported by a model.
 * For multi-head models, returns the heads. For single-head, returns [appliance_id].
 */
export function getModelAppliances(model: Model): string[] {
  if (model.heads && model.heads.length > 0) {
    return model.heads.map((h) => h.appliance_id);
  }
  // Single-head fallback
  return model.appliance_id !== "multi" ? [model.appliance_id] : [];
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

export interface ReportRequest {
  start: string;
  end: string;
  building_id: string;
}

export interface ReportResponse {
  url: string;
  generated_at: string;
}

export interface EnergyInsight {
  type: "anomaly" | "trend" | "saving_opportunity";
  description: string;
  confidence: number;
}

export interface EnergyDataResponse {
  readings: EnergyReading[];
  insights: EnergyInsight[];
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
    api.get<ReadingsResponse>("/analytics/readings", {
      params: {
        ...params,
        include_disaggregation: params.include_disaggregation ?? true
      }
    }),

  /**
   * Fetch predictions from InfluxDB
   * Endpoint: GET /analytics/predictions
   */
  getPredictions: (params: AnalyticsParams) =>
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    api.get<PredictionsResponse>("/analytics/predictions", { params: params as any }),

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
   * Get available buildings (dynamically discovered from InfluxDB)
   * Endpoint: GET /analytics/buildings
   */
  getBuildings: () => api.get<{ buildings: string[] }>("/analytics/buildings"),

  /**
   * Get available appliances for a building (dynamically discovered from InfluxDB)
   * Endpoint: GET /analytics/appliances
   */
  getAppliances: (building: string) =>
    api.get<{ appliances: string[] }>("/analytics/appliances", {
      params: { building_id: building },
    }),
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
