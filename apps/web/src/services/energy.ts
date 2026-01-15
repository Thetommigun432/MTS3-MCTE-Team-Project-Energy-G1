/**
 * Energy API Service
 * Typed interface for energy data endpoints.
 */

import { api, isApiConfigured } from "./api";

export interface EnergyReading {
  timestamp: string;
  building: string;
  appliance: string;
  power_kw: number;
  energy_kwh: number;
  status: "on" | "off";
}

export interface EnergyInsight {
  type: "consumption" | "anomaly" | "saving";
  title: string;
  description: string;
  value?: number;
  unit?: string;
}

export interface EnergyDataResponse {
  readings: EnergyReading[];
  buildings: string[];
  appliances: string[];
  dateRange: {
    min: string;
    max: string;
  };
}

export interface ReportRequest {
  building?: string;
  appliance?: string;
  startDate: string;
  endDate: string;
  format?: "json" | "pdf" | "csv";
}

export interface ReportResponse {
  id: string;
  status: "pending" | "completed" | "failed";
  downloadUrl?: string;
  data?: {
    totalConsumption: number;
    averagePower: number;
    peakPower: number;
    readings: EnergyReading[];
  };
}

/**
 * Energy data API endpoints
 */
export const energyApi = {
  /**
   * Fetch energy readings with optional filters
   */
  getReadings: (params?: {
    building?: string;
    appliance?: string;
    startDate?: string;
    endDate?: string;
    limit?: number;
  }) => api.get<EnergyDataResponse>("/api/energy/readings", { params }),

  /**
   * Fetch insights for the dashboard
   */
  getInsights: (params?: {
    building?: string;
    startDate?: string;
    endDate?: string;
  }) => api.get<EnergyInsight[]>("/api/energy/insights", { params }),

  /**
   * Generate a report
   */
  generateReport: (request: ReportRequest) =>
    api.post<ReportResponse>("/api/energy/reports", request),

  /**
   * Get report status and download URL
   */
  getReport: (reportId: string) =>
    api.get<ReportResponse>(`/api/energy/reports/${reportId}`),

  /**
   * Get available buildings
   */
  getBuildings: () => api.get<string[]>("/api/energy/buildings"),

  /**
   * Get available appliances for a building
   */
  getAppliances: (building?: string) =>
    api.get<string[]>("/api/energy/appliances", {
      params: building ? { building } : undefined,
    }),
};

/**
 * Check if the energy API is available
 */
export function isEnergyApiAvailable(): boolean {
  return isApiConfigured();
}
