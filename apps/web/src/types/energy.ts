export interface ApplianceData {
  est_kW: number;
  on: boolean;
  confidence: number;
}

export interface SeriesEntry {
  ts: string;
  building: string;
  total_kW: number;
  appliances: Record<string, ApplianceData>;
}

export interface DateRange {
  start: Date;
  end: Date;
}

export type DataMode = "demo" | "api";

export interface Building {
  id: string;
  name: string;
  address?: string;
  status?: 'active' | 'inactive' | 'maintenance';
}

export interface ApplianceStatus {
  name: string;
  on: boolean;
  confidence: number;
  est_kW: number;
  rated_kW?: number | null; // From managed appliances
  type?: string; // Appliance type from registration
  building_name?: string; // Which building this appliance belongs to
}

export interface InsightData {
  peakLoad: { kW: number; timestamp: string };
  totalEnergy: number;
  topAppliance: { name: string; confidence: number };
  overallConfidence: { level: "Good" | "Medium" | "Low"; percentage: number };
}
