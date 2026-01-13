import fs from 'fs';
import Papa from 'papaparse';
import path from 'path';

interface CsvRow {
  Time: string;
  Aggregate: number;
  [appliance: string]: string | number;
}

export interface Prediction {
  timestamp: Date;
  appliance: string;
  predicted_kw: number;
  confidence: number;
  building_id: string;
}

// Deterministic weights for each appliance (based on typical consumption patterns)
// These weights roughly sum to 1.0 and represent the proportion of aggregate power
const APPLIANCE_WEIGHTS: Record<string, number> = {
  RangeHood: 0.05,
  Dryer: 0.15,
  Stove: 0.20,
  Dishwasher: 0.10,
  HeatPump: 0.25,
  Washer: 0.08,
  Fridge: 0.07,
  Microwave: 0.04,
  AirConditioner: 0.03,
  ElectricWaterHeater: 0.02,
  Lighting: 0.01,
};

/**
 * Generate deterministic NILM predictions from CSV aggregate data
 * This simulates ML model predictions using a deterministic algorithm
 *
 * @param csvPath Path to the CSV file with aggregate power data
 * @returns Array of predictions for each appliance and timestamp
 */
export function generatePredictions(csvPath: string): Prediction[] {
  console.log(`Reading CSV from: ${csvPath}`);

  const fileContent = fs.readFileSync(csvPath, 'utf-8');
  const parsed = Papa.parse<CsvRow>(fileContent, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });

  if (parsed.errors.length > 0) {
    console.warn('CSV parsing warnings:', parsed.errors.slice(0, 5));
  }

  console.log(`Parsed ${parsed.data.length} rows from CSV`);

  const predictions: Prediction[] = [];

  parsed.data.forEach((row, idx) => {
    try {
      const timestamp = new Date(row.Time);
      if (isNaN(timestamp.getTime())) {
        console.warn(`Invalid timestamp at row ${idx}: ${row.Time}`);
        return;
      }

      const aggregate = Number(row.Aggregate) || 0;

      // Generate prediction for each appliance
      Object.entries(APPLIANCE_WEIGHTS).forEach(([appliance, weight]) => {
        // Time-based features for deterministic variation
        const hourOfDay = timestamp.getHours();
        const dayOfWeek = timestamp.getDay();
        const minuteOfDay = timestamp.getHours() * 60 + timestamp.getMinutes();

        // Time-based multiplier (higher during typical usage hours)
        // Uses sinusoidal pattern to simulate daily usage cycles
        const dailyCycle = 1 + 0.3 * Math.sin((hourOfDay / 24) * 2 * Math.PI);
        const weeklyCycle = 1 + 0.1 * Math.sin((dayOfWeek / 7) * 2 * Math.PI);

        // Base prediction from aggregate power and appliance weight
        let predicted = aggregate * weight * dailyCycle * weeklyCycle;

        // Add deterministic noise based on timestamp and appliance name
        // This makes predictions more realistic by adding variation
        const seed = idx + appliance.length + minuteOfDay;
        const noise1 = Math.sin(seed) * 0.1 * aggregate;
        const noise2 = Math.cos(seed * 2) * 0.05 * aggregate;
        predicted = Math.max(0, predicted + noise1 + noise2);

        // Confidence score: higher for stable patterns, lower for edge cases
        // High aggregate with consistent patterns = high confidence
        // Low aggregate or unusual patterns = lower confidence
        const baseConfidence = 0.7;
        const aggregateFactor = Math.min(0.2, aggregate / 10); // More confidence with higher power
        const consistencyFactor = 0.05 * (1 - Math.abs(noise1 + noise2) / Math.max(aggregate, 0.1));
        const confidence = Math.min(0.95, Math.max(0.50, baseConfidence + aggregateFactor + consistencyFactor));

        predictions.push({
          timestamp,
          appliance,
          predicted_kw: Math.round(predicted * 1000) / 1000, // 3 decimal places
          confidence: Math.round(confidence * 100) / 100, // 2 decimal places
          building_id: 'local',
        });
      });
    } catch (error) {
      console.warn(`Error processing row ${idx}:`, error);
    }
  });

  console.log(`Generated ${predictions.length} predictions (${predictions.length / Object.keys(APPLIANCE_WEIGHTS).length} timestamps)`);

  return predictions;
}

// Allow running directly for testing
if (import.meta.url === `file://${process.argv[1]}`) {
  const csvPath = process.argv[2] || path.join(process.cwd(), 'frontend/public/data/nilm_ready_dataset.csv');
  const predictions = generatePredictions(csvPath);

  console.log('\nSample predictions:');
  predictions.slice(0, 5).forEach(p => {
    console.log(`  ${p.timestamp.toISOString()} | ${p.appliance.padEnd(20)} | ${p.predicted_kw.toFixed(3)} kW | ${(p.confidence * 100).toFixed(1)}% confidence`);
  });
}
