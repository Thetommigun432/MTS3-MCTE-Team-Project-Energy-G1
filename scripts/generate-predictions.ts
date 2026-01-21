import fs from 'fs';
import Papa from 'papaparse';
import path from 'path';
import axios from 'axios';

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
  model_version?: string;
  inference_type?: 'ml' | 'mock';
}

interface InferenceRequest {
  appliance_id: string;
  aggregate_data: number[];
  model_version?: string;
}

interface InferenceResponse {
  predicted_kw: number;
  confidence: number;
  model_version: string;
}

// Backend API configuration
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';
const SEQUENCE_LENGTH = 60;
const USE_ML_INFERENCE = process.env.USE_ML_INFERENCE !== 'false'; // Default to true

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
 * Call backend inference endpoint for a single prediction
 */
async function callInferenceService(request: InferenceRequest): Promise<InferenceResponse> {
  try {
    const response = await axios.post(`${BACKEND_URL}/infer`, request, {
      timeout: 5000, // 5 second timeout
    });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      if (error.code === 'ECONNREFUSED') {
        throw new Error('Backend is not running. Start it with: docker compose up -d');
      }
      throw new Error(`Backend inference error: ${error.message}`);
    }
    throw error;
  }
}

/**
 * Get available models from backend
 */
async function getAvailableModels(): Promise<string[]> {
  try {
    const response = await axios.get(`${BACKEND_URL}/models`, {
      timeout: 3000,
    });
    return response.data.models || [];
  } catch (error) {
    console.warn('Failed to fetch available models from backend');
    return [];
  }
}

/**
 * Generate ML-based NILM predictions using backend API
 *
 * @param csvPath Path to the CSV file with aggregate power data
 * @returns Array of predictions for each appliance and timestamp
 */
export async function generatePredictionsML(csvPath: string): Promise<Prediction[]> {
  console.log(`📊 Reading CSV from: ${csvPath}`);

  const fileContent = fs.readFileSync(csvPath, 'utf-8');
  const parsed = Papa.parse<CsvRow>(fileContent, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });

  if (parsed.errors.length > 0) {
    console.warn('⚠️  CSV parsing warnings:', parsed.errors.slice(0, 5));
  }

  console.log(`✅ Parsed ${parsed.data.length} rows from CSV`);

  // Get available models
  console.log(`🔍 Fetching available models from ${BACKEND_URL}...`);
  const availableAppliances = await getAvailableModels();

  if (availableAppliances.length === 0) {
    throw new Error('No models available in backend');
  }

  console.log(`✅ Found ${availableAppliances.length} model(s): ${availableAppliances.join(', ')}\n`);

  const predictions: Prediction[] = [];

  for (const appliance of availableAppliances) {
    console.log(`  Processing ${appliance}...`);
    let successCount = 0;
    let failCount = 0;

    // Process each timestamp with sliding window
    // Skip first SEQUENCE_LENGTH timestamps (need history for window)
    for (let i = SEQUENCE_LENGTH; i < parsed.data.length; i++) {
      try {
        const row = parsed.data[i];
        const timestamp = new Date(row.Time);

        if (isNaN(timestamp.getTime())) {
          failCount++;
          continue;
        }

        // Extract 60 previous aggregate power values (sliding window)
        const aggregateWindow: number[] = [];
        for (let j = i - SEQUENCE_LENGTH; j < i; j++) {
          aggregateWindow.push(Number(parsed.data[j].Aggregate) || 0);
        }

        // Call backend inference endpoint
        const result = await callInferenceService({
          appliance_id: appliance,
          aggregate_data: aggregateWindow,
        });

        predictions.push({
          timestamp,
          appliance,
          predicted_kw: Math.round(result.predicted_kw * 1000) / 1000, // 3 decimal places
          confidence: Math.round(result.confidence * 100) / 100, // 2 decimal places
          building_id: 'local',
          model_version: result.model_version,
          inference_type: 'ml',
        });

        successCount++;

        // Progress reporting every 1000 points
        if (successCount % 1000 === 0) {
          console.log(`    ✓ ${successCount}/${parsed.data.length - SEQUENCE_LENGTH} predictions`);
        }
      } catch (error) {
        failCount++;
        if (failCount <= 3) { // Only log first few errors
          console.warn(`    ⚠️  Inference failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
      }
    }

    console.log(`  ✅ ${appliance}: ${successCount} predictions generated${failCount > 0 ? ` (${failCount} failed)` : ''}`);
  }

  console.log(`\n✅ Total predictions: ${predictions.length}`);

  return predictions;
}

/**
 * Generate deterministic NILM predictions from CSV aggregate data (MOCK/FALLBACK)
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
          inference_type: 'mock',
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

  (async () => {
    try {
      let predictions: Prediction[];

      if (USE_ML_INFERENCE) {
        console.log('🤖 Using ML inference mode\n');
        predictions = await generatePredictionsML(csvPath);
      } else {
        console.log('🎲 Using mock (deterministic) mode\n');
        predictions = generatePredictions(csvPath);
      }

      console.log('\n📊 Sample predictions:');
      predictions.slice(0, 5).forEach(p => {
        const inferenceLabel = p.inference_type === 'ml' ? '🤖 ML' : '🎲 Mock';
        const versionLabel = p.model_version ? ` (${p.model_version})` : '';
        console.log(`  ${inferenceLabel}${versionLabel} | ${p.timestamp.toISOString()} | ${p.appliance.padEnd(20)} | ${p.predicted_kw.toFixed(3)} kW | ${(p.confidence * 100).toFixed(1)}%`);
      });
    } catch (error) {
      console.error('\n❌ ERROR:', error instanceof Error ? error.message : 'Unknown error');
      console.error('\n💡 TIP: Make sure the backend is running:');
      console.error('   docker compose up -d');
      console.error('\n💡 Or run in mock mode:');
      console.error('   USE_ML_INFERENCE=false npm run predictions:seed');
      process.exit(1);
    }
  })();
}
