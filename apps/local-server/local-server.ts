import express from 'express';
import cors from 'cors';
import { InfluxDB } from '@influxdata/influxdb-client';
import dotenv from 'dotenv';
import axios from 'axios';

// Load environment variables from .env.local
dotenv.config({ path: '.env.local' });

// =============================================================================
// Environment Validation
// =============================================================================

/**
 * Validate required environment variables at startup
 */
function validateEnv(): void {
  const required = ['INFLUX_TOKEN', 'INFLUX_ORG', 'INFLUX_BUCKET'];
  const missing = required.filter((key) => !process.env[key]);

  if (missing.length > 0) {
    console.error('');
    console.error('========================================');
    console.error('  CONFIGURATION ERROR');
    console.error('========================================');
    console.error('');
    console.error(`Missing required environment variables: ${missing.join(', ')}`);
    console.error('');
    console.error('Please ensure you have a .env.local file with the following:');
    missing.forEach((key) => {
      console.error(`  ${key}=<your-value>`);
    });
    console.error('');
    console.error('You can copy the example file:');
    console.error('  cp .env.local.example .env.local');
    console.error('');
    process.exit(1);
  }
}

// Validate environment before proceeding
validateEnv();

// =============================================================================
// Configuration (validated above, safe to use)
// =============================================================================

const PORT = process.env.LOCAL_API_PORT || 3001;
const INFERENCE_SERVICE_URL = process.env.INFERENCE_SERVICE_URL || 'http://localhost:8000';
const INFLUX_URL = process.env.INFLUX_URL || 'http://localhost:8086';
const CORS_ORIGIN = process.env.CORS_ORIGIN || 'http://localhost:8080';
const INFLUX_TOKEN = process.env.INFLUX_TOKEN!;
const INFLUX_ORG = process.env.INFLUX_ORG!;
const INFLUX_BUCKET = process.env.INFLUX_BUCKET!;

// =============================================================================
// Input Validation
// =============================================================================

/**
 * Validate and sanitize query parameters to prevent Flux injection
 */
function validateQueryParam(
  param: unknown,
  pattern: RegExp,
  defaultVal: string,
  paramName: string
): string {
  if (param === undefined || param === null || param === '') {
    return defaultVal;
  }

  // Ensure it's a string (not array or object)
  if (typeof param !== 'string') {
    throw new Error(`Invalid ${paramName}: expected string, got ${typeof param}`);
  }

  // Validate against pattern
  if (!pattern.test(param)) {
    throw new Error(
      `Invalid ${paramName} format: '${param}'. Expected format: ${pattern.toString()}`
    );
  }

  return param;
}

// Validation patterns
const PATTERNS = {
  // Building ID: alphanumeric, dash, underscore (max 64 chars)
  buildingId: /^[a-zA-Z0-9_-]{1,64}$/,
  // Flux time format: -Nd, -Nh, -Nm, -Ns, -Nw, or now()
  fluxTime: /^(-?\d+[smhdw]|now\(\))$/,
  // ISO 8601 datetime
  isoDateTime: /^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}(\.\d{3})?(Z|[+-]\d{2}:\d{2})?)?$/,
};

/**
 * Validate Flux time parameter (supports relative and ISO formats)
 */
function validateFluxTime(param: unknown, defaultVal: string, paramName: string): string {
  if (param === undefined || param === null || param === '') {
    return defaultVal;
  }

  if (typeof param !== 'string') {
    throw new Error(`Invalid ${paramName}: expected string`);
  }

  // Check relative time format first (-7d, -1h, etc.)
  if (PATTERNS.fluxTime.test(param)) {
    return param;
  }

  // Check ISO datetime format
  if (PATTERNS.isoDateTime.test(param)) {
    return param;
  }

  throw new Error(
    `Invalid ${paramName} format: '${param}'. Expected relative time (-7d, -1h) or ISO datetime`
  );
}

// =============================================================================
// InfluxDB Client
// =============================================================================

const influxDB = new InfluxDB({
  url: INFLUX_URL,
  token: INFLUX_TOKEN,
});

const queryApi = influxDB.getQueryApi(INFLUX_ORG);

// =============================================================================
// Express App
// =============================================================================

const app = express();

// Middleware
app.use(cors({ origin: CORS_ORIGIN }));
app.use(express.json());

// =============================================================================
// Endpoints
// =============================================================================

/**
 * Health check endpoint
 */
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    influxOrg: INFLUX_ORG,
    influxBucket: INFLUX_BUCKET,
  });
});

/**
 * Get predictions for a building/time range
 */
app.get('/api/local/predictions', async (req, res) => {
  try {
    // Validate and sanitize inputs
    const buildingId = validateQueryParam(
      req.query.buildingId,
      PATTERNS.buildingId,
      'local',
      'buildingId'
    );
    const start = validateFluxTime(req.query.start, '-7d', 'start');
    const end = validateFluxTime(req.query.end, 'now()', 'end');

    // Build query with validated parameters
    const query = `
      from(bucket: "${INFLUX_BUCKET}")
        |> range(start: ${start}, stop: ${end})
        |> filter(fn: (r) => r._measurement == "appliance_prediction")
        |> filter(fn: (r) => r.building_id == "${buildingId}")
        |> pivot(rowKey:["_time", "appliance_name", "building_id", "inference_type", "model_version"], columnKey: ["_field"], valueColumn: "_value")
    `;

    const rows: any[] = [];

    queryApi.queryRows(query, {
      next(row: string[], tableMeta: any) {
        const record = tableMeta.toObject(row);
        rows.push(record);
      },
      error(error: Error) {
        console.error('InfluxDB query error:', {
          message: error.message,
          buildingId,
          start,
          end,
        });
        // Don't expose query details to client
        res.status(500).json({
          success: false,
          error: 'Database query failed',
        });
      },
      complete() {
        console.log(`Query returned ${rows.length} rows for building: ${buildingId}`);
        res.json({
          success: true,
          data: rows,
          count: rows.length,
          buildingId,
          timeRange: { start, end },
        });
      },
    });
  } catch (error: any) {
    // Handle validation errors
    if (error.message.includes('Invalid')) {
      return res.status(400).json({
        success: false,
        error: error.message,
      });
    }

    console.error('Server error:', error.message);
    res.status(500).json({
      success: false,
      error: 'Internal server error',
    });
  }
});

/**
 * Proxy inference requests to inference service
 */
app.post('/api/local/infer', async (req, res) => {
  try {
    const response = await axios.post(`${INFERENCE_SERVICE_URL}/infer`, req.body, {
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    res.json({
      success: true,
      ...response.data,
    });
  } catch (error: any) {
    console.error('Inference service error:', error.message);

    if (axios.isAxiosError(error)) {
      if (error.code === 'ECONNREFUSED') {
        return res.status(503).json({
          success: false,
          error: 'Inference service is not running',
          hint: 'Make sure Docker Compose is running: docker compose up -d',
        });
      }

      if (error.response) {
        return res.status(error.response.status).json({
          success: false,
          error: error.response.data?.detail || 'Inference request failed',
        });
      }
    }

    res.status(500).json({
      success: false,
      error: 'Inference service error',
    });
  }
});

/**
 * List available models
 */
app.get('/api/local/models', async (req, res) => {
  try {
    const response = await axios.get(`${INFERENCE_SERVICE_URL}/models`, {
      timeout: 5000,
    });

    res.json({
      success: true,
      ...response.data,
    });
  } catch (error: any) {
    console.error('Inference service error:', error.message);

    if (axios.isAxiosError(error) && error.code === 'ECONNREFUSED') {
      return res.status(503).json({
        success: false,
        error: 'Inference service is not running',
      });
    }

    res.status(500).json({
      success: false,
      error: 'Failed to fetch models',
    });
  }
});

/**
 * Check inference service health
 */
app.get('/api/local/inference-health', async (req, res) => {
  try {
    const response = await axios.get(`${INFERENCE_SERVICE_URL}/health`, {
      timeout: 3000,
    });

    res.json({
      success: true,
      inferenceService: response.data,
    });
  } catch (error: any) {
    res.status(503).json({
      success: false,
      error: 'Inference service unavailable',
    });
  }
});

// =============================================================================
// Start Server
// =============================================================================

app.listen(PORT, () => {
  console.log('');
  console.log('========================================');
  console.log('  Local API Server for NILM Monitor');
  console.log('========================================');
  console.log('');
  console.log(`Server running on http://localhost:${PORT}`);
  console.log(`InfluxDB: ${INFLUX_URL}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log('');
  console.log('Available endpoints:');
  console.log(`  GET /health`);
  console.log(`  GET /api/local/predictions?buildingId=local&start=-7d&end=now()`);
  console.log(`  POST /api/local/infer`);
  console.log(`  GET /api/local/models`);
  console.log(`  GET /api/local/inference-health`);
  console.log('');
  console.log('Configuration:');
  console.log(`  InfluxDB Org: ${INFLUX_ORG}`);
  console.log(`  InfluxDB Bucket: ${INFLUX_BUCKET}`);
  console.log(`  Inference Service: ${INFERENCE_SERVICE_URL}`);
  console.log(`  CORS Origin: ${CORS_ORIGIN}`);
  console.log('');
});
