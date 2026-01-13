import express from 'express';
import cors from 'cors';
import { InfluxDB } from '@influxdata/influxdb-client';
import dotenv from 'dotenv';

// Load environment variables from .env.local
dotenv.config({ path: '.env.local' });

const app = express();
const PORT = process.env.LOCAL_API_PORT || 3001;

// InfluxDB client configuration
const influxDB = new InfluxDB({
  url: 'http://localhost:8086',
  token: process.env.INFLUX_TOKEN!,
});

const queryApi = influxDB.getQueryApi(process.env.INFLUX_ORG!);

// Middleware
app.use(cors({ origin: 'http://localhost:8080' }));
app.use(express.json());

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    influxUrl: 'http://localhost:8086',
    influxOrg: process.env.INFLUX_ORG,
    influxBucket: process.env.INFLUX_BUCKET,
  });
});

// Get predictions for a building/time range
app.get('/api/local/predictions', async (req, res) => {
  try {
    const { buildingId = 'local', start, end } = req.query;

    const query = `
      from(bucket: "${process.env.INFLUX_BUCKET}")
        |> range(start: ${start || '-7d'}, stop: ${end || 'now()'})
        |> filter(fn: (r) => r._measurement == "appliance_prediction")
        |> filter(fn: (r) => r.building_id == "${buildingId}")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    `;

    const rows: any[] = [];

    queryApi.queryRows(query, {
      next(row: string[], tableMeta: any) {
        const record = tableMeta.toObject(row);
        rows.push(record);
      },
      error(error: Error) {
        console.error('InfluxDB query error:', error);
        res.status(500).json({
          success: false,
          error: error.message,
          query: query.trim(),
        });
      },
      complete() {
        console.log(`Query returned ${rows.length} rows for building: ${buildingId}`);
        res.json({
          success: true,
          data: rows,
          count: rows.length,
          buildingId,
          timeRange: { start: start || '-7d', end: end || 'now()' }
        });
      },
    });
  } catch (error: any) {
    console.error('Server error:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      stack: process.env.NODE_ENV === 'development' ? error.stack : undefined,
    });
  }
});

// Start server
app.listen(PORT, () => {
  console.log('');
  console.log('========================================');
  console.log('  Local API Server for NILM Monitor');
  console.log('========================================');
  console.log('');
  console.log(`🚀 Server running on http://localhost:${PORT}`);
  console.log(`📊 InfluxDB UI: http://localhost:8086`);
  console.log(`🔍 Health check: http://localhost:${PORT}/health`);
  console.log('');
  console.log('Available endpoints:');
  console.log(`  GET /health`);
  console.log(`  GET /api/local/predictions?buildingId=local&start=-7d&end=now()`);
  console.log('');
  console.log('Configuration:');
  console.log(`  InfluxDB Org: ${process.env.INFLUX_ORG}`);
  console.log(`  InfluxDB Bucket: ${process.env.INFLUX_BUCKET}`);
  console.log('');
});
