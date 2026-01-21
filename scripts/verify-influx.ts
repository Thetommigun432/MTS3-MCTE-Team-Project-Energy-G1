import { InfluxDB } from '@influxdata/influxdb-client';
import dotenv from 'dotenv';

// Load environment variables from .env.local
dotenv.config({ path: '.env.local' });

// Validate required environment variables
const requiredVars = ['INFLUX_TOKEN', 'INFLUX_ORG', 'INFLUX_BUCKET'];
for (const varName of requiredVars) {
  if (!process.env[varName]) {
    console.error(`❌ ERROR: ${varName} not set in .env.local`);
    process.exit(1);
  }
}

const influxDB = new InfluxDB({
  url: 'http://localhost:8086',
  token: process.env.INFLUX_TOKEN!,
});

const queryApi = influxDB.getQueryApi(process.env.INFLUX_ORG!);

async function verifyData() {
  console.log('========================================');
  console.log('  InfluxDB Data Verification');
  console.log('========================================');
  console.log('');
  console.log('Configuration:');
  console.log(`  InfluxDB URL: http://localhost:8086`);
  console.log(`  Organization: ${process.env.INFLUX_ORG}`);
  console.log(`  Bucket: ${process.env.INFLUX_BUCKET}`);
  console.log('');

  try {
    // Query 1: Count total points per appliance
    console.log('🔍 Query 1: Counting predictions per appliance...');
    const countQuery = `
      from(bucket: "${process.env.INFLUX_BUCKET}")
        |> range(start: -30d)
        |> filter(fn: (r) => r._measurement == "appliance_prediction")
        |> filter(fn: (r) => r._field == "predicted_kw")
        |> group(columns: ["appliance_name"])
        |> count()
    `;

    const countRows: any[] = [];
    await new Promise<void>((resolve, reject) => {
      queryApi.queryRows(countQuery, {
        next(row: string[], tableMeta: any) {
          countRows.push(tableMeta.toObject(row));
        },
        error(error: Error) {
          reject(error);
        },
        complete() {
          resolve();
        },
      });
    });

    console.log(`\n✅ Found ${countRows.length} appliances with predictions:\n`);

    let totalPoints = 0;
    countRows.forEach(row => {
      const count = row._value || 0;
      totalPoints += count;
      console.log(`  ${row.appliance_name.padEnd(25)} ${count.toLocaleString().padStart(10)} points`);
    });

    console.log(`  ${'─'.repeat(25)} ${'─'.repeat(10)}`);
    console.log(`  ${'TOTAL'.padEnd(25)} ${totalPoints.toLocaleString().padStart(10)} points`);

    // Query 2: Get time range
    console.log('\n🔍 Query 2: Checking time range...');
    const rangeQuery = `
      from(bucket: "${process.env.INFLUX_BUCKET}")
        |> range(start: -30d)
        |> filter(fn: (r) => r._measurement == "appliance_prediction")
        |> filter(fn: (r) => r._field == "predicted_kw")
        |> keep(columns: ["_time"])
        |> sort(columns: ["_time"])
    `;

    const rangeRows: any[] = [];
    await new Promise<void>((resolve, reject) => {
      queryApi.queryRows(rangeQuery, {
        next(row: string[], tableMeta: any) {
          rangeRows.push(tableMeta.toObject(row));
        },
        error(error: Error) {
          reject(error);
        },
        complete() {
          resolve();
        },
      });
    });

    if (rangeRows.length > 0) {
      const firstTime = new Date(rangeRows[0]._time);
      const lastTime = new Date(rangeRows[rangeRows.length - 1]._time);
      const duration = (lastTime.getTime() - firstTime.getTime()) / (1000 * 60 * 60 * 24);

      console.log(`\n✅ Time range:`);
      console.log(`  First: ${firstTime.toISOString()}`);
      console.log(`  Last:  ${lastTime.toISOString()}`);
      console.log(`  Duration: ${duration.toFixed(1)} days`);
    }

    // Query 3: Sample recent predictions
    console.log('\n🔍 Query 3: Sampling recent predictions...');
    const sampleQuery = `
      from(bucket: "${process.env.INFLUX_BUCKET}")
        |> range(start: -7d)
        |> filter(fn: (r) => r._measurement == "appliance_prediction")
        |> filter(fn: (r) => r.appliance_name == "HeatPump")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> limit(n: 5)
    `;

    const sampleRows: any[] = [];
    await new Promise<void>((resolve, reject) => {
      queryApi.queryRows(sampleQuery, {
        next(row: string[], tableMeta: any) {
          sampleRows.push(tableMeta.toObject(row));
        },
        error(error: Error) {
          reject(error);
        },
        complete() {
          resolve();
        },
      });
    });

    console.log(`\n✅ Sample predictions (HeatPump):\n`);
    sampleRows.forEach(row => {
      const time = new Date(row._time).toISOString();
      const kw = row.predicted_kw?.toFixed(3) || 'N/A';
      const conf = row.confidence ? (row.confidence * 100).toFixed(1) : 'N/A';
      console.log(`  ${time} | ${kw.padStart(7)} kW | ${conf.padStart(5)}% confidence`);
    });

    console.log('');
    console.log('========================================');
    console.log('  ✅ VERIFICATION COMPLETE');
    console.log('========================================');
    console.log('');
    console.log('Next steps:');
    console.log('  • View in UI: http://localhost:8086');
    console.log('  • Start local dev: npm run local:dev');
    console.log('');

  } catch (error: any) {
    console.error('');
    console.error('========================================');
    console.error('  ❌ ERROR');
    console.error('========================================');
    console.error('');
    console.error('Query error:', error.message);
    console.error('');

    if (error.message.includes('ECONNREFUSED')) {
      console.error('ℹ️  InfluxDB may not be running. Start it with:');
      console.error('   docker compose up -d');
    } else if (error.message.includes('unauthorized') || error.message.includes('bucket not found')) {
      console.error('ℹ️  Database may not be seeded yet. Run:');
      console.error('   npm run predictions:seed');
    }

    console.error('');
    process.exit(1);
  }
}

// Run verification
verifyData();
