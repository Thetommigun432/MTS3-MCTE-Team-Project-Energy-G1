import { InfluxDB, Point } from '@influxdata/influxdb-client';
import dotenv from 'dotenv';
import { generatePredictions } from './generate-predictions.js';
import path from 'path';

// Load environment variables from .env.local
dotenv.config({ path: '.env.local' });

// Validate required environment variables
const requiredVars = ['INFLUX_TOKEN', 'INFLUX_ORG', 'INFLUX_BUCKET'];
for (const varName of requiredVars) {
  if (!process.env[varName]) {
    console.error(`❌ ERROR: ${varName} not set in .env.local`);
    console.error('Please copy .env.local.example to .env.local and configure it.');
    process.exit(1);
  }
}

const influxDB = new InfluxDB({
  url: 'http://localhost:8086',
  token: process.env.INFLUX_TOKEN!,
});

const writeApi = influxDB.getWriteApi(
  process.env.INFLUX_ORG!,
  process.env.INFLUX_BUCKET!,
  'ms' // millisecond precision for timestamps
);

// Configure batching for better performance
writeApi.useDefaultTags({ source: 'csv_generator' });

async function writePredictions() {
  try {
    console.log('========================================');
    console.log('  NILM Prediction Seeder');
    console.log('========================================');
    console.log('');
    console.log('Configuration:');
    console.log(`  InfluxDB URL: http://localhost:8086`);
    console.log(`  Organization: ${process.env.INFLUX_ORG}`);
    console.log(`  Bucket: ${process.env.INFLUX_BUCKET}`);
    console.log('');

    // Generate predictions from CSV
    console.log('📊 Step 1: Generating predictions from CSV...');
    const csvPath = path.join(process.cwd(), 'frontend/public/data/nilm_ready_dataset.csv');
    const predictions = generatePredictions(csvPath);

    console.log(`✅ Generated ${predictions.length} predictions`);
    console.log('');

    // Write to InfluxDB
    console.log('📝 Step 2: Writing predictions to InfluxDB...');
    console.log('   (This may take 30-60 seconds for large datasets)');
    console.log('');

    let written = 0;
    const startTime = Date.now();
    const totalPredictions = predictions.length;

    for (const pred of predictions) {
      // Create InfluxDB point
      const point = new Point('appliance_prediction')
        .tag('building_id', pred.building_id)
        .tag('appliance_name', pred.appliance)
        .floatField('predicted_kw', pred.predicted_kw)
        .floatField('confidence', pred.confidence)
        .timestamp(pred.timestamp);

      writeApi.writePoint(point);
      written++;

      // Progress indicator every 1000 points
      if (written % 1000 === 0) {
        const percent = ((written / totalPredictions) * 100).toFixed(1);
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        console.log(`  [${percent}%] Written ${written}/${totalPredictions} points (${elapsed}s elapsed)`);
      }
    }

    // Flush remaining points
    console.log('');
    console.log('⏳ Flushing remaining points to InfluxDB...');
    await writeApi.close();

    const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);

    console.log('');
    console.log('========================================');
    console.log('  ✅ SUCCESS');
    console.log('========================================');
    console.log('');
    console.log(`📊 Total predictions written: ${written}`);
    console.log(`⏱️  Total time: ${totalTime}s`);
    console.log(`📈 Write rate: ${(written / parseFloat(totalTime)).toFixed(0)} points/sec`);
    console.log('');
    console.log('Next steps:');
    console.log('  1. View data in InfluxDB UI: http://localhost:8086');
    console.log('  2. Verify data: npm run predictions:verify');
    console.log('  3. Start local dev: npm run local:dev');
    console.log('');

  } catch (error: any) {
    console.error('');
    console.error('========================================');
    console.error('  ❌ ERROR');
    console.error('========================================');
    console.error('');
    console.error('Error writing predictions:', error.message);
    console.error('');

    if (error.message.includes('ECONNREFUSED')) {
      console.error('ℹ️  InfluxDB may not be running. Start it with:');
      console.error('   docker compose up -d');
    } else if (error.message.includes('unauthorized')) {
      console.error('ℹ️  Check your INFLUX_TOKEN in .env.local');
      console.error('   It should match the token in docker-compose.yml');
    } else {
      console.error('Stack trace:', error.stack);
    }

    console.error('');
    process.exit(1);
  }
}

// Run the seeder
writePredictions();
