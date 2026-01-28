import { describe, it, expect } from 'vitest';
import {
  computeConfidence,
  computeEnergyKwh,
  computeTotalEnergy,
  getTopAppliancesByEnergy,
  ON_THRESHOLD,
  MIN_ON_THRESHOLD,
  ON_RATIO,
  computeOnThreshold,
  isApplianceOn,
  NilmDataRow,
} from './useNilmCsvData';

describe('NILM Data Utilities', () => {
  describe('ON_THRESHOLD constants', () => {
    it('should have correct default thresholds', () => {
      expect(ON_THRESHOLD).toBe(0.05);        // 50W legacy fallback
      expect(MIN_ON_THRESHOLD).toBe(0.02);    // 20W minimum
      expect(ON_RATIO).toBe(0.05);            // 5% of rated power
    });
  });

  describe('computeOnThreshold', () => {
    it('should return legacy threshold when rated power is null/undefined', () => {
      expect(computeOnThreshold(null)).toBe(ON_THRESHOLD);
      expect(computeOnThreshold(undefined)).toBe(ON_THRESHOLD);
    });

    it('should return minimum threshold for small appliances', () => {
      // RangeHood: 0.5 kW rated => 0.5 * 0.05 = 0.025 > 0.02 => 0.025
      expect(computeOnThreshold(0.5)).toBe(0.025);
    });

    it('should scale with rated power for large appliances', () => {
      // HeatPump: 5.0 kW rated => 5.0 * 0.05 = 0.25 kW (250W)
      expect(computeOnThreshold(5.0)).toBe(0.25);
      // EVCharger: 7.5 kW rated => 7.5 * 0.05 = 0.375 kW (375W)
      expect(computeOnThreshold(7.5)).toBe(0.375);
    });

    it('should enforce minimum threshold for very small rated power', () => {
      // 0.1 kW rated => 0.1 * 0.05 = 0.005 < 0.02 => 0.02 (minimum)
      expect(computeOnThreshold(0.1)).toBe(MIN_ON_THRESHOLD);
    });
  });

  describe('isApplianceOn', () => {
    it('should use legacy threshold when rated power unknown', () => {
      expect(isApplianceOn(0.04, null)).toBe(false); // 40W < 50W
      expect(isApplianceOn(0.05, null)).toBe(true);  // 50W >= 50W
    });

    it('should use dynamic threshold based on rated power', () => {
      // HeatPump: 5kW rated, threshold = 250W
      expect(isApplianceOn(0.2, 5.0)).toBe(false);  // 200W < 250W
      expect(isApplianceOn(0.25, 5.0)).toBe(true);  // 250W >= 250W
    });

    it('should correctly detect small appliances ON', () => {
      // RangeHood: 0.5kW rated, threshold = 25W
      expect(isApplianceOn(0.02, 0.5)).toBe(false); // 20W < 25W
      expect(isApplianceOn(0.03, 0.5)).toBe(true);  // 30W >= 25W
    });
  });

  describe('computeConfidence', () => {
    it('should return high confidence (~0.95) for zero power (clearly OFF)', () => {
      const conf = computeConfidence(0);
      expect(conf).toBeGreaterThanOrEqual(0.90);
      expect(conf).toBeLessThanOrEqual(0.96);  // Allow for floating point precision
    });

    it('should return high confidence for high power (clearly ON)', () => {
      const conf = computeConfidence(1.0);
      expect(conf).toBeGreaterThanOrEqual(0.85);
      expect(conf).toBeLessThanOrEqual(0.95);
    });

    it('should return lower confidence for middle-range power (uncertain)', () => {
      const conf = computeConfidence(0.2);
      expect(conf).toBeGreaterThanOrEqual(0.45);
      expect(conf).toBeLessThanOrEqual(0.75);
    });

    it('should handle negative values by taking absolute', () => {
      expect(computeConfidence(-0.5)).toBe(computeConfidence(0.5));
    });
  });

  describe('computeEnergyKwh', () => {
    it('should convert kW to kWh for 15-minute interval', () => {
      // 1 kW for 15 minutes = 0.25 kWh
      expect(computeEnergyKwh(1)).toBe(0.25);
    });

    it('should handle fractional kW values', () => {
      expect(computeEnergyKwh(0.4)).toBe(0.1);
    });

    it('should return 0 for 0 kW', () => {
      expect(computeEnergyKwh(0)).toBe(0);
    });
  });

  describe('computeTotalEnergy', () => {
    it('should sum energy from multiple readings', () => {
      const readings = [1, 2, 3, 4]; // kW values
      // Each reading is 0.25h, so total = (1+2+3+4) * 0.25 = 2.5 kWh
      expect(computeTotalEnergy(readings)).toBe(2.5);
    });

    it('should return 0 for empty array', () => {
      expect(computeTotalEnergy([])).toBe(0);
    });
  });

  describe('getTopAppliancesByEnergy', () => {
    const mockRows: NilmDataRow[] = [
      {
        time: new Date('2024-01-01T00:00:00'),
        aggregate: 2.0,
        appliances: {
          HeatPump: 1.0,
          Dishwasher: 0.5,
          WashingMachine: 0.3,
          Oven: 0.2,
        },
      },
      {
        time: new Date('2024-01-01T00:15:00'),
        aggregate: 2.5,
        appliances: {
          HeatPump: 1.2,
          Dishwasher: 0.6,
          WashingMachine: 0.4,
          Oven: 0.3,
        },
      },
      {
        time: new Date('2024-01-01T00:30:00'),
        aggregate: 1.5,
        appliances: {
          HeatPump: 0.8,
          Dishwasher: 0.4,
          WashingMachine: 0.2,
          Oven: 0.1,
        },
      },
    ];

    const appliances = ['HeatPump', 'Dishwasher', 'WashingMachine', 'Oven'];

    it('should return appliances sorted by total energy', () => {
      const result = getTopAppliancesByEnergy(mockRows, appliances, 4);

      expect(result[0].name).toBe('HeatPump');
      expect(result[1].name).toBe('Dishwasher');
      expect(result[2].name).toBe('WashingMachine');
      expect(result[3].name).toBe('Oven');
    });

    it('should respect topN limit', () => {
      const result = getTopAppliancesByEnergy(mockRows, appliances, 2);
      expect(result).toHaveLength(2);
      expect(result[0].name).toBe('HeatPump');
      expect(result[1].name).toBe('Dishwasher');
    });

    it('should calculate correct energy totals', () => {
      const result = getTopAppliancesByEnergy(mockRows, appliances, 4);

      // HeatPump: (1.0 + 1.2 + 0.8) * 0.25 = 0.75 kWh
      expect(result[0].totalKwh).toBeCloseTo(0.75);

      // Dishwasher: (0.5 + 0.6 + 0.4) * 0.25 = 0.375 kWh
      expect(result[1].totalKwh).toBeCloseTo(0.375);
    });

    it('should handle empty rows', () => {
      const result = getTopAppliancesByEnergy([], appliances, 4);
      expect(result).toHaveLength(4);
      result.forEach((app) => {
        expect(app.totalKwh).toBe(0);
      });
    });

    it('should handle missing appliance data in rows', () => {
      const rowsWithMissing: NilmDataRow[] = [
        {
          time: new Date('2024-01-01T00:00:00'),
          aggregate: 1.0,
          appliances: { HeatPump: 1.0 }, // Only HeatPump present
        },
      ];

      const result = getTopAppliancesByEnergy(rowsWithMissing, appliances, 4);

      // HeatPump should have energy
      const heatPump = result.find((a) => a.name === 'HeatPump');
      expect(heatPump?.totalKwh).toBeCloseTo(0.25);

      // Others should be 0
      const dishwasher = result.find((a) => a.name === 'Dishwasher');
      expect(dishwasher?.totalKwh).toBe(0);
    });
  });
});
