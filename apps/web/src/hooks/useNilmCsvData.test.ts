import { describe, it, expect } from 'vitest';
import {
  computeConfidence,
  computeEnergyKwh,
  computeTotalEnergy,
  getTopAppliancesByEnergy,
  ON_THRESHOLD,
  NilmDataRow,
} from './useNilmCsvData';

describe('NILM Data Utilities', () => {
  describe('ON_THRESHOLD', () => {
    it('should be 0.05 kW', () => {
      expect(ON_THRESHOLD).toBe(0.05);
    });
  });

  describe('computeConfidence', () => {
    it('should return 0 for zero power', () => {
      expect(computeConfidence(0)).toBe(0);
    });

    it('should return 0.5 for 0.5 kW', () => {
      expect(computeConfidence(0.5)).toBe(0.5);
    });

    it('should cap at 1.0 for power >= 1 kW', () => {
      expect(computeConfidence(1.0)).toBe(1);
      expect(computeConfidence(2.0)).toBe(1);
      expect(computeConfidence(10.0)).toBe(1);
    });

    it('should clamp negative values to 0', () => {
      expect(computeConfidence(-0.5)).toBe(0);
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
