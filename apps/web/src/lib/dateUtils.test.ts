import { describe, it, expect } from 'vitest';
import {
  parseLocalDate,
  parseLocalDateEnd,
  startOfDayLocal,
  endOfDayLocal,
  formatDateForInput,
} from './dateUtils';

describe('dateUtils', () => {
  describe('parseLocalDate', () => {
    it('should parse date string to start of day', () => {
      const result = parseLocalDate('2024-03-15');
      expect(result.getFullYear()).toBe(2024);
      expect(result.getMonth()).toBe(2); // March is 2 (0-indexed)
      expect(result.getDate()).toBe(15);
      expect(result.getHours()).toBe(0);
      expect(result.getMinutes()).toBe(0);
      expect(result.getSeconds()).toBe(0);
    });

    it('should handle single-digit months and days', () => {
      const result = parseLocalDate('2024-01-05');
      expect(result.getMonth()).toBe(0); // January
      expect(result.getDate()).toBe(5);
    });
  });

  describe('parseLocalDateEnd', () => {
    it('should parse date string to end of day', () => {
      const result = parseLocalDateEnd('2024-03-15');
      expect(result.getFullYear()).toBe(2024);
      expect(result.getMonth()).toBe(2);
      expect(result.getDate()).toBe(15);
      expect(result.getHours()).toBe(23);
      expect(result.getMinutes()).toBe(59);
      expect(result.getSeconds()).toBe(59);
      expect(result.getMilliseconds()).toBe(999);
    });
  });

  describe('startOfDayLocal', () => {
    it('should normalize date to start of day', () => {
      const input = new Date(2024, 2, 15, 14, 30, 45);
      const result = startOfDayLocal(input);
      expect(result.getHours()).toBe(0);
      expect(result.getMinutes()).toBe(0);
      expect(result.getSeconds()).toBe(0);
      expect(result.getMilliseconds()).toBe(0);
    });

    it('should not mutate the original date', () => {
      const input = new Date(2024, 2, 15, 14, 30, 45);
      startOfDayLocal(input);
      expect(input.getHours()).toBe(14);
    });
  });

  describe('endOfDayLocal', () => {
    it('should normalize date to end of day', () => {
      const input = new Date(2024, 2, 15, 14, 30, 45);
      const result = endOfDayLocal(input);
      expect(result.getHours()).toBe(23);
      expect(result.getMinutes()).toBe(59);
      expect(result.getSeconds()).toBe(59);
      expect(result.getMilliseconds()).toBe(999);
    });

    it('should not mutate the original date', () => {
      const input = new Date(2024, 2, 15, 14, 30, 45);
      endOfDayLocal(input);
      expect(input.getHours()).toBe(14);
    });
  });

  describe('formatDateForInput', () => {
    it('should format date as YYYY-MM-DD', () => {
      const date = new Date(2024, 2, 15);
      expect(formatDateForInput(date)).toBe('2024-03-15');
    });

    it('should pad single-digit months and days', () => {
      const date = new Date(2024, 0, 5);
      expect(formatDateForInput(date)).toBe('2024-01-05');
    });

    it('should handle December correctly', () => {
      const date = new Date(2024, 11, 25);
      expect(formatDateForInput(date)).toBe('2024-12-25');
    });
  });
});
