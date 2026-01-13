/**
 * Date utilities for timezone-safe date parsing
 */

/**
 * Parse a YYYY-MM-DD string to a local Date at 00:00:00
 * Avoids UTC parsing which can shift dates
 */
export function parseLocalDate(dateStr: string): Date {
  const [year, month, day] = dateStr.split('-').map(Number);
  return new Date(year, month - 1, day, 0, 0, 0, 0);
}

/**
 * Parse a YYYY-MM-DD string to end of that day (23:59:59.999)
 */
export function parseLocalDateEnd(dateStr: string): Date {
  const [year, month, day] = dateStr.split('-').map(Number);
  return new Date(year, month - 1, day, 23, 59, 59, 999);
}

/**
 * Normalize a Date to start of day (00:00:00 local)
 */
export function startOfDayLocal(date: Date): Date {
  const d = new Date(date);
  d.setHours(0, 0, 0, 0);
  return d;
}

/**
 * Normalize a Date to end of day (23:59:59.999 local)
 */
export function endOfDayLocal(date: Date): Date {
  const d = new Date(date);
  d.setHours(23, 59, 59, 999);
  return d;
}

/**
 * Format a Date to YYYY-MM-DD for input fields
 */
export function formatDateForInput(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}
