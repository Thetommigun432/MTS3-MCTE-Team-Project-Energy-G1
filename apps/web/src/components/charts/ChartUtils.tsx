import { format } from "date-fns";

/**
 * Colorblind-safe palette for charts (7+ distinct colors)
 */
export const CHART_COLORS = [
  "#E69F00", // Orange
  "#56B4E9", // Sky
  "#009E73", // Green
  "#0072B2", // Blue
  "#D55E00", // Vermillion
  "#CC79A7", // Purple
  "#999999", // Other/Gray
  "#F0E442", // Yellow
  "#0072B2", // Blue (darker)
];

/**
 * Reusable axis/tooltip styles for dark-mode-friendly charts
 */
export const CHART_AXIS_STYLE = {
  tick: { fontSize: 10, fill: "hsl(var(--muted-foreground))" },
  tickLine: false,
  axisLine: false,
};

export const CHART_GRID_STYLE = {
  strokeDasharray: "3 3",
  stroke: "hsl(var(--border))",
  vertical: false,
};

export const CHART_TOOLTIP_STYLE = {
  backgroundColor: "hsl(var(--popover))",
  border: "1px solid hsl(var(--border))",
  borderRadius: "var(--radius)",
  fontSize: 12,
  color: "hsl(var(--foreground))",
};

/**
 * Time axis formatters
 */
export const tickFormatter = (ms: number) => format(new Date(ms), "HH:mm");
export const brushFormatter = (ms: number) => format(new Date(ms), "MM/dd HH:mm");
export const tooltipLabelFormatter = (ms: number) => format(new Date(ms), "yyyy-MM-dd HH:mm:ss");

/**
 * LTTB (Largest Triangle Three Buckets) downsampling
 * Preserves visual features while reducing point count for performance
 */
export function downsampleLTTB<T extends { t: number }>(
  data: T[],
  threshold: number,
  yKey: keyof T = "total" as keyof T
): T[] {
  if (data.length <= threshold) return data;
  
  const sampled: T[] = [data[0]];
  const bucketSize = (data.length - 2) / (threshold - 2);
  let a = 0;
  
  for (let i = 0; i < threshold - 2; i++) {
    const bucketStart = Math.floor((i + 1) * bucketSize) + 1;
    const bucketEnd = Math.min(Math.floor((i + 2) * bucketSize) + 1, data.length - 1);
    let avgX = 0, avgY = 0, count = 0;
    
    for (let j = bucketStart; j < bucketEnd; j++) {
      avgX += data[j].t;
      avgY += Number(data[j][yKey]) || 0;
      count++;
    }
    avgX /= count; 
    avgY /= count;
    
    const rangeStart = Math.floor(i * bucketSize) + 1;
    const rangeEnd = Math.floor((i + 1) * bucketSize) + 1;
    let maxArea = -1, maxIdx = rangeStart;
    const pointA = data[a];
    const pointAY = Number(pointA[yKey]) || 0;
    
    for (let j = rangeStart; j < rangeEnd; j++) {
      const pointJY = Number(data[j][yKey]) || 0;
      const area = Math.abs(
        (pointA.t - avgX) * (pointJY - pointAY) -
        (pointA.t - data[j].t) * (avgY - pointAY)
      ) * 0.5;
      if (area > maxArea) { 
        maxArea = area; 
        maxIdx = j; 
      }
    }
    sampled.push(data[maxIdx]);
    a = maxIdx;
  }
  sampled.push(data[data.length - 1]);
  return sampled;
}

/**
 * Smart downsampling that preserves small-appliance patterns
 * Uses a combined importance metric: total + max appliance deviation
 */
export function smartDownsample<T extends { t: number; total?: number }>(
  data: T[],
  threshold: number,
  applianceKeys: string[]
): T[] {
  if (data.length <= threshold) return data;
  
  // For short ranges (< 200 points), keep all data for detail
  if (data.length < 200) return data;
  
  // Compute importance score for each point
  const withImportance = data.map((d, i) => {
    const maxAppliance = Math.max(...applianceKeys.map(key => Number((d as Record<string, unknown>)[key]) || 0));
    // Importance = total magnitude + deviation of largest appliance from mean
    const importance = (d.total || 0) + maxAppliance * 2;
    return { ...d, importance, originalIndex: i };
  });
  
  // Sort by importance and take top threshold points, then re-sort by time
  const sorted = [...withImportance].sort((a, b) => b.importance - a.importance);
  const selected = sorted.slice(0, threshold);
  return selected.sort((a, b) => a.originalIndex - b.originalIndex) as T[];
}

/**
 * Payload entry type for Recharts tooltip
 */
interface TooltipPayloadEntry {
  dataKey?: string | number;
  value?: number;
  color?: string;
  name?: string;
}

/**
 * Simple tooltip - cleaner, less cluttered
 */
interface SimpleTooltipProps {
  active?: boolean;
  payload?: TooltipPayloadEntry[];
  label?: number;
}

export function SimpleEnergyTooltip({ active, payload, label }: SimpleTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  
  return (
    <div className="bg-popover border border-border rounded-lg px-3 py-2 shadow-lg min-w-[140px] text-xs">
      <div className="text-muted-foreground font-medium mb-1.5">
        {label ? format(new Date(label), "MMM d, HH:mm") : ""}
      </div>
      <div className="space-y-1">
        {payload.slice(0, 5).map((p: TooltipPayloadEntry) => {
          const val = p.value ?? 0;
          const label = String(p.dataKey).replace(/_/g, " ");
          return (
            <div key={String(p.dataKey)} className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-1.5 truncate">
                <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: p.color }} />
                <span className="truncate text-foreground capitalize">{label}</span>
              </div>
              <span className="font-mono text-foreground">{val.toFixed(2)} kW</span>
            </div>
          );
        })}
        {payload.length > 5 && (
          <div className="text-muted-foreground text-center pt-1">+{payload.length - 5} more</div>
        )}
      </div>
    </div>
  );
}

/**
 * Enhanced tooltip component for energy charts (detailed mode)
 */
interface EnhancedTooltipProps {
  active?: boolean;
  payload?: TooltipPayloadEntry[];
  label?: number;
  showResidual?: boolean;
  intervalMinutes?: number;
}

export function EnhancedEnergyTooltip({ 
  active, 
  payload, 
  label, 
  showResidual = true,
  intervalMinutes = 15 
}: EnhancedTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  
  const total = payload.find((p: TooltipPayloadEntry) => p.dataKey === "total")?.value;
  const sum = payload.find((p: TooltipPayloadEntry) => p.dataKey === "sum")?.value;
  const residual = payload.find((p: TooltipPayloadEntry) => p.dataKey === "residual")?.value;
  
  // Calculate % of total for each appliance
  const appliancePayloads = payload.filter((p: TooltipPayloadEntry) => 
    !["total", "sum", "residual"].includes(String(p.dataKey))
  );
  
  return (
    <div 
      className="bg-popover border border-border rounded-lg px-3 py-2 shadow-lg min-w-[180px]"
      style={{ fontSize: 12 }}
    >
      <div className="text-xs text-muted-foreground font-medium mb-2">
        {label ? format(new Date(label), "yyyy-MM-dd HH:mm") : ""}
      </div>
      
      {/* Summary metrics */}
      <div className="space-y-1 border-b border-border pb-2 mb-2">
        {total !== undefined && (
          <div className="flex justify-between gap-4">
            <span className="text-muted-foreground">Total</span>
            <span className="font-mono font-medium text-foreground">
              {total.toFixed(3)} kW
            </span>
          </div>
        )}
        {sum !== undefined && (
          <div className="flex justify-between gap-4">
            <span className="text-muted-foreground">Sum (appliances)</span>
            <span className="font-mono text-foreground">{sum.toFixed(3)} kW</span>
          </div>
        )}
        {showResidual && residual !== undefined && (
          <div className="flex justify-between gap-4">
            <span className="text-muted-foreground">Unexplained</span>
            <span className="font-mono text-orange-500">{residual.toFixed(3)} kW</span>
          </div>
        )}
      </div>
      
      {/* Appliance breakdown */}
      {appliancePayloads.length > 0 && (
        <div className="space-y-1 max-h-40 overflow-y-auto">
          {appliancePayloads.map((p: TooltipPayloadEntry) => {
            const val = p.value ?? 0;
            const pct = total && total > 0 ? ((val / total) * 100).toFixed(1) : "0.0";
            return (
              <div key={String(p.dataKey)} className="flex items-center justify-between gap-2 text-xs">
                <div className="flex items-center gap-1.5 truncate">
                  <span 
                    className="w-2 h-2 rounded-full shrink-0" 
                    style={{ backgroundColor: p.color }}
                  />
                  <span className="truncate text-foreground">
                    {String(p.dataKey).replace(/_/g, " ")}
                  </span>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <span className="font-mono text-muted-foreground">{pct}%</span>
                  <span className="font-mono text-foreground w-16 text-right">
                    {val.toFixed(3)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
      
      {/* Energy for interval */}
      {total !== undefined && intervalMinutes > 0 && (
        <div className="border-t border-border mt-2 pt-2 text-xs text-muted-foreground">
          ≈ {(total * intervalMinutes / 60).toFixed(4)} kWh for {intervalMinutes}min interval
        </div>
      )}
    </div>
  );
}

/**
 * Sort appliances by total energy (for stable stacking order)
 */
export function sortAppliancesByEnergy(
  data: Array<{ t: number; [key: string]: number }>,
  applianceKeys: string[]
): string[] {
  const totals = applianceKeys.map(key => ({
    key,
    total: data.reduce((sum, d) => sum + (d[key] || 0), 0),
  }));
  // Sort ascending so largest is at bottom of stack (rendered first)
  return totals.sort((a, b) => a.total - b.total).map(t => t.key);
}

/**
 * Group small appliances into "Other" category
 */
export function groupSmallAppliances<T extends Record<string, unknown>>(
  data: T[],
  applianceKeys: string[],
  topN: number
): { data: T[]; keys: string[] } {
  // Calculate total energy per appliance
  const totals = applianceKeys.map(key => ({
    key,
    total: data.reduce((sum, d) => sum + (Number(d[key]) || 0), 0),
  }));
  
  // Sort and take top N
  totals.sort((a, b) => b.total - a.total);
  const topKeys = totals.slice(0, topN).map(t => t.key);
  const otherKeys = totals.slice(topN).map(t => t.key);
  
  if (otherKeys.length === 0) {
    return { data, keys: topKeys };
  }
  
  // Add "Other" series
  const newData = data.map(d => {
    const otherSum = otherKeys.reduce((sum, key) => sum + (Number(d[key]) || 0), 0);
    return { ...d, Other: otherSum };
  }) as T[];
  
  return { data: newData, keys: [...topKeys, "Other"] };
}

/**
 * Build 24h usage profile from time-series data
 */
export function buildHourlyProfile(
  data: Array<{ t: number; [key: string]: number }>,
  valueKey: string = "total"
): Array<{ hour: number; avgKw: number; energyKwh: number; count: number }> {
  const hourlyBuckets: Record<number, { sum: number; count: number }> = {};
  
  for (let h = 0; h < 24; h++) {
    hourlyBuckets[h] = { sum: 0, count: 0 };
  }
  
  data.forEach(d => {
    const hour = new Date(d.t).getHours();
    const value = d[valueKey] || 0;
    hourlyBuckets[hour].sum += value;
    hourlyBuckets[hour].count++;
  });
  
  return Array.from({ length: 24 }, (_, hour) => {
    const bucket = hourlyBuckets[hour];
    const avgKw = bucket.count > 0 ? bucket.sum / bucket.count : 0;
    // Assuming 15-minute intervals: energy = kW * 0.25 * count
    const energyKwh = bucket.sum * 0.25;
    return { hour, avgKw, energyKwh, count: bucket.count };
  });
}

/**
 * Build day-of-week × hour heatmap data
 */
export function buildDayHourHeatmap(
  data: Array<{ t: number; [key: string]: number }>,
  valueKey: string = "total"
): Array<{ day: number; hour: number; value: number; count: number }> {
  const buckets: Record<string, { sum: number; count: number }> = {};
  
  // Initialize all buckets
  for (let d = 0; d < 7; d++) {
    for (let h = 0; h < 24; h++) {
      buckets[`${d}-${h}`] = { sum: 0, count: 0 };
    }
  }
  
  data.forEach(d => {
    const date = new Date(d.t);
    const day = date.getDay(); // 0 = Sunday
    const hour = date.getHours();
    const value = d[valueKey] || 0;
    const key = `${day}-${hour}`;
    buckets[key].sum += value;
    buckets[key].count++;
  });
  
  const result: Array<{ day: number; hour: number; value: number; count: number }> = [];
  for (let d = 0; d < 7; d++) {
    for (let h = 0; h < 24; h++) {
      const bucket = buckets[`${d}-${h}`];
      result.push({
        day: d,
        hour: h,
        value: bucket.count > 0 ? bucket.sum / bucket.count : 0,
        count: bucket.count,
      });
    }
  }
  
  return result;
}

/**
 * Day names for heatmap
 */
export const DAY_NAMES = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
