import { useMemo } from "react";
import { cn } from "@/lib/utils";
import { DAY_NAMES } from "./ChartUtils";

interface HeatmapCellData {
  day: number;
  hour: number;
  value: number;
  count: number;
}

interface HourlyHeatmapProps {
  data: HeatmapCellData[];
  className?: string;
  showDayAxis?: boolean; // true for 7-day view, false for 24h-only
}

/**
 * Heatmap visualization for hourly usage patterns
 * Can show either 24h-only or full day×hour matrix
 */
export function HourlyHeatmap({ data, className, showDayAxis = true }: HourlyHeatmapProps) {
  const { maxValue, normalizedData } = useMemo(() => {
    const max = Math.max(...data.map(d => d.value), 0.001);
    return {
      maxValue: max,
      normalizedData: data.map(d => ({
        ...d,
        intensity: d.value / max,
      })),
    };
  }, [data]);
  
  // Group by day if showing full matrix
  const dayGroups = useMemo(() => {
    if (!showDayAxis) {
      // Aggregate across all days for 24h-only view
      const hourAgg: Record<number, { sum: number; count: number }> = {};
      for (let h = 0; h < 24; h++) hourAgg[h] = { sum: 0, count: 0 };
      
      data.forEach(d => {
        hourAgg[d.hour].sum += d.value * d.count;
        hourAgg[d.hour].count += d.count;
      });
      
      return [{
        day: -1,
        cells: Array.from({ length: 24 }, (_, h) => ({
          hour: h,
          value: hourAgg[h].count > 0 ? hourAgg[h].sum / hourAgg[h].count : 0,
          intensity: hourAgg[h].count > 0 ? (hourAgg[h].sum / hourAgg[h].count) / maxValue : 0,
        })),
      }];
    }
    
    // Full 7-day matrix
    return Array.from({ length: 7 }, (_, day) => ({
      day,
      cells: normalizedData.filter(d => d.day === day).sort((a, b) => a.hour - b.hour),
    }));
  }, [data, normalizedData, showDayAxis, maxValue]);
  
  const getColor = (intensity: number) => {
    if (intensity === 0) return "bg-muted/30";
    if (intensity < 0.2) return "bg-primary/20";
    if (intensity < 0.4) return "bg-primary/40";
    if (intensity < 0.6) return "bg-primary/60";
    if (intensity < 0.8) return "bg-primary/80";
    return "bg-primary";
  };
  
  return (
    <div className={cn("space-y-2", className)}>
      {/* Hour labels */}
      <div className={cn("flex gap-px", showDayAxis ? "ml-10" : "ml-0")}>
        {Array.from({ length: 24 }, (_, h) => (
          <div 
            key={h} 
            className="flex-1 text-center text-[9px] text-muted-foreground"
          >
            {h % 3 === 0 ? h : ""}
          </div>
        ))}
      </div>
      
      {/* Heatmap grid */}
      <div className="space-y-px">
        {dayGroups.map(({ day, cells }) => (
          <div key={day} className="flex items-center gap-1">
            {showDayAxis && (
              <div className="w-9 text-xs text-muted-foreground text-right pr-1 shrink-0">
                {day >= 0 ? DAY_NAMES[day] : ""}
              </div>
            )}
            <div className="flex-1 flex gap-px">
              {cells.map(cell => (
                <div
                  key={`${day}-${cell.hour}`}
                  className={cn(
                    "flex-1 h-5 rounded-sm transition-colors cursor-default",
                    getColor(cell.intensity)
                  )}
                  title={`${showDayAxis ? DAY_NAMES[day] + " " : ""}${cell.hour}:00 - ${cell.value.toFixed(3)} kW avg`}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
      
      {/* Legend */}
      <div className="flex items-center justify-end gap-2 text-[10px] text-muted-foreground pt-1">
        <span>Low</span>
        <div className="flex gap-px">
          <div className="w-4 h-3 rounded-sm bg-muted/30" />
          <div className="w-4 h-3 rounded-sm bg-primary/20" />
          <div className="w-4 h-3 rounded-sm bg-primary/40" />
          <div className="w-4 h-3 rounded-sm bg-primary/60" />
          <div className="w-4 h-3 rounded-sm bg-primary/80" />
          <div className="w-4 h-3 rounded-sm bg-primary" />
        </div>
        <span>High ({maxValue.toFixed(2)} kW)</span>
      </div>
    </div>
  );
}
