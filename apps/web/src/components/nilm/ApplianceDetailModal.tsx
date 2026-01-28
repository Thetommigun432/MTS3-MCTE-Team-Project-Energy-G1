import { useMemo, useCallback, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Brush,
} from "recharts";
import { format } from "date-fns";
import { Activity, Zap, Clock, TrendingUp, BarChart2, Percent } from "lucide-react";
import {
  ApplianceStateBadge,
  ConfidenceIndicator,
} from "./ApplianceStateBadge";
import {
  NilmDataRow,
  isApplianceOn,
  computeOnThreshold,
} from "@/hooks/useNilmCsvData";

interface ApplianceDetailModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  applianceName: string;
  filteredRows: NilmDataRow[];
  currentStatus?: {
    name: string;
    on: boolean;
    confidence: number;
    est_kW: number;
    rated_kW?: number | null;
  };
}

export function ApplianceDetailModal({
  open,
  onOpenChange,
  applianceName,
  filteredRows,
  currentStatus,
}: ApplianceDetailModalProps) {
  const displayName = applianceName.replace(/_/g, " ");
  const ratedKw = currentStatus?.rated_kW;
  const onThreshold = computeOnThreshold(ratedKw);
  
  // Chart mode toggle: "prediction" shows kW, "confidence" shows confidence %
  const [detailChartMode, setDetailChartMode] = useState<"prediction" | "confidence">("prediction");

  // Calculate historical data for this appliance with numeric timestamps
  const historicalData = useMemo(() => {
    return filteredRows.map((row) => {
      const kW = row.appliances[applianceName] ?? 0;
      const isOn = isApplianceOn(kW, ratedKw);  // Dynamic threshold
      // Use backend confidence directly from InfluxDB (0 if not available)
      const confidenceRecord = row.confidence || {};
      const backendConfidence = typeof confidenceRecord === 'object' 
        ? (confidenceRecord[applianceName] ?? 0)
        : 0;
      // Return 0-100 percentage for display
      const confidence = backendConfidence * 100;

      return {
        t: row.time instanceof Date ? row.time.getTime() : new Date(row.time).getTime(),
        time: format(row.time, "MM/dd HH:mm"),
        fullTime: row.time,
        kW,
        isOn,
        confidence,
      };
    });
  }, [filteredRows, applianceName, ratedKw]);

  // Formatters for time axis
  const tickFormatter = useCallback((ms: number) => format(new Date(ms), "HH:mm"), []);
  const brushFormatter = useCallback((ms: number) => format(new Date(ms), "MM/dd HH:mm"), []);
  const tooltipLabelFormatter = useCallback((ms: number) => format(new Date(ms), "yyyy-MM-dd HH:mm"), []);

  // Calculate statistics
  const stats = useMemo(() => {
    if (historicalData.length === 0) {
      return {
        totalKwh: 0,
        avgKw: 0,
        hoursOn: 0,
        avgConfidence: 0,
        peakKw: 0,
        peakTime: null as Date | null,
        onPeriods: 0,
      };
    }

    const totalKwh = historicalData.reduce(
      (sum, d) => sum + d.kW * (15 / 60),
      0,
    );
    const avgKw =
      historicalData.reduce((sum, d) => sum + d.kW, 0) / historicalData.length;
    const hoursOn = historicalData.filter((d) => d.isOn).length * (15 / 60);
    const avgConfidence =
      historicalData.reduce((sum, d) => sum + d.confidence, 0) /
      historicalData.length;

    const peakEntry = historicalData.reduce(
      (max, d) => (d.kW > max.kW ? d : max),
      historicalData[0],
    );

    // Count ON periods (transitions from OFF to ON)
    let onPeriods = 0;
    for (let i = 1; i < historicalData.length; i++) {
      if (historicalData[i].isOn && !historicalData[i - 1].isOn) {
        onPeriods++;
      }
    }
    if (historicalData[0]?.isOn) onPeriods++;

    return {
      totalKwh,
      avgKw,
      hoursOn,
      avgConfidence,
      peakKw: peakEntry?.kW ?? 0,
      peakTime: peakEntry?.fullTime ?? null,
      onPeriods,
    };
  }, [historicalData]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex items-center gap-3">
            <DialogTitle className="text-xl">{displayName}</DialogTitle>
            {currentStatus && (
              <ApplianceStateBadge
                on={currentStatus.on}
                confidence={currentStatus.confidence}
                showConfidence
                size="sm"
              />
            )}
          </div>
          <DialogDescription>
            Historical usage patterns and detection confidence over time
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* Quick Stats Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="bg-muted/50 rounded-lg p-3 space-y-1">
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Zap className="h-3 w-3" />
                Total Energy
              </div>
              <div className="text-lg font-semibold">
                {stats.totalKwh.toFixed(2)} kWh
              </div>
            </div>
            <div className="bg-muted/50 rounded-lg p-3 space-y-1">
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Clock className="h-3 w-3" />
                Hours ON
              </div>
              <div className="text-lg font-semibold">
                {stats.hoursOn.toFixed(1)} hrs
              </div>
            </div>
            <div className="bg-muted/50 rounded-lg p-3 space-y-1">
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <TrendingUp className="h-3 w-3" />
                Peak Power
              </div>
              <div className="text-lg font-semibold">
                {stats.peakKw.toFixed(3)} kW
              </div>
            </div>
            <div className="bg-muted/50 rounded-lg p-3 space-y-1">
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Activity className="h-3 w-3" />
                Avg Confidence
              </div>
              <div className="text-lg font-semibold">
                {stats.avgConfidence.toFixed(0)}%
              </div>
            </div>
          </div>

          <Separator />

          {/* Power Usage Over Time */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium">Power Usage Over Time</h4>
              <Badge variant="outline" className="text-xs">
                {stats.onPeriods} ON period{stats.onPeriods !== 1 ? "s" : ""}{" "}
                detected
              </Badge>
            </div>
            <div className="h-48 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={historicalData}
                  margin={{ top: 8, right: 8, left: -16, bottom: 0 }}
                  syncId="appliance-detail"
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="hsl(var(--border))"
                    vertical={false}
                  />
                  <XAxis
                    dataKey="t"
                    type="number"
                    scale="time"
                    domain={["dataMin", "dataMax"]}
                    tickFormatter={tickFormatter}
                    tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }}
                    tickLine={false}
                    axisLine={false}
                    minTickGap={40}
                  />
                  <YAxis
                    tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v) => `${v.toFixed(2)}`}
                  />
                  <ReferenceLine
                    y={onThreshold}
                    stroke="hsl(var(--nilm-state-on))"
                    strokeDasharray="4 4"
                    strokeOpacity={0.5}
                  />
                  <Tooltip
                    labelFormatter={tooltipLabelFormatter}
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "var(--radius)",
                      fontSize: 11,
                    }}
                    formatter={(value: number) => [
                      `${value.toFixed(4)} kW`,
                      "Power",
                    ]}
                  />
                  <Area
                    type="monotone"
                    dataKey="kW"
                    stroke="hsl(var(--primary))"
                    fill="hsl(var(--primary))"
                    fillOpacity={0.3}
                    strokeWidth={1.5}
                  />
                  <Brush
                    dataKey="t"
                    height={20}
                    stroke="hsl(var(--border))"
                    fill="hsl(var(--muted))"
                    tickFormatter={brushFormatter}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <p className="text-xs text-muted-foreground">
              Dashed line indicates ON threshold ({(onThreshold * 1000).toFixed(0)}W) • Drag brush to zoom
            </p>
          </div>

          <Separator />

          {/* Switchable Detail Chart: Prediction (kW) or Confidence */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium">
                {detailChartMode === "prediction" ? "Predicted Power (kW)" : "Model Confidence"}
              </h4>
              <div className="flex items-center gap-1 bg-muted rounded-md p-0.5">
                <button
                  onClick={() => setDetailChartMode("prediction")}
                  className={`flex items-center gap-1 px-2 py-1 rounded text-xs transition-colors ${
                    detailChartMode === "prediction"
                      ? "bg-background text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                  title="Show predicted power values"
                >
                  <BarChart2 className="h-3 w-3" />
                  Power
                </button>
                <button
                  onClick={() => setDetailChartMode("confidence")}
                  className={`flex items-center gap-1 px-2 py-1 rounded text-xs transition-colors ${
                    detailChartMode === "confidence"
                      ? "bg-background text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                  title="Show model confidence"
                >
                  <Percent className="h-3 w-3" />
                  Confidence
                </button>
              </div>
            </div>
            <div className="h-36 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={historicalData}
                  margin={{ top: 8, right: 8, left: -16, bottom: 0 }}
                  syncId="appliance-detail"
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="hsl(var(--border))"
                    vertical={false}
                  />
                  <XAxis
                    dataKey="t"
                    type="number"
                    scale="time"
                    domain={["dataMin", "dataMax"]}
                    tickFormatter={tickFormatter}
                    tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }}
                    tickLine={false}
                    axisLine={false}
                    minTickGap={40}
                  />
                  <YAxis
                    domain={detailChartMode === "confidence" ? [0, 100] : ["auto", "auto"]}
                    tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v) => 
                      detailChartMode === "confidence" ? `${v}%` : `${v.toFixed(2)}`
                    }
                  />
                  <Tooltip
                    labelFormatter={tooltipLabelFormatter}
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "var(--radius)",
                      fontSize: 11,
                    }}
                    formatter={(value: number) => 
                      detailChartMode === "confidence"
                        ? [`${value.toFixed(0)}%`, "Confidence"]
                        : [`${value.toFixed(4)} kW`, "Predicted Power"]
                    }
                  />
                  <Line
                    type="monotone"
                    dataKey={detailChartMode === "confidence" ? "confidence" : "kW"}
                    stroke={detailChartMode === "confidence" 
                      ? "hsl(var(--nilm-confidence-high))" 
                      : "hsl(var(--primary))"
                    }
                    strokeWidth={1.5}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="flex items-center gap-4">
              <span className="text-xs text-muted-foreground">
                {detailChartMode === "confidence" ? "Current:" : "Average:"}
              </span>
              {detailChartMode === "confidence" ? (
                <ConfidenceIndicator
                  confidence={currentStatus?.confidence ?? stats.avgConfidence}
                  showLabel
                  size="sm"
                />
              ) : (
                <span className="text-sm font-medium">{stats.avgKw.toFixed(4)} kW</span>
              )}
            </div>
          </div>

          {/* ON/OFF Timeline */}
          <div className="space-y-3">
            <h4 className="text-sm font-medium">ON/OFF Timeline</h4>
            <div className="h-8 w-full flex rounded-md overflow-hidden border border-border">
              {historicalData.map((d, i) => (
                <div
                  key={i}
                  className={`flex-1 transition-colors ${
                    d.isOn ? "bg-[hsl(var(--nilm-state-on))]" : "bg-muted"
                  }`}
                  title={`${d.time}: ${d.isOn ? "ON" : "OFF"} (${d.kW.toFixed(3)} kW)`}
                />
              ))}
            </div>
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>{historicalData[0]?.time || "—"}</span>
              <span>
                {historicalData[historicalData.length - 1]?.time || "—"}
              </span>
            </div>
          </div>

          {/* Model Note */}
          <div className="rounded-lg bg-muted/30 border border-border p-3 space-y-1">
            <p className="text-xs font-medium text-muted-foreground">
              Explainability Note
            </p>
            <p className="text-xs text-muted-foreground">
              This appliance's state is predicted by our NILM model analyzing
              total meter readings. Confidence values come directly from the ML model.
              Predictions are estimates, not direct measurements.
            </p>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
