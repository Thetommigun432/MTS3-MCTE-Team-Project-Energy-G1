import { useParams, Link } from "react-router-dom";
import { useMemo, useCallback, useState } from "react";
import { useEnergy } from "@/contexts/EnergyContext";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
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
  Brush,
  ReferenceLine,
  ReferenceArea,
  BarChart,
  Bar,
  Cell,
} from "recharts";
import { format } from "date-fns";
import { ArrowLeft, AlertCircle, Info, Activity, Clock, ChevronDown, ChevronUp } from "lucide-react";
import { isApplianceOn, computeOnThreshold, computeEnergyKwh } from "@/hooks/useNilmCsvData";

// NILM Components
import {
  ApplianceStateBadge,
  ConfidenceIndicator,
} from "@/components/nilm/ApplianceStateBadge";
import { DetectionStageBadge } from "@/components/nilm/ModelTrustBadge";
import { NILMPanel, NILMEmptyState } from "@/components/nilm/NILMPanel";
import { WaveformDecoration } from "@/components/brand/WaveformIcon";

// Chart utilities
import {
  CHART_AXIS_STYLE,
  CHART_GRID_STYLE,
  tickFormatter,
  brushFormatter,
  buildHourlyProfile,
  buildDayHourHeatmap,
} from "@/components/charts";
import { HourlyHeatmap } from "@/components/charts/HourlyHeatmap";

export default function ApplianceDetails() {
  const { name } = useParams();
  const { filteredRows, currentApplianceStatus, loading } = useEnergy();

  const decodedName = decodeURIComponent(name || "");
  const displayName = decodedName.replace(/_/g, " ");

  // Find current status for this appliance
  const applianceData = currentApplianceStatus.find(
    (a) => a.name.toLowerCase() === decodedName.toLowerCase(),
  );

  // Determine detection stage heuristically
  const getDetectionStage = (
    confidence: number,
  ): "Learning" | "Stable" | "Uncertain" => {
    if (confidence >= 0.85) return "Stable";
    if (confidence >= 0.6) return "Learning";
    return "Uncertain";
  };

  // Build chart data for this appliance with numeric timestamps
  const chartData = useMemo(() => {
    const ratedKw = applianceData?.rated_kW;
    return filteredRows.map((row) => {
      const estKw = row.appliances[decodedName] || 0;
      return {
        t: row.time instanceof Date ? row.time.getTime() : new Date(row.time).getTime(),
        time: format(row.time, "MM/dd HH:mm"),
        est_kW: estKw,
        on: isApplianceOn(estKw, ratedKw),  // Dynamic threshold
      };
    });
  }, [filteredRows, decodedName, applianceData?.rated_kW]);

  // Compute ON threshold for display
  const onThreshold = computeOnThreshold(applianceData?.rated_kW);

  // Formatters for time axis
  const tickFormatter = useCallback((ms: number) => format(new Date(ms), "HH:mm"), []);
  const brushFormatter = useCallback((ms: number) => format(new Date(ms), "MM/dd HH:mm"), []);
  const tooltipLabelFormatter = useCallback((ms: number) => format(new Date(ms), "yyyy-MM-dd HH:mm:ss"), []);

  // Top 5 usage periods (intervals with highest kW)
  const topPeriods = [...chartData]
    .map((d, i) => ({ ...d, index: i }))
    .sort((a, b) => b.est_kW - a.est_kW)
    .slice(0, 5);

  // Total energy for this appliance
  const totalKwh = chartData.reduce(
    (sum, d) => sum + computeEnergyKwh(d.est_kW),
    0,
  );

  // Compute ON periods for ReferenceArea overlays
  const onPeriods = useMemo(() => {
    const periods: Array<{ start: number; end: number }> = [];
    let currentStart: number | null = null;
    
    chartData.forEach((d, i) => {
      if (d.on && currentStart === null) {
        currentStart = d.t;
      } else if (!d.on && currentStart !== null) {
        periods.push({ start: currentStart, end: chartData[i - 1]?.t || currentStart });
        currentStart = null;
      }
    });
    
    // Close any open period
    if (currentStart !== null && chartData.length > 0) {
      periods.push({ start: currentStart, end: chartData[chartData.length - 1].t });
    }
    
    return periods;
  }, [chartData]);

  // Build 24h usage profile
  const hourlyProfile = useMemo(() => {
    const data = chartData.map(d => ({ t: d.t, est_kW: d.est_kW }));
    return buildHourlyProfile(data as Array<{ t: number; [key: string]: number }>, "est_kW");
  }, [chartData]);

  // Build day×hour heatmap data
  const heatmapData = useMemo(() => {
    const data = chartData.map(d => ({ t: d.t, est_kW: d.est_kW }));
    return buildDayHourHeatmap(data as Array<{ t: number; [key: string]: number }>, "est_kW");
  }, [chartData]);

  // Pattern view mode state
  const [patternView, setPatternView] = useState<"profile" | "heatmap">("profile");
  // Collapsible sections state
  const [showDetails, setShowDetails] = useState(false);

  // Max value for profile chart
  const maxHourlyKwh = useMemo(() => {
    return Math.max(...hourlyProfile.map(h => h.avgKw), 0.01);
  }, [hourlyProfile]);

  if (loading) {
    return (
      <div className="space-y-6 animate-fade-in">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-64 w-full rounded-lg" />
      </div>
    );
  }

  if (!applianceData && chartData.length === 0) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="sm" asChild>
            <Link to="/app/appliances">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Appliances
            </Link>
          </Button>
        </div>
        <NILMEmptyState
          icon={<AlertCircle className="h-8 w-8 text-muted-foreground" />}
          title="Appliance not found"
          description={`No data available for "${displayName}"`}
          action={
            <Button asChild variant="outline" size="sm">
              <Link to="/app/dashboard">Return to Dashboard</Link>
            </Button>
          }
          className="min-h-[400px]"
        />
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Back Button & Header */}
      <div className="flex items-center gap-4">
        <Button variant="ghost" size="sm" asChild>
          <Link to="/app/appliances">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back
          </Link>
        </Button>
      </div>

      <header className="space-y-2 relative">
        <div className="absolute -top-2 -left-4 opacity-5 pointer-events-none">
          <WaveformDecoration className="h-8 w-auto text-primary" />
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          <h1 className="text-2xl font-semibold tracking-tight text-foreground">
            {displayName}
          </h1>
          {applianceData && (
            <DetectionStageBadge
              stage={getDetectionStage(applianceData.confidence)}
            />
          )}
        </div>
        <p className="text-sm text-muted-foreground">
          Estimated consumption and predicted ON/OFF timeline from NILM model
        </p>
      </header>

      {/* Current Status Summary */}
      {applianceData && (
        <Card className="border-border bg-card">
          <CardContent className="pt-5 pb-4">
            <div className="flex flex-wrap gap-6 items-center">
              <div className="space-y-1">
                <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Current State
                </p>
                <ApplianceStateBadge
                  on={applianceData.on}
                  confidence={applianceData.confidence}
                  showConfidence
                />
              </div>
              <div className="space-y-1">
                <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Confidence
                </p>
                <ConfidenceIndicator confidence={applianceData.confidence} />
              </div>
              <div className="space-y-1">
                <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Est. Power
                </p>
                <p className="metric-value text-foreground">
                  {applianceData.est_kW.toFixed(3)} kW
                </p>
              </div>
              <div className="space-y-1">
                <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Total in Range
                </p>
                <p className="metric-value text-foreground">
                  {totalKwh.toFixed(2)} kWh
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Line Chart: Est kW Over Time with ON period overlays */}
      <NILMPanel
        title="Estimated kW Over Time"
        subtitle="AI-predicted power consumption • ON periods highlighted"
        footer={`Threshold: ${(onThreshold * 1000).toFixed(0)}W • ${onPeriods.length} ON periods detected`}
        showWaveform
      >
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart
              data={chartData}
              margin={{ top: 8, right: 8, left: -16, bottom: 0 }}
            >
              <CartesianGrid {...CHART_GRID_STYLE} />
              <XAxis
                dataKey="t"
                type="number"
                scale="time"
                domain={["dataMin", "dataMax"]}
                tickFormatter={tickFormatter}
                {...CHART_AXIS_STYLE}
                minTickGap={40}
              />
              <YAxis
                {...CHART_AXIS_STYLE}
                domain={[0, "auto"]}
                label={{ value: "kW", angle: -90, position: "insideLeft", style: { fontSize: 10, fill: "hsl(var(--muted-foreground))" } }}
              />
              
              {/* ON period background highlights */}
              {onPeriods.map((period, i) => (
                <ReferenceArea
                  key={i}
                  x1={period.start}
                  x2={period.end}
                  fill="hsl(var(--primary))"
                  fillOpacity={0.1}
                />
              ))}
              
              {/* ON threshold reference line */}
              <ReferenceLine 
                y={onThreshold} 
                stroke="hsl(var(--primary))" 
                strokeDasharray="4 4"
                strokeOpacity={0.5}
                label={{ 
                  value: `ON threshold`, 
                  position: "right", 
                  fill: "hsl(var(--muted-foreground))",
                  fontSize: 9
                }}
              />
              
              <Tooltip
                labelFormatter={tooltipLabelFormatter}
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "var(--radius)",
                  fontSize: 12,
                }}
                formatter={(value: number, _name: string, props: { payload?: { on?: boolean } }) => {
                  const isOn = props.payload?.on;
                  return [
                    <span key="val" className={isOn ? "text-primary font-medium" : ""}>
                      {value.toFixed(4)} kW {isOn ? "(ON)" : "(OFF)"}
                    </span>,
                    "Est. kW",
                  ];
                }}
              />
              
              {/* Main power area */}
              <Area
                type="monotone"
                dataKey="est_kW"
                stroke="hsl(var(--primary))"
                strokeWidth={2}
                fill="hsl(var(--primary))"
                fillOpacity={0.2}
              />
              
              <Brush
                dataKey="t"
                height={24}
                stroke="hsl(var(--border))"
                fill="hsl(var(--muted))"
                tickFormatter={brushFormatter}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </NILMPanel>

      {/* Activity Timeline - time-based ON/OFF visualization */}
      <NILMPanel
        title="Activity Timeline"
        subtitle={`Visual representation of detected ON/OFF states (${onPeriods.length} active periods)`}
        footer="Shows when appliance was detected as ON based on power threshold"
      >
        <div className="h-16">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart
              data={chartData}
              margin={{ top: 4, right: 8, left: -16, bottom: 4 }}
            >
              <XAxis
                dataKey="t"
                type="number"
                scale="time"
                domain={["dataMin", "dataMax"]}
                tickFormatter={tickFormatter}
                {...CHART_AXIS_STYLE}
                minTickGap={60}
              />
              <YAxis hide domain={[0, 1]} />
              <Area
                type="stepAfter"
                dataKey={(d: { on: boolean }) => (d.on ? 1 : 0)}
                stroke="none"
                fill="hsl(var(--primary))"
                fillOpacity={0.6}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div className="flex justify-between text-xs text-muted-foreground mt-2">
          <span className="mono">{chartData[0]?.time || "—"}</span>
          <div className="flex gap-4">
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded bg-primary/60" /> ON ({onPeriods.length})
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded bg-muted border border-border" /> OFF
            </span>
          </div>
          <span className="mono">{chartData[chartData.length - 1]?.time || "—"}</span>
        </div>
      </NILMPanel>

      {/* Collapsible: More Details Section */}
      <button
        onClick={() => setShowDetails(!showDetails)}
        className="w-full flex items-center justify-center gap-2 py-2 text-sm text-muted-foreground hover:text-foreground transition-colors border border-border rounded-lg bg-muted/30 hover:bg-muted/50"
      >
        {showDetails ? (
          <>
            <ChevronUp className="h-4 w-4" />
            Hide Additional Details
          </>
        ) : (
          <>
            <ChevronDown className="h-4 w-4" />
            Show Usage Patterns & Top Periods
          </>
        )}
      </button>

      {showDetails && (
        <>
          {/* 24h Usage Pattern */}
          <NILMPanel
            title="Usage Pattern"
            subtitle="When is this appliance typically active?"
            action={
              <div className="flex items-center gap-1 bg-muted rounded-md p-0.5">
                <button
                  onClick={() => setPatternView("profile")}
                  className={`flex items-center gap-1 px-2 py-1 rounded text-xs transition-colors ${
                    patternView === "profile"
                      ? "bg-background text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <Activity className="h-3 w-3" />
                  24h Profile
                </button>
                <button
                  onClick={() => setPatternView("heatmap")}
                  className={`flex items-center gap-1 px-2 py-1 rounded text-xs transition-colors ${
                    patternView === "heatmap"
                      ? "bg-background text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <Clock className="h-3 w-3" />
                  Week Heatmap
                </button>
              </div>
            }
            footer="Aggregated from selected date range"
          >
            {patternView === "profile" ? (
              <div className="h-40">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={hourlyProfile}
                    margin={{ top: 8, right: 8, left: -16, bottom: 0 }}
                  >
                    <CartesianGrid {...CHART_GRID_STYLE} />
                    <XAxis 
                      dataKey="hour" 
                      {...CHART_AXIS_STYLE}
                      tickFormatter={(h) => `${h}:00`}
                    />
                    <YAxis 
                      {...CHART_AXIS_STYLE}
                      label={{ value: "kW", angle: -90, position: "insideLeft", style: { fontSize: 10, fill: "hsl(var(--muted-foreground))" } }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "hsl(var(--card))",
                        border: "1px solid hsl(var(--border))",
                        borderRadius: "var(--radius)",
                        fontSize: 12,
                      }}
                      formatter={(value: number) => [`${value.toFixed(3)} kW avg`, "Power"]}
                      labelFormatter={(h) => `${h}:00 - ${h}:59`}
                    />
                    <Bar dataKey="avgKw" radius={[2, 2, 0, 0]} maxBarSize={20}>
                      {hourlyProfile.map((entry, index) => (
                        <Cell 
                          key={`cell-${index}`}
                          fill={entry.avgKw > maxHourlyKwh * 0.5 ? "hsl(var(--primary))" : "hsl(var(--primary) / 0.5)"}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <HourlyHeatmap data={heatmapData} showDayAxis />
            )}
          </NILMPanel>

          {/* Top Usage Periods Table */}
          <NILMPanel
            title="Top Usage Periods"
            subtitle="Top 5 intervals by estimated power consumption"
            footer="Rankings based on AI-estimated values"
          >
            <div className="overflow-x-auto -mx-5 px-5">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-3 font-medium text-muted-foreground">
                      Time
                    </th>
                    <th className="text-left py-3 font-medium text-muted-foreground">
                      State
                    </th>
                    <th className="text-right py-3 font-medium text-muted-foreground">
                      Est. kW
                    </th>
                    <th className="text-right py-3 font-medium text-muted-foreground">
                      Est. kWh
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {topPeriods.map((period, i) => (
                    <tr
                      key={i}
                      className="border-b border-border last:border-0 hover:bg-muted/30 transition-colors"
                    >
                      <td className="py-3 mono text-foreground">{period.time}</td>
                      <td className="py-3">
                        <ApplianceStateBadge on={period.on} size="sm" />
                      </td>
                      <td className="py-3 text-right metric-value text-foreground">
                        {period.est_kW.toFixed(4)}
                      </td>
                      <td className="py-3 text-right text-muted-foreground">
                        {computeEnergyKwh(period.est_kW).toFixed(4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </NILMPanel>
        </>
      )}

      {/* Explainability Note */}
      <div className="flex items-start gap-3 p-4 rounded-lg bg-muted/30 border border-border">
        <Info className="h-5 w-5 text-primary shrink-0 mt-0.5" />
        <div className="space-y-1">
          <p className="text-sm font-medium text-foreground">
            Why was this predicted?
          </p>
          <p className="text-xs text-muted-foreground">
            The NILM model detected power consumption patterns consistent with{" "}
            <strong>{displayName}</strong> activity. Predictions are based on
            signal characteristics from total meter data, not direct
            measurement. Confidence level:{" "}
            <strong>
              {applianceData
                ? `${(applianceData.confidence * 100).toFixed(0)}%`
                : "—"}
            </strong>
          </p>
        </div>
      </div>
    </div>
  );
}
