import { useParams, Link } from "react-router-dom";
import { useEnergy } from "@/contexts/EnergyContext";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { format } from "date-fns";
import { ArrowLeft, AlertCircle, Info } from "lucide-react";
import { ON_THRESHOLD, computeEnergyKwh } from "@/hooks/useNilmCsvData";

// NILM Components
import {
  ApplianceStateBadge,
  ConfidenceIndicator,
} from "@/components/nilm/ApplianceStateBadge";
import { DetectionStageBadge } from "@/components/nilm/ModelTrustBadge";
import { NILMPanel, NILMEmptyState } from "@/components/nilm/NILMPanel";
import { WaveformDecoration } from "@/components/brand/WaveformIcon";

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

  // Build chart data for this appliance
  const chartData = filteredRows.map((row) => {
    const estKw = row.appliances[decodedName] || 0;
    return {
      time: format(row.time, "MM/dd HH:mm"),
      est_kW: estKw,
      on: estKw >= ON_THRESHOLD,
    };
  });

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

      {/* Line Chart: Est kW Over Time */}
      <NILMPanel
        title="Estimated kW Over Time"
        subtitle="AI-predicted power consumption for this appliance"
        footer="Estimated by AI from total meter data • Not directly measured"
        showWaveform
      >
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={chartData}
              margin={{ top: 8, right: 8, left: -16, bottom: 0 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="hsl(var(--border))"
                vertical={false}
              />
              <XAxis
                dataKey="time"
                tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                tickLine={false}
                axisLine={false}
                interval="preserveStartEnd"
              />
              <YAxis
                tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "var(--radius)",
                  fontSize: 12,
                }}
                formatter={(value: number) => [
                  `${value.toFixed(4)} kW`,
                  "Est. kW",
                ]}
              />
              <Line
                type="monotone"
                dataKey="est_kW"
                stroke="hsl(var(--primary))"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </NILMPanel>

      {/* ON/OFF Timeline */}
      <NILMPanel
        title="Predicted ON/OFF Timeline"
        subtitle={`Visual representation of detected activity (threshold: ${ON_THRESHOLD} kW)`}
        footer="AI-predicted states • Confidence varies by time period"
      >
        <div className="flex gap-0.5 h-10 rounded overflow-hidden">
          {chartData.map((d, i) => (
            <div
              key={i}
              className={`flex-1 transition-colors ${
                d.on ? "bg-state-on" : "bg-state-off"
              }`}
              title={`${d.time}: ${d.on ? "Predicted ON" : "Predicted OFF"} (${d.est_kW.toFixed(3)} kW)`}
            />
          ))}
        </div>
        <div className="flex justify-between text-xs text-muted-foreground mt-3">
          <span className="mono">{chartData[0]?.time || "—"}</span>
          <div className="flex gap-4">
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded bg-state-on" /> Predicted ON
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded bg-state-off border border-border" />{" "}
              Predicted OFF
            </span>
          </div>
          <span className="mono">
            {chartData[chartData.length - 1]?.time || "—"}
          </span>
        </div>
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
