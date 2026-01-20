import { useState, useCallback, useMemo } from "react";
import { useEnergy } from "@/contexts/EnergyContext";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Link } from "react-router-dom";
import {
  Zap,
  TrendingUp,
  Cpu,
  Activity,
  Download,
  AlertCircle,
  ChevronRight,
  RefreshCw,
  FileSpreadsheet,
  Plus,
  Building2,
} from "lucide-react";
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
  Legend,
} from "recharts";
import { format, formatDistanceToNow } from "date-fns";
import { toast } from "sonner";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

// NILM Components
import { ModelTrustBadge } from "@/components/nilm/ModelTrustBadge";
import {
  ApplianceStateBadge,
  ConfidenceIndicator,
} from "@/components/nilm/ApplianceStateBadge";
import { MetricCardContent } from "@/components/nilm/EstimatedValueDisplay";
import { NILMPanel, NILMEmptyState } from "@/components/nilm/NILMPanel";
import { WaveformDecoration } from "@/components/brand/WaveformIcon";
import { ApplianceDetailModal } from "@/components/nilm/ApplianceDetailModal";

// Colorblind-safe palette optimized for dark backgrounds
const CHART_COLORS = [
  "#E69F00", // Orange
  "#56B4E9", // Sky
  "#009E73", // Green
  "#0072B2", // Blue
  "#D55E00", // Vermillion
  "#CC79A7", // Purple
  "#999999", // Other/Gray
];

export default function Dashboard() {
  const {
    mode,
    filteredRows,
    insights,
    currentApplianceStatus,
    topAppliances,
    loading,
    error,
    isRefreshing,
    lastRefreshed,
    refresh,
    dateRange,
    appliances,
    managedAppliances,
    buildings,
    setMode,
  } = useEnergy();
  const [selectedAppliance, setSelectedAppliance] = useState<string | null>(
    null,
  );

  // Filter to show only active appliances in "What's ON Now" section
  const activeAppliances = useMemo(() => {
    return currentApplianceStatus
      .filter((appliance) => appliance.on) // Only show ON appliances
      .slice(0, 5); // Top 5 active
  }, [currentApplianceStatus]);

  // Export data to CSV
  const exportToCSV = useCallback(() => {
    if (filteredRows.length === 0) {
      toast.error("No data to export");
      return;
    }

    // Build CSV header
    const headers = [
      "Timestamp",
      "Aggregate (kW)",
      ...appliances.map((a) => `${a} (kW)`),
    ];

    // Build CSV rows
    const csvRows = filteredRows.map((row) => {
      const applianceValues = appliances.map((a) =>
        (row.appliances[a] || 0).toFixed(4),
      );
      return [
        format(row.time, "yyyy-MM-dd HH:mm:ss"),
        row.aggregate.toFixed(4),
        ...applianceValues,
      ].join(",");
    });

    // Combine header and rows
    const csvContent = [headers.join(","), ...csvRows].join("\n");

    // Create and download file
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `energy-data-${format(dateRange.start, "yyyy-MM-dd")}-to-${format(dateRange.end, "yyyy-MM-dd")}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    toast.success("Data exported successfully", {
      description: `${filteredRows.length} rows exported to CSV`,
    });
  }, [filteredRows, appliances, dateRange]);

  // Export summary report
  const exportSummary = useCallback(() => {
    if (filteredRows.length === 0) {
      toast.error("No data to export");
      return;
    }

    const summaryLines = [
      "Energy Monitor Summary Report",
      "============================",
      "",
      `Date Range: ${format(dateRange.start, "yyyy-MM-dd")} to ${format(dateRange.end, "yyyy-MM-dd")}`,
      `Data Points: ${filteredRows.length}`,
      `Mode: ${mode === "demo" ? "Training Data (Demo)" : "API Predictions"}`,
      "",
      "Key Metrics:",
      `- Peak Load: ${insights.peakLoad.kW.toFixed(2)} kW (at ${insights.peakLoad.timestamp ? format(new Date(insights.peakLoad.timestamp), "MM/dd HH:mm") : "N/A"})`,
      `- Total Energy: ${insights.totalEnergy.toFixed(2)} kWh`,
      `- Top Consumer: ${insights.topAppliance.name}`,
      `- Model Confidence: ${insights.overallConfidence.level} (${insights.overallConfidence.percentage.toFixed(0)}%)`,
      "",
      "Top Appliances by Energy:",
      ...topAppliances.map(
        (a, i) =>
          `${i + 1}. ${a.name.replace(/_/g, " ")}: ${a.totalKwh.toFixed(2)} kWh`,
      ),
      "",
      "Current Appliance Status:",
      ...currentApplianceStatus
        .slice(0, 5)
        .map(
          (a) =>
            `- ${a.name.replace(/_/g, " ")}: ${a.on ? "ON" : "OFF"} (${a.est_kW.toFixed(3)} kW)`,
        ),
      "",
      `Generated: ${format(new Date(), "yyyy-MM-dd HH:mm:ss")}`,
    ];

    const content = summaryLines.join("\n");
    const blob = new Blob([content], { type: "text/plain;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `energy-summary-${format(new Date(), "yyyy-MM-dd")}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    toast.success("Summary report exported", {
      description: "Report saved as text file",
    });
  }, [
    filteredRows,
    dateRange,
    mode,
    insights,
    topAppliances,
    currentApplianceStatus,
  ]);

  // Chart data for total consumption
  const chartData = filteredRows.map((row) => ({
    time: format(row.time, "MM/dd HH:mm"),
    total: row.aggregate,
    ...Object.fromEntries(
      Object.entries(row.appliances).map(([k, v]) => [k, v]),
    ),
  }));

  const topApplianceKeys = topAppliances.map((a) => a.name);

  // Loading state with polished skeletons
  if (loading) {
    return (
      <div className="space-y-8 animate-fade-in">
        {/* Header skeleton */}
        <header className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="space-y-2">
            <Skeleton className="h-7 w-36" />
            <Skeleton className="h-4 w-80" />
          </div>
          <Skeleton className="h-8 w-48 rounded-full" />
        </header>

        {/* Insight cards skeleton */}
        <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i} className="border-border">
              <CardContent className="pt-5 pb-4">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <Skeleton className="h-4 w-24" />
                    <Skeleton className="h-4 w-4 rounded" />
                  </div>
                  <Skeleton className="h-8 w-28" />
                  <Skeleton className="h-3 w-20" />
                </div>
              </CardContent>
            </Card>
          ))}
        </section>

        {/* Charts skeleton */}
        <section className="grid gap-6 lg:grid-cols-2">
          {[1, 2].map((i) => (
            <Card key={i} className="border-border overflow-hidden">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <div className="space-y-1.5">
                    <Skeleton className="h-4 w-32" />
                    <Skeleton className="h-3 w-48" />
                  </div>
                </div>
              </CardHeader>
              <CardContent className="pt-2 pb-4">
                <div className="h-56 flex flex-col justify-end gap-1">
                  {/* Chart bars skeleton */}
                  <div className="flex items-end gap-1 h-44">
                    {Array.from({ length: 24 }).map((_, j) => (
                      <div
                        key={j}
                        className="flex-1 bg-muted rounded-t animate-pulse"
                        style={{
                          height: `${20 + Math.sin(j * 0.5) * 30 + Math.random() * 30}%`,
                          animationDelay: `${j * 50}ms`,
                        }}
                      />
                    ))}
                  </div>
                  {/* X-axis skeleton */}
                  <div className="flex justify-between px-1">
                    <Skeleton className="h-2 w-8" />
                    <Skeleton className="h-2 w-8" />
                    <Skeleton className="h-2 w-8" />
                    <Skeleton className="h-2 w-8" />
                  </div>
                </div>
              </CardContent>
              <div className="px-5 py-3 border-t border-border">
                <Skeleton className="h-3 w-40" />
              </div>
            </Card>
          ))}
        </section>

        {/* Table skeleton */}
        <Card className="border-border overflow-hidden">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <div className="space-y-1.5">
                <Skeleton className="h-4 w-28" />
                <Skeleton className="h-3 w-36" />
              </div>
              <Skeleton className="h-4 w-16" />
            </div>
          </CardHeader>
          <CardContent className="pt-2 pb-4">
            <div className="space-y-0">
              {/* Table header */}
              <div className="flex items-center py-3 border-b border-border gap-4">
                <Skeleton className="h-3 w-24" />
                <Skeleton className="h-3 w-16 ml-auto" />
                <Skeleton className="h-3 w-20" />
                <Skeleton className="h-3 w-12" />
              </div>
              {/* Table rows */}
              {[1, 2, 3, 4, 5].map((i) => (
                <div
                  key={i}
                  className="flex items-center py-3 border-b border-border last:border-0 gap-4"
                >
                  <Skeleton className="h-4 w-28" />
                  <Skeleton className="h-5 w-20 rounded-full ml-auto" />
                  <div className="flex items-center gap-2">
                    <Skeleton className="h-2 w-16 rounded-full" />
                    <Skeleton className="h-3 w-8" />
                  </div>
                  <Skeleton className="h-4 w-12" />
                </div>
              ))}
            </div>
          </CardContent>
          <div className="px-5 py-3 border-t border-border">
            <Skeleton className="h-3 w-64" />
          </div>
        </Card>

        {/* Actions skeleton */}
        <div className="flex flex-wrap gap-3">
          <Skeleton className="h-10 w-44 rounded-md" />
          <Skeleton className="h-10 w-28 rounded-md" />
          <Skeleton className="h-10 w-24 rounded-md" />
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <NILMEmptyState
        icon={<AlertCircle className="h-8 w-8 text-destructive" />}
        title="Failed to load data"
        description={error}
        className="min-h-[400px]"
      />
    );
  }

  // Empty state
  if (filteredRows.length === 0) {
    return (
      <div className="space-y-8">
        <DashboardHeader
          mode={mode}
          isRefreshing={isRefreshing}
          lastRefreshed={lastRefreshed}
          onRefresh={refresh}
          confidenceLevel={insights.overallConfidence.level}
        />
        <NILMEmptyState
          title="No data in this range"
          description="Try selecting a wider date range or check your data source connection."
          action={
            <Button
              variant="outline"
              size="sm"
              onClick={() =>
                toast.info("Adjust your date range in the top bar")
              }
            >
              Adjust Date Range
            </Button>
          }
          className="min-h-[400px]"
        />
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header with Model Trust Badge */}
      <DashboardHeader
        mode={mode}
        isRefreshing={isRefreshing}
        lastRefreshed={lastRefreshed}
        onRefresh={refresh}
        confidenceLevel={insights.overallConfidence.level}
      />

      {/* Demo Building Info Banner */}
      {mode === "demo" && buildings.length > 0 && (
        <Card className="border-blue-500/50 bg-blue-50 dark:bg-blue-950/20">
          <CardContent className="pt-4 pb-4">
            <div className="flex items-start gap-3">
              <Building2 className="h-5 w-5 text-blue-600 dark:text-blue-400 mt-0.5" />
              <div className="flex-1">
                <div className="font-medium text-blue-900 dark:text-blue-100">
                  {buildings[0].name}
                </div>
                <div className="text-sm text-blue-700 dark:text-blue-300 mt-1">
                  {buildings[0].address || "Training dataset from residential building"}
                </div>
                <div className="text-xs text-blue-600 dark:text-blue-400 mt-1">
                  Contains 13 appliances • Oct 2024 - Oct 2025 • 15-minute intervals
                </div>
              </div>
              <Badge variant="secondary" className="bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300">
                Demo Data
              </Badge>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Welcome Message for New Users */}
      {mode === "api" && buildings.length === 0 && (
        <Card className="border-primary bg-primary/5">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Building2 className="h-5 w-5 text-primary" />
              <h2 className="text-lg font-semibold">Welcome to Energy Monitor</h2>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-sm text-muted-foreground">
              To start monitoring your buildings with real-time data from the API,
              create your first building to get started.
            </p>
            <div className="flex gap-3">
              <Button asChild>
                <Link to="/app/buildings">
                  <Plus className="mr-2 h-4 w-4" />
                  Create Your First Building
                </Link>
              </Button>
              <Button variant="outline" onClick={() => setMode("demo")}>
                Switch to Demo Mode
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              Demo mode lets you explore the app with sample data while you set up
              your buildings.
            </p>
          </CardContent>
        </Card>
      )}

      {/* API Mode Status Banner */}
      {mode === "api" && (
        <div
          className={`rounded-lg px-4 py-3 text-sm text-foreground flex items-center gap-2 ${
            error || !filteredRows.length
              ? "bg-destructive/10 border border-destructive/30"
              : "bg-energy-warning-bg/50 border border-energy-warning/30"
          }`}
        >
          <AlertCircle
            className={`h-4 w-4 shrink-0 ${error || !filteredRows.length ? "text-destructive" : "text-energy-warning"}`}
          />
          <span>
            <span
              className={`font-medium ${error || !filteredRows.length ? "text-destructive" : "text-energy-warning"}`}
            >
              API mode
            </span>
            {" — "}
            {error ? (
              <span>{error}</span>
            ) : !filteredRows.length ? (
              <span>
                No data available. Click <strong>Refresh</strong> to fetch from
                API.
              </span>
            ) : (
              <span>Showing live data from backend API</span>
            )}
          </span>
          {(error || !filteredRows.length) && (
            <Button
              variant="outline"
              size="sm"
              onClick={refresh}
              disabled={isRefreshing}
              className="ml-auto"
            >
              <RefreshCw
                className={`h-3 w-3 mr-1 ${isRefreshing ? "animate-spin" : ""}`}
              />
              Refresh
            </Button>
          )}
        </div>
      )}

      {/* Insight Cards - 4 columns */}
      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card className="border-border">
          <CardContent className="pt-5 pb-4">
            <MetricCardContent
              label="Peak Load"
              value={`${insights.peakLoad.kW.toFixed(2)} kW`}
              subtitle={
                insights.peakLoad.timestamp
                  ? format(new Date(insights.peakLoad.timestamp), "MM/dd HH:mm")
                  : "—"
              }
              icon={<Zap className="h-4 w-4" />}
            />
          </CardContent>
        </Card>
        <Card className="border-border">
          <CardContent className="pt-5 pb-4">
            <MetricCardContent
              label="Total Energy"
              value={`${insights.totalEnergy.toFixed(1)} kWh`}
              subtitle="Selected range"
              icon={<TrendingUp className="h-4 w-4" />}
            />
          </CardContent>
        </Card>
        <Card className="border-border">
          <CardContent className="pt-5 pb-4">
            <MetricCardContent
              label="Top Appliance"
              value={insights.topAppliance.name.replace(/_/g, " ")}
              subtitle={`${topAppliances[0]?.totalKwh.toFixed(1) || 0} kWh estimated`}
              icon={<Cpu className="h-4 w-4" />}
            />
          </CardContent>
        </Card>
        <Card className="border-border">
          <CardContent className="pt-5 pb-4">
            <MetricCardContent
              label="Model Confidence"
              value={insights.overallConfidence.level}
              subtitle={`${insights.overallConfidence.percentage.toFixed(0)}% avg confidence`}
              icon={<Activity className="h-4 w-4" />}
            />
          </CardContent>
        </Card>
      </section>

      {/* Charts Row */}
      <section className="grid gap-6 lg:grid-cols-2">
        {/* Total Consumption Chart */}
        <NILMPanel
          title="Total Consumption"
          subtitle="Aggregate power draw over time (kW)"
          footer={
            mode === "demo"
              ? "Ground truth from sub-meters (training data)"
              : filteredRows.length > 0 && filteredRows[0].inferenceType === "ml"
                ? "Data from smart meter • Estimated by AI"
                : "Data from smart meter • Simulated (Demo)"
          }
        >
          <div className="h-56">
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
                    backgroundColor: "hsl(var(--popover))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "var(--radius)",
                    fontSize: 12,
                    color: "hsl(var(--foreground))",
                  }}
                  labelStyle={{ color: "hsl(var(--muted-foreground))" }}
                  formatter={(value: number) => [
                    `${value.toFixed(3)} kW`,
                    "Total",
                  ]}
                />
                <Line
                  type="monotone"
                  dataKey="total"
                  stroke="hsl(var(--primary))"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </NILMPanel>

        {/* Disaggregation Chart */}
        <NILMPanel
          title={mode === "demo" ? "Appliance Breakdown" : "Disaggregation"}
          subtitle="Top 5 appliances by kWh"
          footer={
            mode === "demo"
              ? "Sub-metered readings (not AI predictions)"
              : filteredRows.length > 0 && filteredRows[0].inferenceType === "ml"
                ? `AI-predicted breakdown${filteredRows[0].modelVersion ? ` • Model ${filteredRows[0].modelVersion}` : ""}`
                : "Simulated breakdown (Demo mode)"
          }
        >
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
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
                    backgroundColor: "hsl(var(--popover))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "var(--radius)",
                    fontSize: 12,
                    color: "hsl(var(--foreground))",
                  }}
                  labelStyle={{ color: "hsl(var(--muted-foreground))" }}
                  formatter={(value: number, name: string) => [
                    `${value.toFixed(3)} kW`,
                    name.replace(/_/g, " "),
                  ]}
                />
                <Legend
                  iconType="circle"
                  iconSize={8}
                  wrapperStyle={{
                    fontSize: 10,
                    paddingTop: 8,
                    color: "hsl(var(--muted-foreground))",
                  }}
                  formatter={(value) => value.replace(/_/g, " ")}
                />
                {topApplianceKeys.map((key, i) => (
                  <Area
                    key={key}
                    type="monotone"
                    dataKey={key}
                    stackId="1"
                    stroke={CHART_COLORS[i]}
                    fill={CHART_COLORS[i]}
                    fillOpacity={0.6}
                  />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </NILMPanel>
      </section>

      {/* What's ON Now Table */}
      <NILMPanel
        title={mode === "demo" ? "Appliance Status" : "What's ON now?"}
        subtitle={`${mode === "demo" ? "Actual state" : "Predicted state"} at ${filteredRows.length > 0 ? format(filteredRows[filteredRows.length - 1].time, "MM/dd HH:mm") : "—"}`}
        action={
          <Link
            to="/app/appliances"
            className="text-xs text-primary hover:underline flex items-center gap-1"
          >
            View All <ChevronRight className="h-3 w-3" />
          </Link>
        }
        footer={
          mode === "demo"
            ? "Ground truth from sub-meters — this data trains the NILM model"
            : "States are AI-predicted from total meter data (NILM) • Not directly measured"
        }
      >
        <div className="overflow-x-auto -mx-5 px-5">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-3 font-medium text-muted-foreground">
                  Appliance
                </th>
                <th className="text-left py-3 font-medium text-muted-foreground">
                  State
                </th>
                <th className="text-left py-3 font-medium text-muted-foreground">
                  {mode === "demo" ? "Data Quality" : "Confidence"}
                </th>
                <th className="text-right py-3 font-medium text-muted-foreground">
                  {mode === "demo" ? "Actual kW" : "Est. kW"}
                </th>
                {managedAppliances.length > 0 && (
                  <th className="text-right py-3 font-medium text-muted-foreground hidden sm:table-cell">
                    Rated kW
                  </th>
                )}
              </tr>
            </thead>
            <tbody>
              {activeAppliances.length === 0 ? (
                <tr>
                  <td colSpan={managedAppliances.length > 0 ? 5 : 4} className="py-8 text-center text-muted-foreground">
                    No appliances currently active
                  </td>
                </tr>
              ) : (
                activeAppliances.map((a) => (
                <tr
                  key={a.name}
                  className="border-b border-border last:border-0 hover:bg-muted/30 transition-colors cursor-pointer focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none"
                  onClick={() => setSelectedAppliance(a.name)}
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      setSelectedAppliance(a.name);
                    }
                  }}
                  aria-label={`View details for ${a.name.replace(/_/g, " ")}`}
                >
                  <td className="py-3">
                    <div>
                      <span className="font-medium text-foreground hover:text-primary">
                        {a.name.replace(/_/g, " ")}
                      </span>
                      {a.type && a.type !== "other" && (
                        <span className="block text-xs text-muted-foreground capitalize">
                          {a.type.replace(/_/g, " ")}
                        </span>
                      )}
                      {a.building_name && (
                        <span className="block text-xs text-muted-foreground/70">
                          {a.building_name}
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="py-3">
                    <ApplianceStateBadge
                      on={a.on}
                      confidence={a.confidence}
                      size="sm"
                    />
                  </td>
                  <td className="py-3">
                    <ConfidenceIndicator confidence={a.confidence} size="sm" />
                  </td>
                  <td className="py-3 text-right">
                    <span className="metric-value text-foreground">
                      {a.est_kW.toFixed(3)}
                    </span>
                  </td>
                  {managedAppliances.length > 0 && (
                    <td className="py-3 text-right hidden sm:table-cell">
                      {a.rated_kW ? (
                        <span className="text-muted-foreground">
                          {a.rated_kW} kW
                        </span>
                      ) : (
                        <span className="text-muted-foreground/50">—</span>
                      )}
                    </td>
                  )}
                </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </NILMPanel>

      {/* Actions */}
      <div className="flex flex-wrap gap-3">
        <Button asChild>
          <Link to="/app/appliances">View All Appliances</Link>
        </Button>
        <Button asChild variant="outline">
          <Link to="/app/reports">View Reports</Link>
        </Button>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline">
              <Download className="h-4 w-4 mr-2" />
              Export
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={exportToCSV}>
              <FileSpreadsheet className="h-4 w-4 mr-2" />
              Export to CSV
            </DropdownMenuItem>
            <DropdownMenuItem onClick={exportSummary}>
              <Download className="h-4 w-4 mr-2" />
              Export Summary Report
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Appliance Detail Modal */}
      <ApplianceDetailModal
        open={!!selectedAppliance}
        onOpenChange={(open) => !open && setSelectedAppliance(null)}
        applianceName={selectedAppliance || ""}
        filteredRows={filteredRows}
        currentStatus={currentApplianceStatus.find(
          (a) => a.name === selectedAppliance,
        )}
      />
    </div>
  );
}

// Dashboard Header Component
interface DashboardHeaderProps {
  mode: "demo" | "api";
  isRefreshing: boolean;
  lastRefreshed: Date | null;
  onRefresh: () => void;
  confidenceLevel: "Good" | "Medium" | "Low";
}

function DashboardHeader({
  mode,
  isRefreshing,
  lastRefreshed,
  onRefresh,
  confidenceLevel,
}: DashboardHeaderProps) {
  return (
    <header className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
      <div className="space-y-1 relative">
        <div className="absolute -top-2 -left-4 opacity-5 pointer-events-none">
          <WaveformDecoration className="h-8 w-auto text-primary" />
        </div>
        <h1 className="text-2xl font-semibold tracking-tight text-foreground">
          Dashboard
        </h1>
        <p className="text-sm text-muted-foreground">
          {mode === "demo"
            ? "Ground truth training data — sub-metered readings for model development"
            : "Real-time energy monitoring and appliance activity overview"}
        </p>
      </div>
      <div className="flex items-center gap-3">
        {/* Refresh indicator */}
        <div className="flex items-center gap-2">
          <button
            onClick={onRefresh}
            disabled={isRefreshing}
            aria-label={
              isRefreshing ? "Refreshing data..." : "Refresh energy data"
            }
            aria-busy={isRefreshing}
            className="p-1.5 rounded-md hover:bg-muted transition-colors disabled:opacity-50"
          >
            <RefreshCw
              className={`h-4 w-4 text-muted-foreground ${isRefreshing ? "animate-spin" : ""}`}
            />
          </button>
          {lastRefreshed && (
            <span className="text-xs text-muted-foreground hidden sm:inline">
              {isRefreshing
                ? "Refreshing..."
                : `Updated ${formatDistanceToNow(lastRefreshed, { addSuffix: true })}`}
            </span>
          )}
        </div>

        {mode === "api" && (
          <ModelTrustBadge
            version="v1.0"
            lastTrained="2025-01"
            confidenceLevel={confidenceLevel}
          />
        )}
        {mode === "demo" && (
          <span className="text-xs text-amber-600 dark:text-amber-400 bg-amber-100 dark:bg-amber-900/30 px-2 py-1 rounded font-medium">
            Training Data
          </span>
        )}
        {mode === "api" && (
          <span className="text-xs text-primary bg-primary/10 px-2 py-1 rounded flex items-center gap-1.5">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
            </span>
            Live
          </span>
        )}
      </div>
    </header>
  );
}
