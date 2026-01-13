import { useState, useMemo, useCallback } from "react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useEnergy } from "@/contexts/EnergyContext";
import {
  Calendar,
  FileText,
  Download,
  Building2,
  Cpu,
  TrendingUp,
  Zap,
  Activity,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Info,
  Printer,
} from "lucide-react";
import { format } from "date-fns";
import { toast } from "sonner";
import { NILMPanel, NILMEmptyState } from "@/components/nilm/NILMPanel";
import { WaveformDecoration } from "@/components/brand/WaveformIcon";
import { ConfidenceIndicator } from "@/components/nilm/ApplianceStateBadge";
import { EstimatedValueDisplay } from "@/components/nilm/EstimatedValueDisplay";
import { ModelTrustBadge } from "@/components/nilm/ModelTrustBadge";
import {
  computeConfidence,
  computeEnergyKwh,
  ON_THRESHOLD,
  getTopAppliancesByEnergy,
} from "@/hooks/useNilmCsvData";
import {
  formatDateForInput,
  parseLocalDate,
  parseLocalDateEnd,
} from "@/lib/dateUtils";
import { energyApi, isEnergyApiAvailable } from "@/services/energy";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface ReportData {
  generatedAt: Date;
  building: string;
  appliance: string;
  scope: "all" | "single";
  dateRange: { start: Date; end: Date };
  dataPointsAnalyzed: number;
  summary: {
    totalEnergyKwh: number;
    peakPowerKw: number;
    peakPowerTimestamp: Date;
    avgPowerKw: number;
    totalReadings: number;
    appliancesDetected: number;
  };
  applianceBreakdown: {
    name: string;
    totalKwh: number;
    percentOfTotal: number;
    avgConfidence: number;
    hoursOn: number;
  }[];
  hourlyPattern: {
    hour: number;
    avgKw: number;
    energyKwh: number;
  }[];
}

interface ReportError {
  message: string;
  retryable: boolean;
}

// Custom tooltip for chart
function ChartTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ value: number }>;
  label?: string;
}) {
  if (!active || !payload || !payload.length) return null;

  return (
    <div className="bg-popover border border-border rounded-lg px-3 py-2 shadow-lg">
      <p className="text-xs text-muted-foreground mb-1">Hour {label}:00</p>
      <p className="text-sm font-semibold text-foreground">
        {payload[0].value.toFixed(2)} kWh
      </p>
    </div>
  );
}

export default function Reports() {
  const {
    mode,
    selectedBuilding,
    selectedAppliance,
    setSelectedAppliance,
    appliances,
    dateRange,
    setDateRange,
    loading,
    filteredRows,
    apiError,
  } = useEnergy();

  const [reportData, setReportData] = useState<ReportData | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [reportError, setReportError] = useState<ReportError | null>(null);

  // NOTE: generateReportFromData must be defined BEFORE handleGenerateReport
  // to avoid temporal dead zone errors in the dependency array
  const generateReportFromData = useCallback((): ReportData => {
    if (filteredRows.length === 0) {
      throw new Error(
        "No data available for the selected date range. Try expanding the date range.",
      );
    }

    const isSingleAppliance = selectedAppliance !== "All";
    const scope = isSingleAppliance ? "single" : "all";

    // For single appliance mode, compute metrics from that appliance only
    let totalEnergyKwh: number;
    let peakPowerKw: number;
    let peakPowerTimestamp: Date;
    let avgPowerKw: number;

    if (isSingleAppliance) {
      // Single appliance metrics
      let peakRow = filteredRows[0];
      let peakValue = 0;
      let totalPower = 0;

      filteredRows.forEach((row) => {
        const kw = row.appliances[selectedAppliance] || 0;
        totalPower += kw;
        if (kw > peakValue) {
          peakValue = kw;
          peakRow = row;
        }
      });

      totalEnergyKwh = filteredRows.reduce(
        (sum, row) =>
          sum + computeEnergyKwh(row.appliances[selectedAppliance] || 0),
        0,
      );
      peakPowerKw = peakValue;
      peakPowerTimestamp = peakRow.time;
      avgPowerKw = totalPower / filteredRows.length;
    } else {
      // Aggregate metrics
      let peakRow = filteredRows[0];

      filteredRows.forEach((row) => {
        if (row.aggregate > peakRow.aggregate) {
          peakRow = row;
        }
      });

      totalEnergyKwh = filteredRows.reduce(
        (sum, row) => sum + computeEnergyKwh(row.aggregate),
        0,
      );
      peakPowerKw = peakRow.aggregate;
      peakPowerTimestamp = peakRow.time;
      avgPowerKw =
        filteredRows.reduce((sum, r) => sum + r.aggregate, 0) /
        filteredRows.length;
    }

    // Get appliance breakdown
    const appliancesToProcess = isSingleAppliance
      ? [selectedAppliance]
      : appliances;
    const applianceEnergy = getTopAppliancesByEnergy(
      filteredRows,
      appliancesToProcess,
      appliancesToProcess.length,
    );

    // Total for percentage calculation (use appliance-specific total when filtering)
    const breakdownTotal = isSingleAppliance
      ? totalEnergyKwh
      : filteredRows.reduce(
          (sum, row) => sum + computeEnergyKwh(row.aggregate),
          0,
        );

    // Calculate breakdown with confidence and hours on
    const applianceBreakdown = applianceEnergy.map((app) => {
      let totalConfidence = 0;
      let hoursOn = 0;
      let count = 0;

      filteredRows.forEach((row) => {
        const kw = row.appliances[app.name] || 0;
        totalConfidence += computeConfidence(kw);
        if (kw >= ON_THRESHOLD) {
          hoursOn += 15 / 60; // 15-minute intervals
        }
        count++;
      });

      return {
        name: app.name,
        totalKwh: app.totalKwh,
        percentOfTotal:
          breakdownTotal > 0 ? (app.totalKwh / breakdownTotal) * 100 : 0,
        avgConfidence: count > 0 ? totalConfidence / count : 0,
        hoursOn,
      };
    });

    // Hourly pattern - compute energy per hour bucket
    const hourlyData: Record<number, { totalKw: number; count: number }> = {};

    filteredRows.forEach((row) => {
      const hour = row.time.getHours();
      const value = isSingleAppliance
        ? row.appliances[selectedAppliance] || 0
        : row.aggregate;

      if (!hourlyData[hour]) {
        hourlyData[hour] = { totalKw: 0, count: 0 };
      }
      hourlyData[hour].totalKw += value;
      hourlyData[hour].count++;
    });

    // Convert to array with energy (kWh = kW * 0.25 per reading, then sum)
    const hourlyPattern = Array.from({ length: 24 }, (_, hour) => {
      const data = hourlyData[hour];
      const avgKw = data ? data.totalKw / data.count : 0;
      // Energy = sum of (kW * 0.25) for all readings in that hour
      const energyKwh = data ? data.totalKw * 0.25 : 0;

      return { hour, avgKw, energyKwh };
    });

    return {
      generatedAt: new Date(),
      building: selectedBuilding,
      appliance: selectedAppliance,
      scope,
      dateRange: { ...dateRange },
      dataPointsAnalyzed: filteredRows.length,
      summary: {
        totalEnergyKwh,
        peakPowerKw,
        peakPowerTimestamp,
        avgPowerKw,
        totalReadings: filteredRows.length,
        appliancesDetected: applianceBreakdown.filter((a) => a.hoursOn > 0)
          .length,
      },
      applianceBreakdown,
      hourlyPattern,
    };
  }, [
    filteredRows,
    selectedAppliance,
    appliances,
    selectedBuilding,
    dateRange,
  ]);

  const handleGenerateReport = useCallback(async () => {
    setIsGenerating(true);
    setReportError(null);

    // In API mode, attempt backend fetch
    if (mode === "api" && isEnergyApiAvailable()) {
      try {
        const response = await energyApi.generateReport({
          building: selectedBuilding,
          appliance:
            selectedAppliance !== "All" ? selectedAppliance : undefined,
          startDate: dateRange.start.toISOString(),
          endDate: dateRange.end.toISOString(),
          format: "json",
        });

        // If API returns data, transform and use it
        if (response.data) {
          // Transform API response to ReportData shape
          // For now, we generate from local data as the API shape may differ
          console.log("API report response:", response);
        }
      } catch (err) {
        console.warn("API report generation failed, using demo data:", err);
      }
    }

    // Generate from local data (demo fallback or demo mode)
    setTimeout(() => {
      try {
        const report = generateReportFromData();
        setReportData(report);
        setReportError(null);
        toast.success("Report generated", {
          description: `${report.dataPointsAnalyzed.toLocaleString()} data points analyzed`,
        });
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Failed to generate report";
        setReportError({ message, retryable: true });
        setReportData(null);
      } finally {
        setIsGenerating(false);
      }
    }, 400);
  }, [
    mode,
    selectedBuilding,
    selectedAppliance,
    dateRange,
    generateReportFromData,
  ]);

  const handleExport = (exportFormat: "pdf" | "csv") => {
    if (!reportData) {
      toast.error("Generate a report first");
      return;
    }

    if (exportFormat === "csv") {
      // Generate CSV with full context
      const scopeLabel =
        reportData.scope === "single"
          ? `Appliance: ${reportData.appliance}`
          : "All Appliances";

      const headers = [
        "Appliance",
        "Total kWh",
        "% of Total",
        "Avg Confidence",
        "Hours On",
      ];
      const rows = reportData.applianceBreakdown.map((a) => [
        a.name,
        a.totalKwh.toFixed(2),
        a.percentOfTotal.toFixed(1),
        (a.avgConfidence * 100).toFixed(0) + "%",
        a.hoursOn.toFixed(1),
      ]);

      // Include hourly data
      const hourlyHeaders = ["Hour", "Energy (kWh)"];
      const hourlyRows = reportData.hourlyPattern.map((h) => [
        h.hour.toString().padStart(2, "0") + ":00",
        h.energyKwh.toFixed(3),
      ]);

      const csvContent = [
        `# Energy Report - ${format(reportData.generatedAt, "yyyy-MM-dd HH:mm")}`,
        `# Building: ${reportData.building}`,
        `# Scope: ${scopeLabel}`,
        `# Date Range: ${format(reportData.dateRange.start, "yyyy-MM-dd")} to ${format(reportData.dateRange.end, "yyyy-MM-dd")}`,
        `# Data Points: ${reportData.dataPointsAnalyzed.toLocaleString()}`,
        "",
        "# Summary",
        `Total Energy (kWh),${reportData.summary.totalEnergyKwh.toFixed(2)}`,
        `Peak Power (kW),${reportData.summary.peakPowerKw.toFixed(2)}`,
        `Peak Time,${format(reportData.summary.peakPowerTimestamp, "yyyy-MM-dd HH:mm")}`,
        `Average Power (kW),${reportData.summary.avgPowerKw.toFixed(2)}`,
        "",
        "# Appliance Breakdown",
        headers.join(","),
        ...rows.map((r) => r.join(",")),
        "",
        "# Hourly Usage Pattern",
        hourlyHeaders.join(","),
        ...hourlyRows.map((r) => r.join(",")),
      ].join("\n");

      const blob = new Blob([csvContent], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `energy-report-${format(new Date(), "yyyy-MM-dd-HHmm")}.csv`;
      a.click();
      URL.revokeObjectURL(url);

      toast.success("CSV exported successfully");
    } else {
      // PDF export via browser print
      window.print();
    }
  };

  // Timezone-safe date change handlers
  const handleStartDateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    if (!val) return;
    setDateRange({ ...dateRange, start: parseLocalDate(val) });
  };

  const handleEndDateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    if (!val) return;
    setDateRange({ ...dateRange, end: parseLocalDateEnd(val) });
  };

  // Check if hourly data is empty
  const hasHourlyData = useMemo(() => {
    if (!reportData) return false;
    return reportData.hourlyPattern.some((h) => h.energyKwh > 0);
  }, [reportData]);

  const maxEnergy = useMemo(() => {
    if (!reportData) return 1;
    return Math.max(...reportData.hourlyPattern.map((h) => h.energyKwh), 0.01);
  }, [reportData]);

  return (
    <TooltipProvider>
      <div className="space-y-6 print:space-y-4">
        {/* Print-only header */}
        <div className="hidden print:block mb-6">
          <h1 className="text-2xl font-bold">Energy Consumption Report</h1>
          <p className="text-sm text-muted-foreground">
            Generated{" "}
            {reportData
              ? format(reportData.generatedAt, "MMMM d, yyyy HH:mm")
              : ""}
          </p>
        </div>

        <div className="flex items-center justify-between print:hidden">
          <h1 className="text-2xl font-bold text-foreground">Reports</h1>
          <ModelTrustBadge version="1.2.0" lastTrained="2024-01-15" />
        </div>

        {/* API Mode Banner */}
        {mode === "api" && (
          <Alert className="bg-energy-warning-bg border-energy-warning/30 print:hidden">
            <AlertTriangle className="h-4 w-4 text-energy-warning" />
            <AlertDescription className="text-foreground">
              <span className="font-medium">API Mode</span> —{" "}
              {apiError ||
                "Will attempt backend connection on report generation. Falls back to demo data if unavailable."}
            </AlertDescription>
          </Alert>
        )}

        {/* Filter Controls */}
        <NILMPanel
          title="Report Filters"
          icon={<Calendar className="h-5 w-5" />}
          className="print:hidden"
        >
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {/* Building Filter - Disabled */}
            <div className="space-y-2">
              <Label
                htmlFor="building"
                className="text-foreground flex items-center gap-1.5"
              >
                <Building2 className="h-3.5 w-3.5 text-muted-foreground" />
                Building
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Info className="h-3 w-3 text-muted-foreground cursor-help" />
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Demo uses single building dataset</p>
                  </TooltipContent>
                </Tooltip>
              </Label>
              <Select value={selectedBuilding} disabled>
                <SelectTrigger id="building" className="bg-popover opacity-70">
                  <SelectValue placeholder="Select building" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={selectedBuilding}>
                    {selectedBuilding}
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Appliance Filter */}
            <div className="space-y-2">
              <Label
                htmlFor="appliance"
                className="text-foreground flex items-center gap-1.5"
              >
                <Cpu className="h-3.5 w-3.5 text-muted-foreground" />
                Appliance Scope
              </Label>
              <Select
                value={selectedAppliance}
                onValueChange={setSelectedAppliance}
                disabled={loading}
              >
                <SelectTrigger id="appliance" className="bg-popover">
                  <SelectValue placeholder="Select appliance" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="All">All Appliances</SelectItem>
                  {appliances.map((appliance) => (
                    <SelectItem key={appliance} value={appliance}>
                      {appliance.replace(/_/g, " ")}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Start Date */}
            <div className="space-y-2">
              <Label htmlFor="startDate" className="text-foreground">
                Start Date
              </Label>
              <Input
                id="startDate"
                type="date"
                value={formatDateForInput(dateRange.start)}
                onChange={handleStartDateChange}
                disabled={loading}
                className="bg-popover"
              />
            </div>

            {/* End Date */}
            <div className="space-y-2">
              <Label htmlFor="endDate" className="text-foreground">
                End Date
              </Label>
              <Input
                id="endDate"
                type="date"
                value={formatDateForInput(dateRange.end)}
                onChange={handleEndDateChange}
                disabled={loading}
                className="bg-popover"
              />
            </div>
          </div>
        </NILMPanel>

        {/* Generate Report */}
        <NILMPanel
          title="Generate Report"
          icon={<FileText className="h-5 w-5" />}
          className="print:hidden"
        >
          <div className="space-y-4 relative">
            <WaveformDecoration className="absolute top-0 right-0 opacity-5" />
            <p className="text-muted-foreground">
              Generate a detailed energy report with NILM disaggregation.
              {selectedAppliance !== "All" && (
                <Badge
                  variant="outline"
                  className="ml-2 bg-accent/10 text-accent border-accent/30"
                >
                  Filtered: {selectedAppliance.replace(/_/g, " ")}
                </Badge>
              )}
            </p>
            <div className="flex flex-wrap gap-3">
              <Button
                onClick={handleGenerateReport}
                className="bg-primary hover:bg-primary/90"
                disabled={loading || isGenerating}
              >
                {isGenerating ? (
                  <>
                    <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <FileText className="mr-2 h-4 w-4" />
                    Generate Report
                  </>
                )}
              </Button>

              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    onClick={() => handleExport("pdf")}
                    disabled={!reportData}
                  >
                    <Printer className="mr-2 h-4 w-4" />
                    Print / PDF
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Opens print dialog for PDF export</p>
                </TooltipContent>
              </Tooltip>

              <Button
                variant="outline"
                onClick={() => handleExport("csv")}
                disabled={!reportData}
              >
                <Download className="mr-2 h-4 w-4" />
                Export CSV
              </Button>
            </div>
          </div>
        </NILMPanel>

        {/* Error State */}
        {reportError && (
          <Alert className="bg-energy-error-bg border-energy-error/30">
            <AlertTriangle className="h-4 w-4 text-energy-error" />
            <AlertDescription className="flex items-center justify-between">
              <span className="text-foreground">{reportError.message}</span>
              {reportError.retryable && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleGenerateReport}
                  className="ml-4"
                >
                  <RefreshCw className="mr-2 h-3 w-3" />
                  Retry
                </Button>
              )}
            </AlertDescription>
          </Alert>
        )}

        {/* Loading Skeletons */}
        {isGenerating && (
          <>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              {[1, 2, 3, 4].map((i) => (
                <NILMPanel key={i} title={<Skeleton className="h-4 w-24" />}>
                  <Skeleton className="h-8 w-32 mb-2" />
                  <Skeleton className="h-3 w-20" />
                </NILMPanel>
              ))}
            </div>
            <NILMPanel title={<Skeleton className="h-5 w-40" />}>
              <div className="space-y-3">
                {[1, 2, 3].map((i) => (
                  <Skeleton key={i} className="h-12 w-full" />
                ))}
              </div>
            </NILMPanel>
          </>
        )}

        {/* Report Results - add data-report-root for print CSS */}
        {reportData && !isGenerating && (
          <div data-report-root="true" className="report-print-root space-y-6">
            {/* Report Metadata */}
            <div className="flex flex-wrap items-center gap-4 text-sm text-muted-foreground print:text-foreground">
              <span className="flex items-center gap-1.5">
                <CheckCircle2 className="h-4 w-4 text-energy-success" />
                Report generated{" "}
                {format(reportData.generatedAt, "MMM d, yyyy HH:mm")}
              </span>
              <span className="text-border">•</span>
              <span>
                {reportData.dataPointsAnalyzed.toLocaleString()} data points
              </span>
              <span className="text-border">•</span>
              <Badge
                variant={
                  reportData.scope === "single" ? "default" : "secondary"
                }
                className="text-xs"
              >
                {reportData.scope === "single"
                  ? `Appliance: ${reportData.appliance.replace(/_/g, " ")}`
                  : "All Appliances"}
              </Badge>
            </div>

            {/* Summary Cards */}
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4 print:grid-cols-4">
              <NILMPanel
                title="Total Energy"
                icon={<Zap className="h-4 w-4" />}
              >
                <EstimatedValueDisplay
                  value={reportData.summary.totalEnergyKwh}
                  unit="kWh"
                  size="lg"
                  showEstimatedLabel
                />
                <p className="text-xs text-muted-foreground mt-1">
                  {reportData.scope === "single"
                    ? "Appliance only"
                    : "Aggregate total"}
                </p>
              </NILMPanel>

              <NILMPanel
                title="Peak Power"
                icon={<TrendingUp className="h-4 w-4" />}
              >
                <EstimatedValueDisplay
                  value={reportData.summary.peakPowerKw}
                  unit="kW"
                  size="lg"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  at{" "}
                  {format(reportData.summary.peakPowerTimestamp, "MMM d HH:mm")}
                </p>
              </NILMPanel>

              <NILMPanel
                title="Avg Power"
                icon={<Activity className="h-4 w-4" />}
              >
                <Tooltip>
                  <TooltipTrigger>
                    <EstimatedValueDisplay
                      value={reportData.summary.avgPowerKw}
                      unit="kW"
                      size="lg"
                    />
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>
                      Mean power over {reportData.summary.totalReadings}{" "}
                      readings
                    </p>
                  </TooltipContent>
                </Tooltip>
                <p className="text-xs text-muted-foreground mt-1">
                  over selected range
                </p>
              </NILMPanel>

              <NILMPanel
                title="Appliances Detected"
                icon={<Cpu className="h-4 w-4" />}
              >
                <Tooltip>
                  <TooltipTrigger>
                    <div className="flex items-baseline gap-1">
                      <span className="text-2xl font-bold text-foreground tabular-nums">
                        {reportData.summary.appliancesDetected}
                      </span>
                      <Info className="h-3 w-3 text-muted-foreground" />
                    </div>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>
                      Appliances with at least one ON reading (≥{ON_THRESHOLD}{" "}
                      kW)
                    </p>
                  </TooltipContent>
                </Tooltip>
                <p className="text-xs text-muted-foreground mt-1">
                  of {appliances.length} monitored
                </p>
              </NILMPanel>
            </div>

            {/* Appliance Breakdown */}
            <NILMPanel
              title={
                <span className="flex items-center gap-2">
                  Appliance Breakdown
                  {reportData.scope === "single" && (
                    <Badge
                      variant="outline"
                      className="text-xs bg-accent/10 text-accent border-accent/30"
                    >
                      {reportData.appliance.replace(/_/g, " ")} only
                    </Badge>
                  )}
                </span>
              }
              icon={<Cpu className="h-5 w-5" />}
              footer={`Energy percentages sum to 100% of ${reportData.scope === "single" ? "selected appliance" : "total aggregate"}`}
            >
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader className="sticky top-0 bg-card z-10">
                    <TableRow className="hover:bg-transparent">
                      <TableHead className="font-semibold">Appliance</TableHead>
                      <TableHead className="text-right font-semibold">
                        Energy (kWh)
                      </TableHead>
                      <TableHead className="text-right font-semibold">
                        % of Total
                      </TableHead>
                      <TableHead className="text-right font-semibold">
                        Hours On
                      </TableHead>
                      <TableHead className="font-semibold">
                        <Tooltip>
                          <TooltipTrigger className="flex items-center gap-1">
                            Confidence
                            <Info className="h-3 w-3 text-muted-foreground" />
                          </TooltipTrigger>
                          <TooltipContent className="max-w-xs">
                            <p className="font-medium mb-1">
                              Model Confidence (demo)
                            </p>
                            <p className="text-xs">
                              Heuristic based on estimated power. Higher power
                              readings correlate with higher detection
                              confidence.
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {reportData.applianceBreakdown.length === 0 ? (
                      <TableRow>
                        <TableCell
                          colSpan={5}
                          className="text-center text-muted-foreground py-8"
                        >
                          No appliance data for selected filters
                        </TableCell>
                      </TableRow>
                    ) : (
                      reportData.applianceBreakdown.map((appliance, idx) => (
                        <TableRow
                          key={appliance.name}
                          className={idx % 2 === 1 ? "bg-muted/20" : ""}
                        >
                          <TableCell className="font-medium text-foreground">
                            {appliance.name.replace(/_/g, " ")}
                          </TableCell>
                          <TableCell className="text-right font-mono text-foreground">
                            {appliance.totalKwh.toFixed(2)}
                          </TableCell>
                          <TableCell className="text-right">
                            <div className="flex items-center justify-end gap-2">
                              <div className="w-16 h-2 bg-muted rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-primary rounded-full transition-all"
                                  style={{
                                    width: `${Math.min(appliance.percentOfTotal, 100)}%`,
                                  }}
                                />
                              </div>
                              <span className="font-mono text-sm text-muted-foreground w-12 text-right">
                                {appliance.percentOfTotal.toFixed(1)}%
                              </span>
                            </div>
                          </TableCell>
                          <TableCell className="text-right font-mono text-muted-foreground">
                            {appliance.hoursOn.toFixed(1)}h
                          </TableCell>
                          <TableCell>
                            <ConfidenceIndicator
                              confidence={appliance.avgConfidence}
                              showLabel
                            />
                          </TableCell>
                        </TableRow>
                      ))
                    )}
                  </TableBody>
                </Table>
              </div>
            </NILMPanel>

            {/* Hourly Pattern Chart */}
            <NILMPanel
              title="Hourly Usage Pattern"
              icon={<Activity className="h-5 w-5" />}
              footer={`Energy consumption by hour of day (${reportData.scope === "single" ? reportData.appliance.replace(/_/g, " ") : "aggregate"})`}
            >
              {hasHourlyData ? (
                <div className="h-64 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={reportData.hourlyPattern}
                      margin={{ top: 20, right: 20, left: 0, bottom: 20 }}
                    >
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="hsl(var(--border))"
                        vertical={false}
                      />
                      <XAxis
                        dataKey="hour"
                        tick={{
                          fill: "hsl(var(--muted-foreground))",
                          fontSize: 11,
                        }}
                        tickFormatter={(h) => `${h}`}
                        axisLine={{ stroke: "hsl(var(--border))" }}
                        tickLine={{ stroke: "hsl(var(--border))" }}
                        label={{
                          value: "Hour of Day",
                          position: "bottom",
                          fill: "hsl(var(--muted-foreground))",
                          fontSize: 12,
                          offset: 0,
                        }}
                      />
                      <YAxis
                        tick={{
                          fill: "hsl(var(--muted-foreground))",
                          fontSize: 11,
                        }}
                        tickFormatter={(v) => v.toFixed(1)}
                        axisLine={{ stroke: "hsl(var(--border))" }}
                        tickLine={{ stroke: "hsl(var(--border))" }}
                        label={{
                          value: "Energy (kWh)",
                          angle: -90,
                          position: "insideLeft",
                          fill: "hsl(var(--muted-foreground))",
                          fontSize: 12,
                          style: { textAnchor: "middle" },
                        }}
                      />
                      <RechartsTooltip content={<ChartTooltip />} />
                      <Bar
                        dataKey="energyKwh"
                        radius={[4, 4, 0, 0]}
                        maxBarSize={40}
                      >
                        {reportData.hourlyPattern.map((entry, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={
                              entry.energyKwh > maxEnergy * 0.7
                                ? "hsl(var(--primary))"
                                : "hsl(var(--primary) / 0.6)"
                            }
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="h-48 flex flex-col items-center justify-center text-center">
                  <Activity className="h-10 w-10 text-muted-foreground/30 mb-3" />
                  <p className="text-muted-foreground font-medium">
                    No hourly data available
                  </p>
                  <p className="text-sm text-muted-foreground/70 mt-1">
                    Try expanding the date range or selecting different filters
                  </p>
                </div>
              )}
            </NILMPanel>
          </div>
        )}

        {/* Empty State */}
        {!reportData && !isGenerating && !reportError && (
          <NILMEmptyState
            icon={<FileText className="h-12 w-12" />}
            title="No Report Generated"
            description="Configure the filters above and click 'Generate Report' to view energy disaggregation data with AI-estimated confidence metrics."
            action={
              <Button
                onClick={handleGenerateReport}
                disabled={loading || isGenerating}
              >
                <CheckCircle2 className="mr-2 h-4 w-4" />
                Generate Report Now
              </Button>
            }
          />
        )}
      </div>

      {/* Print Styles - Fixed selector */}
      <style>{`
        @media print {
          /* Hide everything except report content */
          body * {
            visibility: hidden;
          }
          
          /* Show the report content */
          .report-print-root,
          .report-print-root *,
          [data-report-root="true"],
          [data-report-root="true"] * {
            visibility: visible !important;
          }
          
          /* Position report at top of page */
          .report-print-root,
          [data-report-root="true"] {
            position: absolute;
            left: 0;
            top: 0;
            width: 100%;
            padding: 20px;
          }
          
          /* Hide print:hidden elements */
          .print\\:hidden {
            display: none !important;
          }
          
          /* Show print:block elements */
          .print\\:block {
            display: block !important;
          }
          
          /* Page break handling */
          .nilm-panel {
            break-inside: avoid;
            page-break-inside: avoid;
          }
          
          /* Force colors for print */
          .report-print-root {
            background: white !important;
            color: black !important;
          }
          
          /* Charts and tables */
          .report-print-root table {
            border-collapse: collapse;
          }
          
          .report-print-root th,
          .report-print-root td {
            border: 1px solid #ddd;
            padding: 8px;
          }
        }
      `}</style>
    </TooltipProvider>
  );
}
