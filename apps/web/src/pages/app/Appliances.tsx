import { Link } from "react-router-dom";
import { useEnergy } from "@/contexts/EnergyContext";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { ChevronRight, AlertCircle, RefreshCw } from "lucide-react";
import {
  ApplianceStateBadge,
  ConfidenceIndicator,
} from "@/components/nilm/ApplianceStateBadge";
import { NILMPanel, NILMEmptyState } from "@/components/nilm/NILMPanel";
import { WaveformDecoration } from "@/components/brand/WaveformIcon";
import { computeEnergyKwh } from "@/hooks/useNilmCsvData";
import { cn } from "@/lib/utils";
import { useMemo } from "react";

export default function Appliances() {
  const {
    mode,
    currentApplianceStatus,
    filteredRows,
    loading,
    managedAppliances,
    error,
    isRefreshing,
    refresh,
    dateRange,
  } = useEnergy();

  // Calculate historical consumption metrics per appliance
  const applianceEnergy = useMemo(() => {
    const ON_THRESHOLD = 0.01; // kW threshold for "on" state

    return currentApplianceStatus.map((appliance) => {
      let totalKwh = 0;
      let totalKw = 0;
      let count = 0;
      let peakKw = 0;
      let onCount = 0;

      filteredRows.forEach((row) => {
        const kw = row.appliances[appliance.name] || 0;
        totalKwh += computeEnergyKwh(kw);
        totalKw += kw;
        count++;
        if (kw > peakKw) peakKw = kw;
        if (kw >= ON_THRESHOLD) onCount++;
      });

      return {
        ...appliance,
        totalKwh,
        avgKw: count > 0 ? totalKw / count : 0,
        peakKw,
        onHours: onCount * 0.25, // Assuming 15-min intervals
      };
    });
  }, [currentApplianceStatus, filteredRows]);

  if (loading) {
    return (
      <div className="space-y-6 animate-fade-in">
        <div className="space-y-2">
          <Skeleton className="h-7 w-32" />
          <Skeleton className="h-4 w-64" />
        </div>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <Skeleton key={i} className="h-32 rounded-lg" />
          ))}
        </div>
      </div>
    );
  }

  // Empty state for API mode with no data
  if (filteredRows.length === 0 || currentApplianceStatus.length === 0) {
    return (
      <div className="space-y-6 animate-fade-in">
        <header className="space-y-2 relative">
          <div className="absolute -top-2 -left-4 opacity-5 pointer-events-none">
            <WaveformDecoration className="h-8 w-auto text-primary" />
          </div>
          <h1 className="text-2xl font-semibold tracking-tight text-foreground">
            Appliances
          </h1>
          <p className="text-sm text-muted-foreground">
            Select an appliance to view detailed consumption patterns and
            detection confidence
          </p>
        </header>

        {/* API Mode Banner */}
        {mode === "api" && (
          <div
            className={`rounded-lg px-4 py-3 text-sm text-foreground flex items-center gap-2 ${
              error
                ? "bg-destructive/10 border border-destructive/30"
                : "bg-energy-warning-bg/50 border border-energy-warning/30"
            }`}
          >
            <AlertCircle
              className={`h-4 w-4 shrink-0 ${error ? "text-destructive" : "text-energy-warning"}`}
            />
            <span>
              <span
                className={`font-medium ${error ? "text-destructive" : "text-energy-warning"}`}
              >
                API mode
              </span>
              {" — "}
              {error
                ? error
                : "No appliance data available. Click Refresh to fetch from API."}
            </span>
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
          </div>
        )}

        <NILMEmptyState
          title="No appliance data"
          description={
            mode === "api"
              ? "Click Refresh to fetch appliance data from the API, or switch to Demo mode to see sample data."
              : "No appliance data found in the selected date range. Try selecting a wider date range."
          }
        />
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <header className="space-y-2 relative">
        <div className="absolute -top-2 -left-4 opacity-5 pointer-events-none">
          <WaveformDecoration className="h-8 w-auto text-primary" />
        </div>
        <h1 className="text-2xl font-semibold tracking-tight text-foreground">
          Appliances
        </h1>
        <p className="text-sm text-muted-foreground">
          Comprehensive appliance inventory with historical consumption analysis for{" "}
          {dateRange.start.toLocaleDateString()} - {dateRange.end.toLocaleDateString()}
        </p>
      </header>

      {/* Appliances Grid */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {applianceEnergy.map((appliance) => (
          <Link
            key={appliance.name}
            to={`/app/appliances/${encodeURIComponent(appliance.name)}`}
            className="group"
          >
            <Card className="border-border hover:border-primary/50 transition-all duration-200 hover:shadow-md">
              <CardContent className="pt-5 pb-4">
                <div className="flex items-start justify-between gap-4">
                  <div className="space-y-3 flex-1">
                    <div className="flex items-center gap-2">
                      <h3 className="font-medium text-foreground group-hover:text-primary transition-colors">
                        {appliance.name.replace(/_/g, " ")}
                      </h3>
                      <ChevronRight className="h-4 w-4 text-muted-foreground group-hover:text-primary transition-colors" />
                    </div>
                    {appliance.type && appliance.type !== "other" && (
                      <span className="text-xs text-muted-foreground capitalize">
                        {appliance.type.replace(/_/g, " ")}
                      </span>
                    )}
                    <ApplianceStateBadge
                      on={appliance.on}
                      confidence={appliance.confidence}
                      showConfidence
                      size="sm"
                    />
                    <div className="space-y-1 text-xs text-muted-foreground">
                      <div className="flex items-center gap-3">
                        <span>
                          Now:{" "}
                          <span className="font-medium text-foreground">
                            {appliance.est_kW.toFixed(3)}
                          </span>{" "}
                          kW
                        </span>
                        <span>
                          Avg:{" "}
                          <span className="font-medium text-foreground">
                            {appliance.avgKw.toFixed(3)}
                          </span>{" "}
                          kW
                        </span>
                        <span>
                          Peak:{" "}
                          <span className="font-medium text-foreground">
                            {appliance.peakKw.toFixed(3)}
                          </span>{" "}
                          kW
                        </span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span>
                          Total:{" "}
                          <span className="font-medium text-foreground">
                            {appliance.totalKwh.toFixed(2)}
                          </span>{" "}
                          kWh
                        </span>
                        <span>
                          On:{" "}
                          <span className="font-medium text-foreground">
                            {appliance.onHours.toFixed(1)}
                          </span>{" "}
                          hrs
                        </span>
                        {appliance.rated_kW && (
                          <span>
                            Rated:{" "}
                            <span className="font-medium text-foreground">
                              {appliance.rated_kW}
                            </span>{" "}
                            kW
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="w-16">
                    <ConfidenceIndicator
                      confidence={appliance.confidence}
                      showLabel
                      size="sm"
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>

      {/* Summary */}
      <NILMPanel
        title="Detection Summary"
        subtitle="Overview of all monitored appliances"
        footer="States and confidence are AI-predicted from total meter data"
      >
        <div className="overflow-x-auto -mx-5 px-5">
          <table className="w-full text-sm">
            <thead>
              <tr>
                <th className="text-left py-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  Appliance
                </th>
                <th className="text-left py-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  State
                </th>
                <th className="text-left py-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  Confidence
                </th>
                <th className="text-right py-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  Current kW
                </th>
                {managedAppliances.length > 0 && (
                  <th className="text-right py-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground hidden sm:table-cell">
                    Rated kW
                  </th>
                )}
                <th className="text-right py-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  Total kWh
                </th>
              </tr>
            </thead>
            <tbody>
              {applianceEnergy.map((appliance, idx) => (
                <tr
                  key={appliance.name}
                  className={cn(
                    "hover:bg-muted/40 transition-colors",
                    idx % 2 === 1 && "bg-muted/20",
                  )}
                >
                  <td className="py-3">
                    <Link
                      to={`/app/appliances/${encodeURIComponent(appliance.name)}`}
                      className="font-medium text-foreground hover:text-primary hover:underline"
                    >
                      {appliance.name.replace(/_/g, " ")}
                    </Link>
                    {appliance.type && appliance.type !== "other" && (
                      <span className="block text-xs text-muted-foreground capitalize">
                        {appliance.type.replace(/_/g, " ")}
                      </span>
                    )}
                    {appliance.building_name && (
                      <span className="block text-xs text-muted-foreground/70">
                        {appliance.building_name}
                      </span>
                    )}
                  </td>
                  <td className="py-3">
                    <ApplianceStateBadge on={appliance.on} size="sm" />
                  </td>
                  <td className="py-3">
                    <ConfidenceIndicator
                      confidence={appliance.confidence}
                      size="sm"
                    />
                  </td>
                  <td className="py-3 text-right font-mono text-foreground">
                    {appliance.est_kW.toFixed(3)}
                  </td>
                  {managedAppliances.length > 0 && (
                    <td className="py-3 text-right font-mono text-muted-foreground hidden sm:table-cell">
                      {appliance.rated_kW ? appliance.rated_kW : "—"}
                    </td>
                  )}
                  <td className="py-3 text-right font-mono text-foreground">
                    {appliance.totalKwh.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </NILMPanel>
    </div>
  );
}
