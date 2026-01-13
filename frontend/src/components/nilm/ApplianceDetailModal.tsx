import { useMemo } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { format } from 'date-fns';
import { Activity, Zap, Clock, TrendingUp } from 'lucide-react';
import { ApplianceStateBadge, ConfidenceIndicator } from './ApplianceStateBadge';
import { NilmDataRow, ON_THRESHOLD, computeConfidence } from '@/hooks/useNilmCsvData';

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
  };
}

export function ApplianceDetailModal({
  open,
  onOpenChange,
  applianceName,
  filteredRows,
  currentStatus,
}: ApplianceDetailModalProps) {
  const displayName = applianceName.replace(/_/g, ' ');

  // Calculate historical data for this appliance
  const historicalData = useMemo(() => {
    return filteredRows.map((row) => {
      const kW = row.appliances[applianceName] ?? 0;
      const isOn = kW >= ON_THRESHOLD;
      // Use standardized confidence calculation (returns 0-1, multiply by 100 for percentage)
      const confidence = computeConfidence(kW) * 100;
      
      return {
        time: format(row.time, 'MM/dd HH:mm'),
        fullTime: row.time,
        kW,
        isOn,
        confidence,
      };
    });
  }, [filteredRows, applianceName]);

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

    const totalKwh = historicalData.reduce((sum, d) => sum + d.kW * (15 / 60), 0);
    const avgKw = historicalData.reduce((sum, d) => sum + d.kW, 0) / historicalData.length;
    const hoursOn = historicalData.filter((d) => d.isOn).length * (15 / 60);
    const avgConfidence = historicalData.reduce((sum, d) => sum + d.confidence, 0) / historicalData.length;
    
    const peakEntry = historicalData.reduce((max, d) => (d.kW > max.kW ? d : max), historicalData[0]);
    
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

  // Confidence over time data
  const confidenceData = useMemo(() => {
    return historicalData.map((d) => ({
      time: d.time,
      confidence: d.confidence,
    }));
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
              <div className="text-lg font-semibold">{stats.totalKwh.toFixed(2)} kWh</div>
            </div>
            <div className="bg-muted/50 rounded-lg p-3 space-y-1">
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Clock className="h-3 w-3" />
                Hours ON
              </div>
              <div className="text-lg font-semibold">{stats.hoursOn.toFixed(1)} hrs</div>
            </div>
            <div className="bg-muted/50 rounded-lg p-3 space-y-1">
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <TrendingUp className="h-3 w-3" />
                Peak Power
              </div>
              <div className="text-lg font-semibold">{stats.peakKw.toFixed(3)} kW</div>
            </div>
            <div className="bg-muted/50 rounded-lg p-3 space-y-1">
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Activity className="h-3 w-3" />
                Avg Confidence
              </div>
              <div className="text-lg font-semibold">{stats.avgConfidence.toFixed(0)}%</div>
            </div>
          </div>

          <Separator />

          {/* Power Usage Over Time */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium">Power Usage Over Time</h4>
              <Badge variant="outline" className="text-xs">
                {stats.onPeriods} ON period{stats.onPeriods !== 1 ? 's' : ''} detected
              </Badge>
            </div>
            <div className="h-40 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={historicalData} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                  <XAxis
                    dataKey="time"
                    tick={{ fontSize: 9, fill: 'hsl(var(--muted-foreground))' }}
                    tickLine={false}
                    axisLine={false}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    tick={{ fontSize: 9, fill: 'hsl(var(--muted-foreground))' }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v) => `${v.toFixed(2)}`}
                  />
                  <ReferenceLine
                    y={ON_THRESHOLD}
                    stroke="hsl(var(--nilm-state-on))"
                    strokeDasharray="4 4"
                    strokeOpacity={0.5}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: 'var(--radius)',
                      fontSize: 11,
                    }}
                    formatter={(value: number) => [`${value.toFixed(4)} kW`, 'Power']}
                  />
                  <Area
                    type="monotone"
                    dataKey="kW"
                    stroke="hsl(var(--primary))"
                    fill="hsl(var(--primary))"
                    fillOpacity={0.3}
                    strokeWidth={1.5}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <p className="text-xs text-muted-foreground">
              Dashed line indicates ON threshold ({ON_THRESHOLD} kW)
            </p>
          </div>

          <Separator />

          {/* Confidence Over Time */}
          <div className="space-y-3">
            <h4 className="text-sm font-medium">Detection Confidence Over Time</h4>
            <div className="h-32 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={confidenceData} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                  <XAxis
                    dataKey="time"
                    tick={{ fontSize: 9, fill: 'hsl(var(--muted-foreground))' }}
                    tickLine={false}
                    axisLine={false}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    domain={[0, 100]}
                    tick={{ fontSize: 9, fill: 'hsl(var(--muted-foreground))' }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v) => `${v}%`}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: 'var(--radius)',
                      fontSize: 11,
                    }}
                    formatter={(value: number) => [`${value.toFixed(0)}%`, 'Confidence']}
                  />
                  <Line
                    type="monotone"
                    dataKey="confidence"
                    stroke="hsl(var(--nilm-confidence-high))"
                    strokeWidth={1.5}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="flex items-center gap-4">
              <span className="text-xs text-muted-foreground">Current:</span>
              <ConfidenceIndicator
                confidence={currentStatus?.confidence ?? stats.avgConfidence}
                showLabel
                size="sm"
              />
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
                    d.isOn
                      ? 'bg-[hsl(var(--nilm-state-on))]'
                      : 'bg-muted'
                  }`}
                  title={`${d.time}: ${d.isOn ? 'ON' : 'OFF'} (${d.kW.toFixed(3)} kW)`}
                />
              ))}
            </div>
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>{historicalData[0]?.time || '—'}</span>
              <span>{historicalData[historicalData.length - 1]?.time || '—'}</span>
            </div>
          </div>

          {/* Model Note */}
          <div className="rounded-lg bg-muted/30 border border-border p-3 space-y-1">
            <p className="text-xs font-medium text-muted-foreground">Explainability Note</p>
            <p className="text-xs text-muted-foreground">
              This appliance's state is predicted by our NILM model analyzing total meter readings. 
              Confidence varies based on signal clarity and typical usage patterns. 
              Predictions are estimates, not direct measurements.
            </p>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
