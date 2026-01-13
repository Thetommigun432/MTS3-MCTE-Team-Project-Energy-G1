import { cn } from '@/lib/utils';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { Info } from 'lucide-react';

interface EstimatedValueDisplayProps {
  value: number;
  unit: 'kW' | 'kWh' | '%';
  precision?: number;
  showEstimatedLabel?: boolean;
  size?: 'sm' | 'default' | 'lg';
  className?: string;
}

/**
 * Estimated Value Display
 * Shows metric values with clear indication that they're AI-estimated
 */
export function EstimatedValueDisplay({
  value,
  unit,
  precision = 3,
  showEstimatedLabel = false,
  size = 'default',
  className,
}: EstimatedValueDisplayProps) {
  const sizeClasses = {
    sm: 'text-sm',
    default: 'text-lg',
    lg: 'text-2xl',
  };

  const formattedValue = value.toFixed(precision);

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <span
            className={cn(
              'inline-flex items-center gap-1 cursor-help',
              sizeClasses[size],
              className
            )}
          >
            <span className="metric-value text-foreground">{formattedValue}</span>
            <span className="text-muted-foreground font-normal">{unit}</span>
            {showEstimatedLabel && (
              <Info className="h-3 w-3 text-muted-foreground/50" />
            )}
          </span>
        </TooltipTrigger>
        <TooltipContent side="top">
          <p className="text-xs">
            Estimated by AI (not directly measured)
          </p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

interface MetricCardContentProps {
  label: string;
  value: string;
  subtitle: string;
  icon: React.ReactNode;
  className?: string;
}

/**
 * NILM Metric Card Content
 * Standardized layout: big metrics, small labels, helper text smaller
 */
export function MetricCardContent({
  label,
  value,
  subtitle,
  icon,
  className,
}: MetricCardContentProps) {
  return (
    <div className={cn('flex items-start justify-between gap-4', className)}>
      <div className="space-y-1 min-w-0">
        <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
          {label}
        </p>
        <p className="text-xl font-semibold truncate metric-value text-foreground">
          {value}
        </p>
        <p className="text-xs text-muted-foreground">{subtitle}</p>
      </div>
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
        {icon}
      </div>
    </div>
  );
}
