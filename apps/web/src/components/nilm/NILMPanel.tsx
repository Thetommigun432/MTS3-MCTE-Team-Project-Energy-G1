import { cn } from "@/lib/utils";
import { WaveformDecoration } from "@/components/brand/WaveformIcon";

interface NILMPanelProps {
  title: React.ReactNode;
  subtitle?: string;
  icon?: React.ReactNode;
  action?: React.ReactNode;
  footer?: React.ReactNode;
  children: React.ReactNode;
  showWaveform?: boolean;
  className?: string;
}

/**
 * NILM Panel Card
 * Standard panel structure: title row, body, footer with AI disclaimer
 */
export function NILMPanel({
  title,
  subtitle,
  icon,
  action,
  footer,
  children,
  showWaveform = false,
  className,
}: NILMPanelProps) {
  return (
    <div className={cn("nilm-panel relative overflow-hidden", className)}>
      {/* Waveform watermark */}
      {showWaveform && (
        <div className="absolute top-0 right-0 opacity-5 pointer-events-none">
          <WaveformDecoration className="h-16 w-auto text-primary" />
        </div>
      )}

      {/* Header */}
      <div className="nilm-panel-header">
        <div className="flex items-center gap-2">
          {icon && <span className="text-muted-foreground">{icon}</span>}
          <div className="space-y-1">
            <h3 className="text-base font-semibold text-foreground">{title}</h3>
            {subtitle && (
              <p className="text-xs text-muted-foreground">{subtitle}</p>
            )}
          </div>
        </div>
        {action && <div className="shrink-0">{action}</div>}
      </div>

      {/* Body */}
      <div className="nilm-panel-body">{children}</div>

      {/* Footer */}
      {footer && <div className="nilm-panel-footer">{footer}</div>}
    </div>
  );
}

interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description: string;
  action?: React.ReactNode;
  className?: string;
}

/**
 * NILM Empty State
 * Used when no data is available with waveform decoration
 */
export function NILMEmptyState({
  icon,
  title,
  description,
  action,
  className,
}: EmptyStateProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center text-center p-8 rounded-lg bg-muted/30 border border-dashed border-border min-h-[200px]",
        className,
      )}
    >
      {/* Waveform illustration */}
      <div className="mb-4 relative">
        <WaveformDecoration className="h-8 w-auto text-primary" />
        {icon && (
          <div className="absolute inset-0 flex items-center justify-center">
            {icon}
          </div>
        )}
      </div>

      <h3 className="font-medium text-foreground mb-1">{title}</h3>
      <p className="text-sm text-muted-foreground max-w-xs mb-4">
        {description}
      </p>

      {action}
    </div>
  );
}
