import { cn } from '@/lib/utils';
import { Cpu, CheckCircle2, AlertTriangle, AlertCircle } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

interface ModelTrustBadgeProps {
  version?: string;
  lastTrained?: string;
  confidenceLevel?: 'Good' | 'Medium' | 'Low';
  className?: string;
}

/**
 * Model Trust Indicator
 * Shows model version, training date, and confidence level
 */
export function ModelTrustBadge({
  version = 'v1.2',
  lastTrained = '2024-01-15',
  confidenceLevel = 'Good',
  className,
}: ModelTrustBadgeProps) {
  const confidenceConfig = {
    Good: {
      icon: CheckCircle2,
      color: 'text-confidence-high',
      bgColor: 'bg-confidence-high/10',
      borderColor: 'border-confidence-high/20',
    },
    Medium: {
      icon: AlertTriangle,
      color: 'text-confidence-medium',
      bgColor: 'bg-confidence-medium/10',
      borderColor: 'border-confidence-medium/20',
    },
    Low: {
      icon: AlertCircle,
      color: 'text-confidence-low',
      bgColor: 'bg-confidence-low/10',
      borderColor: 'border-confidence-low/20',
    },
  };

  const config = confidenceConfig[confidenceLevel];
  const Icon = config.icon;

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge
            variant="outline"
            className={cn(
              'cursor-help border-border text-muted-foreground text-xs font-medium gap-1.5',
              className
            )}
          >
            <Cpu className="h-3 w-3" />
            <span>Model {version}</span>
            <span className="text-muted-foreground/50">•</span>
            <span className={config.color}>{confidenceLevel}</span>
          </Badge>
        </TooltipTrigger>
        <TooltipContent side="bottom" className="max-w-xs">
          <div className="space-y-2 text-xs">
            <div className="flex items-center gap-2">
              <Icon className={cn('h-4 w-4', config.color)} />
              <span className="font-medium">Confidence: {confidenceLevel}</span>
            </div>
            <p className="text-muted-foreground">
              NILM Model {version} • Last trained: {lastTrained}
            </p>
            <p className="text-muted-foreground">
              Predictions are AI-estimated from total meter data, not directly measured.
            </p>
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

interface DetectionStageBadgeProps {
  stage: 'Learning' | 'Stable' | 'Uncertain';
  className?: string;
}

/**
 * Detection Stage Badge
 * Shows whether an appliance detection is stable, learning, or uncertain
 */
export function DetectionStageBadge({ stage, className }: DetectionStageBadgeProps) {
  const stageConfig = {
    Learning: {
      color: 'text-secondary',
      bgColor: 'bg-secondary/10',
      description: 'Model is still learning this pattern',
    },
    Stable: {
      color: 'text-confidence-high',
      bgColor: 'bg-confidence-high/10',
      description: 'Detection pattern is stable',
    },
    Uncertain: {
      color: 'text-confidence-medium',
      bgColor: 'bg-confidence-medium/10',
      description: 'Detection may need verification',
    },
  };

  const config = stageConfig[stage];

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <span
            className={cn(
              'inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium',
              config.bgColor,
              config.color,
              className
            )}
          >
            {stage}
          </span>
        </TooltipTrigger>
        <TooltipContent side="top">
          <p className="text-xs">{config.description}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
