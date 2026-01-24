import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Power, PowerOff } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface ApplianceStateBadgeProps {
  on: boolean;
  confidence?: number;
  showConfidence?: boolean;
  size?: "sm" | "default";
  className?: string;
}

/**
 * Appliance State Badge with NILM-specific terminology
 * Shows "Predicted ON/OFF" with confidence indicator
 */
export function ApplianceStateBadge({
  on,
  confidence = 0.8,
  showConfidence = false,
  size = "default",
  className,
}: ApplianceStateBadgeProps) {
  const confidencePercent = Math.round(confidence * 100);
  const confidenceLevel =
    confidence >= 0.8 ? "high" : confidence >= 0.5 ? "medium" : "low";

  const confidenceColors = {
    high: "text-confidence-high",
    medium: "text-confidence-medium",
    low: "text-confidence-low",
  };

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge
            variant={on ? "default" : "outline"}
            className={cn(
              "cursor-help transition-all gap-1.5",
              on
                ? "bg-primary text-primary-foreground hover:bg-primary/90"
                : "border-muted-foreground/30 bg-muted/50 text-muted-foreground hover:bg-muted",
              size === "sm" && "text-[10px] px-1.5 py-0",
              className,
            )}
          >
            {on ? (
              <Power
                className={cn(
                  "shrink-0",
                  size === "sm" ? "h-2.5 w-2.5" : "h-3 w-3",
                )}
              />
            ) : (
              <PowerOff
                className={cn(
                  "shrink-0",
                  size === "sm" ? "h-2.5 w-2.5" : "h-3 w-3",
                )}
              />
            )}
            <span>{on ? "ON" : "OFF"}</span>
            {showConfidence && (
              <span
                className={cn("opacity-70", confidenceColors[confidenceLevel])}
              >
                {confidencePercent}%
              </span>
            )}
          </Badge>
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-xs">
          <div className="space-y-1 text-xs">
            <p className="font-medium">
              {on ? "Predicted ON" : "Predicted OFF"} • {confidencePercent}%
              confidence
            </p>
            <p className="text-muted-foreground">
              Estimated by AI from total meter data, not directly measured.
            </p>
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

interface ConfidenceIndicatorProps {
  confidence: number;
  showLabel?: boolean;
  size?: "sm" | "default";
  className?: string;
}

/**
 * Confidence Indicator
 * Visual representation of AI prediction confidence
 */
export function ConfidenceIndicator({
  confidence,
  showLabel = true,
  size = "default",
  className,
}: ConfidenceIndicatorProps) {
  const percent = Math.round(confidence * 100);
  const level =
    confidence >= 0.8 ? "high" : confidence >= 0.5 ? "medium" : "low";

  const levelConfig = {
    high: { color: "bg-confidence-high", label: "High" },
    medium: { color: "bg-confidence-medium", label: "Medium" },
    low: { color: "bg-confidence-low", label: "Low" },
  };

  const config = levelConfig[level];

  return (
    <div className={cn("flex items-center gap-2", className)}>
      {/* Visual bar */}
      <div
        className={cn(
          "relative rounded-full bg-muted overflow-hidden",
          size === "sm" ? "h-1.5 w-12" : "h-2 w-16",
        )}
      >
        <div
          className={cn(
            "absolute inset-y-0 left-0 rounded-full transition-all",
            config.color,
          )}
          style={{ width: `${percent}%` }}
        />
      </div>

      {showLabel && (
        <span
          className={cn(
            "text-muted-foreground",
            size === "sm" ? "text-[10px]" : "text-xs",
          )}
        >
          {percent}%
        </span>
      )}
    </div>
  );
}
