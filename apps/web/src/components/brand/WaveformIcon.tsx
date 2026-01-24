import { cn } from "@/lib/utils";

interface WaveformIconProps {
  className?: string;
  size?: "sm" | "md" | "lg";
  animated?: boolean;
}

/**
 * NILM Brand Mark: Waveform + Appliance Pulse
 * Represents energy signal disaggregation visually
 */
export function WaveformIcon({
  className,
  size = "md",
  animated = false,
}: WaveformIconProps) {
  const sizeClasses = {
    sm: "h-6 w-6",
    md: "h-8 w-8",
    lg: "h-12 w-12",
  };

  return (
    <svg
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={cn(
        sizeClasses[size],
        animated && "animate-pulse-glow",
        className,
      )}
    >
      {/* Background circle */}
      <circle cx="16" cy="16" r="16" className="fill-primary" />

      {/* Waveform signal - main energy wave */}
      <path
        d="M4 16 Q8 8, 12 16 T20 16 T28 16"
        stroke="white"
        strokeWidth="2"
        strokeLinecap="round"
        fill="none"
        className={animated ? "animate-pulse" : ""}
      />

      {/* Appliance pulses - disaggregation points */}
      <circle cx="10" cy="12" r="2" className="fill-white/80" />
      <circle cx="16" cy="20" r="2" className="fill-white/80" />
      <circle cx="22" cy="14" r="2" className="fill-white/80" />
    </svg>
  );
}

/**
 * Inline waveform decoration for cards/sections
 */
export function WaveformDecoration({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 120 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={cn("h-6 w-auto opacity-10", className)}
    >
      <path
        d="M0 12 Q10 4, 20 12 T40 12 T60 12 T80 12 T100 12 T120 12"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        fill="none"
      />
    </svg>
  );
}
