import { cn } from '@/lib/utils';

interface IllustrationProps {
  className?: string;
  accentColor?: string;
}

/**
 * Meter → AI → Appliances Flow Illustration
 * Shows the NILM disaggregation concept: single meter input, AI processing, multiple appliance outputs
 */
export function MeterToAppliancesIllustration({ className, accentColor = 'currentColor' }: IllustrationProps) {
  return (
    <svg
      viewBox="0 0 200 80"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={cn('text-muted-foreground', className)}
    >
      {/* Meter (left) */}
      <rect x="10" y="25" width="30" height="30" rx="4" />
      <circle cx="25" cy="40" r="8" />
      <path d="M25 35 L25 40 L29 42" />
      
      {/* Arrow from meter */}
      <path d="M45 40 L65 40" />
      <path d="M60 36 L65 40 L60 44" />
      
      {/* AI Brain (center) */}
      <g stroke={accentColor} className="text-primary">
        <rect x="70" y="20" width="40" height="40" rx="8" />
        {/* Neural network dots */}
        <circle cx="82" cy="32" r="2" fill={accentColor} />
        <circle cx="98" cy="32" r="2" fill={accentColor} />
        <circle cx="90" cy="40" r="2.5" fill={accentColor} />
        <circle cx="82" cy="48" r="2" fill={accentColor} />
        <circle cx="98" cy="48" r="2" fill={accentColor} />
        {/* Connections */}
        <path d="M84 33 L88 38" strokeWidth="1" />
        <path d="M96 33 L92 38" strokeWidth="1" />
        <path d="M84 47 L88 42" strokeWidth="1" />
        <path d="M96 47 L92 42" strokeWidth="1" />
      </g>
      
      {/* Arrow from AI */}
      <path d="M115 40 L135 40" />
      <path d="M130 36 L135 40 L130 44" />
      
      {/* Appliances (right) - three stacked */}
      <g>
        {/* Appliance 1 - top */}
        <rect x="140" y="12" width="24" height="16" rx="3" />
        <path d="M148 17 L156 17" />
        <path d="M148 23 L152 23" />
        
        {/* Appliance 2 - middle */}
        <rect x="150" y="32" width="24" height="16" rx="3" />
        <circle cx="162" cy="40" r="4" />
        
        {/* Appliance 3 - bottom */}
        <rect x="140" y="52" width="24" height="16" rx="3" />
        <path d="M145 60 L159 60" />
        <path d="M152 56 L152 64" />
      </g>
    </svg>
  );
}

/**
 * Waveform Signal Illustration
 * Energy consumption waveform with highlighted detection zone
 */
export function WaveformSignalIllustration({ className, accentColor = 'currentColor' }: IllustrationProps) {
  return (
    <svg
      viewBox="0 0 160 60"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={cn('text-muted-foreground', className)}
    >
      {/* Base line */}
      <path d="M10 45 L150 45" strokeWidth="1" opacity="0.3" />
      
      {/* Main waveform */}
      <path 
        d="M10 45 L20 45 L22 40 L28 40 L30 45 L45 45 L47 25 L55 25 L57 45 L75 45 L77 35 L85 35 L87 45 L100 45 L102 15 L115 15 L117 45 L130 45 L132 38 L140 38 L142 45 L150 45"
        strokeWidth="1.5"
      />
      
      {/* Highlighted detection zone */}
      <g stroke={accentColor} className="text-primary">
        <rect x="96" y="10" width="26" height="40" rx="2" fill={accentColor} fillOpacity="0.1" strokeDasharray="3 2" />
        <path d="M102 15 L115 15 L117 45" strokeWidth="2" />
      </g>
      
      {/* Detection marker */}
      <g stroke={accentColor} className="text-primary">
        <circle cx="109" cy="5" r="3" fill={accentColor} />
        <path d="M109 8 L109 12" strokeWidth="1" />
      </g>
      
      {/* Label dots */}
      <circle cx="52" cy="22" r="1.5" fill="currentColor" opacity="0.5" />
      <circle cx="82" cy="32" r="1.5" fill="currentColor" opacity="0.5" />
    </svg>
  );
}

/**
 * Confidence Gauge Illustration
 * Semi-circular gauge showing confidence level with indicator
 */
export function ConfidenceGaugeIllustration({ className, accentColor = 'currentColor' }: IllustrationProps) {
  return (
    <svg
      viewBox="0 0 100 70"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={cn('text-muted-foreground', className)}
    >
      {/* Gauge arc background */}
      <path 
        d="M15 55 A35 35 0 0 1 85 55"
        strokeWidth="6"
        opacity="0.2"
      />
      
      {/* Low zone (red hint) */}
      <path 
        d="M15 55 A35 35 0 0 1 30 28"
        strokeWidth="6"
        opacity="0.3"
      />
      
      {/* Medium zone */}
      <path 
        d="M30 28 A35 35 0 0 1 70 28"
        strokeWidth="6"
        opacity="0.4"
      />
      
      {/* High zone (accent color) */}
      <path 
        d="M70 28 A35 35 0 0 1 85 55"
        stroke={accentColor}
        strokeWidth="6"
        className="text-primary"
      />
      
      {/* Needle */}
      <g stroke={accentColor} className="text-primary">
        <circle cx="50" cy="55" r="4" fill={accentColor} />
        <path d="M50 55 L72 32" strokeWidth="2" />
        <circle cx="72" cy="32" r="2" fill={accentColor} />
      </g>
      
      {/* Tick marks */}
      <path d="M15 55 L20 55" strokeWidth="1" />
      <path d="M22 35 L26 38" strokeWidth="1" />
      <path d="M50 20 L50 25" strokeWidth="1" />
      <path d="M78 35 L74 38" strokeWidth="1" />
      <path d="M85 55 L80 55" strokeWidth="1" />
      
      {/* Labels */}
      <text x="18" y="65" fontSize="6" fill="currentColor" opacity="0.6" stroke="none">Low</text>
      <text x="44" y="12" fontSize="6" fill="currentColor" opacity="0.6" stroke="none">Med</text>
      <text x="74" y="65" fontSize="6" fill={accentColor} stroke="none" className="text-primary">High</text>
    </svg>
  );
}

/**
 * Energy Flow Illustration
 * Simple meter to home energy flow
 */
export function EnergyFlowIllustration({ className, accentColor = 'currentColor' }: IllustrationProps) {
  return (
    <svg
      viewBox="0 0 120 60"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={cn('text-muted-foreground', className)}
    >
      {/* Power lines */}
      <path d="M10 15 L30 15 L35 25 L40 5 L45 25 L50 5 L55 25 L60 15 L80 15" />
      
      {/* Meter */}
      <rect x="80" y="5" width="30" height="25" rx="3" />
      <circle cx="95" cy="17" r="7" />
      <path d="M95 12 L95 17 L99 19" />
      <text x="85" y="27" fontSize="5" fill="currentColor" stroke="none">kWh</text>
      
      {/* Flow arrow */}
      <path d="M95 35 L95 45" stroke={accentColor} className="text-primary" />
      <path d="M91 42 L95 47 L99 42" stroke={accentColor} className="text-primary" />
      
      {/* House */}
      <path d="M80 55 L95 45 L110 55" />
      <rect x="82" y="55" width="26" height="5" rx="1" />
    </svg>
  );
}

/**
 * Appliance Detection Illustration
 * Shows appliance with detection status indicators
 */
export function ApplianceDetectionIllustration({ className, accentColor = 'currentColor' }: IllustrationProps) {
  return (
    <svg
      viewBox="0 0 80 80"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={cn('text-muted-foreground', className)}
    >
      {/* Appliance body */}
      <rect x="15" y="25" width="50" height="40" rx="6" />
      
      {/* Appliance details (washing machine style) */}
      <circle cx="40" cy="48" r="12" />
      <circle cx="40" cy="48" r="6" strokeDasharray="2 2" />
      
      {/* Control panel */}
      <rect x="20" y="30" width="40" height="8" rx="2" fill="currentColor" fillOpacity="0.1" />
      <circle cx="27" cy="34" r="2" />
      <circle cx="35" cy="34" r="2" />
      <path d="M42 32 L55 32" />
      <path d="M42 36 L50 36" />
      
      {/* Detection waves */}
      <g stroke={accentColor} className="text-primary">
        <path d="M68 35 Q75 40 68 45" strokeWidth="1.5" />
        <path d="M72 32 Q82 40 72 48" strokeWidth="1.5" />
        <path d="M76 29 Q88 40 76 51" strokeWidth="1.5" />
      </g>
      
      {/* Status indicator */}
      <circle cx="40" cy="12" r="6" stroke={accentColor} className="text-primary" />
      <path d="M37 12 L39 14 L44 9" stroke={accentColor} strokeWidth="2" className="text-primary" />
    </svg>
  );
}
