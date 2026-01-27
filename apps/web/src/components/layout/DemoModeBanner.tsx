/**
 * Demo Mode Banner
 *
 * Displays a visible warning when the app is running in demo mode.
 * This prevents user confusion about whether data is real or simulated.
 */
import { AlertTriangle, X } from "lucide-react";
import { useState, useEffect } from "react";
import { isDemoMode, onModeChange, DataSource } from "@/lib/dataSource";

export function DemoModeBanner() {
  const [dismissed, setDismissed] = useState(false);
  const [isDemo, setIsDemo] = useState(isDemoMode);

  // Subscribe to mode changes
  useEffect(() => {
    const unsubscribe = onModeChange((_mode: DataSource) => {
      setIsDemo(isDemoMode());
      // Reset dismissed state when switching to demo mode
      setDismissed(false);
    });
    return unsubscribe;
  }, []);

  // Only show in demo mode
  if (!isDemo || dismissed) {
    return null;
  }

  return (
    <div className="bg-amber-500 text-amber-950 px-4 py-2 flex items-center justify-center gap-2 text-sm font-medium">
      <AlertTriangle className="h-4 w-4 flex-shrink-0" />
      <span>
        <strong>DEMO MODE:</strong> Data is simulated locally. Not using backend
        API or InfluxDB.
      </span>
      <button
        onClick={() => setDismissed(true)}
        className="ml-2 p-1 hover:bg-amber-600/20 rounded"
        aria-label="Dismiss demo mode banner"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  );
}
