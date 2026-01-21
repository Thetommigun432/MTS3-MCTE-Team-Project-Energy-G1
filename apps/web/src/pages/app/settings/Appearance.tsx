import { NILMPanel } from "@/components/nilm/NILMPanel";
import { Palette, Sun, Moon, Monitor, Eye } from "lucide-react";
import { useTheme } from "@/contexts/ThemeContext";
import { useAppearance } from "@/contexts/AppearanceContext";
import { cn } from "@/lib/utils";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";

const themeOptions = [
  {
    value: "light",
    label: "Light",
    icon: Sun,
    description: "Bright interface for daytime use",
  },
  {
    value: "dark",
    label: "Dark",
    icon: Moon,
    description: "Easy on the eyes in low light",
  },
  {
    value: "system",
    label: "System",
    icon: Monitor,
    description: "Follows your device settings",
  },
] as const;

export default function Appearance() {
  const { theme, setTheme } = useTheme();
  const { settings, setHighContrast, setCompactMode, setShowAnimations } =
    useAppearance();

  return (
    <div className="space-y-6">
      <NILMPanel
        title="Theme"
        icon={<Palette className="h-5 w-5" />}
        footer="Choose how Energy Monitor looks to you"
      >
        <div className="grid gap-3 sm:grid-cols-3">
          {themeOptions.map((option) => {
            const isActive = theme === option.value;
            return (
              <button
                key={option.value}
                onClick={() => setTheme(option.value)}
                className={cn(
                  "flex flex-col items-center gap-3 p-4 rounded-lg border-2 transition-all text-left",
                  isActive
                    ? "border-primary bg-primary/5"
                    : "border-transparent bg-muted/30 hover:bg-muted/50",
                )}
              >
                <div
                  className={cn(
                    "p-3 rounded-full",
                    isActive
                      ? "bg-primary/10 text-primary"
                      : "bg-muted text-muted-foreground",
                  )}
                >
                  <option.icon className="h-5 w-5" />
                </div>
                <div className="text-center">
                  <p
                    className={cn(
                      "font-medium text-sm",
                      isActive ? "text-primary" : "text-foreground",
                    )}
                  >
                    {option.label}
                  </p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {option.description}
                  </p>
                </div>
              </button>
            );
          })}
        </div>
      </NILMPanel>

      <NILMPanel
        title="Accessibility"
        icon={<Eye className="h-5 w-5" />}
        footer="Adjust visual settings for better accessibility"
      >
        <div className="space-y-4">
          <div className="flex items-center justify-between py-2">
            <div className="space-y-0.5">
              <Label className="text-sm font-medium text-foreground">
                High Contrast
              </Label>
              <p className="text-sm text-muted-foreground">
                Increase contrast for better visibility (WCAG-friendly)
              </p>
            </div>
            <Switch
              checked={settings.highContrast}
              onCheckedChange={setHighContrast}
            />
          </div>
        </div>
      </NILMPanel>

      <NILMPanel
        title="Display Options"
        footer="Customize the interface layout and behavior"
      >
        <div className="space-y-4">
          <div className="flex items-center justify-between py-2">
            <div className="space-y-0.5">
              <Label className="text-sm font-medium text-foreground">
                Compact Mode
              </Label>
              <p className="text-sm text-muted-foreground">
                Reduce spacing for more data density
              </p>
            </div>
            <Switch
              checked={settings.compactMode}
              onCheckedChange={setCompactMode}
            />
          </div>
          <div className="flex items-center justify-between py-2">
            <div className="space-y-0.5">
              <Label className="text-sm font-medium text-foreground">
                Show Animations
              </Label>
              <p className="text-sm text-muted-foreground">
                Enable chart and UI animations
              </p>
            </div>
            <Switch
              checked={settings.showAnimations}
              onCheckedChange={setShowAnimations}
            />
          </div>
        </div>
      </NILMPanel>
    </div>
  );
}
