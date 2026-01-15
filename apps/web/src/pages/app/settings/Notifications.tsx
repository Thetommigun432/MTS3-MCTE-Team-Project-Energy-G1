import { NILMPanel } from "@/components/nilm/NILMPanel";
import { Mail, Smartphone, Zap } from "lucide-react";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { useState } from "react";

interface NotificationSetting {
  id: string;
  label: string;
  description: string;
  enabled: boolean;
}

export default function Notifications() {
  const [emailSettings, setEmailSettings] = useState<NotificationSetting[]>([
    {
      id: "daily-summary",
      label: "Daily Summary",
      description: "Receive a daily email with energy usage highlights",
      enabled: true,
    },
    {
      id: "weekly-report",
      label: "Weekly Report",
      description: "Get a detailed weekly energy report",
      enabled: true,
    },
    {
      id: "anomaly-alerts",
      label: "Anomaly Alerts",
      description: "Be notified when unusual consumption patterns are detected",
      enabled: false,
    },
  ]);

  const [pushSettings, setPushSettings] = useState<NotificationSetting[]>([
    {
      id: "high-usage",
      label: "High Usage Warning",
      description: "Alert when consumption exceeds threshold",
      enabled: true,
    },
    {
      id: "appliance-on",
      label: "Appliance Left On",
      description: "Notify when appliances are on for too long",
      enabled: false,
    },
    {
      id: "peak-hours",
      label: "Peak Hours Reminder",
      description: "Remind before peak pricing hours",
      enabled: true,
    },
  ]);

  const toggleSetting = (
    settings: NotificationSetting[],
    setSettings: React.Dispatch<React.SetStateAction<NotificationSetting[]>>,
    id: string,
  ) => {
    setSettings(
      settings.map((s) => (s.id === id ? { ...s, enabled: !s.enabled } : s)),
    );
  };

  return (
    <div className="space-y-6">
      <NILMPanel
        title="Email Notifications"
        icon={<Mail className="h-5 w-5" />}
        footer="Email notifications are sent to your registered email address"
      >
        <div className="space-y-4">
          {emailSettings.map((setting) => (
            <div
              key={setting.id}
              className="flex items-start justify-between gap-4 py-2"
            >
              <div className="space-y-0.5">
                <Label
                  htmlFor={setting.id}
                  className="text-sm font-medium text-foreground cursor-pointer"
                >
                  {setting.label}
                </Label>
                <p className="text-sm text-muted-foreground">
                  {setting.description}
                </p>
              </div>
              <Switch
                id={setting.id}
                checked={setting.enabled}
                onCheckedChange={() =>
                  toggleSetting(emailSettings, setEmailSettings, setting.id)
                }
              />
            </div>
          ))}
        </div>
      </NILMPanel>

      <NILMPanel
        title="Push Notifications"
        icon={<Smartphone className="h-5 w-5" />}
        footer="Push notifications require browser permissions"
      >
        <div className="space-y-4">
          {pushSettings.map((setting) => (
            <div
              key={setting.id}
              className="flex items-start justify-between gap-4 py-2"
            >
              <div className="space-y-0.5">
                <Label
                  htmlFor={setting.id}
                  className="text-sm font-medium text-foreground cursor-pointer"
                >
                  {setting.label}
                </Label>
                <p className="text-sm text-muted-foreground">
                  {setting.description}
                </p>
              </div>
              <Switch
                id={setting.id}
                checked={setting.enabled}
                onCheckedChange={() =>
                  toggleSetting(pushSettings, setPushSettings, setting.id)
                }
              />
            </div>
          ))}
        </div>
      </NILMPanel>

      <NILMPanel title="Real-time Alerts" icon={<Zap className="h-5 w-5" />}>
        <div className="flex items-center justify-between py-2">
          <div className="space-y-0.5">
            <Label className="text-sm font-medium text-foreground">
              Enable Real-time Monitoring
            </Label>
            <p className="text-sm text-muted-foreground">
              Get instant alerts for critical energy events
            </p>
          </div>
          <Switch defaultChecked />
        </div>
      </NILMPanel>
    </div>
  );
}
