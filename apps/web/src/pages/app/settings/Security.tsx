import { useState, useEffect } from "react";
import { NILMPanel } from "@/components/nilm/NILMPanel";
import {
  Key,
  Smartphone,
  History,
  Loader2,
  Eye,
  EyeOff,
  Monitor,
  Globe,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import { formatDistanceToNow, format } from "date-fns";

interface LoginEvent {
  id: string;
  created_at: string;
  device_label: string | null;
  user_agent: string | null;
  ip_address: string | null;
}

function parseUserAgent(ua: string | null): string {
  if (!ua) return "Unknown device";

  // Simple browser detection
  if (ua.includes("Chrome") && !ua.includes("Edg")) return "Chrome";
  if (ua.includes("Firefox")) return "Firefox";
  if (ua.includes("Safari") && !ua.includes("Chrome")) return "Safari";
  if (ua.includes("Edg")) return "Microsoft Edge";
  if (ua.includes("Opera") || ua.includes("OPR")) return "Opera";

  return "Web Browser";
}

export default function Security() {
  const { user } = useAuth();
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [isChangingPassword, setIsChangingPassword] = useState(false);
  const [twoFactorEnabled, setTwoFactorEnabled] = useState(false);
  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [loginHistory, setLoginHistory] = useState<LoginEvent[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [historyUnavailable, setHistoryUnavailable] = useState(false);

  // Fetch login history
  useEffect(() => {
    async function fetchLoginHistory() {
      if (!user) return;

      setLoadingHistory(true);
      try {
        const { data, error } = await supabase
          .from("login_events")
          .select("*")
          .eq("user_id", user.id)
          .order("created_at", { ascending: false })
          .limit(10);

        if (error) throw error;
        setLoginHistory(data || []);
        setHistoryUnavailable(false);
      } catch (err) {
        console.error("Error fetching login history:", err);
        setHistoryUnavailable(true);
        setLoginHistory([]);
      } finally {
        setLoadingHistory(false);
      }
    }

    fetchLoginHistory();
  }, [user]);

  const handlePasswordChange = async (e: React.FormEvent) => {
    e.preventDefault();

    // Validate inputs
    if (!currentPassword || !newPassword || !confirmPassword) {
      toast.error("Please fill in all password fields");
      return;
    }

    if (newPassword !== confirmPassword) {
      toast.error("New passwords do not match");
      return;
    }

    if (newPassword.length < 6) {
      toast.error("New password must be at least 6 characters");
      return;
    }

    if (!user?.email) {
      toast.error("Unable to verify user email");
      return;
    }

    setIsChangingPassword(true);

    try {
      // First, verify the current password by re-authenticating
      const { error: authError } = await supabase.auth.signInWithPassword({
        email: user.email,
        password: currentPassword,
      });

      if (authError) {
        toast.error("Current password is incorrect", {
          description: "Please enter your correct current password",
        });
        setIsChangingPassword(false);
        return;
      }

      // Current password verified, now update to new password
      const { error: updateError } = await supabase.auth.updateUser({
        password: newPassword,
      });

      if (updateError) {
        toast.error("Failed to change password", {
          description: updateError.message,
        });
      } else {
        toast.success("Password changed successfully", {
          description: "Your password has been updated",
        });
        // Clear the form
        setCurrentPassword("");
        setNewPassword("");
        setConfirmPassword("");
      }
    } catch (err) {
      console.error("Password change error:", err);
      toast.error("An unexpected error occurred");
    } finally {
      setIsChangingPassword(false);
    }
  };

  const handleTwoFactorToggle = (enabled: boolean) => {
    setTwoFactorEnabled(enabled);
    if (enabled) {
      toast.info("Two-factor authentication", {
        description:
          "This feature requires additional setup. Contact your administrator.",
      });
    } else {
      toast.info("Two-factor authentication disabled");
    }
  };

  return (
    <div className="space-y-6">
      <NILMPanel
        title="Change Password"
        icon={<Key className="h-5 w-5" />}
        footer="Use a strong, unique password to protect your account"
      >
        <form onSubmit={handlePasswordChange} className="space-y-4 max-w-md">
          <div className="space-y-2">
            <Label htmlFor="current-password">Current Password</Label>
            <div className="relative">
              <Input
                id="current-password"
                type={showCurrentPassword ? "text" : "password"}
                placeholder="••••••••"
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                disabled={isChangingPassword}
                className="pr-10"
              />
              <button
                type="button"
                onClick={() => setShowCurrentPassword(!showCurrentPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              >
                {showCurrentPassword ? (
                  <EyeOff className="h-4 w-4" />
                ) : (
                  <Eye className="h-4 w-4" />
                )}
              </button>
            </div>
          </div>
          <div className="space-y-2">
            <Label htmlFor="new-password">New Password</Label>
            <div className="relative">
              <Input
                id="new-password"
                type={showNewPassword ? "text" : "password"}
                placeholder="••••••••"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                disabled={isChangingPassword}
                className="pr-10"
              />
              <button
                type="button"
                onClick={() => setShowNewPassword(!showNewPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              >
                {showNewPassword ? (
                  <EyeOff className="h-4 w-4" />
                ) : (
                  <Eye className="h-4 w-4" />
                )}
              </button>
            </div>
            <p className="text-xs text-muted-foreground">
              Must be at least 6 characters
            </p>
          </div>
          <div className="space-y-2">
            <Label htmlFor="confirm-password">Confirm New Password</Label>
            <div className="relative">
              <Input
                id="confirm-password"
                type={showConfirmPassword ? "text" : "password"}
                placeholder="••••••••"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                disabled={isChangingPassword}
                className="pr-10"
              />
              <button
                type="button"
                onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              >
                {showConfirmPassword ? (
                  <EyeOff className="h-4 w-4" />
                ) : (
                  <Eye className="h-4 w-4" />
                )}
              </button>
            </div>
          </div>
          <Button type="submit" disabled={isChangingPassword}>
            {isChangingPassword && (
              <Loader2 className="h-4 w-4 animate-spin mr-2" />
            )}
            {isChangingPassword ? "Updating..." : "Update Password"}
          </Button>
        </form>
      </NILMPanel>

      <NILMPanel
        title="Two-Factor Authentication"
        icon={<Smartphone className="h-5 w-5" />}
        footer="Adds an extra layer of security to your account"
      >
        <div className="flex items-center justify-between py-2">
          <div className="space-y-0.5">
            <Label className="text-sm font-medium text-foreground">
              Enable 2FA
            </Label>
            <p className="text-sm text-muted-foreground">
              Use an authenticator app for additional security
            </p>
          </div>
          <Switch
            checked={twoFactorEnabled}
            onCheckedChange={handleTwoFactorToggle}
          />
        </div>
      </NILMPanel>

      <NILMPanel
        title="Login History"
        icon={<History className="h-5 w-5" />}
        footer="Recent login activity for your account"
      >
        {loadingHistory ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : historyUnavailable || loginHistory.length === 0 ? (
          <p className="text-sm text-muted-foreground py-4">
            Login history not available
          </p>
        ) : (
          <div className="space-y-3">
            {loginHistory.map((session, index) => (
              <div
                key={session.id}
                className="flex items-center justify-between py-3 px-4 rounded-lg bg-muted/30"
              >
                <div className="flex items-center gap-3">
                  <Monitor className="h-5 w-5 text-muted-foreground" />
                  <div className="space-y-0.5">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-foreground">
                        {session.device_label ||
                          parseUserAgent(session.user_agent)}
                      </span>
                      {index === 0 && (
                        <Badge
                          variant="outline"
                          className="bg-energy-success/10 text-energy-success border-energy-success/20 text-xs"
                        >
                          Current
                        </Badge>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground flex items-center gap-1">
                      {session.ip_address && (
                        <>
                          <Globe className="h-3 w-3" />
                          <span>{session.ip_address}</span>
                          <span className="mx-1">•</span>
                        </>
                      )}
                      <span>
                        {format(
                          new Date(session.created_at),
                          "MMM d, yyyy h:mm a",
                        )}
                      </span>
                      <span className="mx-1">•</span>
                      <span>
                        {formatDistanceToNow(new Date(session.created_at), {
                          addSuffix: true,
                        })}
                      </span>
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </NILMPanel>
    </div>
  );
}
