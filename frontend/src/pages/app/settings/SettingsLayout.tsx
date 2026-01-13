import { NavLink, Outlet, useLocation } from "react-router-dom";
import { User, Users, Bell, Shield, Palette } from "lucide-react";
import { cn } from "@/lib/utils";

const settingsNav = [
  { label: "Profile", path: "/app/settings/profile", icon: User },
  { label: "Team", path: "/app/settings/users", icon: Users },
  { label: "Notifications", path: "/app/settings/notifications", icon: Bell },
  { label: "Security", path: "/app/settings/security", icon: Shield },
  { label: "Appearance", path: "/app/settings/appearance", icon: Palette },
];

export default function SettingsLayout() {
  const location = useLocation();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Settings</h1>
        <p className="text-muted-foreground mt-1">
          Manage your account and preferences
        </p>
      </div>

      <div className="flex flex-col md:flex-row gap-6">
        {/* Settings Navigation */}
        <nav className="w-full md:w-56 shrink-0">
          <div className="flex md:flex-col gap-1 overflow-x-auto md:overflow-visible pb-2 md:pb-0">
            {settingsNav.map((item) => {
              const isActive = location.pathname === item.path;
              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={cn(
                    "flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors whitespace-nowrap",
                    isActive
                      ? "bg-primary/10 text-primary"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted/50",
                  )}
                >
                  <item.icon className="h-4 w-4 shrink-0" />
                  {item.label}
                </NavLink>
              );
            })}
          </div>
        </nav>

        {/* Settings Content */}
        <div className="flex-1 min-w-0">
          <Outlet />
        </div>
      </div>
    </div>
  );
}
