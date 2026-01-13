import { Link, useLocation } from "react-router-dom";
import {
  LayoutDashboard,
  Cpu,
  Building2,
  FileText,
  BrainCircuit,
  Settings,
  Users,
  HelpCircle,
  ChevronLeft,
  ChevronRight,
  Menu,
  Home,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { WaveformIcon } from "@/components/brand/WaveformIcon";

const navItems = [
  { label: "Home", icon: Home, path: "/" },
  { label: "Dashboard", icon: LayoutDashboard, path: "/app/dashboard" },
  { label: "Appliances", icon: Cpu, path: "/app/appliances" },
  { label: "Buildings", icon: Building2, path: "/app/buildings" },
  { label: "Reports", icon: FileText, path: "/app/reports" },
  { label: "Model", icon: BrainCircuit, path: "/app/model" },
];

const settingsItems = [
  { label: "Profile", icon: Settings, path: "/app/settings/profile" },
  { label: "Team", icon: Users, path: "/app/settings/users" },
  { label: "Help", icon: HelpCircle, path: "/app/help" },
];

function SidebarContent({
  collapsed,
  onNavigate,
}: {
  collapsed: boolean;
  onNavigate?: () => void;
}) {
  const location = useLocation();

  const isActive = (path: string) => {
    if (path === "/app/appliances") {
      return location.pathname.startsWith("/app/appliances");
    }
    return location.pathname === path;
  };

  return (
    <>
      {/* Logo - links to public home */}
      <div className="flex items-center gap-2 px-4 py-5 border-b border-sidebar-border">
        <Link to="/" className="flex items-center gap-2" onClick={onNavigate}>
          <WaveformIcon size="md" />
          {!collapsed && (
            <span className="font-semibold text-lg text-sidebar-foreground">
              Energy Monitor
            </span>
          )}
        </Link>
      </div>

      {/* Main Navigation */}
      <nav className="flex-1 px-2 py-4 space-y-1">
        {navItems.map((item) => {
          const active = isActive(item.path);
          return (
            <Link
              key={item.path}
              to={item.path}
              onClick={onNavigate}
              className={cn(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200",
                active
                  ? "bg-sidebar-primary text-sidebar-primary-foreground"
                  : "text-sidebar-foreground/70 hover:text-sidebar-foreground hover:bg-sidebar-accent",
              )}
            >
              <item.icon className="h-5 w-5 shrink-0" />
              {!collapsed && <span>{item.label}</span>}
            </Link>
          );
        })}
      </nav>

      {/* Settings */}
      <div className="px-2 py-4 border-t border-sidebar-border space-y-1">
        {settingsItems.map((item) => {
          const active = location.pathname === item.path;
          return (
            <Link
              key={item.path}
              to={item.path}
              onClick={onNavigate}
              className={cn(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200",
                active
                  ? "bg-sidebar-primary text-sidebar-primary-foreground"
                  : "text-sidebar-foreground/70 hover:text-sidebar-foreground hover:bg-sidebar-accent",
              )}
            >
              <item.icon className="h-5 w-5 shrink-0" />
              {!collapsed && <span>{item.label}</span>}
            </Link>
          );
        })}
      </div>

      {/* NILM Info Footer */}
      {!collapsed && (
        <div className="px-4 py-3 border-t border-sidebar-border">
          <p className="text-[10px] text-sidebar-foreground/40 leading-relaxed">
            NILM: Non-Intrusive Load Monitoring
          </p>
        </div>
      )}
    </>
  );
}

export function AppSidebar() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={cn(
        "hidden md:flex flex-col bg-sidebar text-sidebar-foreground border-r border-sidebar-border transition-all duration-300",
        collapsed ? "w-16" : "w-60",
      )}
    >
      <SidebarContent collapsed={collapsed} />

      {/* Collapse Toggle */}
      <div className="p-2 border-t border-sidebar-border">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setCollapsed(!collapsed)}
          className="w-full justify-center text-sidebar-foreground/70 hover:text-sidebar-foreground hover:bg-sidebar-accent"
        >
          {collapsed ? (
            <ChevronRight className="h-4 w-4" />
          ) : (
            <ChevronLeft className="h-4 w-4" />
          )}
        </Button>
      </div>
    </aside>
  );
}

export function MobileSidebar() {
  const [open, setOpen] = useState(false);

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="md:hidden"
          aria-label="Open menu"
        >
          <Menu className="h-5 w-5" />
        </Button>
      </SheetTrigger>
      <SheetContent
        side="left"
        className="w-64 p-0 bg-sidebar text-sidebar-foreground border-sidebar-border"
      >
        <div className="flex flex-col h-full">
          <SidebarContent collapsed={false} onNavigate={() => setOpen(false)} />
        </div>
      </SheetContent>
    </Sheet>
  );
}
