import {
  Calendar,
  ChevronDown,
  LogOut,
  Settings,
  Filter,
  Plus,
  Building2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ThemeToggleCompact } from "@/components/theme/ThemeToggle";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { useEnergy } from "@/contexts/EnergyContext";
import { useAuth } from "@/contexts/AuthContext";
import { useNavigate, useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";
import { MobileSidebar } from "./AppSidebar";
import {
  formatDateForInput,
  parseLocalDate,
  parseLocalDateEnd,
} from "@/lib/dateUtils";
import { useSignedAvatarUrl } from "@/hooks/useSignedAvatarUrl";

export function TopBar() {
  const {
    mode,
    setMode,
    selectedBuildingId,
    setSelectedBuildingId,
    selectedAppliance,
    setSelectedAppliance,
    buildings,
    appliances,
    dateRange,
    setDateRange,
    dataDateRange,
    loading,
  } = useEnergy();
  const { user, profile, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  // Only show filters on the dashboard page
  const showFilters = location.pathname === "/app" || location.pathname === "/app/dashboard";

  const handleLogout = async () => {
    await logout();
    navigate("/login");
  };

  const setPreset = (days: number) => {
    if (!dataDateRange) return;
    const end = dataDateRange.max;
    const start = new Date(end);
    start.setDate(start.getDate() - days + 1);
    start.setHours(0, 0, 0, 0);
    if (start < dataDateRange.min) {
      setDateRange({ start: dataDateRange.min, end });
    } else {
      setDateRange({ start, end });
    }
  };

  const handleStartDateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    if (!val) return;
    setDateRange({ ...dateRange, start: parseLocalDate(val) });
  };

  const handleEndDateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    if (!val) return;
    // Use parseLocalDateEnd to include the full end date
    setDateRange({ ...dateRange, end: parseLocalDateEnd(val) });
  };

  const displayName =
    profile?.display_name || user?.email?.split("@")[0] || "User";
  const signedAvatarUrl = useSignedAvatarUrl(profile?.avatar_url);

  return (
    <header className="flex items-center justify-between gap-4 px-4 md:px-6 py-3 bg-background border-b border-border">
      {/* Left: Mobile menu + Building + Appliance + Date Range (Dashboard only) */}
      <div className="flex items-center gap-3 flex-wrap">
        <MobileSidebar />

        {/* Filters - Only shown on Dashboard */}
        {showFilters && (
          <>
            {/* Building Selector */}
            {mode === "api" && buildings.length === 0 ? (
              <Button
                size="sm"
                variant="outline"
                onClick={() => navigate("/app/buildings")}
                className="gap-2"
              >
                <Plus className="h-4 w-4" />
                Create Building
              </Button>
            ) : (
              <Select
                value={selectedBuildingId || ""}
                onValueChange={(value) => {
                  setSelectedBuildingId(value);
                  // Automatically switch to API mode when building selected (if not already)
                  if (value && value !== "demo" && mode !== "api") {
                    setMode("api");
                  }
                }}
                disabled={mode === "demo"}
              >
                <SelectTrigger className={cn(
                  "w-36 md:w-48 border-border bg-background text-foreground",
                  mode === "demo" && "opacity-70 cursor-not-allowed"
                )}>
                  <Building2 className="h-4 w-4 mr-2 text-muted-foreground" />
                  <SelectValue placeholder="Select Building" />
                </SelectTrigger>
                <SelectContent>
                  {buildings.map((building) => (
                    <SelectItem key={building.id} value={building.id}>
                      <div className="flex flex-col">
                        <span>{building.name}</span>
                        {building.address && (
                          <span className="text-xs text-muted-foreground">{building.address}</span>
                        )}
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}

            {/* Appliance Filter */}
            <Select
              value={selectedAppliance}
              onValueChange={setSelectedAppliance}
              disabled={loading}
            >
              <SelectTrigger className="w-36 md:w-44 border-border bg-background text-foreground">
                <Filter className="h-4 w-4 mr-2 text-muted-foreground" />
                <SelectValue placeholder="Appliance" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="All">All Appliances</SelectItem>
                {appliances.map((appliance) => (
                  <SelectItem key={appliance} value={appliance}>
                    {appliance.replace(/_/g, " ")}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {/* Date Range - Hidden on small screens */}
            <div className="hidden lg:flex items-center gap-2">
              <Calendar className="h-4 w-4 text-muted-foreground" />
              <Input
                type="date"
                value={formatDateForInput(dateRange.start)}
                onChange={handleStartDateChange}
                className="w-36 border-border bg-background text-foreground"
                disabled={loading}
              />
              <span className="text-muted-foreground">to</span>
              <Input
                type="date"
                value={formatDateForInput(dateRange.end)}
                onChange={handleEndDateChange}
                className="w-36 border-border bg-background text-foreground"
                disabled={loading}
              />
            </div>

            {/* Preset Buttons */}
            <div className="hidden md:flex gap-1">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPreset(1)}
                disabled={loading}
                className="border-border text-foreground hover:bg-muted"
              >
                Today
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPreset(7)}
                disabled={loading}
                className="border-border text-foreground hover:bg-muted"
              >
                7 days
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPreset(30)}
                disabled={loading}
                className="border-border text-foreground hover:bg-muted"
              >
                30 days
              </Button>
            </div>
          </>
        )}
      </div>

      {/* Mode-specific help text */}
      <div className="hidden xl:flex items-center text-xs text-muted-foreground">
        {mode === "demo" && (
          <span>
            Viewing sample data •{" "}
            <button
              onClick={() => setMode("api")}
              className="text-primary hover:underline"
            >
              Switch to live data
            </button>
          </span>
        )}
        {mode === "api" && !selectedBuildingId && (
          <span className="text-amber-600 dark:text-amber-400">
            Select a building to view live data
          </span>
        )}
      </div>

      {/* Right: Theme + Mode Toggle + User Menu */}
      <div className="flex items-center gap-2 md:gap-3">
        {/* Theme Toggle */}
        <ThemeToggleCompact />

        {/* Mode Toggle */}
        <div className="inline-flex rounded-lg bg-popover border border-border p-0.5">
          <button
            onClick={() => setMode("demo")}
            disabled={loading}
            className={cn(
              "rounded-md px-2 md:px-3 py-1.5 text-xs md:text-sm font-medium transition-all focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
              mode === "demo"
                ? "bg-primary text-primary-foreground shadow-xs"
                : "text-muted-foreground hover:text-foreground hover:bg-muted",
            )}
          >
            Demo
          </button>
          <button
            onClick={() => setMode("api")}
            disabled={loading}
            className={cn(
              "rounded-md px-2 md:px-3 py-1.5 text-xs md:text-sm font-medium transition-all focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
              mode === "api"
                ? "bg-primary text-primary-foreground shadow-xs"
                : "text-muted-foreground hover:text-foreground hover:bg-muted",
            )}
          >
            API
          </button>
        </div>

        {/* User Menu */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              size="sm"
              className="flex items-center gap-1 md:gap-2 text-foreground hover:bg-muted"
            >
              <Avatar className="h-8 w-8">
                <AvatarImage src={signedAvatarUrl || undefined} />
                <AvatarFallback className="bg-primary text-primary-foreground text-sm font-medium">
                  {displayName.charAt(0).toUpperCase()}
                </AvatarFallback>
              </Avatar>
              <ChevronDown className="h-4 w-4 hidden md:block text-muted-foreground" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            align="end"
            className="w-48 bg-popover border-border"
          >
            <div className="px-2 py-1.5">
              <p className="text-sm font-medium truncate text-foreground">
                {displayName}
              </p>
              <p className="text-xs text-muted-foreground truncate font-mono">
                {user?.email}
              </p>
            </div>
            <DropdownMenuSeparator className="bg-border" />
            <DropdownMenuItem
              onClick={() => navigate("/app/settings/profile")}
              className="text-foreground focus:bg-muted"
            >
              <Settings className="mr-2 h-4 w-4" />
              Settings
            </DropdownMenuItem>
            <DropdownMenuItem
              onClick={handleLogout}
              className="text-foreground focus:bg-muted"
            >
              <LogOut className="mr-2 h-4 w-4" />
              Logout
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  );
}
