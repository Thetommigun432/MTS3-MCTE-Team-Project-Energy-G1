import { Calendar, Filter, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useEnergy } from "@/contexts/EnergyContext";
import {
  formatDateForInput,
  parseLocalDate,
  parseLocalDateEnd,
} from "@/lib/dateUtils";
import { cn } from "@/lib/utils";

interface PageFiltersProps {
  showBuilding?: boolean;
  showAppliance?: boolean;
  showDateRange?: boolean;
  className?: string;
}

export function PageFilters({
  showBuilding = false,
  showAppliance = true,
  showDateRange = true,
  className,
}: PageFiltersProps) {
  const {
    selectedBuilding,
    setSelectedBuilding,
    selectedAppliance,
    setSelectedAppliance,
    buildings,
    appliances,
    dateRange,
    setDateRange,
    dataDateRange,
    loading,
  } = useEnergy();

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
    setDateRange({ ...dateRange, end: parseLocalDateEnd(val) });
  };

  return (
    <div
      className={cn(
        "flex flex-wrap items-center gap-3 p-4 bg-card rounded-lg border border-border",
        className,
      )}
    >
      <span className="text-sm font-medium text-muted-foreground flex items-center gap-1.5">
        <Filter className="h-4 w-4" />
        Filters
      </span>

      {showBuilding && (
        <Tooltip>
          <TooltipTrigger asChild>
            <div>
              <Select
                value={selectedBuilding}
                onValueChange={setSelectedBuilding}
                disabled
              >
                <SelectTrigger className="w-40 border-border bg-background text-foreground opacity-70 cursor-not-allowed">
                  <SelectValue placeholder="Building" />
                </SelectTrigger>
                <SelectContent>
                  {buildings.map((building) => (
                    <SelectItem key={building.id} value={building.id}>
                      {building.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </TooltipTrigger>
          <TooltipContent>
            <p className="flex items-center gap-1">
              <Info className="h-3 w-3" />
              Demo uses single building dataset
            </p>
          </TooltipContent>
        </Tooltip>
      )}

      {showAppliance && (
        <Select
          value={selectedAppliance}
          onValueChange={setSelectedAppliance}
          disabled={loading}
        >
          <SelectTrigger className="w-40 border-border bg-background text-foreground">
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
      )}

      {showDateRange && (
        <>
          <div className="flex items-center gap-2">
            <Calendar className="h-4 w-4 text-muted-foreground" />
            <Input
              type="date"
              value={formatDateForInput(dateRange.start)}
              onChange={handleStartDateChange}
              className="w-36 border-border bg-background text-foreground"
              disabled={loading}
            />
            <span className="text-muted-foreground text-sm">to</span>
            <Input
              type="date"
              value={formatDateForInput(dateRange.end)}
              onChange={handleEndDateChange}
              className="w-36 border-border bg-background text-foreground"
              disabled={loading}
            />
          </div>

          <div className="flex gap-1">
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
  );
}
