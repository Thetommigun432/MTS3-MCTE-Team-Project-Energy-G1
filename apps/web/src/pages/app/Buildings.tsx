import { useState, useEffect, useCallback, useRef } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Plus,
  Building2,
  MapPin,
  Pencil,
  Trash2,
  Zap,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import { useEnergy } from "@/contexts/EnergyContext";
import { isSupabaseEnabled } from "@/lib/env";
import { toast } from "sonner";

// Track schema warnings (session-level)
let schemaWarningLogged = false;

/**
 * Check if error is a Supabase schema/table missing error
 */
function isSchemaError(err: unknown): boolean {
  if (!err || typeof err !== 'object') return false;
  const error = err as { code?: string; message?: string };
  return ['PGRST205', '42703', '42P01', 'PGRST200'].includes(error.code || '') ||
    (error.message?.includes('does not exist') ?? false);
}

interface Building {
  id: string;
  name: string;
  address: string | null;
  description: string | null;
  status: string;
  total_appliances: number;
  created_at: string;
  updated_at: string;
}

interface Appliance {
  id: string; // building_appliance_id
  building_id: string;
  org_appliance_id: string;
  name: string;
  type: string;
  rated_power_kw: number | null;
  status: string; // derived from is_enabled
  notes: string | null; // not in schema but kept for interface/state compat if needed
  created_at: string;
  updated_at: string;
}

interface OrgAppliance {
  id: string;
  name: string;
  type: string;
  rated_power_kw: number | null;
  slug: string;
}

const APPLIANCE_TYPES = [
  { value: "hvac", label: "HVAC" },
  { value: "lighting", label: "Lighting" },
  { value: "refrigeration", label: "Refrigeration" },
  { value: "computing", label: "Computing" },
  { value: "kitchen", label: "Kitchen" },
  { value: "laundry", label: "Laundry" },
  { value: "water_heater", label: "Water Heater" },
  { value: "ev_charger", label: "EV Charger" },
  { value: "other", label: "Other" },
];

export default function Buildings() {
  const { user } = useAuth();
  const { mode, buildings: contextBuildings, appliances: contextAppliances } = useEnergy();
  const [buildings, setBuildings] = useState<Building[]>([]);
  const [appliances, setAppliances] = useState<Record<string, Appliance[]>>({});
  const [orgAppliances, setOrgAppliances] = useState<OrgAppliance[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedBuildings, setExpandedBuildings] = useState<Set<string>>(
    new Set(),
  );
  // Track if Supabase schema is unavailable (don't retry)
  const schemaUnavailable = useRef(false);

  // Building dialogs
  const [isAddBuildingOpen, setIsAddBuildingOpen] = useState(false);
  const [isEditBuildingOpen, setIsEditBuildingOpen] = useState(false);
  const [editingBuilding, setEditingBuilding] = useState<Building | null>(null);
  const [buildingForm, setBuildingForm] = useState({
    name: "",
    address: "",
    description: "",
    status: "active" as "active" | "inactive" | "maintenance",
  });

  // Appliance dialogs
  const [isAddApplianceOpen, setIsAddApplianceOpen] = useState(false);
  const [isEditApplianceOpen, setIsEditApplianceOpen] = useState(false);
  const [selectedBuildingId, setSelectedBuildingId] = useState<string | null>(
    null,
  );
  const [editingAppliance, setEditingAppliance] = useState<Appliance | null>(
    null,
  );
  const [applianceForm, setApplianceForm] = useState({
    org_appliance_id: "",
    status: "active" as "active" | "inactive",
  });

  const [saving, setSaving] = useState(false);

  // Fetch available organization appliances
  const fetchOrgAppliances = useCallback(async () => {
    if (!user || mode === "demo") return;
    try {
      // Using 'appliances' table (the actual table name in the schema)
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data, error } = await (supabase as any)
        .from("appliances")
        .select("id, name, category, typical_power_kw, is_enabled")
        .eq("is_enabled", true)
        .order("name");

      if (error) {
        if (isSchemaError(error)) {
          if (!schemaWarningLogged) {
            console.warn("[Buildings] appliances table not available. Using empty state.");
            schemaWarningLogged = true;
          }
          return;
        }
        throw error;
      }
      // Transform to match expected OrgAppliance shape
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const transformed = (data || []).map((item: any) => ({
        id: item.id,
        name: item.name,
        type: item.category || "appliance",
        rated_power_kw: item.typical_power_kw,
      }));
      setOrgAppliances(transformed);
    } catch (error) {
      console.error("Error fetching org appliances:", error);
    }
  }, [user, mode]);

  useEffect(() => {
    fetchOrgAppliances();
  }, [fetchOrgAppliances]);

  const fetchBuildings = useCallback(async () => {
    try {
      setLoading(true);

      // In demo mode, show demo building from context
      if (mode === "demo") {
        const demoBuildings = contextBuildings.map(b => ({
          id: b.id,
          name: b.name,
          address: b.address || null,
          description: "Training dataset from residential building with 13 appliances",
          status: b.status || "active",
          total_appliances: contextAppliances.length,
          created_at: "2024-10-20T00:00:00Z",
          updated_at: "2025-10-20T00:00:00Z"
        }));
        setBuildings(demoBuildings);
        setLoading(false);
        return;
      }

      // In API mode, fetch from Supabase
      if (!user || !isSupabaseEnabled() || schemaUnavailable.current) {
        // In API mode without Supabase access, show empty state (not demo data)
        setBuildings([]);
        setLoading(false);
        return;
      }

      const { data, error } = await supabase
        .from("buildings")
        .select("*")
        .neq("is_demo", true)
        .order("created_at", { ascending: false });

      if (error) {
        if (isSchemaError(error)) {
          schemaUnavailable.current = true;
          if (!schemaWarningLogged) {
            console.warn("[Buildings] Supabase buildings table not available.");
            schemaWarningLogged = true;
          }
          // In API mode, show empty state rather than fallback to demo
          setBuildings([]);
          setLoading(false);
          return;
        }
        throw error;
      }
      setBuildings(data || []);
    } catch (error) {
      console.error("Error fetching buildings:", error);
      toast.error("Failed to load buildings");
    } finally {
      setLoading(false);
    }
  }, [user, mode, contextBuildings, contextAppliances]);

  const fetchAppliances = async (buildingId: string) => {
    try {
      // In demo mode, show demo appliances from context
      if (mode === "demo") {
        const demoAppliances = contextAppliances.map((name, index) => ({
          id: `demo-appliance-${index}`,
          building_id: buildingId,
          org_appliance_id: `demo-org-app-${index}`,
          name: name,
          type: "other",
          rated_power_kw: null as number | null,
          status: "active",
          notes: "From training dataset",
          created_at: "2024-10-20T00:00:00Z",
          updated_at: "2025-10-20T00:00:00Z"
        }));
        setAppliances((prev) => ({ ...prev, [buildingId]: demoAppliances }));
        return;
      }

      // In API mode, fetch from Supabase using actual schema
      // Join building_appliances with appliances table
      const { data, error } = await supabase
        .from("building_appliances")
        .select(`
          id,
          building_id,
          alias,
          is_active,
          created_at,
          appliances!inner (
            id,
            name,
            category,
            typical_power_kw
          )
        `)
        .eq("building_id", buildingId)
        .eq("is_active", true);

      if (error) throw error;

      // Transform result to match expected Appliance shape
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const transformed: Appliance[] = (data || []).map((item: any) => ({
        id: item.id,
        building_id: item.building_id,
        org_appliance_id: item.appliances?.id,
        name: item.alias || item.appliances?.name || "Unknown",
        type: item.appliances?.category || "other",
        rated_power_kw: item.appliances?.typical_power_kw,
        status: item.is_active ? "active" : "inactive",
        notes: "", // Default to empty string for notes
        created_at: item.created_at,
        updated_at: item.created_at // Use created_at as updated_at fallback
      })).sort((a, b) => a.name.localeCompare(b.name));

      setAppliances((prev) => ({ ...prev, [buildingId]: transformed }));
    } catch (error) {
      console.error("Error fetching appliances:", error);
    }
  };

  useEffect(() => {
    fetchBuildings();
  }, [fetchBuildings]);

  const toggleBuilding = (buildingId: string) => {
    const newExpanded = new Set(expandedBuildings);
    if (newExpanded.has(buildingId)) {
      newExpanded.delete(buildingId);
    } else {
      newExpanded.add(buildingId);
      if (!appliances[buildingId]) {
        fetchAppliances(buildingId);
      }
    }
    setExpandedBuildings(newExpanded);
  };

  // Building CRUD
  const handleAddBuilding = async () => {
    const trimmedName = buildingForm.name.trim();
    const trimmedAddress = buildingForm.address.trim();
    const trimmedDescription = buildingForm.description.trim();

    // Client-side validation matching database constraints
    if (!user || !trimmedName) {
      toast.error("Building name is required");
      return;
    }
    if (trimmedName.length > 200) {
      toast.error("Building name must be 200 characters or less");
      return;
    }
    if (trimmedAddress.length > 500) {
      toast.error("Address must be 500 characters or less");
      return;
    }
    if (trimmedDescription.length > 1000) {
      toast.error("Description must be 1000 characters or less");
      return;
    }

    try {
      setSaving(true);
      const { error } = await supabase.from("buildings").insert({
        user_id: user.id,
        name: trimmedName,
        address: trimmedAddress || null,
        description: trimmedDescription || null,
        status: buildingForm.status,
      });

      if (error) throw error;
      toast.success("Building added successfully");
      setIsAddBuildingOpen(false);
      resetBuildingForm();
      fetchBuildings();
    } catch (error) {
      console.error("Error adding building:", error);
      toast.error("Failed to add building");
    } finally {
      setSaving(false);
    }
  };

  const handleEditBuilding = async () => {
    const trimmedName = buildingForm.name.trim();
    const trimmedAddress = buildingForm.address.trim();
    const trimmedDescription = buildingForm.description.trim();

    // Client-side validation matching database constraints
    if (!editingBuilding || !trimmedName) {
      toast.error("Building name is required");
      return;
    }
    if (trimmedName.length > 200) {
      toast.error("Building name must be 200 characters or less");
      return;
    }
    if (trimmedAddress.length > 500) {
      toast.error("Address must be 500 characters or less");
      return;
    }
    if (trimmedDescription.length > 1000) {
      toast.error("Description must be 1000 characters or less");
      return;
    }

    try {
      setSaving(true);
      const { error } = await supabase
        .from("buildings")
        .update({
          name: trimmedName,
          address: trimmedAddress || null,
          description: trimmedDescription || null,
          status: buildingForm.status,
        })
        .eq("id", editingBuilding.id);

      if (error) throw error;
      toast.success("Building updated");
      setIsEditBuildingOpen(false);
      setEditingBuilding(null);
      resetBuildingForm();
      fetchBuildings();
    } catch (error) {
      console.error("Error updating building:", error);
      toast.error("Failed to update building");
    } finally {
      setSaving(false);
    }
  };

  const handleDeleteBuilding = async (id: string) => {
    try {
      const { error } = await supabase.from("buildings").delete().eq("id", id);
      if (error) throw error;
      toast.success("Building deleted");
      fetchBuildings();
    } catch (error) {
      console.error("Error deleting building:", error);
      toast.error("Failed to delete building");
    }
  };

  const openEditBuildingDialog = (building: Building) => {
    setEditingBuilding(building);
    setBuildingForm({
      name: building.name,
      address: building.address || "",
      description: building.description || "",
      status: building.status as "active" | "inactive" | "maintenance",
    });
    setIsEditBuildingOpen(true);
  };

  const resetBuildingForm = () => {
    setBuildingForm({
      name: "",
      address: "",
      description: "",
      status: "active",
    });
  };

  // Appliance CRUD (Refactored to select from Org Appliances)
  const handleAddAppliance = async () => {
    if (!user || !selectedBuildingId || !applianceForm.org_appliance_id) {
      toast.error("Please select an appliance");
      return;
    }

    try {
      setSaving(true);
      // Insert into building_appliances linking building and appliance
      // Using actual schema: appliance_id (not org_appliance_id), is_active (not is_enabled)
      const { error } = await supabase.from("building_appliances").insert({
        building_id: selectedBuildingId,
        appliance_id: applianceForm.org_appliance_id, // Maps to appliances.id
        is_active: applianceForm.status === "active",
        alias: applianceForm.name || null, // Optional custom name
      });

      if (error) throw error;
      toast.success("Appliance added");
      setIsAddApplianceOpen(false);
      resetApplianceForm();
      fetchAppliances(selectedBuildingId);
    } catch (error) {
      console.error("Error adding appliance:", error);
      toast.error("Failed to add appliance");
    } finally {
      setSaving(false);
    }
  };

  const handleEditAppliance = async () => {
    // Editing only allows changing enablement status now, since properties are on appliance
    if (!editingAppliance) return;

    try {
      setSaving(true);
      const { error } = await supabase
        .from("building_appliances")
        .update({
          is_active: applianceForm.status === "active",
          alias: applianceForm.name || null,
        })
        .eq("id", editingAppliance.id);

      if (error) throw error;
      toast.success("Appliance updated");
      setIsEditApplianceOpen(false);
      if (editingAppliance.building_id) {
        fetchAppliances(editingAppliance.building_id);
      }
      setEditingAppliance(null);
      resetApplianceForm();
    } catch (error) {
      console.error("Error updating appliance:", error);
      toast.error("Failed to update appliance");
    } finally {
      setSaving(false);
    }
  };

  const handleDeleteAppliance = async (appliance: Appliance) => {
    try {
      // Deleting means hard delete from link table
      const { error } = await supabase
        .from("building_appliances")
        .delete()
        .eq("id", appliance.id);
      if (error) throw error;
      toast.success("Appliance removed");
      fetchAppliances(appliance.building_id);
    } catch (error) {
      console.error("Error deleting appliance:", error);
      toast.error("Failed to delete appliance");
    }
  };

  const openAddApplianceDialog = (buildingId: string) => {
    setSelectedBuildingId(buildingId);
    setIsAddApplianceOpen(true);
  };

  const openEditApplianceDialog = (appliance: Appliance) => {
    setEditingAppliance(appliance);
    setApplianceForm({
      org_appliance_id: appliance.org_appliance_id,
      status: appliance.status as "active" | "inactive",
    });
    setIsEditApplianceOpen(true);
  };

  const resetApplianceForm = () => {
    setApplianceForm({
      org_appliance_id: "",
      status: "active",
    });
    setSelectedBuildingId(null);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "active":
        return "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400";
      case "inactive":
        return "bg-muted text-muted-foreground";
      case "maintenance":
      case "unknown":
        return "bg-amber-500/15 text-amber-600 dark:text-amber-400";
      default:
        return "bg-muted text-muted-foreground";
    }
  };

  const getTypeLabel = (type: string) => {
    return APPLIANCE_TYPES.find((t) => t.value === type)?.label || type;
  };

  // Check if a building is a demo building
  const isDemoBuilding = (buildingId: string) => {
    return buildingId.startsWith("demo-");
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Buildings</h1>
          <p className="text-muted-foreground mt-1">
            Manage buildings and their appliances
            {mode === "demo" && (
              <span className="text-blue-600 dark:text-blue-400">
                {" "}• Viewing demo building
              </span>
            )}
          </p>
        </div>
        {mode === "api" && (
          <Dialog
            open={isAddBuildingOpen}
            onOpenChange={(open) => {
              setIsAddBuildingOpen(open);
              if (!open) resetBuildingForm();
            }}
          >
            <DialogTrigger asChild>
              <Button>
                <Plus className="h-4 w-4 mr-2" />
                Add Building
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Add New Building</DialogTitle>
                <DialogDescription>
                  Add a new building to monitor energy consumption.
                </DialogDescription>
              </DialogHeader>
              <BuildingFormFields form={buildingForm} setForm={setBuildingForm} />
              <DialogFooter>
                <Button onClick={handleAddBuilding} disabled={saving}>
                  {saving ? "Saving..." : "Add Building"}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        )}
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Building2 className="h-5 w-5" />
            Registered Buildings
          </CardTitle>
          <CardDescription>
            {buildings.length} building{buildings.length !== 1 ? "s" : ""}{" "}
            registered. Click to expand and manage appliances.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-16 w-full" />
              ))}
            </div>
          ) : buildings.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              <Building2 className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p className="font-medium">No buildings registered yet</p>
              <p className="text-sm mt-1">
                Add your first building to start monitoring
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {buildings.map((building) => {
                const isExpanded = expandedBuildings.has(building.id);
                const buildingAppliances = appliances[building.id] || [];

                return (
                  <Collapsible
                    key={building.id}
                    open={isExpanded}
                    onOpenChange={() => toggleBuilding(building.id)}
                  >
                    <div className="border rounded-lg">
                      <CollapsibleTrigger asChild>
                        <div className="flex items-center justify-between p-4 cursor-pointer hover:bg-muted/50 transition-colors">
                          <div className="flex items-center gap-3">
                            {isExpanded ? (
                              <ChevronDown className="h-4 w-4" />
                            ) : (
                              <ChevronRight className="h-4 w-4" />
                            )}
                            <div>
                              <div className="font-medium flex items-center gap-2">
                                {building.name}
                                {isDemoBuilding(building.id) && (
                                  <Badge
                                    variant="secondary"
                                    className="bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300"
                                  >
                                    Demo
                                  </Badge>
                                )}
                                <Badge
                                  variant="secondary"
                                  className={getStatusColor(building.status)}
                                >
                                  {building.status}
                                </Badge>
                              </div>
                              {building.address && (
                                <div className="flex items-center gap-1 text-sm text-muted-foreground mt-0.5">
                                  <MapPin className="h-3 w-3" />
                                  {building.address}
                                </div>
                              )}
                            </div>
                          </div>
                          <div
                            className="flex items-center gap-2"
                            onClick={(e) => e.stopPropagation()}
                          >
                            <Badge variant="outline" className="gap-1">
                              <Zap className="h-3 w-3" />
                              {buildingAppliances.length} appliances
                            </Badge>
                            {!isDemoBuilding(building.id) && (
                              <>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  onClick={() => openEditBuildingDialog(building)}
                                >
                                  <Pencil className="h-4 w-4" />
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  onClick={() => handleDeleteBuilding(building.id)}
                                >
                                  <Trash2 className="h-4 w-4 text-destructive" />
                                </Button>
                              </>
                            )}
                          </div>
                        </div>
                      </CollapsibleTrigger>

                      <CollapsibleContent>
                        <div className="border-t px-4 py-3 bg-muted/30">
                          <div className="flex items-center justify-between mb-3">
                            <h4 className="text-sm font-medium flex items-center gap-2">
                              <Zap className="h-4 w-4" />
                              Appliances
                            </h4>
                            {!isDemoBuilding(building.id) && (
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={() =>
                                  openAddApplianceDialog(building.id)
                                }
                              >
                                <Plus className="h-3 w-3 mr-1" />
                                Add Appliance
                              </Button>
                            )}
                          </div>

                          {buildingAppliances.length === 0 ? (
                            <p className="text-sm text-muted-foreground text-center py-4">
                              No appliances registered for this building yet.
                            </p>
                          ) : (
                            <Table>
                              <TableHeader>
                                <TableRow>
                                  <TableHead>Name</TableHead>
                                  <TableHead>Type</TableHead>
                                  <TableHead>Power (kW)</TableHead>
                                  <TableHead>Status</TableHead>
                                  <TableHead className="text-right">
                                    Actions
                                  </TableHead>
                                </TableRow>
                              </TableHeader>
                              <TableBody>
                                {buildingAppliances.map((appliance) => (
                                  <TableRow key={appliance.id}>
                                    <TableCell>
                                      <div className="font-medium">
                                        {appliance.name}
                                      </div>
                                      {appliance.notes && (
                                        <div className="text-xs text-muted-foreground truncate max-w-[150px]">
                                          {appliance.notes}
                                        </div>
                                      )}
                                    </TableCell>
                                    <TableCell>
                                      <Badge variant="outline">
                                        {getTypeLabel(appliance.type)}
                                      </Badge>
                                    </TableCell>
                                    <TableCell>
                                      {appliance.rated_power_kw
                                        ? `${appliance.rated_power_kw} kW`
                                        : "—"}
                                    </TableCell>
                                    <TableCell>
                                      <Badge
                                        variant="secondary"
                                        className={getStatusColor(
                                          appliance.status,
                                        )}
                                      >
                                        {appliance.status}
                                      </Badge>
                                    </TableCell>
                                    <TableCell className="text-right">
                                      {!isDemoBuilding(building.id) && (
                                        <div className="flex justify-end gap-1">
                                          <Button
                                            variant="ghost"
                                            size="icon"
                                            onClick={() =>
                                              openEditApplianceDialog(appliance)
                                            }
                                          >
                                            <Pencil className="h-3 w-3" />
                                          </Button>
                                          <Button
                                            variant="ghost"
                                            size="icon"
                                            onClick={() =>
                                              handleDeleteAppliance(appliance)
                                            }
                                          >
                                            <Trash2 className="h-3 w-3 text-destructive" />
                                          </Button>
                                        </div>
                                      )}
                                    </TableCell>
                                  </TableRow>
                                ))}
                              </TableBody>
                            </Table>
                          )}
                        </div>
                      </CollapsibleContent>
                    </div>
                  </Collapsible>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Edit Building Dialog */}
      <Dialog
        open={isEditBuildingOpen}
        onOpenChange={(open) => {
          setIsEditBuildingOpen(open);
          if (!open) {
            setEditingBuilding(null);
            resetBuildingForm();
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Building</DialogTitle>
            <DialogDescription>Update building information.</DialogDescription>
          </DialogHeader>
          <BuildingFormFields form={buildingForm} setForm={setBuildingForm} />
          <DialogFooter>
            <Button onClick={handleEditBuilding} disabled={saving}>
              {saving ? "Saving..." : "Save Changes"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Add Appliance Dialog */}
      <Dialog
        open={isAddApplianceOpen}
        onOpenChange={(open) => {
          setIsAddApplianceOpen(open);
          if (!open) resetApplianceForm();
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add Appliance</DialogTitle>
            <DialogDescription>
              Add a new appliance to this building.
            </DialogDescription>
          </DialogHeader>
          <ApplianceFormFields
            form={applianceForm}
            setForm={setApplianceForm}
            orgAppliances={orgAppliances}
            mode="add"
          />
          <DialogFooter>
            <Button onClick={handleAddAppliance} disabled={saving}>
              {saving ? "Saving..." : "Add Appliance"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit Appliance Dialog */}
      <Dialog
        open={isEditApplianceOpen}
        onOpenChange={(open) => {
          setIsEditApplianceOpen(open);
          if (!open) {
            setEditingAppliance(null);
            resetApplianceForm();
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Appliance</DialogTitle>
            <DialogDescription>Update appliance information.</DialogDescription>
          </DialogHeader>
          <ApplianceFormFields
            form={applianceForm}
            setForm={setApplianceForm}
            orgAppliances={orgAppliances}
            mode="edit"
          />
          <DialogFooter>
            <Button onClick={handleEditAppliance} disabled={saving}>
              {saving ? "Saving..." : "Save Changes"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

// Form Components
function BuildingFormFields({
  form,
  setForm,
}: {
  form: {
    name: string;
    address: string;
    description: string;
    status: "active" | "inactive" | "maintenance";
  };
  setForm: React.Dispatch<React.SetStateAction<typeof form>>;
}) {
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="building-name">Building Name *</Label>
        <Input
          id="building-name"
          value={form.name}
          onChange={(e) => setForm({ ...form, name: e.target.value })}
          placeholder="e.g., Main Office"
        />
      </div>
      <div className="space-y-2">
        <Label htmlFor="building-address">Address</Label>
        <Input
          id="building-address"
          value={form.address}
          onChange={(e) => setForm({ ...form, address: e.target.value })}
          placeholder="e.g., 123 Main Street"
        />
      </div>
      <div className="space-y-2">
        <Label htmlFor="building-description">Description</Label>
        <Input
          id="building-description"
          value={form.description}
          onChange={(e) => setForm({ ...form, description: e.target.value })}
          placeholder="Brief description of the building"
        />
      </div>
      <div className="space-y-2">
        <Label htmlFor="building-status">Status</Label>
        <Select
          value={form.status}
          onValueChange={(value: "active" | "inactive" | "maintenance") =>
            setForm({ ...form, status: value })
          }
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="active">Active</SelectItem>
            <SelectItem value="inactive">Inactive</SelectItem>
            <SelectItem value="maintenance">Maintenance</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}

function ApplianceFormFields({
  form,
  setForm,
  orgAppliances,
  mode,
}: {
  form: {
    org_appliance_id: string;
    status: "active" | "inactive";
  };
  setForm: React.Dispatch<React.SetStateAction<typeof form>>;
  orgAppliances: OrgAppliance[];
  mode: "add" | "edit";
}) {
  return (
    <div className="space-y-4">
      {mode === "add" && (
        <div className="space-y-2">
          <Label htmlFor="appliance-select">Select Appliance *</Label>
          <Select
            value={form.org_appliance_id}
            onValueChange={(value) => setForm({ ...form, org_appliance_id: value })}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select an appliance type" />
            </SelectTrigger>
            <SelectContent>
              {orgAppliances.map((app) => (
                <SelectItem key={app.id} value={app.id}>
                  {app.name} ({app.type})
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}

      <div className="space-y-2">
        <Label htmlFor="appliance-status">Status</Label>
        <Select
          value={form.status}
          onValueChange={(value: "active" | "inactive") =>
            setForm({ ...form, status: value })
          }
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="active">Active</SelectItem>
            <SelectItem value="inactive">Inactive</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}
