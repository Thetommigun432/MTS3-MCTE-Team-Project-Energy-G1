import { useState, useEffect, useCallback } from "react";
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
import { toast } from "sonner";

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
  id: string;
  building_id: string;
  name: string;
  type: string;
  rated_power_kw: number | null;
  status: string;
  notes: string | null;
  created_at: string;
  updated_at: string;
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
  const [buildings, setBuildings] = useState<Building[]>([]);
  const [appliances, setAppliances] = useState<Record<string, Appliance[]>>({});
  const [loading, setLoading] = useState(true);
  const [expandedBuildings, setExpandedBuildings] = useState<Set<string>>(
    new Set(),
  );

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
    name: "",
    type: "other",
    rated_power_kw: "",
    status: "active" as "active" | "inactive" | "unknown",
    notes: "",
  });

  const [saving, setSaving] = useState(false);

  const fetchBuildings = useCallback(async () => {
    if (!user) return;

    try {
      setLoading(true);
      const { data, error } = await supabase
        .from("buildings")
        .select("*")
        .order("created_at", { ascending: false });

      if (error) throw error;
      setBuildings(data || []);
    } catch (error) {
      console.error("Error fetching buildings:", error);
      toast.error("Failed to load buildings");
    } finally {
      setLoading(false);
    }
  }, [user]);

  const fetchAppliances = async (buildingId: string) => {
    try {
      const { data, error } = await supabase
        .from("appliances")
        .select("*")
        .eq("building_id", buildingId)
        .order("name");

      if (error) throw error;
      setAppliances((prev) => ({ ...prev, [buildingId]: data || [] }));
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

  // Appliance CRUD
  const handleAddAppliance = async () => {
    const trimmedName = applianceForm.name.trim();
    const trimmedNotes = applianceForm.notes.trim();
    const ratedPower = applianceForm.rated_power_kw
      ? parseFloat(applianceForm.rated_power_kw)
      : null;

    // Client-side validation matching database constraints
    if (!user || !selectedBuildingId || !trimmedName) {
      toast.error("Appliance name is required");
      return;
    }
    if (trimmedName.length > 200) {
      toast.error("Appliance name must be 200 characters or less");
      return;
    }
    if (trimmedNotes.length > 1000) {
      toast.error("Notes must be 1000 characters or less");
      return;
    }
    if (ratedPower !== null && (ratedPower <= 0 || ratedPower > 10000)) {
      toast.error("Power rating must be between 0 and 10,000 kW");
      return;
    }

    try {
      setSaving(true);
      const { error } = await supabase.from("appliances").insert({
        user_id: user.id,
        building_id: selectedBuildingId,
        name: trimmedName,
        type: applianceForm.type,
        rated_power_kw: ratedPower,
        status: applianceForm.status,
        notes: trimmedNotes || null,
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
    const trimmedName = applianceForm.name.trim();
    const trimmedNotes = applianceForm.notes.trim();
    const ratedPower = applianceForm.rated_power_kw
      ? parseFloat(applianceForm.rated_power_kw)
      : null;

    // Client-side validation matching database constraints
    if (!editingAppliance || !trimmedName) {
      toast.error("Appliance name is required");
      return;
    }
    if (trimmedName.length > 200) {
      toast.error("Appliance name must be 200 characters or less");
      return;
    }
    if (trimmedNotes.length > 1000) {
      toast.error("Notes must be 1000 characters or less");
      return;
    }
    if (ratedPower !== null && (ratedPower <= 0 || ratedPower > 10000)) {
      toast.error("Power rating must be between 0 and 10,000 kW");
      return;
    }

    try {
      setSaving(true);
      const { error } = await supabase
        .from("appliances")
        .update({
          name: trimmedName,
          type: applianceForm.type,
          rated_power_kw: ratedPower,
          status: applianceForm.status,
          notes: trimmedNotes || null,
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
      const { error } = await supabase
        .from("appliances")
        .delete()
        .eq("id", appliance.id);
      if (error) throw error;
      toast.success("Appliance deleted");
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
      name: appliance.name,
      type: appliance.type,
      rated_power_kw: appliance.rated_power_kw?.toString() || "",
      status: appliance.status as "active" | "inactive" | "unknown",
      notes: appliance.notes || "",
    });
    setIsEditApplianceOpen(true);
  };

  const resetApplianceForm = () => {
    setApplianceForm({
      name: "",
      type: "other",
      rated_power_kw: "",
      status: "active",
      notes: "",
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

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Buildings</h1>
          <p className="text-muted-foreground mt-1">
            Manage buildings and their appliances
          </p>
        </div>
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
}: {
  form: {
    name: string;
    type: string;
    rated_power_kw: string;
    status: "active" | "inactive" | "unknown";
    notes: string;
  };
  setForm: React.Dispatch<React.SetStateAction<typeof form>>;
}) {
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="appliance-name">Appliance Name *</Label>
        <Input
          id="appliance-name"
          value={form.name}
          onChange={(e) => setForm({ ...form, name: e.target.value })}
          placeholder="e.g., Main HVAC Unit"
        />
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="appliance-type">Type</Label>
          <Select
            value={form.type}
            onValueChange={(value) => setForm({ ...form, type: value })}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {APPLIANCE_TYPES.map((type) => (
                <SelectItem key={type.value} value={type.value}>
                  {type.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-2">
          <Label htmlFor="appliance-power">Rated Power (kW)</Label>
          <Input
            id="appliance-power"
            type="number"
            step="0.001"
            value={form.rated_power_kw}
            onChange={(e) =>
              setForm({ ...form, rated_power_kw: e.target.value })
            }
            placeholder="e.g., 2.5"
          />
        </div>
      </div>
      <div className="space-y-2">
        <Label htmlFor="appliance-status">Status</Label>
        <Select
          value={form.status}
          onValueChange={(value: "active" | "inactive" | "unknown") =>
            setForm({ ...form, status: value })
          }
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="active">Active</SelectItem>
            <SelectItem value="inactive">Inactive</SelectItem>
            <SelectItem value="unknown">Unknown</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div className="space-y-2">
        <Label htmlFor="appliance-notes">Notes</Label>
        <Input
          id="appliance-notes"
          value={form.notes}
          onChange={(e) => setForm({ ...form, notes: e.target.value })}
          placeholder="Optional notes about this appliance"
        />
      </div>
    </div>
  );
}
