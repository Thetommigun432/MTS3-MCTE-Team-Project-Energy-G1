import { useState, useMemo, useRef } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Plus,
  Upload,
  CheckCircle2,
  AlertCircle,
  Clock,
  Cpu,
  Layers,
  RefreshCw,
  Settings2,
  ChevronRight,
  FileUp,
} from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { useEnergy } from "@/contexts/EnergyContext";
import { useModels, type Model as ModelType } from "@/hooks/useModels";
import { useOrgAppliances } from "@/hooks/useOrgAppliances";
import { useBuildings } from "@/hooks/useBuildings";
import { format, formatDistanceToNow } from "date-fns";

// NILM Components
import { ModelTrustBadge } from "@/components/nilm/ModelTrustBadge";
import { NILMPanel, NILMEmptyState } from "@/components/nilm/NILMPanel";
import { WaveformDecoration } from "@/components/brand/WaveformIcon";

function StatusBadge({ status }: { status: string }) {
  const variants: Record<string, { className: string; icon: React.ReactNode }> =
    {
      ready: {
        className: "bg-green-500/10 text-green-500 border-green-500/20",
        icon: <CheckCircle2 className="h-3 w-3" />,
      },
      pending: {
        className: "bg-yellow-500/10 text-yellow-500 border-yellow-500/20",
        icon: <Clock className="h-3 w-3" />,
      },
      uploading: {
        className: "bg-blue-500/10 text-blue-500 border-blue-500/20",
        icon: <Upload className="h-3 w-3" />,
      },
      failed: {
        className: "bg-red-500/10 text-red-500 border-red-500/20",
        icon: <AlertCircle className="h-3 w-3" />,
      },
    };

  const variant = variants[status] || variants.pending;

  return (
    <Badge variant="outline" className={cn("gap-1", variant.className)}>
      {variant.icon}
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </Badge>
  );
}

function RegisterModelDialog({
  orgAppliances,
  onRegister,
}: {
  orgAppliances: { id: string; name: string; slug: string }[];
  onRegister: (
    orgApplianceId: string,
    name: string,
    architecture?: string,
  ) => Promise<string | null>;
}) {
  const [open, setOpen] = useState(false);
  const [selectedAppliance, setSelectedAppliance] = useState("");
  const [modelName, setModelName] = useState("");
  const [architecture, setArchitecture] = useState("seq2point");
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async () => {
    if (!selectedAppliance || !modelName.trim()) {
      toast.error("Please fill in all required fields");
      return;
    }

    setSubmitting(true);
    const result = await onRegister(
      selectedAppliance,
      modelName.trim(),
      architecture,
    );
    setSubmitting(false);

    if (result) {
      setOpen(false);
      setSelectedAppliance("");
      setModelName("");
      setArchitecture("seq2point");
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button className="gap-2">
          <Plus className="h-4 w-4" />
          Register Model
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Register New Model</DialogTitle>
          <DialogDescription>
            Create a new NILM model for an appliance type. You can upload model
            versions after registration.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <Label>Target Appliance *</Label>
            <Select
              value={selectedAppliance}
              onValueChange={setSelectedAppliance}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select appliance type" />
              </SelectTrigger>
              <SelectContent>
                {orgAppliances.map((app) => (
                  <SelectItem key={app.id} value={app.id}>
                    {app.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label>Model Name *</Label>
            <Input
              placeholder="e.g., Refrigerator Detector v1"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
            />
          </div>

          <div className="space-y-2">
            <Label>Architecture</Label>
            <Select value={architecture} onValueChange={setArchitecture}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="seq2point">Seq2Point CNN</SelectItem>
                <SelectItem value="seq2seq">Seq2Seq LSTM</SelectItem>
                <SelectItem value="transformer">Transformer</SelectItem>
                <SelectItem value="custom">Custom</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={submitting}>
            {submitting ? "Registering..." : "Register Model"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function UploadVersionDialog({
  model,
  onUpload,
}: {
  model: ModelType;
  onUpload: (
    modelId: string,
    version: string,
    modelFile: File,
    scalerFile?: File,
  ) => Promise<boolean>;
}) {
  const [open, setOpen] = useState(false);
  const [version, setVersion] = useState("");
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [scalerFile, setScalerFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const modelInputRef = useRef<HTMLInputElement>(null);
  const scalerInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = async () => {
    if (!version.trim() || !modelFile) {
      toast.error("Please provide a version and model file");
      return;
    }

    setSubmitting(true);
    const success = await onUpload(
      model.id,
      version.trim(),
      modelFile,
      scalerFile || undefined,
    );
    setSubmitting(false);

    if (success) {
      setOpen(false);
      setVersion("");
      setModelFile(null);
      setScalerFile(null);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" className="gap-1.5">
          <Upload className="h-3.5 w-3.5" />
          Upload Version
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Upload Model Version</DialogTitle>
          <DialogDescription>
            Upload a new version for {model.name}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <Label>Version *</Label>
            <Input
              placeholder="e.g., v1.2.0"
              value={version}
              onChange={(e) => setVersion(e.target.value)}
            />
          </div>

          <div className="space-y-2">
            <Label>Model File (.h5, .pkl, .pt) *</Label>
            <input
              ref={modelInputRef}
              type="file"
              accept=".h5,.pkl,.pt,.onnx"
              className="hidden"
              onChange={(e) => setModelFile(e.target.files?.[0] || null)}
            />
            <Button
              variant="outline"
              className="w-full justify-start gap-2"
              onClick={() => modelInputRef.current?.click()}
            >
              <FileUp className="h-4 w-4" />
              {modelFile ? modelFile.name : "Choose model file..."}
            </Button>
          </div>

          <div className="space-y-2">
            <Label>Scaler File (optional)</Label>
            <input
              ref={scalerInputRef}
              type="file"
              accept=".pkl,.joblib"
              className="hidden"
              onChange={(e) => setScalerFile(e.target.files?.[0] || null)}
            />
            <Button
              variant="outline"
              className="w-full justify-start gap-2"
              onClick={() => scalerInputRef.current?.click()}
            >
              <FileUp className="h-4 w-4" />
              {scalerFile
                ? scalerFile.name
                : "Choose scaler file (optional)..."}
            </Button>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={submitting}>
            {submitting ? "Uploading..." : "Upload"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function ModelCard({
  model,
  onUploadVersion,
  onSetActive,
}: {
  model: ModelType;
  onUploadVersion: (
    modelId: string,
    version: string,
    modelFile: File,
    scalerFile?: File,
  ) => Promise<boolean>;
  onSetActive: (versionId: string) => Promise<boolean>;
}) {
  const activeVersion = model.active_version;
  const metrics = activeVersion?.metrics;

  return (
    <Card className="overflow-hidden">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <CardTitle className="text-lg">{model.name}</CardTitle>
            <CardDescription className="flex items-center gap-2">
              <span className="font-medium text-foreground/80">
                {model.org_appliance_name}
              </span>
              <ChevronRight className="h-3 w-3 text-muted-foreground" />
              <span className="font-mono text-xs">
                {model.org_appliance_slug}
              </span>
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            {activeVersion ? (
              <StatusBadge status={activeVersion.status} />
            ) : (
              <Badge variant="outline" className="text-muted-foreground">
                No versions
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Active Version Info */}
        {activeVersion ? (
          <div className="rounded-lg bg-muted/50 p-3 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                Active Version
              </span>
              <Badge variant="secondary" className="font-mono text-xs">
                {activeVersion.version}
              </Badge>
            </div>

            {metrics && (
              <div className="grid grid-cols-3 gap-3 text-sm">
                {metrics.accuracy !== undefined && (
                  <div>
                    <p className="text-muted-foreground text-xs">Accuracy</p>
                    <p className="font-semibold">
                      {(metrics.accuracy * 100).toFixed(1)}%
                    </p>
                  </div>
                )}
                {metrics.mae !== undefined && (
                  <div>
                    <p className="text-muted-foreground text-xs">MAE</p>
                    <p className="font-semibold font-mono">
                      {metrics.mae.toFixed(3)} kW
                    </p>
                  </div>
                )}
                {metrics.f1_score !== undefined && (
                  <div>
                    <p className="text-muted-foreground text-xs">F1 Score</p>
                    <p className="font-semibold font-mono">
                      {metrics.f1_score.toFixed(2)}
                    </p>
                  </div>
                )}
              </div>
            )}

            {activeVersion.trained_at && (
              <p className="text-xs text-muted-foreground">
                Trained{" "}
                {formatDistanceToNow(new Date(activeVersion.trained_at), {
                  addSuffix: true,
                })}
              </p>
            )}
          </div>
        ) : (
          <div className="rounded-lg border border-dashed p-4 text-center">
            <p className="text-sm text-muted-foreground">
              No model version deployed yet
            </p>
          </div>
        )}

        {/* Architecture & Versions */}
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center gap-2 text-muted-foreground">
            <Cpu className="h-4 w-4" />
            <span>{model.architecture || "Unknown"}</span>
          </div>
          <div className="flex items-center gap-2 text-muted-foreground">
            <Layers className="h-4 w-4" />
            <span>
              {model.versions.length} version
              {model.versions.length !== 1 ? "s" : ""}
            </span>
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-2">
          <UploadVersionDialog model={model} onUpload={onUploadVersion} />

          {model.versions.length > 1 && (
            <Select onValueChange={onSetActive}>
              <SelectTrigger className="w-auto">
                <Settings2 className="h-3.5 w-3.5 mr-1.5" />
                Set Active
              </SelectTrigger>
              <SelectContent>
                {model.versions.map((v) => (
                  <SelectItem
                    key={v.id}
                    value={v.id}
                    disabled={v.is_active || v.status !== "ready"}
                  >
                    {v.version} {v.is_active && "(current)"}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

export default function Model() {
  const { mode, selectedBuildingId, insights } = useEnergy();
  const { buildings, loading: buildingsLoading } = useBuildings();
  const { appliances: orgAppliances, loading: appliancesLoading } =
    useOrgAppliances();
  const [selectedBuilding, setSelectedBuilding] = useState<string | null>(null);

  // Use selected building from context or local selection
  const effectiveBuildingId = selectedBuildingId || selectedBuilding;

  const {
    models,
    loading: modelsLoading,
    refetch,
    registerModel,
    uploadModelVersion,
    setActiveVersion,
  } = useModels(effectiveBuildingId);

  const loading = buildingsLoading || appliancesLoading || modelsLoading;

  // Filter org appliances that don't have models yet
  const appliancesWithoutModels = useMemo(() => {
    const modelsApplianceIds = new Set(models.map((m) => m.org_appliance_id));
    return orgAppliances.filter((a) => !modelsApplianceIds.has(a.id));
  }, [orgAppliances, models]);

  if (loading) {
    return (
      <div className="space-y-8 animate-fade-in">
        <header className="space-y-2">
          <h1 className="text-2xl font-semibold tracking-tight text-foreground">
            Model Manager
          </h1>
          <p className="text-sm text-muted-foreground">
            Loading model information...
          </p>
        </header>
        <NILMEmptyState
          title="Loading..."
          description="Fetching models and configurations"
        />
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <header className="space-y-2 relative">
        <div className="absolute -top-2 -left-4 opacity-5 pointer-events-none">
          <WaveformDecoration className="h-8 w-auto text-primary" />
        </div>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight text-foreground">
              Model Manager
            </h1>
            <p className="text-sm text-muted-foreground">
              Manage NILM models for each appliance type
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Button
              variant="outline"
              size="sm"
              onClick={() => refetch()}
              className="gap-2"
            >
              <RefreshCw className="h-4 w-4" />
              Refresh
            </Button>
            <RegisterModelDialog
              orgAppliances={appliancesWithoutModels}
              onRegister={registerModel}
            />
          </div>
        </div>
      </header>

      {/* Mode Notice */}
      {mode === "demo" && (
        <div className="rounded-lg bg-amber-100 dark:bg-amber-900/30 border border-amber-300 dark:border-amber-700 px-4 py-3 text-sm flex items-center gap-2">
          <AlertCircle className="h-4 w-4 text-amber-600 dark:text-amber-400 shrink-0" />
          <span className="text-amber-800 dark:text-amber-200">
            <span className="font-medium">Demo Mode</span> — Model management
            requires API mode with a connected backend.
          </span>
        </div>
      )}

      {/* Building Filter */}
      {buildings.length > 0 && (
        <div className="flex items-center gap-3">
          <Label className="text-sm text-muted-foreground">
            Filter by Building:
          </Label>
          <Select
            value={effectiveBuildingId || "all"}
            onValueChange={(v) => setSelectedBuilding(v === "all" ? null : v)}
          >
            <SelectTrigger className="w-64">
              <SelectValue placeholder="All buildings" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All buildings</SelectItem>
              {buildings.map((b) => (
                <SelectItem key={b.id} value={b.id}>
                  {b.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}

      {/* Tabs for different views */}
      <Tabs defaultValue="models" className="space-y-6">
        <TabsList>
          <TabsTrigger value="models">Models ({models.length})</TabsTrigger>
          <TabsTrigger value="versions">All Versions</TabsTrigger>
        </TabsList>

        {/* Models Grid */}
        <TabsContent value="models" className="space-y-6">
          {models.length === 0 ? (
            <NILMEmptyState
              title="No models registered"
              description="Register your first model to start disaggregating appliance consumption"
            />
          ) : (
            <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
              {models.map((model) => (
                <ModelCard
                  key={model.id}
                  model={model}
                  onUploadVersion={uploadModelVersion}
                  onSetActive={setActiveVersion}
                />
              ))}
            </div>
          )}
        </TabsContent>

        {/* All Versions Table */}
        <TabsContent value="versions" className="space-y-6">
          <NILMPanel
            title="Model Versions"
            subtitle="All uploaded model versions across appliances"
          >
            {models.flatMap((m) => m.versions).length === 0 ? (
              <p className="text-sm text-muted-foreground py-4">
                No versions uploaded yet
              </p>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Appliance</TableHead>
                    <TableHead>Model</TableHead>
                    <TableHead>Version</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Trained</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {models.flatMap((model) =>
                    model.versions.map((version) => (
                      <TableRow key={version.id}>
                        <TableCell className="font-medium">
                          {model.org_appliance_name}
                        </TableCell>
                        <TableCell>{model.name}</TableCell>
                        <TableCell className="font-mono text-sm">
                          {version.version}
                          {version.is_active && (
                            <Badge variant="secondary" className="ml-2 text-xs">
                              Active
                            </Badge>
                          )}
                        </TableCell>
                        <TableCell>
                          <StatusBadge status={version.status} />
                        </TableCell>
                        <TableCell className="text-muted-foreground text-sm">
                          {version.trained_at
                            ? format(
                                new Date(version.trained_at),
                                "MMM d, yyyy",
                              )
                            : "-"}
                        </TableCell>
                        <TableCell className="text-right">
                          {!version.is_active && version.status === "ready" && (
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => setActiveVersion(version.id)}
                            >
                              Set Active
                            </Button>
                          )}
                        </TableCell>
                      </TableRow>
                    )),
                  )}
                </TableBody>
              </Table>
            )}
          </NILMPanel>
        </TabsContent>
      </Tabs>

      {/* Quick Stats */}
      <div className="grid gap-6 lg:grid-cols-3">
        <NILMPanel title="Model Overview" subtitle="Summary statistics">
          <div className="space-y-3 text-sm">
            <div className="flex justify-between py-2">
              <span className="text-muted-foreground">Total Models</span>
              <span className="font-semibold">{models.length}</span>
            </div>
            <div className="flex justify-between py-2">
              <span className="text-muted-foreground">Active Versions</span>
              <span className="font-semibold">
                {models.filter((m) => m.active_version).length}
              </span>
            </div>
            <div className="flex justify-between py-2">
              <span className="text-muted-foreground">Org Appliances</span>
              <span className="font-semibold">{orgAppliances.length}</span>
            </div>
            <div className="flex justify-between py-2">
              <span className="text-muted-foreground">Without Models</span>
              <span className="font-semibold text-amber-500">
                {appliancesWithoutModels.length}
              </span>
            </div>
          </div>
        </NILMPanel>

        <NILMPanel title="Overall Performance" subtitle="Aggregated metrics">
          <div className="space-y-3 text-sm">
            <div className="flex justify-between py-2">
              <span className="text-muted-foreground">Avg Confidence</span>
              <span className="font-semibold">
                {insights.overallConfidence.percentage.toFixed(0)}%
              </span>
            </div>
            <div className="flex justify-between py-2 items-center">
              <span className="text-muted-foreground">Confidence Level</span>
              <ModelTrustBadge
                confidenceLevel={insights.overallConfidence.level}
              />
            </div>
          </div>
        </NILMPanel>

        <NILMPanel title="Quick Actions" subtitle="Common operations">
          <div className="space-y-2">
            <Button
              variant="outline"
              className="w-full justify-start gap-2"
              disabled={mode === "demo"}
            >
              <RefreshCw className="h-4 w-4" />
              Retrain All Models
            </Button>
            <Button
              variant="outline"
              className="w-full justify-start gap-2"
              disabled={mode === "demo"}
            >
              <Cpu className="h-4 w-4" />
              Run Batch Inference
            </Button>
          </div>
        </NILMPanel>
      </div>
    </div>
  );
}
