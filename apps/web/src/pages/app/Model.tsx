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
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { RefreshCw, CheckCircle2, AlertCircle } from "lucide-react";
import { useModels } from "@/hooks/useModels";
import { useEnergy } from "@/contexts/EnergyContext";
import { NILMPanel, NILMEmptyState } from "@/components/nilm/NILMPanel";
import { WaveformDecoration } from "@/components/brand/WaveformIcon";

export default function Model() {
  const { mode } = useEnergy();
  const { models, loading, refetch } = useModels(mode);

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
          description="Fetching models from backend registry"
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
              Model Registry
            </h1>
            <p className="text-sm text-muted-foreground">
              View active NILM models served by the backend
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
          </div>
        </div>
      </header>

      {/* Demo Mode Notice */}
      {mode === 'demo' && (
        <div className="rounded-lg bg-amber-100 dark:bg-amber-900/30 border border-amber-300 dark:border-amber-700 px-4 py-3 text-sm flex items-center gap-2">
          <AlertCircle className="h-4 w-4 text-amber-600 dark:text-amber-400 shrink-0" />
          <span className="text-amber-800 dark:text-amber-200">
            <span className="font-medium">Demo Mode Active</span> — Models shown may be simulated.
          </span>
        </div>
      )}

      {/* Models List */}
      <NILMPanel
        title="Active Models"
        subtitle="Models currently loaded in memory for inference"
      >
        {models.length === 0 ? (
          <NILMEmptyState
            title="No models found"
            description="The backend registry is empty. Check backend configuration."
          />
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Model ID</TableHead>
                <TableHead>Version</TableHead>
                <TableHead>Architecture</TableHead>
                <TableHead>Window Size</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Cached</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {models.map((model) => (
                <TableRow key={model.model_id}>
                  <TableCell className="font-medium">{model.model_id}</TableCell>
                  <TableCell className="font-mono text-sm">
                    {model.model_version}
                  </TableCell>
                  <TableCell>{model.architecture}</TableCell>
                  <TableCell>{model.input_window_size}</TableCell>
                  <TableCell>
                    {model.is_active ? (
                      <Badge variant="outline" className="bg-green-500/10 text-green-500 border-green-500/20 gap-1">
                        <CheckCircle2 className="h-3 w-3" />
                        Active
                      </Badge>
                    ) : (
                      <Badge variant="secondary">Inactive</Badge>
                    )}
                  </TableCell>
                  <TableCell>
                    {model.cached ? (
                      <Badge variant="outline" className="gap-1">
                        <CheckCircle2 className="h-3 w-3" /> Yes
                      </Badge>
                    ) : (
                      <span className="text-muted-foreground">-</span>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </NILMPanel>

      {/* Info Card */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Architecture Info</CardTitle>
            <CardDescription>System deployment details</CardDescription>
          </CardHeader>
          <CardContent className="text-sm space-y-2">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Inference Engine</span>
              <span>On-Demand / Worker</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Registry Type</span>
              <span>Local File System</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Multi-Head Support</span>
              <span>Enabled</span>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
