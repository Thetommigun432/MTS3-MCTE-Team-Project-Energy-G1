import { PublicNavbar } from '@/components/layout/PublicNavbar';
import { PublicFooter } from '@/components/layout/PublicFooter';
import { WaveformDecoration } from '@/components/brand/WaveformIcon';
import { 
  MeterToAppliancesIllustration, 
  WaveformSignalIllustration, 
  ConfidenceGaugeIllustration,
  ApplianceDetectionIllustration,
  EnergyFlowIllustration
} from '@/components/brand/NILMIllustrations';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { BookOpen, Cpu, Activity, Gauge, Zap, HelpCircle } from 'lucide-react';

export default function Docs() {
  return (
    <div className="min-h-screen flex flex-col bg-background">
      <PublicNavbar />
      
      <main className="flex-1">
        {/* Hero */}
        <section className="py-16 px-6 border-b border-border relative overflow-hidden">
          <div className="absolute inset-0 opacity-[0.02] pointer-events-none">
            <WaveformDecoration className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 h-64 w-auto text-primary" />
          </div>
          <div className="container max-w-4xl text-center relative z-10">
            <Badge variant="outline" className="mb-4 border-primary/20 text-primary">
              <BookOpen className="h-3 w-3 mr-1" />
              Documentation
            </Badge>
            <h1 className="text-3xl md:text-4xl font-bold mb-4 text-foreground">
              Understanding NILM Technology
            </h1>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Learn how Non-Intrusive Load Monitoring disaggregates total building consumption into individual appliance usage.
            </p>
          </div>
        </section>

        {/* Core Concept */}
        <section className="py-12 px-6">
          <div className="container max-w-5xl">
            <div className="grid lg:grid-cols-2 gap-8 items-center">
              <div>
                <h2 className="text-2xl font-semibold text-foreground mb-4 flex items-center gap-2">
                  <Cpu className="h-6 w-6 text-primary" />
                  What is NILM?
                </h2>
                <p className="text-muted-foreground mb-4">
                  <strong>Non-Intrusive Load Monitoring (NILM)</strong> is an AI-powered technique that analyzes 
                  the total electrical consumption of a building to identify and estimate the usage of individual appliances.
                </p>
                <p className="text-muted-foreground mb-4">
                  Instead of installing separate meters on each device (which is expensive and intrusive), 
                  NILM uses machine learning to recognize the unique electrical signatures of different appliances.
                </p>
                <div className="bg-muted/50 rounded-lg p-4 border border-border">
                  <p className="text-sm text-muted-foreground">
                    <strong className="text-foreground">Key insight:</strong> Each appliance has a distinctive power consumption 
                    pattern — like a fingerprint — that can be detected in the aggregate signal.
                  </p>
                </div>
              </div>
              <div className="bg-card rounded-xl border border-border p-8">
                <MeterToAppliancesIllustration 
                  className="w-full h-auto" 
                  accentColor="hsl(var(--primary))" 
                />
                <p className="text-center text-sm text-muted-foreground mt-4">
                  Single meter input → AI processing → Per-appliance output
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Signal Detection */}
        <section className="py-12 px-6 bg-muted/30 border-y border-border">
          <div className="container max-w-5xl">
            <div className="grid lg:grid-cols-2 gap-8 items-center">
              <div className="order-2 lg:order-1 bg-card rounded-xl border border-border p-8">
                <WaveformSignalIllustration 
                  className="w-full h-auto" 
                  accentColor="hsl(var(--primary))" 
                />
                <p className="text-center text-sm text-muted-foreground mt-4">
                  Highlighted zone shows detected appliance event
                </p>
              </div>
              <div className="order-1 lg:order-2">
                <h2 className="text-2xl font-semibold text-foreground mb-4 flex items-center gap-2">
                  <Activity className="h-6 w-6 text-primary" />
                  Signal Detection
                </h2>
                <p className="text-muted-foreground mb-4">
                  The AI continuously monitors the power consumption waveform, looking for characteristic changes that indicate appliances turning on or off.
                </p>
                <ul className="space-y-3 text-sm text-muted-foreground">
                  <li className="flex items-start gap-2">
                    <span className="h-1.5 w-1.5 rounded-full bg-primary mt-1.5 shrink-0" />
                    <span><strong className="text-foreground">Step changes:</strong> Sudden power increases/decreases when appliances switch states</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="h-1.5 w-1.5 rounded-full bg-primary mt-1.5 shrink-0" />
                    <span><strong className="text-foreground">Pattern matching:</strong> Recognizing known appliance signatures in the signal</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="h-1.5 w-1.5 rounded-full bg-primary mt-1.5 shrink-0" />
                    <span><strong className="text-foreground">Time correlation:</strong> Using usage patterns (e.g., HVAC cycles) to improve accuracy</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Confidence Levels */}
        <section className="py-12 px-6">
          <div className="container max-w-5xl">
            <div className="grid lg:grid-cols-2 gap-8 items-center">
              <div>
                <h2 className="text-2xl font-semibold text-foreground mb-4 flex items-center gap-2">
                  <Gauge className="h-6 w-6 text-primary" />
                  Confidence Levels
                </h2>
                <p className="text-muted-foreground mb-4">
                  Every prediction includes a confidence score indicating how certain the model is about its estimate. 
                  This transparency helps you understand the reliability of each data point.
                </p>
                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <Badge className="bg-confidence-high/10 text-confidence-high border-confidence-high/20">High</Badge>
                    <span className="text-sm text-muted-foreground">Clear signal, high certainty (&gt;80%)</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <Badge className="bg-confidence-medium/10 text-confidence-medium border-confidence-medium/20">Medium</Badge>
                    <span className="text-sm text-muted-foreground">Reasonable confidence (50-80%)</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <Badge className="bg-confidence-low/10 text-confidence-low border-confidence-low/20">Low</Badge>
                    <span className="text-sm text-muted-foreground">Uncertain, overlapping signals (&lt;50%)</span>
                  </div>
                </div>
              </div>
              <div className="bg-card rounded-xl border border-border p-8 flex justify-center">
                <ConfidenceGaugeIllustration 
                  className="w-48 h-auto" 
                  accentColor="hsl(var(--primary))" 
                />
              </div>
            </div>
          </div>
        </section>

        {/* Appliance Detection States */}
        <section className="py-12 px-6 bg-muted/30 border-y border-border">
          <div className="container max-w-5xl">
            <h2 className="text-2xl font-semibold text-foreground mb-8 text-center flex items-center justify-center gap-2">
              <Zap className="h-6 w-6 text-primary" />
              Detection Stages
            </h2>
            <div className="grid md:grid-cols-3 gap-6">
              <Card className="border-border">
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center gap-2">
                    <Badge variant="outline" className="bg-energy-warning-bg text-energy-warning border-energy-warning/20">Learning</Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    The model is still gathering data on this appliance. Predictions may be less accurate during this initial phase.
                  </p>
                </CardContent>
              </Card>

              <Card className="border-border">
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center gap-2">
                    <Badge variant="outline" className="bg-energy-success/10 text-energy-success border-energy-success/20">Stable</Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Enough data collected. The model confidently recognizes this appliance's signature and provides reliable predictions.
                  </p>
                </CardContent>
              </Card>

              <Card className="border-border">
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center gap-2">
                    <Badge variant="outline" className="bg-muted text-muted-foreground border-border">Uncertain</Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Signal overlap or unusual patterns detected. The model flags these cases for manual review or additional training.
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>
        </section>

        {/* API Coming Soon */}
        <section className="py-12 px-6">
          <div className="container max-w-3xl">
            <Card className="border-border relative overflow-hidden">
              <div className="absolute top-4 right-4 opacity-10">
                <WaveformDecoration className="h-16 w-auto text-primary" />
              </div>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <HelpCircle className="h-5 w-5 text-primary" />
                  API Documentation
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4">
                  Full API documentation and integration guides are coming soon. You'll be able to:
                </p>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-center gap-2">
                    <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                    Connect your building's smart meter data
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                    Receive real-time disaggregation via REST API or webhooks
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                    Train custom models for your specific appliances
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                    Export historical data and generate reports
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </section>
      </main>

      <PublicFooter />
    </div>
  );
}
