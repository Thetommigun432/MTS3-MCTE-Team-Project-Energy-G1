import { Link } from "react-router-dom";
import { BarChart3, Cpu, LineChart, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { PublicNavbar } from "@/components/layout/PublicNavbar";
import { PublicFooter } from "@/components/layout/PublicFooter";
import {
  WaveformIcon,
  WaveformDecoration,
} from "@/components/brand/WaveformIcon";
import {
  MeterToAppliancesIllustration,
  WaveformSignalIllustration,
  ConfidenceGaugeIllustration,
} from "@/components/brand/NILMIllustrations";

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col bg-background">
      <PublicNavbar />

      <main className="flex-1">
        {/* Hero with waveform motif */}
        <section className="py-20 px-6 relative overflow-hidden">
          {/* Background waveform pattern */}
          <div className="absolute inset-0 flex items-center justify-center opacity-[0.03] pointer-events-none">
            <svg viewBox="0 0 800 200" className="w-full h-auto max-w-4xl">
              <path
                d="M0 100 Q50 20, 100 100 T200 100 T300 100 T400 100 T500 100 T600 100 T700 100 T800 100"
                fill="none"
                stroke="currentColor"
                strokeWidth="3"
                className="text-primary"
              />
            </svg>
          </div>

          <div className="container max-w-5xl relative z-10">
            <div className="grid lg:grid-cols-2 gap-12 items-center">
              {/* Left: Text content */}
              <div className="text-center lg:text-left">
                <div className="flex justify-center lg:justify-start mb-6">
                  <WaveformIcon size="lg" animated />
                </div>
                <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-4 text-foreground">
                  Energy Monitor
                </h1>
                <p className="text-xl text-primary font-medium mb-2">
                  Non-Intrusive Load Monitoring (NILM)
                </p>
                <p className="text-lg text-muted-foreground mb-8 max-w-xl">
                  Identify which appliances are ON using AI — from total
                  building consumption alone
                </p>
                <div className="flex flex-col sm:flex-row gap-3 justify-center lg:justify-start">
                  <Button asChild size="lg" className="font-medium">
                    <Link to="/login">Open Dashboard</Link>
                  </Button>
                  <Button
                    asChild
                    size="lg"
                    variant="outline"
                    className="font-medium"
                  >
                    <Link to="/docs">Learn How It Works</Link>
                  </Button>
                </div>
              </div>

              {/* Right: Hero illustration */}
              <div className="flex justify-center lg:justify-end">
                <div className="bg-card rounded-2xl border border-border p-8 shadow-lg max-w-sm w-full">
                  <MeterToAppliancesIllustration
                    className="w-full h-auto"
                    accentColor="hsl(var(--primary))"
                  />
                  <div className="mt-6 text-center">
                    <p className="text-sm font-medium text-foreground">
                      Single Meter → AI → Appliances
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">
                      No sub-meters required
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Features - NILM specific with illustrations */}
        <section className="py-16 px-6 bg-muted/30 border-y border-border">
          <div className="container max-w-5xl">
            <div className="text-center mb-12">
              <h2 className="text-2xl font-semibold text-foreground mb-2">
                How NILM Works
              </h2>
              <p className="text-muted-foreground">
                From a single meter to per-appliance visibility
              </p>
            </div>

            <div className="grid md:grid-cols-3 gap-8">
              {/* Problem */}
              <div className="bg-card p-6 rounded-xl border border-border relative overflow-hidden">
                <div className="absolute top-0 right-0 opacity-5">
                  <WaveformDecoration className="h-12 w-auto text-primary" />
                </div>
                <BarChart3 className="h-10 w-10 text-primary mb-4" />
                <h3 className="font-semibold text-lg mb-2 text-foreground">
                  The Problem
                </h3>
                <p className="text-sm text-muted-foreground">
                  Buildings only measure total electricity. Understanding
                  appliance-level usage typically requires expensive sub-meters
                  on each device.
                </p>
              </div>

              {/* Solution */}
              <div className="bg-card p-6 rounded-xl border border-border relative overflow-hidden">
                <div className="absolute top-0 right-0 opacity-5">
                  <WaveformDecoration className="h-12 w-auto text-primary" />
                </div>
                <Cpu className="h-10 w-10 text-primary mb-4" />
                <h3 className="font-semibold text-lg mb-2 text-foreground">
                  AI Disaggregation
                </h3>
                <p className="text-sm text-muted-foreground">
                  Our NILM model analyzes total consumption patterns to predict
                  which appliances are running and estimate their individual
                  usage.
                </p>
              </div>

              {/* Result */}
              <div className="bg-card p-6 rounded-xl border border-border relative overflow-hidden">
                <div className="absolute top-0 right-0 opacity-5">
                  <WaveformDecoration className="h-12 w-auto text-primary" />
                </div>
                <LineChart className="h-10 w-10 text-primary mb-4" />
                <h3 className="font-semibold text-lg mb-2 text-foreground">
                  What You Get
                </h3>
                <p className="text-sm text-muted-foreground">
                  Real-time visibility into <strong>Predicted ON/OFF</strong>{" "}
                  states, estimated consumption, and confidence scores for each
                  appliance.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Visual explanation section */}
        <section className="py-16 px-6">
          <div className="container max-w-5xl">
            <div className="grid md:grid-cols-2 gap-12 items-center">
              {/* Waveform illustration */}
              <div className="bg-card rounded-xl border border-border p-8">
                <WaveformSignalIllustration
                  className="w-full h-auto"
                  accentColor="hsl(var(--primary))"
                />
                <div className="mt-6">
                  <h3 className="font-semibold text-foreground mb-2">
                    Signal Detection
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    AI identifies unique power signatures when appliances turn
                    on or off, detecting patterns in the total consumption
                    waveform.
                  </p>
                </div>
              </div>

              {/* Confidence gauge */}
              <div className="bg-card rounded-xl border border-border p-8">
                <div className="flex justify-center">
                  <ConfidenceGaugeIllustration
                    className="w-48 h-auto"
                    accentColor="hsl(var(--primary))"
                  />
                </div>
                <div className="mt-6 text-center">
                  <h3 className="font-semibold text-foreground mb-2">
                    Confidence Scoring
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    Every prediction comes with a confidence level so you know
                    how certain the model is about each appliance state.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Trust indicators */}
        <section className="py-12 px-6 bg-muted/30 border-y border-border">
          <div className="container max-w-3xl">
            <div className="bg-card rounded-xl border border-border p-8 text-center relative overflow-hidden">
              <div className="absolute top-4 left-4 opacity-10">
                <WaveformDecoration className="h-6 w-auto text-primary" />
              </div>
              <h3 className="text-lg font-semibold text-foreground mb-4">
                Transparency by Design
              </h3>
              <p className="text-muted-foreground text-sm leading-relaxed max-w-xl mx-auto mb-6">
                All predictions show <strong>confidence levels</strong> and are
                clearly labeled as
                <em> "Estimated by AI (not directly measured)"</em>. Our model
                displays version info and last training date so you always know
                what's powering the predictions.
              </p>
              <Button asChild variant="outline" size="sm">
                <Link to="/docs" className="inline-flex items-center gap-2">
                  Learn more about our methodology
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
            </div>
          </div>
        </section>
      </main>

      <PublicFooter />
    </div>
  );
}
