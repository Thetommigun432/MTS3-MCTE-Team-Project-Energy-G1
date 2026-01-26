import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import {
  BarChart3,
  Building2,
  FileText,
  Settings,
  Zap,
  HelpCircle,
  BookOpen,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Users,
  Shield,
} from "lucide-react";

export default function Help() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Help & Support</h1>
        <p className="text-muted-foreground mt-1">
          Learn how to use Energy Monitor effectively
        </p>
      </div>

      {/* Quick Start Guide */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BookOpen className="h-5 w-5 text-primary" />
            Getting Started
          </CardTitle>
          <CardDescription>
            Essential steps to begin monitoring your energy consumption
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="flex gap-3 p-4 rounded-lg bg-muted/50">
              <div className="flex-shrink-0 h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center text-primary font-semibold">
                1
              </div>
              <div>
                <h4 className="font-medium">Register Buildings</h4>
                <p className="text-sm text-muted-foreground">
                  Add your buildings in the Buildings section to organize your
                  monitoring data.
                </p>
              </div>
            </div>
            <div className="flex gap-3 p-4 rounded-lg bg-muted/50">
              <div className="flex-shrink-0 h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center text-primary font-semibold">
                2
              </div>
              <div>
                <h4 className="font-medium">View Dashboard</h4>
                <p className="text-sm text-muted-foreground">
                  Monitor real-time power consumption and appliance predictions
                  on the dashboard.
                </p>
              </div>
            </div>
            <div className="flex gap-3 p-4 rounded-lg bg-muted/50">
              <div className="flex-shrink-0 h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center text-primary font-semibold">
                3
              </div>
              <div>
                <h4 className="font-medium">Analyze Appliances</h4>
                <p className="text-sm text-muted-foreground">
                  Click on individual appliances to see detailed consumption
                  patterns and trends.
                </p>
              </div>
            </div>
            <div className="flex gap-3 p-4 rounded-lg bg-muted/50">
              <div className="flex-shrink-0 h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center text-primary font-semibold">
                4
              </div>
              <div>
                <h4 className="font-medium">Generate Reports</h4>
                <p className="text-sm text-muted-foreground">
                  Create detailed reports for specific time periods and export
                  them as CSV or PDF.
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Feature Guides */}
      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <BarChart3 className="h-5 w-5 text-primary" />
              Dashboard
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <p className="text-muted-foreground">
              The dashboard provides a real-time overview of your energy
              consumption with NILM-based appliance disaggregation.
            </p>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-emerald-500 mt-0.5 flex-shrink-0" />
                <span>View aggregate power consumption over time</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-emerald-500 mt-0.5 flex-shrink-0" />
                <span>See which appliances are currently on/off</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-emerald-500 mt-0.5 flex-shrink-0" />
                <span>Monitor model confidence levels</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-emerald-500 mt-0.5 flex-shrink-0" />
                <span>Filter by date range and appliance</span>
              </li>
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Zap className="h-5 w-5 text-primary" />
              Appliances
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <p className="text-muted-foreground">
              View detailed consumption data for each detected appliance in your
              building.
            </p>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-emerald-500 mt-0.5 flex-shrink-0" />
                <span>Individual appliance power profiles</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-emerald-500 mt-0.5 flex-shrink-0" />
                <span>Historical consumption patterns</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-emerald-500 mt-0.5 flex-shrink-0" />
                <span>On/Off state predictions with confidence</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-emerald-500 mt-0.5 flex-shrink-0" />
                <span>Energy usage rankings</span>
              </li>
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Building2 className="h-5 w-5 text-primary" />
              Buildings
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <p className="text-muted-foreground">
              Manage multiple buildings and their monitoring configurations.
            </p>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-emerald-500 mt-0.5 flex-shrink-0" />
                <span>Add and organize multiple buildings</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-emerald-500 mt-0.5 flex-shrink-0" />
                <span>Set building status (active, inactive, maintenance)</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-emerald-500 mt-0.5 flex-shrink-0" />
                <span>Track building-specific consumption</span>
              </li>
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <FileText className="h-5 w-5 text-primary" />
              Reports
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <p className="text-muted-foreground">
              Generate comprehensive energy consumption reports for analysis and
              record-keeping.
            </p>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-emerald-500 mt-0.5 flex-shrink-0" />
                <span>Custom date range selection</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-emerald-500 mt-0.5 flex-shrink-0" />
                <span>Filter by building and appliance</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-emerald-500 mt-0.5 flex-shrink-0" />
                <span>Export to CSV or PDF formats</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-emerald-500 mt-0.5 flex-shrink-0" />
                <span>Summary statistics and charts</span>
              </li>
            </ul>
          </CardContent>
        </Card>
      </div>

      {/* Understanding NILM */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-primary" />
            Understanding NILM
          </CardTitle>
          <CardDescription>
            How Non-Intrusive Load Monitoring works
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-muted-foreground">
            Non-Intrusive Load Monitoring (NILM) is a technology that
            disaggregates total power consumption into individual
            appliance-level usage using machine learning algorithms.
          </p>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="p-4 rounded-lg border bg-card">
              <div className="flex items-center gap-2 mb-2">
                <Badge
                  variant="secondary"
                  className="bg-blue-500/15 text-blue-600 dark:text-blue-400"
                >
                  Signal Analysis
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground">
                NILM analyzes the aggregate power signal to detect unique
                electrical signatures of each appliance.
              </p>
            </div>
            <div className="p-4 rounded-lg border bg-card">
              <div className="flex items-center gap-2 mb-2">
                <Badge
                  variant="secondary"
                  className="bg-purple-500/15 text-purple-600 dark:text-purple-400"
                >
                  Pattern Recognition
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground">
                Machine learning models learn appliance patterns to accurately
                predict which devices are active.
              </p>
            </div>
            <div className="p-4 rounded-lg border bg-card">
              <div className="flex items-center gap-2 mb-2">
                <Badge
                  variant="secondary"
                  className="bg-emerald-500/15 text-emerald-600 dark:text-emerald-400"
                >
                  Confidence Scoring
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground">
                Each prediction includes a confidence score indicating how
                certain the model is about the detection.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* FAQ */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <HelpCircle className="h-5 w-5 text-primary" />
            Frequently Asked Questions
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Accordion type="single" collapsible className="w-full">
            <AccordionItem value="item-1">
              <AccordionTrigger>
                What does the confidence percentage mean?
              </AccordionTrigger>
              <AccordionContent>
                The confidence percentage indicates how certain the NILM model
                is about its prediction. Higher confidence (green, &gt;50%)
                means the model is more sure about the appliance state. Lower
                confidence (yellow/red) may occur during transitional states or
                when multiple appliances have similar power signatures.
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="item-2">
              <AccordionTrigger>
                Why is an appliance shown as "off" when it's actually on?
              </AccordionTrigger>
              <AccordionContent>
                NILM predictions are based on power consumption patterns. If an
                appliance is in standby mode or consuming very little power, it
                may be classified as "off." The model also requires some
                learning time to accurately identify all appliances in your
                building.
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="item-3">
              <AccordionTrigger>
                How accurate are the energy consumption estimates?
              </AccordionTrigger>
              <AccordionContent>
                NILM typically achieves 80-95% accuracy for major appliances
                with distinct power signatures. Accuracy may vary based on the
                complexity of your electrical setup and how many appliances
                operate simultaneously. The model improves over time as it
                learns your usage patterns.
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="item-4">
              <AccordionTrigger>Can I add custom appliances?</AccordionTrigger>
              <AccordionContent>
                Currently, the system automatically detects appliances based on
                their power signatures. Custom appliance naming and manual
                additions are planned features for future updates. The model
                continuously learns and may detect new appliances over time.
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="item-5">
              <AccordionTrigger>
                How do I switch between demo data and live data?
              </AccordionTrigger>
              <AccordionContent>
                Use the data mode toggle in the top bar to switch between "Demo"
                (sample data) and "API" (live data from your connected
                monitoring system). Demo mode is useful for exploring the
                features without a live data connection.
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="item-6">
              <AccordionTrigger>How do I manage team members?</AccordionTrigger>
              <AccordionContent>
                Navigate to Settings → Team to invite new members, assign roles
                (Admin, Member, Viewer), and manage existing team access. Admins
                can invite and manage all users, while Members have limited
                administrative capabilities.
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </CardContent>
      </Card>

      {/* Settings Reference */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5 text-primary" />
            Settings Reference
          </CardTitle>
          <CardDescription>
            Quick overview of available settings
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <div className="flex items-start gap-3">
              <div className="p-2 rounded-lg bg-muted">
                <Users className="h-4 w-4" />
              </div>
              <div>
                <h4 className="font-medium text-sm">Profile</h4>
                <p className="text-xs text-muted-foreground">
                  Update your name, email, and avatar
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="p-2 rounded-lg bg-muted">
                <Users className="h-4 w-4" />
              </div>
              <div>
                <h4 className="font-medium text-sm">Team</h4>
                <p className="text-xs text-muted-foreground">
                  Invite members and manage roles
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="p-2 rounded-lg bg-muted">
                <AlertTriangle className="h-4 w-4" />
              </div>
              <div>
                <h4 className="font-medium text-sm">Notifications</h4>
                <p className="text-xs text-muted-foreground">
                  Configure alerts and email preferences
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="p-2 rounded-lg bg-muted">
                <Shield className="h-4 w-4" />
              </div>
              <div>
                <h4 className="font-medium text-sm">Security</h4>
                <p className="text-xs text-muted-foreground">
                  Change password and security settings
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="p-2 rounded-lg bg-muted">
                <Settings className="h-4 w-4" />
              </div>
              <div>
                <h4 className="font-medium text-sm">Appearance</h4>
                <p className="text-xs text-muted-foreground">
                  Toggle light/dark mode theme
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
      {/* Debug Information */}
      <Card className="border-dashed border-muted-foreground/20 bg-muted/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm text-muted-foreground">
            <AlertTriangle className="h-4 w-4" />
            Debug Information
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-xs font-mono space-y-2 text-muted-foreground">
            <div className="flex justify-between border-b pb-1">
              <span>App Mode:</span>
              <span className="text-foreground">{import.meta.env.VITE_DEMO_MODE === "true" ? "Demo (Forced)" : "Production"}</span>
            </div>
            <div className="flex justify-between border-b pb-1">
              <span>API URL:</span>
              <span className="text-foreground">{import.meta.env.VITE_BACKEND_URL || "Not configured"}</span>
            </div>
            <div className="flex justify-between border-b pb-1">
              <span>Frontend Build:</span>
              <span className="text-foreground">{new Date().toISOString().split('T')[0]}</span>
            </div>
            <div className="flex justify-between">
              <span>Supabase URL:</span>
              <span className="text-foreground">{import.meta.env.VITE_SUPABASE_URL ? "Configured" : "Missing"}</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
