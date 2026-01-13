import { PublicNavbar } from "@/components/layout/PublicNavbar";
import { PublicFooter } from "@/components/layout/PublicFooter";
export default function About() {
  return (
    <div className="min-h-screen flex flex-col">
      <PublicNavbar />
      <main className="flex-1 container py-16">
        <h1 className="text-3xl font-bold mb-6">About Energy Monitor</h1>
        <p className="text-muted-foreground max-w-2xl">
          Energy Monitor uses Non-Intrusive Load Monitoring (NILM) to
          disaggregate total building electricity consumption into individual
          appliance-level estimates using AI.
        </p>
      </main>
      <PublicFooter />
    </div>
  );
}
