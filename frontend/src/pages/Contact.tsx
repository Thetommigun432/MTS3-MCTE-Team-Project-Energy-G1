import { PublicNavbar } from "@/components/layout/PublicNavbar";
import { PublicFooter } from "@/components/layout/PublicFooter";
export default function Contact() {
  return (
    <div className="min-h-screen flex flex-col">
      <PublicNavbar />
      <main className="flex-1 container py-16">
        <h1 className="text-3xl font-bold mb-6">Contact Us</h1>
        <p className="text-muted-foreground">
          For inquiries, please email support@energymonitor.example
        </p>
      </main>
      <PublicFooter />
    </div>
  );
}
