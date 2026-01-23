import { Outlet, useLocation } from "react-router-dom";
import { AppSidebar } from "./AppSidebar";
import { TopBar } from "./TopBar";
import { DemoModeBanner } from "./DemoModeBanner";
import { EnergyProvider } from "@/contexts/EnergyContext";
import { useEffect } from "react";

export function AppShell() {
  const location = useLocation();

  // Scroll to top on route change
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [location.pathname]);

  return (
    <EnergyProvider>
      <div className="flex h-screen w-full flex-col overflow-hidden">
        {/* Demo mode warning banner - visible when VITE_DEMO_MODE=true */}
        <DemoModeBanner />
        <div className="flex flex-1 overflow-hidden">
          <AppSidebar />
          <div className="flex flex-1 flex-col overflow-hidden">
            <TopBar />
            <main className="flex-1 overflow-auto bg-background p-4 md:p-6">
              <div className="animate-fade-in">
                <Outlet />
              </div>
            </main>
          </div>
        </div>
      </div>
    </EnergyProvider>
  );
}
