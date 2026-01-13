import { lazy, Suspense, useEffect } from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "@/contexts/AuthContext";
import { ThemeProvider } from "@/contexts/ThemeContext";
import { AppearanceProvider } from "@/contexts/AppearanceContext";
import { ProtectedRoute } from "@/components/auth/ProtectedRoute";
import { AppShell } from "@/components/layout/AppShell";
import { Skeleton } from "@/components/ui/skeleton";
import { ErrorBoundary } from "@/components/ErrorBoundary";

// Lazy load route components for code splitting
const Home = lazy(() => import("./pages/Home"));
const Login = lazy(() => import("./pages/auth/Login"));
const Signup = lazy(() => import("./pages/auth/Signup"));
const ForgotPassword = lazy(() => import("./pages/auth/ForgotPassword"));
const ResetPassword = lazy(() => import("./pages/auth/ResetPassword"));
const VerifyEmail = lazy(() => import("./pages/auth/VerifyEmail"));
const Dashboard = lazy(() => import("./pages/app/Dashboard"));
const Appliances = lazy(() => import("./pages/app/Appliances"));
const ApplianceDetails = lazy(() => import("./pages/app/ApplianceDetails"));
const Buildings = lazy(() => import("./pages/app/Buildings"));
const Reports = lazy(() => import("./pages/app/Reports"));
const Model = lazy(() => import("./pages/app/Model"));
const SettingsLayout = lazy(
  () => import("./pages/app/settings/SettingsLayout"),
);
const Profile = lazy(() => import("./pages/app/settings/Profile"));
const UsersSettings = lazy(() => import("./pages/app/settings/Users"));
const Notifications = lazy(() => import("./pages/app/settings/Notifications"));
const Security = lazy(() => import("./pages/app/settings/Security"));
const Appearance = lazy(() => import("./pages/app/settings/Appearance"));
const Help = lazy(() => import("./pages/app/Help"));
const About = lazy(() => import("./pages/About"));
const Docs = lazy(() => import("./pages/Docs"));
const Contact = lazy(() => import("./pages/Contact"));
const NotFound = lazy(() => import("./pages/NotFound"));

// Preload critical routes after initial render
const preloadCriticalRoutes = () => {
  // Preload auth routes for unauthenticated users
  import("./pages/auth/Login");
  import("./pages/auth/Signup");
  // Preload dashboard for authenticated users
  import("./pages/app/Dashboard");
  import("./pages/Home");
};

const queryClient = new QueryClient();

// Loading fallback for lazy-loaded routes
function PageLoader() {
  return (
    <div className="flex items-center justify-center min-h-screen bg-background">
      <div className="space-y-4 w-full max-w-md px-4">
        <Skeleton className="h-8 w-3/4 mx-auto" />
        <Skeleton className="h-4 w-1/2 mx-auto" />
        <Skeleton className="h-32 w-full" />
      </div>
    </div>
  );
}

// Hook to trigger preloading after app mounts
function usePreloadRoutes() {
  useEffect(() => {
    // Use requestIdleCallback if available, otherwise setTimeout
    if ("requestIdleCallback" in window) {
      window.requestIdleCallback(preloadCriticalRoutes);
    } else {
      setTimeout(preloadCriticalRoutes, 1000);
    }
  }, []);
}

const App = () => {
  usePreloadRoutes();

  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <ThemeProvider>
          <AppearanceProvider>
            <AuthProvider>
              <TooltipProvider>
                <Toaster />
                <Sonner />
                <BrowserRouter>
                  <Suspense fallback={<PageLoader />}>
                    <Routes>
                      {/* Public */}
                      <Route path="/" element={<Home />} />
                      <Route path="/about" element={<About />} />
                      <Route path="/docs" element={<Docs />} />
                      <Route path="/contact" element={<Contact />} />
                      <Route path="/login" element={<Login />} />
                      <Route path="/signup" element={<Signup />} />
                      <Route
                        path="/forgot-password"
                        element={<ForgotPassword />}
                      />
                      <Route
                        path="/reset-password"
                        element={<ResetPassword />}
                      />
                      <Route path="/verify-email" element={<VerifyEmail />} />

                      {/* Protected App */}
                      <Route
                        path="/app"
                        element={
                          <ProtectedRoute>
                            <AppShell />
                          </ProtectedRoute>
                        }
                      >
                        <Route
                          index
                          element={<Navigate to="/app/dashboard" replace />}
                        />
                        <Route path="dashboard" element={<Dashboard />} />
                        <Route path="appliances" element={<Appliances />} />
                        <Route
                          path="appliances/:name"
                          element={<ApplianceDetails />}
                        />
                        <Route path="buildings" element={<Buildings />} />
                        <Route path="reports" element={<Reports />} />
                        <Route path="model" element={<Model />} />
                        <Route path="settings" element={<SettingsLayout />}>
                          <Route
                            index
                            element={
                              <Navigate to="/app/settings/profile" replace />
                            }
                          />
                          <Route path="profile" element={<Profile />} />
                          <Route path="users" element={<UsersSettings />} />
                          <Route
                            path="notifications"
                            element={<Notifications />}
                          />
                          <Route path="security" element={<Security />} />
                          <Route path="appearance" element={<Appearance />} />
                        </Route>
                        <Route path="help" element={<Help />} />
                      </Route>

                      <Route path="*" element={<NotFound />} />
                    </Routes>
                  </Suspense>
                </BrowserRouter>
              </TooltipProvider>
            </AuthProvider>
          </AppearanceProvider>
        </ThemeProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  );
};

export default App;
