import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { Loader2 } from 'lucide-react';

/**
 * AdminRoute component - Guards routes that require admin privileges
 *
 * SECURITY NOTE: This is a UI-ONLY guard. The backend MUST validate
 * all admin operations via Supabase Row Level Security (RLS) policies
 * and edge function permission checks. Never trust client-side role checks.
 *
 * @param children - React components to render if user is an admin
 */
export function AdminRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, profile, loading } = useAuth();
  const location = useLocation();

  // Show loading spinner while checking auth state
  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  // Redirect to login if not authenticated
  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  // Redirect to dashboard if authenticated but not an admin
  // IMPORTANT: Backend must also validate this on every API call
  if (profile?.role !== 'admin') {
    return <Navigate to="/app/dashboard" replace />;
  }

  // User is authenticated and has admin role - render protected content
  return <>{children}</>;
}
