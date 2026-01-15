import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  CardFooter,
} from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Loader2, Eye, EyeOff, UserCircle } from "lucide-react";
import {
  WaveformIcon,
  WaveformDecoration,
} from "@/components/brand/WaveformIcon";
import { supabase } from "@/integrations/supabase/client";
import { getRememberMe, setRememberMe } from "@/lib/authStorage";

// Demo credentials - always available for teacher presentations
const DEMO_EMAIL = "demo@energy-monitor.app";
const DEMO_PASSWORD = "DemoPass2026!";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [demoLoading, setDemoLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMeChecked, setRememberMeChecked] = useState(() =>
    getRememberMe(),
  );
  const { login, isAuthenticated, loading } = useAuth();
  const navigate = useNavigate();

  // Redirect if already authenticated
  useEffect(() => {
    if (!loading && isAuthenticated) {
      navigate("/app/dashboard", { replace: true });
    }
  }, [isAuthenticated, loading, navigate]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    // Allow "demo" shortcut to log in as demo user
    let loginEmail = email;
    if (email.toLowerCase() === "demo" && !email.includes("@")) {
      loginEmail = DEMO_EMAIL;
    }

    // Validate email
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!loginEmail || !emailRegex.test(loginEmail)) {
      setError("Please enter a valid email address.");
      return;
    }

    if (!password) {
      setError("Please enter your password.");
      return;
    }

    if (password.length < 6) {
      setError("Password must be at least 6 characters.");
      return;
    }

    // Save remember me preference before login
    setRememberMe(rememberMeChecked);

    setSubmitting(true);
    const { error: loginError } = await login(loginEmail, password);
    setSubmitting(false);

    if (loginError) {
      setError(loginError);
    }
  };

  // Handle demo login - tries direct login first, then signup if user doesn't exist
  const handleDemoLogin = async () => {
    setError("");
    setDemoLoading(true);

    try {
      // First, try to sign in directly with demo credentials
      setRememberMe(true);
      const { error: loginError } = await login(
        DEMO_EMAIL,
        DEMO_PASSWORD,
        true,
      );

      if (!loginError) {
        // Login successful, we're done
        return;
      }

      // If login failed, try to create the demo user via signup
      if (
        loginError.includes("Invalid login credentials") ||
        loginError.includes("invalid_credentials")
      ) {
        console.log("Demo user does not exist, attempting to create...");

        // Try to sign up the demo user
        const { error: signupError } = await supabase.auth.signUp({
          email: DEMO_EMAIL,
          password: DEMO_PASSWORD,
          options: {
            data: {
              display_name: "Demo Admin",
            },
          },
        });

        if (signupError) {
          console.error("Failed to create demo user:", signupError);
          setError(
            "Unable to create demo account. Please try again or contact support.",
          );
          return;
        }

        // Wait a moment for the user to be created, then try to sign in
        await new Promise((resolve) => setTimeout(resolve, 500));

        const { error: retryError } = await login(
          DEMO_EMAIL,
          DEMO_PASSWORD,
          true,
        );
        if (retryError) {
          // If email confirmation is required
          if (
            retryError.includes("confirm") ||
            retryError.includes("Email not confirmed")
          ) {
            setError(
              "Demo account created but requires email confirmation. Please check with your administrator.",
            );
          } else {
            setError(retryError);
          }
        }
      } else {
        setError(loginError);
      }
    } catch (err) {
      console.error("Demo login error:", err);
      setError("An unexpected error occurred. Please try again.");
    } finally {
      setDemoLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-6 bg-background relative overflow-hidden">
      {/* Background waveform decoration */}
      <div className="absolute inset-0 flex items-center justify-center opacity-[0.02] pointer-events-none">
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

      <Card className="w-full max-w-md border-border relative">
        {/* Watermark */}
        <div className="absolute top-4 right-4 opacity-5 pointer-events-none">
          <WaveformDecoration className="h-8 w-auto text-primary" />
        </div>

        <CardHeader className="text-center space-y-2">
          <Link to="/" className="flex justify-center mb-2">
            <WaveformIcon size="lg" />
          </Link>
          <CardTitle className="text-2xl text-foreground">
            Welcome back
          </CardTitle>
          <CardDescription>
            Sign in to your Energy Monitor account
          </CardDescription>
        </CardHeader>

        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <div className="text-sm text-destructive bg-energy-error-bg p-3 rounded-lg border border-destructive/20">
                {error}
              </div>
            )}

            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                autoComplete="email"
                required
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="password">Password</Label>
                <Link
                  to="/forgot-password"
                  className="text-xs text-primary hover:underline"
                >
                  Forgot password?
                </Link>
              </div>
              <div className="relative">
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  autoComplete="current-password"
                  className="pr-10"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                >
                  {showPassword ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </button>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <Checkbox
                id="remember"
                checked={rememberMeChecked}
                onCheckedChange={(checked) =>
                  setRememberMeChecked(checked === true)
                }
              />
              <Label
                htmlFor="remember"
                className="text-sm text-muted-foreground cursor-pointer"
              >
                Remember me
              </Label>
            </div>

            <Button
              type="submit"
              className="w-full"
              disabled={submitting || demoLoading}
            >
              {submitting ? (
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
              ) : null}
              {submitting ? "Signing in..." : "Sign in"}
            </Button>
          </form>

          {/* Demo login button - always visible for teacher presentations */}
          <div className="mt-6 pt-4 border-t border-border">
            <div className="text-center mb-3">
              <span className="text-xs text-muted-foreground uppercase tracking-wide">
                Quick Access
              </span>
            </div>
            <Button
              type="button"
              variant="outline"
              className="w-full bg-primary/5 hover:bg-primary/10 border-primary/20"
              onClick={handleDemoLogin}
              disabled={demoLoading || submitting}
            >
              {demoLoading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  Logging in...
                </>
              ) : (
                <>
                  <UserCircle className="h-4 w-4 mr-2" />
                  Login as Demo
                </>
              )}
            </Button>
            <p className="text-xs text-muted-foreground text-center mt-2">
              No account needed - explore the dashboard instantly
            </p>
          </div>
        </CardContent>

        <CardFooter className="flex flex-col space-y-4">
          <div className="relative w-full">
            <div className="absolute inset-0 flex items-center">
              <span className="w-full border-t border-border" />
            </div>
            <div className="relative flex justify-center text-xs uppercase">
              <span className="bg-card px-2 text-muted-foreground">
                New to Energy Monitor?
              </span>
            </div>
          </div>
          <Link to="/signup" className="w-full">
            <Button variant="outline" className="w-full">
              Create an account
            </Button>
          </Link>
        </CardFooter>
      </Card>
    </div>
  );
}
