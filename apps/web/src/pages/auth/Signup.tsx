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
import { Loader2, Eye, EyeOff, CheckCircle2 } from "lucide-react";
import {
  WaveformIcon,
  WaveformDecoration,
} from "@/components/brand/WaveformIcon";
import { cn } from "@/lib/utils";

export default function Signup() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const { signup, isAuthenticated, loading } = useAuth();
  const navigate = useNavigate();

  // Password validation
  const passwordLength = password.length >= 6;
  const passwordHasContent = password.length > 0;

  useEffect(() => {
    if (!loading && isAuthenticated) {
      navigate("/app/dashboard", { replace: true });
    }
  }, [isAuthenticated, loading, navigate]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    // Validate email
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!email || !emailRegex.test(email)) {
      setError("Please enter a valid email address.");
      return;
    }

    if (!password) {
      setError("Please enter a password.");
      return;
    }

    if (password.length < 6) {
      setError("Password must be at least 6 characters.");
      return;
    }

    setSubmitting(true);
    const { error: signupError } = await signup(
      email,
      password,
      displayName || undefined,
    );
    setSubmitting(false);

    if (signupError) {
      setError(signupError);
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
            Create an account
          </CardTitle>
          <CardDescription>
            Start monitoring your energy consumption today
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
              <Label htmlFor="displayName">
                Display Name{" "}
                <span className="text-muted-foreground">(optional)</span>
              </Label>
              <Input
                id="displayName"
                type="text"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                placeholder="John Doe"
                autoComplete="name"
              />
            </div>

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
              <Label htmlFor="password">Password</Label>
              <div className="relative">
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  autoComplete="new-password"
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
              {/* Password requirements */}
              <div className="space-y-1 pt-1">
                <div
                  className={cn(
                    "flex items-center gap-2 text-xs",
                    passwordHasContent
                      ? passwordLength
                        ? "text-energy-success"
                        : "text-muted-foreground"
                      : "text-muted-foreground",
                  )}
                >
                  <CheckCircle2
                    className={cn(
                      "h-3.5 w-3.5",
                      passwordLength ? "opacity-100" : "opacity-40",
                    )}
                  />
                  At least 6 characters
                </div>
              </div>
            </div>

            <Button type="submit" className="w-full" disabled={submitting}>
              {submitting ? (
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
              ) : null}
              {submitting ? "Creating account..." : "Create account"}
            </Button>
          </form>
        </CardContent>

        <CardFooter className="flex flex-col space-y-4">
          <div className="relative w-full">
            <div className="absolute inset-0 flex items-center">
              <span className="w-full border-t border-border" />
            </div>
            <div className="relative flex justify-center text-xs uppercase">
              <span className="bg-card px-2 text-muted-foreground">
                Already have an account?
              </span>
            </div>
          </div>
          <Link to="/login" className="w-full">
            <Button variant="outline" className="w-full">
              Sign in
            </Button>
          </Link>
        </CardFooter>
      </Card>
    </div>
  );
}
