import { useState } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ArrowLeft, Mail, CheckCircle2, Clock } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";
import { NILMPanel } from "@/components/nilm/NILMPanel";
import { WaveformIcon } from "@/components/brand/WaveformIcon";
import { useRateLimit } from "@/hooks/useRateLimit";

export default function ForgotPassword() {
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  // Rate limit: 3 attempts per minute, 60 second cooldown
  const { isLimited, remainingTime, attempt } = useRateLimit(3, 60000, 60000);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Check rate limit before proceeding
    if (!attempt()) {
      toast.error(`Too many requests. Please wait ${remainingTime} seconds.`);
      return;
    }

    setLoading(true);

    const { error } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo: `${window.location.origin}/reset-password`,
    });

    setLoading(false);

    if (error) {
      // Don't expose internal error details - provide generic message
      toast.error("Unable to send reset email. Please try again later.");
    } else {
      setSubmitted(true);
      toast.success("Check your email for the reset link");
    }
  };

  if (submitted) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background px-4">
        <div className="w-full max-w-md space-y-6">
          {/* Logo */}
          <div className="flex flex-col items-center mb-8">
            <Link to="/" className="flex items-center gap-2 mb-2">
              <WaveformIcon className="h-10 w-10 text-primary" />
              <span className="text-2xl font-bold text-foreground">
                Energy Monitor
              </span>
            </Link>
          </div>

          <NILMPanel
            title="Check Your Email"
            icon={<Mail className="h-5 w-5" />}
          >
            <div className="text-center py-4">
              <CheckCircle2 className="h-12 w-12 text-energy-success mx-auto mb-4" />
              <p className="text-foreground mb-2">
                We've sent a password reset link to
              </p>
              <p className="font-medium text-foreground mb-4">{email}</p>
              <p className="text-sm text-muted-foreground mb-6">
                Didn't receive the email? Check your spam folder or try again.
              </p>
              <div className="space-y-3">
                <Button
                  variant="outline"
                  className="w-full"
                  onClick={() => setSubmitted(false)}
                >
                  Try again
                </Button>
                <Link
                  to="/login"
                  className="block text-center text-sm text-primary hover:underline"
                >
                  Back to login
                </Link>
              </div>
            </div>
          </NILMPanel>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-background px-4">
      <div className="w-full max-w-md space-y-6">
        {/* Logo */}
        <div className="flex flex-col items-center mb-8">
          <Link to="/" className="flex items-center gap-2 mb-2">
            <WaveformIcon className="h-10 w-10 text-primary" />
            <span className="text-2xl font-bold text-foreground">
              Energy Monitor
            </span>
          </Link>
          <p className="text-muted-foreground text-sm">Reset your password</p>
        </div>

        <NILMPanel title="Forgot Password" icon={<Mail className="h-5 w-5" />}>
          <Link
            to="/login"
            className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground mb-4"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to login
          </Link>

          <p className="text-muted-foreground text-sm mb-4">
            Enter your email and we'll send you a reset link.
          </p>

          {isLimited && (
            <div className="flex items-center gap-2 p-3 mb-4 rounded-md bg-amber-500/10 border border-amber-500/20 text-amber-600 dark:text-amber-400">
              <Clock className="h-4 w-4 shrink-0" />
              <span className="text-sm">
                Too many attempts. Please wait {remainingTime} seconds.
              </span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                disabled={isLimited}
              />
            </div>
            <Button
              type="submit"
              className="w-full"
              disabled={loading || isLimited}
            >
              {loading ? "Sending..." : "Send reset link"}
            </Button>
          </form>
        </NILMPanel>
      </div>
    </div>
  );
}
