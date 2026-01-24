import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Mail, CheckCircle, XCircle, Loader2 } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";

export default function VerifyEmail() {
  const [status, setStatus] = useState<
    "loading" | "success" | "error" | "pending"
  >("loading");
  const navigate = useNavigate();

  useEffect(() => {
    // Check if we have a hash fragment (email confirmation flow)
    const hashParams = new URLSearchParams(window.location.hash.substring(1));
    const accessToken = hashParams.get("access_token");
    const type = hashParams.get("type");

    /* eslint-disable react-hooks/set-state-in-effect -- legitimate state update based on URL params */
    if (accessToken && type === "signup") {
      // User clicked confirmation link - they're now verified
      setStatus("success");
      // Redirect to app after a brief delay
      setTimeout(() => navigate("/app/dashboard"), 2000);
    } else {
      // Check current session
      supabase.auth.getSession().then(({ data: { session } }) => {
        if (session?.user?.email_confirmed_at) {
          setStatus("success");
        } else if (session?.user) {
          setStatus("pending");
        } else {
          setStatus("error");
        }
      });
    }
    /* eslint-enable react-hooks/set-state-in-effect */
  }, [navigate]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50 px-4">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          {status === "loading" && (
            <>
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-slate-100">
                <Loader2 className="h-6 w-6 text-slate-600 animate-spin" />
              </div>
              <CardTitle className="text-xl">Verifying your email...</CardTitle>
              <CardDescription>
                Please wait while we confirm your email address.
              </CardDescription>
            </>
          )}

          {status === "success" && (
            <>
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-green-100">
                <CheckCircle className="h-6 w-6 text-green-600" />
              </div>
              <CardTitle className="text-xl">Email verified!</CardTitle>
              <CardDescription>
                Your email has been successfully verified. Redirecting to
                dashboard...
              </CardDescription>
            </>
          )}

          {status === "pending" && (
            <>
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-amber-100">
                <Mail className="h-6 w-6 text-amber-600" />
              </div>
              <CardTitle className="text-xl">Check your inbox</CardTitle>
              <CardDescription>
                We've sent a verification link to your email address. Please
                click the link to verify your account.
              </CardDescription>
            </>
          )}

          {status === "error" && (
            <>
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-red-100">
                <XCircle className="h-6 w-6 text-red-600" />
              </div>
              <CardTitle className="text-xl">Verification failed</CardTitle>
              <CardDescription>
                We couldn't verify your email. The link may have expired or is
                invalid.
              </CardDescription>
            </>
          )}
        </CardHeader>

        <CardContent className="space-y-4">
          {status === "success" && (
            <Button asChild className="w-full">
              <Link to="/app/dashboard">Go to Dashboard</Link>
            </Button>
          )}

          {status === "pending" && (
            <>
              <p className="text-sm text-muted-foreground text-center">
                Didn't receive the email? Check your spam folder.
              </p>
              <Button asChild variant="outline" className="w-full">
                <Link to="/login">Back to login</Link>
              </Button>
            </>
          )}

          {status === "error" && (
            <>
              <Button asChild className="w-full">
                <Link to="/signup">Try signing up again</Link>
              </Button>
              <Button asChild variant="outline" className="w-full">
                <Link to="/login">Back to login</Link>
              </Button>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
