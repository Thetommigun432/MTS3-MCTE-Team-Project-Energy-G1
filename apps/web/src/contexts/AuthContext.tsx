import {
  createContext,
  useContext,
  useState,
  useEffect,
  ReactNode,
} from "react";
import { User, Session } from "@supabase/supabase-js";
import { supabase } from "@/integrations/supabase/client";
import { invokeFunction } from "@/lib/supabaseHelpers";
import { setRememberMe } from "@/lib/authStorage";
import { isSupabaseEnabled } from "@/lib/env";

interface Profile {
  id: string;
  email: string | null;
  display_name: string | null;
  avatar_url: string | null;
  role: "admin" | "user" | "member" | "viewer" | null;
}

// Type for User Agent Client Hints API (not in all TypeScript libs)
interface UADataBrand {
  brand: string;
  version: string;
}
interface UADataValues {
  brands?: UADataBrand[];
}

interface AuthContextType {
  isAuthenticated: boolean;
  user: User | null;
  session: Session | null;
  profile: Profile | null;
  loading: boolean;
  login: (
    email: string,
    password: string,
    rememberMe?: boolean,
  ) => Promise<{ error: string | null }>;
  signup: (
    email: string,
    password: string,
    displayName?: string,
  ) => Promise<{ error: string | null }>;
  logout: () => Promise<void>;
  refreshProfile: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | null>(null);

// Demo user for when Supabase is not configured
const DEMO_USER: User = {
  id: "demo-user-id",
  email: "demo@example.com",
  app_metadata: {},
  user_metadata: { display_name: "Demo User" },
  aud: "authenticated",
  created_at: new Date().toISOString(),
} as User;

const DEMO_SESSION: Session = {
  access_token: "demo-token",
  refresh_token: "demo-refresh",
  expires_in: 3600,
  expires_at: Math.floor(Date.now() / 1000) + 3600,
  token_type: "bearer",
  user: DEMO_USER,
} as Session;

const DEMO_PROFILE: Profile = {
  id: "demo-user-id",
  email: "demo@example.com",
  display_name: "Demo User",
  avatar_url: null,
  role: "admin",
};

export function AuthProvider({ children }: { children: ReactNode }) {
  const supabaseEnabled = isSupabaseEnabled();
  const [user, setUser] = useState<User | null>(supabaseEnabled ? null : DEMO_USER);
  const [session, setSession] = useState<Session | null>(supabaseEnabled ? null : DEMO_SESSION);
  const [profile, setProfile] = useState<Profile | null>(supabaseEnabled ? null : DEMO_PROFILE);
  const [loading, setLoading] = useState(supabaseEnabled);

  const fetchProfile = async (userId: string) => {
    if (!supabaseEnabled) return;

    const { data, error } = await supabase
      .from("profiles")
      .select("*")
      .eq("id", userId)
      .maybeSingle();

    if (!error && data) {
      setProfile({
        id: data.id,
        email: data.email,
        display_name: data.display_name,
        avatar_url: data.avatar_url,
        role: (data as { role?: Profile["role"] }).role ?? null,
      });
    }
  };

  // Log login event to Supabase
  const logLoginEvent = async (userId: string) => {
    if (!supabaseEnabled) return;

    try {
      const uaBrands = (navigator as Navigator & { userAgentData?: UADataValues })
        .userAgentData?.brands
        ?.map((entry: UADataBrand) => `${entry.brand}/${entry.version}`)
        .join(" ");

      await invokeFunction("log-login-event", {
        user_id: userId,
        user_agent: uaBrands || navigator.userAgent,
      });
    } catch (err) {
      console.warn("Failed to log login event:", err);
    }
  };

  useEffect(() => {
    // Skip Supabase setup if not enabled (demo/local mode)
    if (!supabaseEnabled) {
      return;
    }

    // Set up auth state listener FIRST
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
      setUser(session?.user ?? null);

      // Defer profile fetch to avoid deadlock
      // Capture user ID safely to prevent race condition
      if (session?.user) {
        const userId = session.user.id;
        setTimeout(() => {
          fetchProfile(userId);
        }, 0);
      } else {
        setProfile(null);
      }

      setLoading(false);
    });

    // THEN check for existing session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      setUser(session?.user ?? null);

      if (session?.user) {
        fetchProfile(session.user.id);
      }

      setLoading(false);
    });

    return () => subscription.unsubscribe();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- fetchProfile is stable, only re-run on supabaseEnabled change
  }, [supabaseEnabled]);

  const login = async (
    email: string,
    password: string,
    rememberMe = true,
  ): Promise<{ error: string | null }> => {
    // In demo mode, auto-login as demo user
    if (!supabaseEnabled) {
      setUser(DEMO_USER);
      setSession(DEMO_SESSION);
      setProfile(DEMO_PROFILE);
      return { error: null };
    }

    // Set remember me preference before login
    setRememberMe(rememberMe);

    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password,
    });

    if (error) {
      return { error: error.message };
    }

    // Log successful login event
    if (data.user) {
      logLoginEvent(data.user.id);
    }

    return { error: null };
  };

  const signup = async (
    email: string,
    password: string,
    displayName?: string,
  ): Promise<{ error: string | null }> => {
    // In demo mode, just set demo user
    if (!supabaseEnabled) {
      setUser(DEMO_USER);
      setSession(DEMO_SESSION);
      setProfile({ ...DEMO_PROFILE, display_name: displayName || email.split("@")[0] });
      return { error: null };
    }

    const redirectUrl = `${window.location.origin}/`;

    const { error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        emailRedirectTo: redirectUrl,
        data: {
          display_name: displayName || email.split("@")[0],
        },
      },
    });

    if (error) {
      if (error.message.includes("already registered")) {
        return {
          error: "This email is already registered. Please log in instead.",
        };
      }
      return { error: error.message };
    }
    return { error: null };
  };

  const logout = async () => {
    if (supabaseEnabled) {
      await supabase.auth.signOut();
    }
    // In demo mode without Supabase, restore demo user state
    if (!supabaseEnabled) {
      setUser(DEMO_USER);
      setSession(DEMO_SESSION);
      setProfile(DEMO_PROFILE);
    } else {
      setUser(null);
      setSession(null);
      setProfile(null);
    }
  };

  const refreshProfile = async () => {
    if (!supabaseEnabled) return;
    if (user) {
      await fetchProfile(user.id);
    }
  };

  return (
    <AuthContext.Provider
      value={{
        isAuthenticated: !!session,
        user,
        session,
        profile,
        loading,
        login,
        signup,
        logout,
        refreshProfile,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return context;
}
