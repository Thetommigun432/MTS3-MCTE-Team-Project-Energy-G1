type RequiredEnv = {
  VITE_SUPABASE_URL?: string;
  VITE_SUPABASE_PUBLISHABLE_KEY?: string;
};

export type AppEnv = {
  supabaseUrl: string;
  supabaseAnonKey: string;
  apiBaseUrl: string;
  demoMode: boolean;
  localMode: boolean;
};

let cachedEnv: AppEnv | null = null;

function readEnv(): AppEnv {
  const env = import.meta.env as RequiredEnv & Record<string, string | undefined>;
  const supabaseUrl = env.VITE_SUPABASE_URL?.trim();
  const supabaseAnonKey = env.VITE_SUPABASE_PUBLISHABLE_KEY?.trim();

  const missing: string[] = [];
  if (!supabaseUrl) missing.push("VITE_SUPABASE_URL");
  if (!supabaseAnonKey) missing.push("VITE_SUPABASE_PUBLISHABLE_KEY");

  if (missing.length) {
    const message = `Missing required environment variables: ${missing.join(", ")}`;
    console.error(message);
    throw new Error(message);
  }

  return {
    supabaseUrl,
    supabaseAnonKey,
    apiBaseUrl: env.VITE_API_BASE_URL?.trim() || "",
    demoMode: env.VITE_DEMO_MODE === "true",
    localMode: env.VITE_LOCAL_MODE === "true",
  };
}

export function getEnv(): AppEnv {
  if (!cachedEnv) {
    cachedEnv = readEnv();
  }
  return cachedEnv;
}

export function hasApiBaseUrl(): boolean {
  return Boolean(getEnv().apiBaseUrl);
}
