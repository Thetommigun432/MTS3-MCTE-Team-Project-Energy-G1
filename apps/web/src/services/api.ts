/**
 * API Service Layer
 * Provides a typed interface for making HTTP requests to the backend API.
 */

import { getEnv } from "@/lib/env";

const { backendBaseUrl: API_BASE_URL } = getEnv();

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public code?: string,
    public requestId?: string,
    public details?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

interface RequestOptions extends Omit<RequestInit, "body"> {
  body?: unknown;
  params?: Record<string, string | number | boolean | undefined>;
}

/**
 * Build URL with query parameters
 */
function buildUrl(endpoint: string, params?: RequestOptions["params"]): string {
  if (!API_BASE_URL) {
    throw new ApiError("API base URL is not configured", 0);
  }

  const url = new URL(endpoint, API_BASE_URL);

  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        url.searchParams.append(key, String(value));
      }
    });
  }

  return url.toString();
}

/**
 * Generate a unique request ID for tracing
 */
function generateRequestId(): string {
  return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Core fetch wrapper with error handling
 */
async function request<T>(
  endpoint: string,
  options: RequestOptions = {},
): Promise<T> {
  const { body, params, headers: customHeaders, ...restOptions } = options;

  // Generate request ID for tracing
  const requestId = generateRequestId();

  const headers: HeadersInit = {
    "Content-Type": "application/json",
    "X-Request-ID": requestId,
    ...customHeaders,
  };

  // Add auth token if available - dynamically construct key from Supabase URL
  const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
  if (supabaseUrl) {
    try {
      const projectRef = new URL(supabaseUrl).hostname.split(".")[0];
      const token = localStorage.getItem(`sb-${projectRef}-auth-token`);
      if (token) {
        const parsed = JSON.parse(token);
        if (parsed?.access_token) {
          (headers as Record<string, string>)["Authorization"] =
            `Bearer ${parsed.access_token}`;
        }
      }
    } catch {
      // Invalid URL or token format, skip
    }
  }

  const config: RequestInit = {
    ...restOptions,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  };

  const url = buildUrl(endpoint, params);

  // Log in development
  if (import.meta.env.DEV) {
    console.log(`[API] ${options.method || 'GET'} ${endpoint} [${requestId}]`);
  }

  try {
    const response = await fetch(url, config);

    if (!response.ok) {
      let errorData: unknown;
      try {
        errorData = await response.json();
      } catch {
        errorData = await response.text();
      }

      // Parse backend error format: { error: { code, message, details }, request_id }
      if (errorData && typeof errorData === 'object' && 'error' in errorData) {
        const backendError = errorData as {
          error?: { code?: string; message?: string; details?: unknown };
          request_id?: string;
        };

        throw new ApiError(
          backendError.error?.message || `Request failed with status ${response.status}`,
          response.status,
          backendError.error?.code,
          backendError.request_id || requestId,
          backendError.error?.details,
        );
      }

      // Fallback for non-standard error format
      throw new ApiError(
        `Request failed with status ${response.status}`,
        response.status,
        undefined,
        requestId,
        errorData,
      );
    }

    // Handle empty responses
    const contentType = response.headers.get("content-type");
    if (contentType?.includes("application/json")) {
      return response.json();
    }

    return {} as T;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError(
      error instanceof Error ? error.message : "Network error",
      0,
      "NETWORK_ERROR",
      requestId,
    );
  }
}

/**
 * API client with typed methods
 */
export const api = {
  get: <T>(endpoint: string, options?: Omit<RequestOptions, "body">) =>
    request<T>(endpoint, { ...options, method: "GET" }),

  post: <T>(endpoint: string, body?: unknown, options?: RequestOptions) =>
    request<T>(endpoint, { ...options, method: "POST", body }),

  put: <T>(endpoint: string, body?: unknown, options?: RequestOptions) =>
    request<T>(endpoint, { ...options, method: "PUT", body }),

  patch: <T>(endpoint: string, body?: unknown, options?: RequestOptions) =>
    request<T>(endpoint, { ...options, method: "PATCH", body }),

  delete: <T>(endpoint: string, options?: RequestOptions) =>
    request<T>(endpoint, { ...options, method: "DELETE" }),
};

/**
 * Check if API is configured
 */
export function isApiConfigured(): boolean {
  return Boolean(API_BASE_URL);
}

/**
 * Get the configured API base URL
 */
export function getApiBaseUrl(): string {
  return API_BASE_URL;
}
