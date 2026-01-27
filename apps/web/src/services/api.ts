/**
 * API Service Layer
 * Provides a typed interface for making HTTP requests to the backend API.
 */

import { getEnv } from "@/lib/env";
import { supabase } from "@/integrations/supabase/client";

/**
 * Error types for better error handling and user feedback.
 */
export enum ApiErrorType {
  /** Network error - DNS failure, no connection, timeout */
  NETWORK = "NETWORK",
  /** CORS blocked - request made but blocked by browser */
  CORS = "CORS",
  /** 401/403 - unauthorized or forbidden */
  AUTH = "AUTH",
  /** 404 - endpoint not found */
  NOT_FOUND = "NOT_FOUND",
  /** 5xx - server error */
  SERVER_ERROR = "SERVER_ERROR",
  /** 4xx (other than 401/403/404) - client error */
  CLIENT_ERROR = "CLIENT_ERROR",
  /** Configuration error (e.g., missing base URL) */
  CONFIG = "CONFIG",
  /** Unknown error */
  UNKNOWN = "UNKNOWN",
}

export class ApiError extends Error {
  public readonly errorType: ApiErrorType;

  constructor(
    message: string,
    public status: number,
    public code?: string,
    public requestId?: string,
    public details?: unknown,
    errorType?: ApiErrorType,
  ) {
    super(message);
    this.name = "ApiError";
    this.errorType = errorType ?? ApiError.inferErrorType(status, code);
  }

  /**
   * Infer error type from status code and error code.
   */
  private static inferErrorType(status: number, code?: string): ApiErrorType {
    if (status === 0) {
      // Status 0 typically means network error or CORS
      if (code === "CORS_ERROR") return ApiErrorType.CORS;
      return ApiErrorType.NETWORK;
    }
    if (status === 401 || status === 403) return ApiErrorType.AUTH;
    if (status === 404) return ApiErrorType.NOT_FOUND;
    if (status >= 500) return ApiErrorType.SERVER_ERROR;
    if (status >= 400) return ApiErrorType.CLIENT_ERROR;
    return ApiErrorType.UNKNOWN;
  }

  /**
   * Get a user-friendly error message based on error type.
   */
  getUserMessage(): string {
    switch (this.errorType) {
      case ApiErrorType.NETWORK:
        return "Cannot reach the server. Check your internet connection.";
      case ApiErrorType.CORS:
        return "Request blocked by browser security (CORS). Backend may need to allow this origin.";
      case ApiErrorType.AUTH:
        return this.status === 401
          ? "Please log in to continue."
          : "You don't have permission to access this resource.";
      case ApiErrorType.NOT_FOUND:
        return "The requested resource was not found.";
      case ApiErrorType.SERVER_ERROR:
        return "Server error. Please try again later.";
      case ApiErrorType.CONFIG:
        return "API not configured. Please set VITE_BACKEND_URL.";
      default:
        return this.message;
    }
  }
}


interface RequestOptions extends Omit<RequestInit, "body"> {
  body?: unknown;
  params?: Record<string, string | number | boolean | undefined>;
}

/**
 * Build URL with query parameters
 * Handles both absolute URLs and relative paths (e.g. /api proxy)
 */
function buildUrl(endpoint: string, params?: RequestOptions["params"]): string {
  const { backendBaseUrl: API_BASE_URL } = getEnv();

  if (!API_BASE_URL) {
    throw new ApiError("API base URL is not configured", 0);
  }

  let urlString: string;

  // Handle relative base URLs (e.g. "/api" for proxying)
  if (API_BASE_URL.startsWith("http")) {
    const url = new URL(endpoint, API_BASE_URL);
    urlString = url.toString();
  } else {
    // Enforce /api prefix normalization
    // If endpoint is "models", it becomes "/api/models"
    // If endpoint is "/api/models", it stays "/api/models"
    let cleanEndpoint = endpoint.startsWith("/") ? endpoint : `/${endpoint}`;
    if (!cleanEndpoint.startsWith("/api") && !cleanEndpoint.startsWith("/health") && !cleanEndpoint.startsWith("/live") && !cleanEndpoint.startsWith("/ready")) {
      cleanEndpoint = `/api${cleanEndpoint}`;
    }

    // Clean up slashes for manual concatenation
    const base = API_BASE_URL.replace(/\/$/, "");
    const path = cleanEndpoint;
    urlString = `${base}${path}`;
  }

  // Append query params manually since we might have a relative URL string
  if (params) {
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        searchParams.append(key, String(value));
      }
    });
    const queryString = searchParams.toString();
    if (queryString) {
      urlString += (urlString.includes("?") ? "&" : "?") + queryString;
    }
  }

  return urlString;
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

  // Add auth token from Supabase session
  if (import.meta.env.VITE_SUPABASE_URL) {
    try {
      const { data } = await supabase.auth.getSession();
      const token = data.session?.access_token;

      if (token) {
        (headers as Record<string, string>)["Authorization"] = `Bearer ${token}`;
      }
    } catch {
      // Session fetch failed, skip auth header
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

    // Detect CORS errors - they typically manifest as TypeError with specific patterns
    // CORS errors don't provide status codes, so we detect them heuristically
    const isCorsError = error instanceof TypeError && (
      error.message.includes("Failed to fetch") ||
      error.message.includes("NetworkError") ||
      error.message.includes("CORS")
    );

    if (isCorsError) {
      // Check if this might be a CORS issue by trying a simple check
      // In browsers, CORS errors often appear as generic network errors
      throw new ApiError(
        "Request may be blocked by CORS policy. Check backend CORS configuration.",
        0,
        "CORS_ERROR",
        requestId,
        undefined,
        ApiErrorType.CORS,
      );
    }

    throw new ApiError(
      error instanceof Error ? error.message : "Network error",
      0,
      "NETWORK_ERROR",
      requestId,
      undefined,
      ApiErrorType.NETWORK,
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
  return Boolean(getEnv().backendBaseUrl);
}

/**
 * Get the configured API base URL
 */
export function getApiBaseUrl(): string {
  return getEnv().backendBaseUrl;
}
