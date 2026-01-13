/**
 * Supabase Helper Functions
 * Typed wrapper for edge function calls with consistent error handling
 */

import { supabase } from '@/integrations/supabase/client';

export class EdgeFunctionError extends Error {
  constructor(
    message: string,
    public code?: string,
    public details?: unknown
  ) {
    super(message);
    this.name = 'EdgeFunctionError';
  }
}

interface EdgeFunctionResponse<T> {
  data: T | null;
  error: EdgeFunctionError | null;
}

/**
 * Invoke a Supabase edge function with typed request/response
 */
export async function invokeFunction<TRequest, TResponse>(
  functionName: string,
  payload?: TRequest
): Promise<EdgeFunctionResponse<TResponse>> {
  try {
    const { data, error } = await supabase.functions.invoke<TResponse>(functionName, {
      body: payload,
    });

    if (error) {
      return {
        data: null,
        error: new EdgeFunctionError(
          error.message || 'Edge function call failed',
          'FUNCTION_ERROR',
          error
        ),
      };
    }

    // Check if response contains an error field
    if (data && typeof data === 'object' && 'error' in data) {
      const errorData = data as { error: string; code?: string };
      return {
        data: null,
        error: new EdgeFunctionError(
          errorData.error,
          errorData.code || 'FUNCTION_ERROR'
        ),
      };
    }

    return { data, error: null };
  } catch (err) {
    return {
      data: null,
      error: new EdgeFunctionError(
        err instanceof Error ? err.message : 'Unknown error',
        'NETWORK_ERROR',
        err
      ),
    };
  }
}

// =====================================================
// Edge Function Types
// =====================================================

// Invite user
export interface InviteUserRequest {
  email: string;
  role?: string;
}

export interface InviteUserResponse {
  success: boolean;
  invitation_id?: string;
  message?: string;
}

// Delete account
export interface DeleteAccountRequest {
  confirmation: string;
}

export interface DeleteAccountResponse {
  success: boolean;
  message?: string;
}

// Accept invite
export interface AcceptInviteRequest {
  token: string;
}

export interface AcceptInviteResponse {
  success: boolean;
  message?: string;
}

// Log login event
export interface LogLoginEventRequest {
  user_agent?: string;
  success?: boolean;
}

export interface LogLoginEventResponse {
  success: boolean;
  event_id?: string;
}

// Register model
export interface RegisterModelRequest {
  org_appliance_id: string;
  name: string;
  architecture?: string;
}

export interface RegisterModelResponse {
  success: boolean;
  model_id?: string;
  message?: string;
}

// Create model version upload
export interface CreateModelVersionUploadRequest {
  model_id: string;
  version: string;
  has_scaler?: boolean;
}

export interface CreateModelVersionUploadResponse {
  success: boolean;
  version_id?: string;
  model_upload_url?: string;
  scaler_upload_url?: string;
  message?: string;
}

// Finalize model version
export interface FinalizeModelVersionRequest {
  version_id: string;
  metrics?: Record<string, number>;
  training_config?: Record<string, unknown>;
}

export interface FinalizeModelVersionResponse {
  success: boolean;
  message?: string;
}

// Set active model version
export interface SetActiveVersionRequest {
  version_id: string;
}

export interface SetActiveVersionResponse {
  success: boolean;
  message?: string;
}

// Run inference
export interface RunInferenceRequest {
  building_id: string;
  org_appliance_id: string;
  start_date: string;
  end_date: string;
}

export interface RunInferenceResponse {
  success: boolean;
  predictions_count?: number;
  message?: string;
}

// Get readings (predictions + aggregate)
export interface GetReadingsRequest {
  building_id: string;
  start_date?: string;
  end_date?: string;
}

export interface ReadingEntry {
  ts: string;
  aggregate_kw: number;
  appliance_estimates: Record<string, number>;
  confidence?: Record<string, number>;
}

export interface GetReadingsResponse {
  success: boolean;
  readings: ReadingEntry[];
  message?: string;
}

// Generate report
export interface GenerateReportRequest {
  building_id: string;
  start_date: string;
  end_date: string;
  appliance_ids?: string[];
}

export interface ReportSummary {
  total_energy_kwh: number;
  average_power_kw: number;
  peak_power_kw: number;
  peak_timestamp: string;
  data_points_analyzed: number;
}

export interface ApplianceBreakdown {
  appliance_name: string;
  appliance_slug: string;
  energy_kwh: number;
  percentage: number;
  avg_confidence: number;
}

export interface HourlyPattern {
  hour: number;
  avg_power_kw: number;
}

export interface GenerateReportResponse {
  success: boolean;
  summary?: ReportSummary;
  breakdown?: ApplianceBreakdown[];
  hourly_pattern?: HourlyPattern[];
  message?: string;
}

// Invite user (with org support)
export interface InviteUserWithOrgRequest {
  org_id: string;
  email: string;
  role: 'admin' | 'member' | 'viewer';
  redirect_to?: string;
}

export interface InviteUserWithOrgResponse {
  success: boolean;
  message?: string;
  user_id?: string;
  invited_user_id?: string;
}

// Get dashboard data
export interface GetDashboardDataRequest {
  building_id: string;
  start?: string;
  end?: string;
}

export interface ApplianceStatus {
  appliance_id: string;
  name: string;
  slug: string;
  current_power_kw: number;
  is_on: boolean;
  confidence: number;
}

export interface DashboardInsight {
  type: 'peak_load' | 'daily_usage' | 'top_consumer' | 'efficiency';
  label: string;
  value: string;
  trend?: 'up' | 'down' | 'stable';
}

export interface GetDashboardDataResponse {
  building_id: string;
  period: {
    start: string;
    end: string;
  };
  aggregate_series: { ts: string; power_kw: number }[];
  appliance_series: Record<string, { ts: string; power_kw: number; confidence: number }[]>;
  whats_on_now: ApplianceStatus[];
  insights: DashboardInsight[];
  data_points: number;
}

// Log auth event
export interface LogAuthEventRequest {
  event: 'login' | 'logout' | 'signup' | 'password_reset';
  user_agent?: string;
  success?: boolean;
}

export interface LogAuthEventResponse {
  success: boolean;
  event_id?: string;
}

// Upsert avatar
export interface UpsertAvatarRequest {
  avatar_base64: string;
  filename: string;
}

export interface UpsertAvatarResponse {
  success: boolean;
  avatar_url?: string;
  path?: string;
}

// =====================================================
// Typed Edge Function Wrappers
// =====================================================

export const edgeFunctions = {
  inviteUser: (payload: InviteUserRequest) =>
    invokeFunction<InviteUserRequest, InviteUserResponse>('admin-invite', payload),
  
  // Invite user to organization
  inviteUserToOrg: (payload: InviteUserWithOrgRequest) =>
    invokeFunction<InviteUserWithOrgRequest, InviteUserWithOrgResponse>('invite-user', payload),

  deleteAccount: (payload: DeleteAccountRequest) =>
    invokeFunction<DeleteAccountRequest, DeleteAccountResponse>('delete-account', payload),

  acceptInvite: (payload: AcceptInviteRequest) =>
    invokeFunction<AcceptInviteRequest, AcceptInviteResponse>('accept-invite', payload),

  logLoginEvent: (payload: LogLoginEventRequest) =>
    invokeFunction<LogLoginEventRequest, LogLoginEventResponse>('log-login-event', payload),

  registerModel: (payload: RegisterModelRequest) =>
    invokeFunction<RegisterModelRequest, RegisterModelResponse>('register-model', payload),

  createModelVersionUpload: (payload: CreateModelVersionUploadRequest) =>
    invokeFunction<CreateModelVersionUploadRequest, CreateModelVersionUploadResponse>(
      'create-model-version-upload',
      payload
    ),

  finalizeModelVersion: (payload: FinalizeModelVersionRequest) =>
    invokeFunction<FinalizeModelVersionRequest, FinalizeModelVersionResponse>(
      'finalize-model-version',
      payload
    ),

  setActiveVersion: (payload: SetActiveVersionRequest) =>
    invokeFunction<SetActiveVersionRequest, SetActiveVersionResponse>(
      'set-active-model-version',
      payload
    ),

  runInference: (payload: RunInferenceRequest) =>
    invokeFunction<RunInferenceRequest, RunInferenceResponse>('run-inference', payload),

  getReadings: (payload: GetReadingsRequest) =>
    invokeFunction<GetReadingsRequest, GetReadingsResponse>('get-readings', payload),

  generateReport: (payload: GenerateReportRequest) =>
    invokeFunction<GenerateReportRequest, GenerateReportResponse>('generate-report', payload),

  // New functions
  getDashboardData: (payload: GetDashboardDataRequest) =>
    invokeFunction<GetDashboardDataRequest, GetDashboardDataResponse>('get-dashboard-data', payload),

  logAuthEvent: (payload: LogAuthEventRequest) =>
    invokeFunction<LogAuthEventRequest, LogAuthEventResponse>('log-auth-event', payload),

  upsertAvatar: (payload: UpsertAvatarRequest) =>
    invokeFunction<UpsertAvatarRequest, UpsertAvatarResponse>('upsert-avatar', payload),
};
