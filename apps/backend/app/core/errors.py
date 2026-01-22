"""
Error codes and exception classes for consistent error handling.
All API errors follow a standard schema with error code, message, and request_id.
"""

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Standardized error codes for API responses."""

    # Authentication (401)
    AUTH_MISSING_TOKEN = "AUTH_MISSING_TOKEN"
    AUTH_INVALID_TOKEN = "AUTH_INVALID_TOKEN"
    AUTH_EXPIRED_TOKEN = "AUTH_EXPIRED_TOKEN"
    AUTH_VERIFICATION_FAILED = "AUTH_VERIFICATION_FAILED"

    # Authorization (403)
    AUTHZ_FORBIDDEN = "AUTHZ_FORBIDDEN"
    AUTHZ_INSUFFICIENT_ROLE = "AUTHZ_INSUFFICIENT_ROLE"
    AUTHZ_BUILDING_ACCESS_DENIED = "AUTHZ_BUILDING_ACCESS_DENIED"
    AUTHZ_APPLIANCE_ACCESS_DENIED = "AUTHZ_APPLIANCE_ACCESS_DENIED"

    # Validation (422)
    VALIDATION_ERROR = "VALIDATION_ERROR"
    VALIDATION_WINDOW_LENGTH = "VALIDATION_WINDOW_LENGTH"
    VALIDATION_WINDOW_NAN_INF = "VALIDATION_WINDOW_NAN_INF"
    VALIDATION_INVALID_ID = "VALIDATION_INVALID_ID"
    VALIDATION_INVALID_TIMESTAMP = "VALIDATION_INVALID_TIMESTAMP"
    VALIDATION_INVALID_RESOLUTION = "VALIDATION_INVALID_RESOLUTION"

    # Request Limits (413, 429)
    REQUEST_TOO_LARGE = "REQUEST_TOO_LARGE"
    RATE_LIMITED = "RATE_LIMITED"

    # Model/Inference (500)
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_LOAD_ERROR = "MODEL_LOAD_ERROR"
    MODEL_FACTORY_ERROR = "MODEL_FACTORY_ERROR"
    MODEL_ARTIFACT_INVALID = "MODEL_ARTIFACT_INVALID"
    INFERENCE_FAILED = "INFERENCE_FAILED"

    # InfluxDB (503)
    INFLUX_CONNECTION_ERROR = "INFLUX_CONNECTION_ERROR"
    INFLUX_QUERY_ERROR = "INFLUX_QUERY_ERROR"
    INFLUX_WRITE_FAILED = "INFLUX_WRITE_FAILED"
    INFLUX_BUCKET_MISSING = "INFLUX_BUCKET_MISSING"

    # Internal (500)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    REGISTRY_ERROR = "REGISTRY_ERROR"


class AppError(Exception):
    """Base application exception with error code and details."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class AuthenticationError(AppError):
    """Authentication failure (401)."""

    def __init__(
        self,
        code: ErrorCode = ErrorCode.AUTH_INVALID_TOKEN,
        message: str = "Authentication failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code, message, status_code=401, details=details)


class AuthorizationError(AppError):
    """Authorization failure (403)."""

    def __init__(
        self,
        code: ErrorCode = ErrorCode.AUTHZ_FORBIDDEN,
        message: str = "Access denied",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code, message, status_code=403, details=details)


class ValidationError(AppError):
    """Validation failure (422)."""

    def __init__(
        self,
        code: ErrorCode = ErrorCode.VALIDATION_ERROR,
        message: str = "Validation failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code, message, status_code=422, details=details)


class RequestTooLargeError(AppError):
    """Request body too large (413)."""

    def __init__(
        self,
        message: str = "Request body too large",
        max_bytes: int | None = None,
    ) -> None:
        details = {"max_bytes": max_bytes} if max_bytes else None
        super().__init__(ErrorCode.REQUEST_TOO_LARGE, message, status_code=413, details=details)


class RateLimitError(AppError):
    """Rate limit exceeded (429)."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
    ) -> None:
        details = {"retry_after_seconds": retry_after} if retry_after else None
        super().__init__(ErrorCode.RATE_LIMITED, message, status_code=429, details=details)


class ModelError(AppError):
    """Model-related error (500)."""

    def __init__(
        self,
        code: ErrorCode = ErrorCode.MODEL_LOAD_ERROR,
        message: str = "Model error",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code, message, status_code=500, details=details)


class InfluxError(AppError):
    """InfluxDB-related error (503)."""

    def __init__(
        self,
        code: ErrorCode = ErrorCode.INFLUX_CONNECTION_ERROR,
        message: str = "InfluxDB error",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code, message, status_code=503, details=details)


def error_response(
    error: AppError,
    request_id: str | None = None,
) -> dict[str, Any]:
    """
    Build a standardized error response dict.

    Returns:
        {
            "error": {"code": "...", "message": "...", "details": {...}},
            "request_id": "..."
        }
    """
    return {
        "error": {
            "code": error.code.value,
            "message": error.message,
            "details": error.details,
        },
        "request_id": request_id,
    }
