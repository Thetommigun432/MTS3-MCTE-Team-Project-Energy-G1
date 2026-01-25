"""
Structured JSON logging configuration.
Outputs JSON logs to stdout for cloud-native deployments.
"""

import logging
import sys
from contextvars import ContextVar
from typing import Any

from pythonjsonlogger import jsonlogger

# Context variable for request ID propagation
request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)


class RequestIdFilter(logging.Filter):
    """Add request_id to log records from context variable."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()  # type: ignore[attr-defined]
        return True


class SafeJsonFormatter(jsonlogger.JsonFormatter):
    """
    JSON formatter that:
    - Adds standard fields (timestamp, level, logger)
    - Excludes sensitive data
    - Truncates large arrays/strings
    """

    SENSITIVE_KEYS = frozenset({
        "password", "token", "secret", "authorization", "api_key",
        "jwt", "bearer", "credential", "auth",
    })
    MAX_ARRAY_LENGTH = 10
    MAX_STRING_LENGTH = 500

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)

        # Add standard fields
        log_record["timestamp"] = self.formatTime(record)
        log_record["level"] = record.levelname
        log_record["logger"] = record.name

        # Add request_id if present
        if hasattr(record, "request_id") and record.request_id:
            log_record["request_id"] = record.request_id

        # Sanitize sensitive data
        self._sanitize_dict(log_record)

    def _sanitize_dict(self, d: dict[str, Any]) -> None:
        """Recursively sanitize dictionary, removing sensitive data."""
        for key in list(d.keys()):
            lower_key = key.lower()

            # Check for sensitive keys
            if any(sensitive in lower_key for sensitive in self.SENSITIVE_KEYS):
                d[key] = "[REDACTED]"
                continue

            value = d[key]

            # Truncate large arrays (e.g., window data)
            if isinstance(value, (list, tuple)):
                if len(value) > self.MAX_ARRAY_LENGTH:
                    d[key] = f"[{len(value)} items, truncated]"
                continue

            # Truncate long strings
            if isinstance(value, str) and len(value) > self.MAX_STRING_LENGTH:
                d[key] = value[: self.MAX_STRING_LENGTH] + "...[truncated]"
                continue

            # Recurse into nested dicts
            if isinstance(value, dict):
                self._sanitize_dict(value)


def setup_logging(level: str = "INFO") -> None:
    """
    Configure structured JSON logging to stdout.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        SafeJsonFormatter(
            fmt="%(timestamp)s %(level)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    handler.addFilter(RequestIdFilter())

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper()))

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)
