"""
FastAPI dependencies for route handlers.
"""

from typing import Annotated

from fastapi import Depends, Header, Request

from app.core.config import Settings, get_settings
from app.core.errors import AuthenticationError, ErrorCode
from app.core.security import TokenPayload, extract_token_from_header, verify_token
from app.schemas.auth import CurrentUser


def get_request_id(request: Request) -> str:
    """Get request ID from request state."""
    return getattr(request.state, "request_id", "unknown")


async def get_current_user(
    authorization: Annotated[str | None, Header()] = None,
) -> TokenPayload:
    """
    Dependency to get the current authenticated user.

    Extracts and verifies the JWT from the Authorization header.

    Raises:
        AuthenticationError: If token is missing or invalid
    """
    token = extract_token_from_header(authorization)
    return verify_token(token)


async def get_optional_user(
    authorization: Annotated[str | None, Header()] = None,
) -> TokenPayload | None:
    """
    Dependency to get the current user if authenticated.

    Returns None if no valid token is provided.
    """
    if not authorization:
        return None

    try:
        token = extract_token_from_header(authorization)
        return verify_token(token)
    except AuthenticationError:
        return None


async def require_admin_token(
    request: Request,
    admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
) -> None:
    """
    Dependency to require admin token in production.

    Admin endpoints can be protected by both:
    1. User role (admin)
    2. Admin token header (X-Admin-Token)
    """
    settings = get_settings()

    # In prod with ADMIN_TOKEN configured, require the header
    if settings.env == "prod" and settings.admin_token:
        if admin_token != settings.admin_token:
            raise AuthenticationError(
                code=ErrorCode.AUTH_MISSING_TOKEN,
                message="Admin token required",
            )


# Type aliases for dependency injection
CurrentUserDep = Annotated[TokenPayload, Depends(get_current_user)]
OptionalUserDep = Annotated[TokenPayload | None, Depends(get_optional_user)]
RequestIdDep = Annotated[str, Depends(get_request_id)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
