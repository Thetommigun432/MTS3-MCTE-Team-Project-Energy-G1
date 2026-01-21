"""
JWT verification and authentication.
Supports HS256 (default) and RS256 (via JWKS).
"""

import time
from typing import Any

import httpx
import jwt
from jwt import PyJWKClient

from app.core.config import get_settings
from app.core.errors import AuthenticationError, ErrorCode
from app.core.logging import get_logger
from app.core.telemetry import AUTH_VERIFY_LATENCY

logger = get_logger(__name__)


class JWKSCache:
    """Cached JWKS client for RS256 verification."""

    def __init__(self, jwks_url: str, cache_ttl_hours: int = 6) -> None:
        self.jwks_url = jwks_url
        self.cache_ttl_seconds = cache_ttl_hours * 3600
        self._client: PyJWKClient | None = None
        self._last_refresh: float = 0

    def get_signing_key(self, token: str) -> Any:
        """Get signing key from JWKS, refreshing cache if needed."""
        now = time.time()
        if self._client is None or (now - self._last_refresh) > self.cache_ttl_seconds:
            self._refresh_client()
        return self._client.get_signing_key_from_jwt(token)  # type: ignore[union-attr]

    def _refresh_client(self) -> None:
        """Refresh the JWKS client."""
        logger.info("Refreshing JWKS cache", extra={"jwks_url": self.jwks_url})
        self._client = PyJWKClient(self.jwks_url)
        self._last_refresh = time.time()


# Global JWKS cache (initialized lazily)
_jwks_cache: JWKSCache | None = None


def _get_jwks_cache() -> JWKSCache | None:
    """Get or create JWKS cache if configured."""
    global _jwks_cache
    settings = get_settings()

    if not settings.supabase_jwks_url:
        return None

    if _jwks_cache is None:
        _jwks_cache = JWKSCache(
            settings.supabase_jwks_url,
            settings.jwks_cache_ttl_hours,
        )
    return _jwks_cache


class TokenPayload:
    """Parsed JWT token payload."""

    def __init__(
        self,
        sub: str,
        email: str | None = None,
        role: str | None = None,
        exp: int | None = None,
        iat: int | None = None,
        iss: str | None = None,
        aud: str | None = None,
        raw: dict[str, Any] | None = None,
    ) -> None:
        self.sub = sub  # User ID
        self.email = email
        self.role = role
        self.exp = exp
        self.iat = iat
        self.iss = iss
        self.aud = aud
        self.raw = raw or {}

    @property
    def user_id(self) -> str:
        """Alias for sub (user ID)."""
        return self.sub


def verify_token(token: str) -> TokenPayload:
    """
    Verify a JWT token and return the payload.

    Verification order:
    1. If SUPABASE_JWKS_URL is set, try RS256 verification
    2. Fall back to HS256 using SUPABASE_JWT_SECRET
    3. In test mode (ENV=test), also accept TEST_JWT_SECRET

    Raises:
        AuthenticationError: If verification fails
    """
    settings = get_settings()
    start_time = time.time()
    algorithm = "unknown"

    try:
        payload: dict[str, Any] | None = None

        # Try RS256 via JWKS if configured
        jwks_cache = _get_jwks_cache()
        if jwks_cache:
            try:
                algorithm = "RS256"
                signing_key = jwks_cache.get_signing_key(token)
                payload = jwt.decode(
                    token,
                    signing_key.key,
                    algorithms=["RS256"],
                    audience="authenticated" if settings.auth_verify_aud else None,
                    issuer=f"{settings.supabase_url}/auth/v1" if settings.supabase_url else None,
                )
            except jwt.PyJWTError as e:
                logger.debug("RS256 verification failed, trying HS256", extra={"error": str(e)})
                payload = None

        # Try HS256 with Supabase JWT secret
        if payload is None and settings.supabase_jwt_secret:
            try:
                algorithm = "HS256"
                payload = jwt.decode(
                    token,
                    settings.supabase_jwt_secret,
                    algorithms=["HS256"],
                    audience="authenticated" if settings.auth_verify_aud else None,
                    issuer=f"{settings.supabase_url}/auth/v1" if settings.supabase_url else None,
                )
            except jwt.PyJWTError:
                payload = None

        # In test mode, try TEST_JWT_SECRET
        if payload is None and settings.env == "test" and settings.test_jwt_secret:
            try:
                algorithm = "HS256-test"
                payload = jwt.decode(
                    token,
                    settings.test_jwt_secret,
                    algorithms=["HS256"],
                    options={"verify_aud": False, "verify_iss": False},
                )
            except jwt.PyJWTError:
                payload = None

        # Verification failed
        if payload is None:
            raise AuthenticationError(
                code=ErrorCode.AUTH_VERIFICATION_FAILED,
                message="Token verification failed",
            )

        # Validate required claims
        sub = payload.get("sub")
        if not sub:
            raise AuthenticationError(
                code=ErrorCode.AUTH_INVALID_TOKEN,
                message="Token missing 'sub' claim",
            )

        return TokenPayload(
            sub=sub,
            email=payload.get("email"),
            role=payload.get("role"),
            exp=payload.get("exp"),
            iat=payload.get("iat"),
            iss=payload.get("iss"),
            aud=payload.get("aud"),
            raw=payload,
        )

    except jwt.ExpiredSignatureError:
        raise AuthenticationError(
            code=ErrorCode.AUTH_EXPIRED_TOKEN,
            message="Token has expired",
        )
    except jwt.InvalidTokenError as e:
        raise AuthenticationError(
            code=ErrorCode.AUTH_INVALID_TOKEN,
            message=f"Invalid token: {e}",
        )
    except AuthenticationError:
        raise
    except Exception as e:
        logger.error("Unexpected error during token verification", extra={"error": str(e)})
        raise AuthenticationError(
            code=ErrorCode.AUTH_VERIFICATION_FAILED,
            message="Token verification failed",
        )
    finally:
        duration = time.time() - start_time
        AUTH_VERIFY_LATENCY.labels(algorithm=algorithm).observe(duration)


def extract_token_from_header(authorization: str | None) -> str:
    """
    Extract Bearer token from Authorization header.

    Args:
        authorization: Authorization header value

    Returns:
        The token string

    Raises:
        AuthenticationError: If header is missing or malformed
    """
    if not authorization:
        raise AuthenticationError(
            code=ErrorCode.AUTH_MISSING_TOKEN,
            message="Authorization header required",
        )

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise AuthenticationError(
            code=ErrorCode.AUTH_INVALID_TOKEN,
            message="Invalid Authorization header format. Expected 'Bearer <token>'",
        )

    return parts[1]
