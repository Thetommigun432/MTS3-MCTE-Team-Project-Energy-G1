"""
JWT verification and authentication.
Supports HS256 (default), RS256, and ES256 (via JWKS).
"""

import threading
import time
from typing import Any

import jwt
from jwt import PyJWKClient
from jwt.exceptions import PyJWKClientError, InvalidTokenError

from app.core.config import get_settings
from app.core.errors import AuthenticationError, ErrorCode
from app.core.logging import get_logger
from app.core.telemetry import AUTH_VERIFY_LATENCY

logger = get_logger(__name__)


class JWKSCache:
    """
    Cached JWKS client for RS256/ES256 verification.
    
    Implements concurrent-safe refreshing:
    - Caches keys in-memory via PyJWKClient.
    - On key lookup failure (kid mismatch), refreshes JWKS once.
    - Uses a lock to prevent thundering herd during refresh.
    """

    def __init__(self, jwks_url: str, cache_ttl_hours: int = 6) -> None:
        self.jwks_url = jwks_url
        self.cache_ttl_seconds = cache_ttl_hours * 3600
        self._client: PyJWKClient | None = None
        self._last_refresh: float = 0
        self._lock = threading.Lock()

    def get_signing_key(self, token: str) -> Any:
        """
        Get signing key from JWKS, refreshing cache if needed.
        """
        # 1. Lazy init or TTL expiry check (optimistic reading)
        if self._should_refresh():
            with self._lock:
                if self._should_refresh():
                    self._refresh_client()

        # 2. Try to get key
        try:
            return self._client.get_signing_key_from_jwt(token)  # type: ignore[union-attr]
        except (PyJWKClientError, InvalidTokenError) as e:
            # 3. If key not found, try FORCE refresh (handling rotation)
            # We use a smaller logic here: fail if we just refreshed recently
            if time.time() - self._last_refresh < 10:
                logger.warning("JWKS key not found and recently refreshed, rejecting token", extra={"error": str(e)})
                raise AuthenticationError(
                    code=ErrorCode.AUTH_VERIFICATION_FAILED,
                    message="Invalid token key ID"
                )

            logger.info("Key ID not found in cache, triggering refresh")
            with self._lock:
                # Check again in case another thread refreshed
                try:
                     return self._client.get_signing_key_from_jwt(token)  # type: ignore[union-attr]
                except (PyJWKClientError, InvalidTokenError):
                   self._refresh_client()

            # 4. Try again after refresh
            try:
                return self._client.get_signing_key_from_jwt(token)  # type: ignore[union-attr]
            except Exception as e:
                logger.error("Failed to retrieve signing key after refresh", extra={"error": str(e)})
                raise AuthenticationError(
                    code=ErrorCode.AUTH_VERIFICATION_FAILED,
                    message="Unable to verify token signature key"
                )

    def _should_refresh(self) -> bool:
        """Check if client is uninitialized or TTL expired."""
        now = time.time()
        return self._client is None or (now - self._last_refresh) > self.cache_ttl_seconds

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

    # Auto-derived in config, so this should usually be set if URL is set
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


def extract_token_from_header(authorization: str | None) -> str:
    """
    Extract Bearer token from Authorization header.
    
    Raises:
        AuthenticationError: If header is missing or invalid format
    """
    if not authorization:
        raise AuthenticationError(
            code=ErrorCode.AUTH_MISSING_TOKEN,
            message="Missing authorization header",
        )

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise AuthenticationError(
            code=ErrorCode.AUTH_INVALID_TOKEN,
            message="Invalid authorization header format. Expected 'Bearer <token>'",
        )

    return parts[1]



def verify_token(token: str) -> TokenPayload:
    """
    Verify a JWT token and return the payload.

    Verification mode (Auto):
    1. Parse header (unverified) to get 'alg'.
    2. If RS256 or ES256: Verify via JWKS (SUPABASE_JWKS_URL).
    3. If HS256: Verify via SUPABASE_JWT_SECRET (Legacy).
    4. If Unknown: Reject.

    Raises:
        AuthenticationError: If verification fails
    """
    settings = get_settings()
    start_time = time.time()
    algorithm = "unknown"
    
    try:
        # 1. Parse Header (Unverified)
        try:
            header = jwt.get_unverified_header(token)
            algorithm = header.get("alg", "unknown")
        except jwt.PyJWTError:
             raise AuthenticationError(
                code=ErrorCode.AUTH_INVALID_TOKEN,
                message="Invalid token header",
            )

        payload: dict[str, Any] | None = None

        # 2. Route Verification based on Algorithm
        if algorithm in ("RS256", "ES256"):
            jwks_cache = _get_jwks_cache()
            if not jwks_cache:
                # Should not happen if config is correct, but safe fallback
                raise AuthenticationError(
                    code=ErrorCode.AUTH_CONFIGURATION_ERROR,
                    message=f"{algorithm} token received but JWKS URL not configured"
                )
            
            signing_key = jwks_cache.get_signing_key(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=[algorithm],
                audience="authenticated" if settings.auth_verify_aud else None,
                issuer=f"{settings.supabase_url}/auth/v1" if settings.supabase_url else None,
                options={
                    "require": ["exp", "sub"],
                    "verify_aud": settings.auth_verify_aud,
                }
            )

        elif algorithm == "HS256":
            # Check for Test Secret first (Hard-gated)
            if settings.env == "test" and settings.test_jwt_secret:
                # In test mode with test secret, we might accept HS256 signed with it
                 payload = jwt.decode(
                    token,
                    settings.test_jwt_secret,
                    algorithms=["HS256"],
                    options={"verify_aud": False, "verify_iss": False, "require": ["sub"]}
                )
                 algorithm = "HS256-test"
            
            # Legacy Fallback
            elif settings.supabase_jwt_secret:
                payload = jwt.decode(
                    token,
                    settings.supabase_jwt_secret,
                    algorithms=["HS256"],
                    audience="authenticated" if settings.auth_verify_aud else None,
                    issuer=f"{settings.supabase_url}/auth/v1" if settings.supabase_url else None,
                    options={
                        "require": ["exp", "sub"],
                        "verify_aud": settings.auth_verify_aud,
                    }
                )
            else:
                # Fail closed if HS256 received but no secret configured
                raise AuthenticationError(
                    code=ErrorCode.AUTH_VERIFICATION_FAILED,
                    message="HS256 token received but legacy secret not configured"
                )
        
        else:
             raise AuthenticationError(
                code=ErrorCode.AUTH_INVALID_TOKEN,
                message=f"Unsupported token algorithm: {algorithm}"
            )

        # Verification failed (should match raise in blocks above, but double check)
        if payload is None:
            raise AuthenticationError(
                code=ErrorCode.AUTH_VERIFICATION_FAILED,
                message="Token verification failed",
            )

        return TokenPayload(
            sub=payload["sub"],
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
        logger.error("Unexpected error during token verification", extra={"error": str(e), "alg": algorithm})
        raise AuthenticationError(
            code=ErrorCode.AUTH_VERIFICATION_FAILED,
            message="Token verification failed",
        )
    finally:
        duration = time.time() - start_time
        AUTH_VERIFY_LATENCY.labels(algorithm=algorithm).observe(duration)
