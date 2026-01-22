"""
Authorization caching with TTL.
Thread-safe in async context using asyncio locks.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.telemetry import AUTHZ_CACHE_HIT, AUTHZ_CACHE_MISS

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """A cached permission entry with TTL."""

    data: Any
    expires_at: float


@dataclass
class PermissionGraph:
    """Cached permission data for a user."""

    buildings: list[str] = field(default_factory=list)
    building_appliances: dict[str, list[str]] = field(default_factory=dict)
    role: str = "user"


class AuthzCache:
    """Thread-safe authorization cache with TTL."""

    def __init__(self, ttl_seconds: int | None = None) -> None:
        settings = get_settings()
        self._ttl_seconds = ttl_seconds or settings.authz_cache_ttl_seconds
        self._cache: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()

    async def get(self, user_id: str) -> PermissionGraph | None:
        """
        Get cached permission graph for a user.

        Returns:
            PermissionGraph or None if not cached/expired
        """
        async with self._lock:
            entry = self._cache.get(user_id)

            if entry is None:
                AUTHZ_CACHE_MISS.inc()
                return None

            if time.time() > entry.expires_at:
                # Expired
                del self._cache[user_id]
                AUTHZ_CACHE_MISS.inc()
                return None

            AUTHZ_CACHE_HIT.inc()
            return entry.data

    async def set(self, user_id: str, graph: PermissionGraph) -> None:
        """Cache permission graph for a user."""
        async with self._lock:
            self._cache[user_id] = CacheEntry(
                data=graph,
                expires_at=time.time() + self._ttl_seconds,
            )

    async def invalidate(self, user_id: str) -> bool:
        """
        Invalidate cache for a specific user.

        Returns:
            True if entry was removed
        """
        async with self._lock:
            if user_id in self._cache:
                del self._cache[user_id]
                logger.info("Authz cache invalidated", extra={"user_id": user_id})
                return True
            return False

    async def invalidate_all(self) -> int:
        """
        Invalidate all cached entries.

        Returns:
            Number of entries removed
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info("Authz cache cleared", extra={"entries_removed": count})
            return count

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            now = time.time()
            active_entries = sum(1 for e in self._cache.values() if e.expires_at > now)
            return {
                "total_entries": len(self._cache),
                "active_entries": active_entries,
                "ttl_seconds": self._ttl_seconds,
            }


# Global cache instance
_authz_cache: AuthzCache | None = None


def get_authz_cache() -> AuthzCache:
    """Get the global authz cache instance."""
    global _authz_cache
    if _authz_cache is None:
        _authz_cache = AuthzCache()
    return _authz_cache
