"""
Redis client with graceful fallback to in-memory cache.
Provides caching infrastructure for idempotency and AuthZ.
"""

import asyncio
import time
from collections import OrderedDict
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.telemetry import REDIS_UNAVAILABLE, CACHE_FALLBACK_IN_USE

logger = get_logger(__name__)


class InMemoryCache:
    """
    Bounded in-memory cache with TTL support.
    Used as fallback when Redis is unavailable.
    """

    def __init__(self, max_size: int = 10000, default_ttl: int = 600) -> None:
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl

    async def get(self, key: str) -> Any | None:
        """Get value from cache if exists and not expired."""
        entry = self._cache.get(key)
        if entry is None:
            return None

        expires_at, value = entry
        if time.time() > expires_at:
            del self._cache[key]
            return None

        # Move to end (LRU)
        self._cache.move_to_end(key)
        return value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache with TTL."""
        if ttl is None:
            ttl = self._default_ttl

        expires_at = time.time() + ttl

        # Evict oldest if at capacity
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)

        self._cache[key] = (expires_at, value)
        self._cache.move_to_end(key)

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return await self.get(key) is not None

    def size(self) -> int:
        """Return current cache size."""
        return len(self._cache)

    def cleanup(self) -> int:
        """Remove expired entries. Returns count removed."""
        now = time.time()
        expired = [k for k, (exp, _) in self._cache.items() if now > exp]
        for k in expired:
            del self._cache[k]
        return len(expired)


class RedisCache:
    """
    Redis-backed async cache with automatic fallback to in-memory.
    """

    def __init__(self) -> None:
        self._redis: Any = None  # redis.asyncio.Redis instance
        self._fallback = InMemoryCache()
        self._using_fallback = False
        self._last_redis_check = 0.0
        self._redis_check_interval = 30.0  # Retry Redis every 30s

    async def connect(self) -> bool:
        """
        Connect to Redis. Returns True if connected, False if falling back.
        """
        settings = get_settings()

        if not settings.redis_url:
            logger.info("REDIS_URL not configured, using in-memory cache")
            self._using_fallback = True
            CACHE_FALLBACK_IN_USE.set(1)
            return False

        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(
                settings.redis_url,
                max_connections=settings.redis_pool_size,
                socket_connect_timeout=settings.redis_connect_timeout_ms / 1000,
                decode_responses=True,
            )
            # Test connection
            await self._redis.ping()
            self._using_fallback = False
            CACHE_FALLBACK_IN_USE.set(0)
            logger.info("Redis connected successfully")
            return True

        except ImportError:
            logger.warning("redis package not installed, using in-memory cache")
            self._using_fallback = True
            CACHE_FALLBACK_IN_USE.set(1)
            return False

        except Exception as e:
            logger.warning("Redis connection failed, using in-memory fallback", extra={"error": str(e)})
            REDIS_UNAVAILABLE.inc()
            self._using_fallback = True
            CACHE_FALLBACK_IN_USE.set(1)
            return False

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None

    async def _try_redis_reconnect(self) -> bool:
        """Periodically try to reconnect to Redis."""
        now = time.time()
        if now - self._last_redis_check < self._redis_check_interval:
            return False

        self._last_redis_check = now
        settings = get_settings()

        if not settings.redis_url:
            return False

        try:
            import redis.asyncio as aioredis

            if self._redis is None:
                self._redis = aioredis.from_url(
                    settings.redis_url,
                    max_connections=settings.redis_pool_size,
                    socket_connect_timeout=settings.redis_connect_timeout_ms / 1000,
                    decode_responses=True,
                )

            await self._redis.ping()
            self._using_fallback = False
            CACHE_FALLBACK_IN_USE.set(0)
            logger.info("Redis reconnected successfully")
            return True

        except Exception:
            return False

    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        if self._using_fallback:
            await self._try_redis_reconnect()
            if self._using_fallback:
                return await self._fallback.get(key)

        try:
            import json
            value = await self._redis.get(key)
            if value is None:
                return None
            return json.loads(value)

        except Exception as e:
            logger.warning("Redis get failed, falling back", extra={"key": key, "error": str(e)})
            REDIS_UNAVAILABLE.inc()
            self._using_fallback = True
            CACHE_FALLBACK_IN_USE.set(1)
            return await self._fallback.get(key)

    async def set(self, key: str, value: Any, ttl: int = 600) -> None:
        """Set value in cache with TTL."""
        # Always set in fallback for consistency during transitions
        await self._fallback.set(key, value, ttl)

        if self._using_fallback:
            await self._try_redis_reconnect()
            if self._using_fallback:
                return

        try:
            import json
            await self._redis.setex(key, ttl, json.dumps(value))

        except Exception as e:
            logger.warning("Redis set failed, using fallback only", extra={"key": key, "error": str(e)})
            REDIS_UNAVAILABLE.inc()
            self._using_fallback = True
            CACHE_FALLBACK_IN_USE.set(1)

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        await self._fallback.delete(key)

        if self._using_fallback:
            return True

        try:
            result = await self._redis.delete(key)
            return result > 0

        except Exception as e:
            logger.warning("Redis delete failed", extra={"key": key, "error": str(e)})
            REDIS_UNAVAILABLE.inc()
            self._using_fallback = True
            CACHE_FALLBACK_IN_USE.set(1)
            return True

    @property
    def is_using_fallback(self) -> bool:
        """Check if currently using fallback cache."""
        return self._using_fallback


# Global cache instance
_redis_cache: RedisCache | None = None


def get_redis_cache() -> RedisCache:
    """Get the global Redis cache instance."""
    global _redis_cache
    if _redis_cache is None:
        _redis_cache = RedisCache()
    return _redis_cache


async def init_redis_cache() -> RedisCache:
    """Initialize the global Redis cache."""
    cache = get_redis_cache()
    await cache.connect()
    return cache


async def close_redis_cache() -> None:
    """Close the global Redis cache."""
    global _redis_cache
    if _redis_cache:
        await _redis_cache.close()
        _redis_cache = None
