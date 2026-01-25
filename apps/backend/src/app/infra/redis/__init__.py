"""
Redis infrastructure layer.
Provides caching with graceful fallback to in-memory.
"""

from app.infra.redis.client import (
    RedisCache,
    InMemoryCache,
    get_redis_cache,
    init_redis_cache,
    close_redis_cache,
)

__all__ = [
    "RedisCache",
    "InMemoryCache",
    "get_redis_cache",
    "init_redis_cache",
    "close_redis_cache",
]
