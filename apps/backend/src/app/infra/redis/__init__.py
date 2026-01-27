"""
Redis infrastructure layer.
Provides caching with graceful fallback to in-memory.
Includes rolling window support for NILM pipeline.
"""

from app.infra.redis.client import (
    RedisCache,
    InMemoryCache,
    get_redis_cache,
    init_redis_cache,
    close_redis_cache,
)

from app.infra.redis.rolling_window import (
    update_rolling_window,
    get_rolling_window,
    get_window_values,
    get_window_samples,
    get_window_length,
    clear_window,
)

__all__ = [
    # Cache
    "RedisCache",
    "InMemoryCache",
    "get_redis_cache",
    "init_redis_cache",
    "close_redis_cache",
    # Rolling window
    "update_rolling_window",
    "get_rolling_window",
    "get_window_values",
    "get_window_samples",
    "get_window_length",
    "clear_window",
]
