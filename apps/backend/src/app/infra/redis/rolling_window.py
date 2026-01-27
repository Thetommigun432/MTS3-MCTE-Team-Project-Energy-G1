"""
Redis Rolling Window for NILM pipeline.
Maintains a fixed-size sliding window of preprocessed readings per building.

Key design:
- Uses Redis LIST per building: nilm:{building_id}:window
- Each element is JSON: {"ts": "<iso>", "value": <float>}
- RPUSH + LTRIM enforces max size (4100 samples for largest model window)
- Window is deterministic: oldest samples are dropped when full
"""

import json
from typing import Any, List, Tuple

from app.core.logging import get_logger
from app.infra.redis.client import get_redis_cache

logger = get_logger(__name__)


def _window_key(building_id: str) -> str:
    """Generate Redis key for building's rolling window."""
    return f"nilm:{building_id}:window"


async def get_redis_client() -> Any:
    """
    Get the underlying redis client from the cache wrapper.
    Raises RuntimeError if Redis is not available.
    """
    cache = get_redis_cache()
    if cache.is_using_fallback or cache._redis is None:
        raise RuntimeError("Redis is not available for rolling window operations")
    return cache._redis


async def update_rolling_window(
    building_id: str,
    timestamp: str,
    value: float,
    max_size: int = 4100,
) -> int:
    """
    Add a preprocessed reading to the rolling window.

    Enforces the rolling window invariant:
    - After update, window has at most `max_size` elements
    - Oldest elements are dropped when window is full

    Args:
        building_id: Building identifier
        timestamp: ISO8601 timestamp string
        value: Preprocessed power value (float)
        max_size: Maximum window size (default 4100 for largest model window)

    Returns:
        Current window length after update
    """
    redis = await get_redis_client()
    key = _window_key(building_id)

    # Create element as JSON
    element = json.dumps({"ts": timestamp, "value": value})

    # RPUSH: Add to end of list
    await redis.rpush(key, element)

    # LTRIM: Keep only last max_size elements (enforces ring buffer)
    # LTRIM key -max_size -1 keeps the last max_size elements
    await redis.ltrim(key, -max_size, -1)

    # Get current length for logging/metrics
    length = await redis.llen(key)

    return length


async def get_rolling_window(
    building_id: str,
) -> List[Tuple[str, float]]:
    """
    Get all readings from the rolling window.

    Returns:
        List of (timestamp, value) tuples, oldest first
    """
    redis = await get_redis_client()
    key = _window_key(building_id)

    # LRANGE 0 -1 gets all elements
    elements = await redis.lrange(key, 0, -1)

    result = []
    for elem in elements:
        try:
            data = json.loads(elem)
            result.append((data["ts"], float(data["value"])))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Invalid window element: {elem}, error: {e}")
            continue

    return result


async def get_window_values(
    building_id: str,
    last_n: int | None = None,
) -> List[float]:
    """
    Get power values from the rolling window.

    Args:
        building_id: Building identifier
        last_n: If specified, return only the last N values

    Returns:
        List of power values (floats), oldest first
    """
    redis = await get_redis_client()
    key = _window_key(building_id)

    if last_n:
        # Get last N elements
        elements = await redis.lrange(key, -last_n, -1)
    else:
        # Get all elements
        elements = await redis.lrange(key, 0, -1)

    values = []
    for elem in elements:
        try:
            data = json.loads(elem)
            values.append(float(data["value"]))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    return values


async def get_window_samples(
    building_id: str,
    last_n: int | None = None,
) -> List[Tuple[str, float]]:
    """
    Get full window samples (timestamp and value).

    Args:
        building_id: Building identifier
        last_n: If specified, return only the last N elements

    Returns:
        List of (timestamp, value) tuples, oldest first
    """
    redis = await get_redis_client()
    key = _window_key(building_id)

    if last_n:
        elements = await redis.lrange(key, -last_n, -1)
    else:
        elements = await redis.lrange(key, 0, -1)

    result = []
    for elem in elements:
        try:
            data = json.loads(elem)
            result.append((data["ts"], float(data["value"])))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    return result


async def get_window_length(building_id: str) -> int:
    """
    Get current length of the rolling window.

    Returns:
        Number of elements in window
    """
    redis = await get_redis_client()
    key = _window_key(building_id)
    return await redis.llen(key)


async def clear_window(building_id: str) -> bool:
    """
    Clear the rolling window for a building.

    Returns:
        True if window was deleted
    """
    redis = await get_redis_client()
    key = _window_key(building_id)
    result = await redis.delete(key)
    return result > 0
