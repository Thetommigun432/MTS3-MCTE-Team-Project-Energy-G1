"""
Redis Streams helper functions.
Provides XADD, XREADGROUP, and XACK functionality using the existing Redis infrastructure.
"""

from typing import Any, List, Optional, Tuple

from app.core.logging import get_logger
from app.infra.redis.client import get_redis_cache

logger = get_logger(__name__)


async def get_redis_client() -> Any:
    """
    Get the underlying redis client from the cache wrapper.
    Raises RuntimeError if Redis is not available (i.e., using in-memory fallback).
    """
    cache = get_redis_cache()
    if cache.is_using_fallback or cache._redis is None:
        raise RuntimeError("Redis is not available for streaming operations")
    return cache._redis


async def ensure_group(
    stream_key: str, group_name: str, start_id: str = "$"
) -> None:
    """
    Ensure a consumer group exists for a stream.
    Creates the stream if it doesn't exist (MKSTREAM).
    Ignores BUSYGROUP error if group already exists.
    """
    redis = await get_redis_client()
    try:
        await redis.xgroup_create(stream_key, group_name, id=start_id, mkstream=True)
        logger.info(f"Created consumer group {group_name} for stream {stream_key}")
    except Exception as e:
        if "BUSYGROUP" in str(e):
            # Group already exists, which is fine
            pass
        else:
            logger.error(f"Failed to create consumer group: {e}")
            raise


async def xadd_reading(
    stream_key: str, fields: dict[str, str | float | int], max_len: int = 10000
) -> str:
    """
    Add a reading to the stream.
    """
    redis = await get_redis_client()
    # Ensure all values are strings/bytes for Redis
    # redis-py handles generic types, but explicit conversion is safer for streams
    data = {k: str(v) for k, v in fields.items()}
    
    msg_id = await redis.xadd(stream_key, data, maxlen=max_len)
    return msg_id


async def read_group(
    stream_key: str,
    group_name: str,
    consumer_name: str,
    count: int = 10,
    block_ms: int = 2000,
) -> List[Tuple[str, dict[str, str]]]:
    """
    Read new messages from a consumer group.
    Returns list of (msg_id, fields).
    """
    redis = await get_redis_client()
    # '>' means read new messages never delivered to this consumer
    streams = {stream_key: ">"}
    
    results = await redis.xreadgroup(
        group_name, consumer_name, streams=streams, count=count, block=block_ms
    )
    
    # xreadgroup returns [[stream_name, [[msg_id, fields], ...]], ...]
    if not results:
        return []
    
    # We only queried one stream
    _, messages = results[0]
    return messages


async def ack(stream_key: str, group_name: str, msg_ids: List[str]) -> int:
    """
    Acknowledge processed messages.
    """
    if not msg_ids:
        return 0
    redis = await get_redis_client()
    return await redis.xack(stream_key, group_name, *msg_ids)
