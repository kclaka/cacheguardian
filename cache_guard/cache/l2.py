"""L2 Cache: Optional Redis backend for cross-worker prefix sharing and coordination."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger("cache_guard")


class L2Cache:
    """Shared cache for distributed environments.

    Stores:
    - Cross-worker prefix hashes (so workers know about each other's sessions)
    - Gemini CachedContent IDs (prevents redundant cache creation across workers)
    - Rate limit coordination metadata

    Falls back gracefully when Redis is unavailable.
    """

    def __init__(self, redis_url: Optional[str] = None) -> None:
        self._client = None
        self._available = False

        if redis_url:
            try:
                import redis
                self._client = redis.from_url(redis_url, decode_responses=True)
                self._client.ping()
                self._available = True
                logger.debug("L2 cache connected: %s", redis_url)
            except Exception as e:
                logger.warning("L2 cache unavailable (falling back to L1-only): %s", e)
                self._client = None
                self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def store_prefix_hash(
        self, prefix_hash: str, provider: str, model: str, token_count: int, ttl_seconds: int = 3600,
    ) -> None:
        """Store a prefix hash so other workers can detect shared prefixes."""
        if not self._available:
            return
        try:
            key = f"cache_guard:prefix:{prefix_hash}"
            value = json.dumps({
                "provider": provider,
                "model": model,
                "token_count": token_count,
                "stored_at": datetime.now().isoformat(),
            })
            self._client.setex(key, ttl_seconds, value)
        except Exception as e:
            logger.debug("L2 store_prefix_hash failed: %s", e)

    def check_prefix_hash(self, prefix_hash: str) -> Optional[dict[str, Any]]:
        """Check if a prefix hash exists in the shared cache."""
        if not self._available:
            return None
        try:
            key = f"cache_guard:prefix:{prefix_hash}"
            value = self._client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.debug("L2 check_prefix_hash failed: %s", e)
        return None

    # -- Gemini CachedContent coordination --

    def store_gemini_cache_id(
        self, content_hash: str, cache_name: str, ttl_seconds: int = 3600,
    ) -> None:
        """Store a Gemini CachedContent ID so other workers can reuse it."""
        if not self._available:
            return
        try:
            key = f"cache_guard:gemini_cache:{content_hash}"
            value = json.dumps({
                "cache_name": cache_name,
                "stored_at": datetime.now().isoformat(),
            })
            self._client.setex(key, ttl_seconds, value)
        except Exception as e:
            logger.debug("L2 store_gemini_cache_id failed: %s", e)

    def get_gemini_cache_id(self, content_hash: str) -> Optional[str]:
        """Check if another worker already created a CachedContent for this hash."""
        if not self._available:
            return None
        try:
            key = f"cache_guard:gemini_cache:{content_hash}"
            value = self._client.get(key)
            if value:
                data = json.loads(value)
                return data.get("cache_name")
        except Exception as e:
            logger.debug("L2 get_gemini_cache_id failed: %s", e)
        return None

    def remove_gemini_cache_id(self, content_hash: str) -> None:
        """Remove a Gemini cache ID (e.g., after deletion)."""
        if not self._available:
            return
        try:
            key = f"cache_guard:gemini_cache:{content_hash}"
            self._client.delete(key)
        except Exception as e:
            logger.debug("L2 remove_gemini_cache_id failed: %s", e)

    # -- Rate limit coordination --

    def increment_request_count(
        self, provider: str, window_key: str, ttl_seconds: int = 60,
    ) -> int:
        """Increment and return the request count for a time window."""
        if not self._available:
            return 0
        try:
            key = f"cache_guard:rate:{provider}:{window_key}"
            count = self._client.incr(key)
            if count == 1:
                self._client.expire(key, ttl_seconds)
            return count
        except Exception as e:
            logger.debug("L2 increment_request_count failed: %s", e)
            return 0

    def close(self) -> None:
        """Close the Redis connection."""
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
            self._available = False
