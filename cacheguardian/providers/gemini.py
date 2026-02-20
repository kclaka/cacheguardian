"""Gemini provider: CachedContent lifecycle, promotion, safety lock, storage tracking."""

from __future__ import annotations

import copy
import logging
from typing import Any, Optional

from cacheguardian.config import CacheGuardConfig
from cacheguardian.core.metrics import MetricsCollector
from cacheguardian.core.optimizer import sort_tools
from cacheguardian.core.promoter import CachePromoter
from cacheguardian.persistence.cache_registry import CacheRegistry
from cacheguardian.providers.base import CacheProvider
from cacheguardian.types import CacheMetrics, Provider, SessionState

logger = logging.getLogger("cacheguardian")


class GeminiProvider(CacheProvider):
    """Optimizes Google Gemini API requests for prompt caching.

    Key optimizations:
    - Implicit→Explicit promotion when cost-benefit analysis is positive
    - TTL optimization based on request frequency
    - Safety lock: disk-persisted cache registry for zombie cache cleanup
    - Storage cost tracking separate from input savings
    - Cross-worker awareness via L2 (if Redis available)
    """

    def __init__(
        self,
        config: CacheGuardConfig,
        gemini_client: Any = None,
        registry: CacheRegistry | None = None,
    ) -> None:
        super().__init__(config)
        self._metrics = MetricsCollector(config)
        self._promoter = CachePromoter(config)
        self._gemini_client = gemini_client
        self._registry = registry or CacheRegistry()

    def get_provider(self) -> Provider:
        return Provider.GEMINI

    def intercept_request(self, kwargs: dict[str, Any], session: SessionState) -> dict[str, Any]:
        """Apply Gemini-specific cache optimizations."""
        kwargs = copy.copy(kwargs)

        # 1. Sort tools if present (Gemini uses function declarations)
        if self.config.auto_fix:
            tools = kwargs.get("tools")
            if tools and isinstance(tools, list):
                kwargs["tools"] = sort_tools(tools)

        # 2. Check if we should use an existing explicit cache
        if session.gemini_cache_name:
            config = kwargs.get("config", {})
            if isinstance(config, dict):
                config["cached_content"] = session.gemini_cache_name
                kwargs["config"] = config

        # 3. Evaluate promotion from implicit to explicit
        elif self.config.auto_fix and self._gemini_client and session.request_count >= 3:
            token_estimate = self._estimate_tokens(kwargs)
            if token_estimate >= 1024:
                decision = self._promoter.evaluate_gemini_explicit(
                    session, token_estimate,
                )
                if decision.should_promote:
                    cache_name = self._create_explicit_cache(kwargs, session)
                    if cache_name:
                        session.gemini_cache_name = cache_name
                        config = kwargs.get("config", {})
                        if isinstance(config, dict):
                            config["cached_content"] = cache_name
                            kwargs["config"] = config

        return kwargs

    def extract_metrics(self, response: Any, model: str) -> CacheMetrics:
        return self._metrics.extract_gemini(response, model)

    def extract_request_parts(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Extract parts from Gemini's request format."""
        return {
            "model": kwargs.get("model", ""),
            "system": kwargs.get("system_instruction"),
            "tools": kwargs.get("tools"),
            "messages": kwargs.get("contents"),
        }

    def get_min_cache_tokens(self, model: str) -> int:
        if "2.5" in model:
            return 1024
        return 4096  # conservative default

    def cleanup_stale_caches(self, max_age_hours: float = 2.0) -> int:
        """Clean up stale caches from the registry.

        This is the safety lock: even if the process crashes, the registry
        persists to disk and can clean up on next startup.

        Returns the number of caches cleaned up.
        """
        if not self._gemini_client:
            return 0

        stale = self._registry.get_stale_entries(max_age_hours)
        cleaned = 0

        for entry in stale:
            try:
                self._gemini_client.caches.delete(entry["cache_name"])
                self._registry.remove(entry["cache_name"])
                logger.info("Cleaned up stale Gemini cache: %s", entry["cache_name"])
                cleaned += 1
            except Exception as e:
                logger.warning("Failed to clean up cache %s: %s", entry["cache_name"], e)

        return cleaned

    def _create_explicit_cache(self, kwargs: dict[str, Any], session: SessionState) -> str | None:
        """Create an explicit CachedContent on Gemini."""
        try:
            from google.genai import types

            system_instruction = kwargs.get("system_instruction", "")
            contents = kwargs.get("contents", [])

            avg_interval = session.average_request_interval_seconds()
            ttl_seconds = self._calculate_optimal_ttl(avg_interval)

            cache = self._gemini_client.caches.create(
                model=kwargs.get("model", session.model),
                config=types.CreateCachedContentConfig(
                    system_instruction=system_instruction,
                    contents=contents,
                    ttl=f"{ttl_seconds}s",
                ),
            )

            # Register in safety lock
            self._registry.add(
                cache_name=cache.name,
                model=session.model,
                session_id=session.session_id,
                ttl_seconds=ttl_seconds,
            )

            logger.info("Created Gemini explicit cache: %s (TTL: %ds)", cache.name, ttl_seconds)
            return cache.name

        except Exception as e:
            logger.warning("Failed to create Gemini explicit cache: %s", e)
            return None

    def _calculate_optimal_ttl(self, avg_interval_seconds: float | None) -> int:
        """Calculate optimal TTL to minimize storage while avoiding expiry."""
        if avg_interval_seconds is None:
            return 3600  # 1 hour default

        # TTL should be at least 3x the average interval to handle variance
        min_ttl = int(avg_interval_seconds * 3)
        # Cap at 24 hours
        max_ttl = 86400
        # Floor at 5 minutes
        return max(300, min(min_ttl, max_ttl))

    def _estimate_tokens(self, kwargs: dict[str, Any]) -> int:
        """Rough token estimate (4 chars per token heuristic)."""
        total_chars = 0
        for key in ("system_instruction", "contents"):
            val = kwargs.get(key)
            if isinstance(val, str):
                total_chars += len(val)
            elif isinstance(val, list):
                total_chars += len(str(val))
        return total_chars // 4
