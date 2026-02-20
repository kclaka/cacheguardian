"""OpenAI provider: prompt_cache_key, retention, content reordering, thresholds."""

from __future__ import annotations

import copy
import hashlib
from typing import Any

from cacheguardian.config import CacheGuardConfig
from cacheguardian.core.metrics import MetricsCollector
from cacheguardian.core.optimizer import reorder_static_first, sort_tools
from cacheguardian.providers.base import CacheProvider
from cacheguardian.types import CacheMetrics, Provider, SessionState

# OpenAI caching minimum threshold
_MIN_CACHE_TOKENS = 1024


class OpenAIProvider(CacheProvider):
    """Optimizes OpenAI API requests for prompt caching.

    Key optimizations:
    - Auto-derive prompt_cache_key from session context or user-provided function
    - Set prompt_cache_retention='24h' for slow sessions
    - Sort tools deterministically
    - Reorder content: static first, dynamic last
    - Suppress false-positive warnings for prompts < 1024 tokens
    """

    def __init__(self, config: CacheGuardConfig) -> None:
        super().__init__(config)
        self._metrics = MetricsCollector(config)

    def get_provider(self) -> Provider:
        return Provider.OPENAI

    def intercept_request(self, kwargs: dict[str, Any], session: SessionState) -> dict[str, Any]:
        """Apply OpenAI-specific cache optimizations."""
        kwargs = copy.copy(kwargs)

        # 1. Sort tools deterministically
        if self.config.auto_fix and "tools" in kwargs and kwargs["tools"]:
            kwargs["tools"] = sort_tools(kwargs["tools"])

        # 2. Reorder messages: system first, then stable context, then dynamic
        if self.config.auto_fix and "messages" in kwargs:
            kwargs["messages"] = reorder_static_first(kwargs["messages"])

        # 3. Add prompt_cache_key for routing
        if self.config.auto_fix and "prompt_cache_key" not in kwargs:
            cache_key = self._derive_cache_key(kwargs, session)
            if cache_key:
                kwargs["prompt_cache_key"] = cache_key

        # 4. Set prompt_cache_retention based on request frequency
        if self.config.auto_fix and "prompt_cache_retention" not in kwargs:
            avg_interval = session.average_request_interval_seconds()
            if avg_interval is not None and avg_interval > 600:
                kwargs["prompt_cache_retention"] = "24h"

        return kwargs

    def extract_metrics(self, response: Any, model: str) -> CacheMetrics:
        return self._metrics.extract_openai(response, model)

    def extract_request_parts(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Extract parts from OpenAI's message format."""
        messages = kwargs.get("messages", [])

        # OpenAI uses messages for system prompts too
        system = None
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
                break

        return {
            "model": kwargs.get("model", ""),
            "system": system,
            "tools": kwargs.get("tools"),
            "messages": messages,
        }

    def get_min_cache_tokens(self, model: str) -> int:
        return _MIN_CACHE_TOKENS

    def _derive_cache_key(self, kwargs: dict[str, Any], session: SessionState) -> str | None:
        """Derive a stable prompt_cache_key for routing."""
        # Use user-provided function if available
        if self.config.cache_key_fn is not None:
            try:
                return self.config.cache_key_fn(session)
            except Exception:
                pass

        # Use stored key if we already derived one
        if session.openai_cache_key:
            return session.openai_cache_key

        # Auto-derive from session fingerprint
        if session.system_hash:
            key = f"cg_{session.session_id}"
            session.openai_cache_key = key
            return key

        return None
