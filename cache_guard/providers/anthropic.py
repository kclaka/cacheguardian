"""Anthropic provider: cache_control injection, breakpoints, TTL, JSON stabilization."""

from __future__ import annotations

import copy
from typing import Any

from cache_guard.config import CacheGuardConfig
from cache_guard.core.metrics import MetricsCollector
from cache_guard.core.optimizer import sort_tools, stabilize_json_keys
from cache_guard.providers.base import CacheProvider
from cache_guard.types import CacheMetrics, Provider, SessionState


# Minimum cacheable tokens per model family
_MIN_TOKENS: dict[str, int] = {
    "claude-opus-4": 4096,
    "claude-sonnet-4": 1024,
    "claude-haiku-4": 4096,
    "claude-3-5-haiku": 2048,
    "claude-3-haiku": 2048,
}


class AnthropicProvider(CacheProvider):
    """Optimizes Anthropic Claude API requests for prompt caching.

    Key optimizations:
    - Auto-inject cache_control at top level if missing
    - Sort tools deterministically by name
    - Add intermediate breakpoints when >20 content blocks
    - Smart TTL selection (5min vs 1hr) based on user think time
    - Stabilize JSON key ordering in all content blocks
    """

    def __init__(self, config: CacheGuardConfig) -> None:
        super().__init__(config)
        self._metrics = MetricsCollector(config)

    def get_provider(self) -> Provider:
        return Provider.ANTHROPIC

    def intercept_request(self, kwargs: dict[str, Any], session: SessionState) -> dict[str, Any]:
        """Apply Anthropic-specific cache optimizations."""
        kwargs = copy.copy(kwargs)

        # 1. Auto-inject top-level cache_control if missing
        if self.config.auto_fix and "cache_control" not in kwargs:
            ttl = self._select_ttl(session)
            kwargs["cache_control"] = {"type": "ephemeral"}

        # 2. Sort tools deterministically
        if self.config.auto_fix and "tools" in kwargs and kwargs["tools"]:
            kwargs["tools"] = sort_tools(kwargs["tools"])
            # Stabilize JSON keys within each tool
            kwargs["tools"] = [stabilize_json_keys(t) for t in kwargs["tools"]]

        # 3. Handle 20-block rule: add intermediate breakpoints
        if self.config.auto_fix and "messages" in kwargs:
            kwargs["messages"] = self._ensure_intermediate_breakpoints(kwargs["messages"])

        # 4. Stabilize system prompt if it's a list of content blocks
        if self.config.auto_fix and "system" in kwargs and isinstance(kwargs["system"], list):
            kwargs["system"] = [stabilize_json_keys(block) for block in kwargs["system"]]

        return kwargs

    def extract_metrics(self, response: Any, model: str) -> CacheMetrics:
        return self._metrics.extract_anthropic(response, model)

    def extract_request_parts(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": kwargs.get("model", ""),
            "system": kwargs.get("system"),
            "tools": kwargs.get("tools"),
            "messages": kwargs.get("messages"),
        }

    def get_min_cache_tokens(self, model: str) -> int:
        for prefix, min_tokens in _MIN_TOKENS.items():
            if model.startswith(prefix):
                return min_tokens
        return 1024  # default

    def _select_ttl(self, session: SessionState) -> str:
        """Select TTL based on strategy and observed think time."""
        if self.config.ttl_strategy == "1h":
            return "1h"
        elif self.config.ttl_strategy == "5m":
            return "5m"
        else:  # "auto"
            avg = session.average_request_interval_seconds()
            if avg is not None and avg > 300:  # > 5 minutes
                return "1h"
            return "5m"

    def _ensure_intermediate_breakpoints(
        self, messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Add cache_control breakpoints every 15 blocks if total > 20.

        Anthropic's auto-caching searches backward at most 20 blocks from each
        breakpoint. If you have more than 20 blocks without intermediate breakpoints,
        changes early in the chain can invalidate everything.
        """
        messages = [copy.copy(m) for m in messages]

        # Count total content blocks
        total_blocks = sum(
            len(m.get("content", [])) if isinstance(m.get("content"), list) else 1
            for m in messages
        )

        if total_blocks <= 20:
            return messages

        # Add breakpoints every 15 messages (leaving headroom within the 20-block window)
        for i in range(14, len(messages), 15):
            if i < len(messages):
                msg = messages[i]
                content = msg.get("content")
                if isinstance(content, list) and content:
                    # Add cache_control to last content block
                    last_block = copy.copy(content[-1])
                    if "cache_control" not in last_block:
                        last_block["cache_control"] = {"type": "ephemeral"}
                        content = content[:-1] + [last_block]
                        msg = {**msg, "content": content}
                        messages[i] = msg

        return messages
