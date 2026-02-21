"""Anthropic provider: cache_control injection, breakpoints, TTL, JSON stabilization."""

from __future__ import annotations

import copy
from typing import Any

from cacheguardian.config import CacheGuardConfig
from cacheguardian.core.metrics import MetricsCollector
from cacheguardian.core.optimizer import sort_tools, stabilize_json_keys
from cacheguardian.providers.base import CacheProvider
from cacheguardian.types import CacheMetrics, Provider, SessionState


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

        if not self.config.auto_fix:
            return kwargs

        # 1. Sort tools deterministically, stabilize JSON keys, inject cache_control
        if "tools" in kwargs and kwargs["tools"]:
            tools = sort_tools(kwargs["tools"])
            tools = [stabilize_json_keys(t) for t in tools]
            # Add cache_control to last tool so the entire tool block is cached
            if tools:
                last = dict(tools[-1])
                if "cache_control" not in last:
                    last["cache_control"] = {"type": "ephemeral"}
                tools[-1] = last
            kwargs["tools"] = tools

        # 2. Convert system prompt to content blocks with cache_control
        if "system" in kwargs and kwargs["system"] is not None:
            system = kwargs["system"]
            if isinstance(system, str):
                system = [{"type": "text", "text": system}]
            if isinstance(system, list):
                system = [stabilize_json_keys(b) for b in system]
                if system:
                    last = dict(system[-1])
                    if "cache_control" not in last:
                        last["cache_control"] = {"type": "ephemeral"}
                    system[-1] = last
            kwargs["system"] = system

        # 3. Conversation caching: breakpoints on message prefix
        if "messages" in kwargs:
            kwargs["messages"] = self._add_message_breakpoints(kwargs["messages"])

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

    def _add_message_breakpoints(
        self, messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Add cache_control breakpoints to messages for conversation caching.

        Strategy:
        1. Always add a breakpoint on the second-to-last message.  This caches
           the entire conversation prefix (system + tools + all prior messages)
           so only the newest user message is uncached on each turn.
        2. For very long conversations (>20 content blocks), also add
           intermediate breakpoints every 15 blocks.

        Anthropic allows up to 4 breakpoints per request.
        """
        messages = [copy.copy(m) for m in messages]

        # Anthropic allows max 4 breakpoints per request.
        # System + tools already use 2, leaving 2 for messages.
        max_msg_breakpoints = 2
        breakpoint_indices: list[int] = []

        # 1. Conversation prefix breakpoint: second-to-last message
        #    (the last assistant response before the new user message).
        if len(messages) >= 2:
            breakpoint_indices.append(len(messages) - 2)

        # 2. Intermediate breakpoint for long conversations (>20 blocks)
        total_blocks = sum(
            len(m.get("content", [])) if isinstance(m.get("content"), list) else 1
            for m in messages
        )
        if total_blocks > 20:
            for i in range(14, len(messages) - 2, 15):
                if len(breakpoint_indices) < max_msg_breakpoints:
                    breakpoint_indices.append(i)

        for idx in breakpoint_indices:
            self._inject_breakpoint(messages, idx)

        return messages

    @staticmethod
    def _inject_breakpoint(messages: list[dict[str, Any]], idx: int) -> None:
        """Add cache_control to a message's last content block."""
        if idx < 0 or idx >= len(messages):
            return
        msg = messages[idx]
        content = msg.get("content")
        if isinstance(content, list) and content:
            last_block = copy.copy(content[-1])
            if "cache_control" not in last_block:
                last_block["cache_control"] = {"type": "ephemeral"}
                content = content[:-1] + [last_block]
                messages[idx] = {**msg, "content": content}
        elif isinstance(content, str):
            # Convert string content to a content block with cache_control
            messages[idx] = {
                **msg,
                "content": [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}],
            }
