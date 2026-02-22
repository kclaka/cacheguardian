"""Anthropic provider: cache_control injection, breakpoints, TTL, JSON stabilization."""

from __future__ import annotations

import copy
import json
from typing import Any

from cacheguardian.config import CacheGuardConfig
from cacheguardian.core.metrics import MetricsCollector
from cacheguardian.core.optimizer import sort_tools, stabilize_json_keys
from cacheguardian.providers.base import CacheProvider
from cacheguardian.types import CacheMetrics, Provider, SessionState

# Fixed token cost for image content blocks (Anthropic's standard image grid)
_IMAGE_FIXED_TOKENS = 1568


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
        model = kwargs.get("model", "")
        if "messages" in kwargs:
            kwargs["messages"] = self._add_message_breakpoints(
                kwargs["messages"], model=model,
            )

        # 4. Enforce the 4-breakpoint hard limit (Anthropic API constraint)
        kwargs = self._enforce_breakpoint_limit(kwargs)

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
        self,
        messages: list[dict[str, Any]],
        *,
        model: str = "",
    ) -> list[dict[str, Any]]:
        """Add cache_control breakpoints to messages for conversation caching.

        Strategy:
        1. Gate on minimum cacheable token threshold for the model — skip
           breakpoint injection when the message prefix is too small to
           benefit from caching.
        2. Place the primary breakpoint at the static/dynamic boundary
           (just before dynamic tool results begin) rather than blindly
           at ``len-2``.  This avoids caching tool results that won't be
           reused (paper Section 5.1 — "Exclude Tool Results").
        3. For very long conversations (>20 content blocks), add an
           intermediate breakpoint every 15 blocks.

        Anthropic allows up to 4 breakpoints per request.
        """
        # Gate: skip message breakpoints if below minimum cacheable tokens
        min_tokens = self.get_min_cache_tokens(model)
        if self._estimate_message_tokens(messages) < min_tokens:
            return messages

        messages = [copy.copy(m) for m in messages]

        # Anthropic allows max 4 breakpoints per request.
        # System + tools already use 2, leaving 2 for messages.
        max_msg_breakpoints = 2
        breakpoint_indices: list[int] = []

        # 1. Primary breakpoint at the static/dynamic boundary
        boundary = self._find_static_boundary(messages)
        if boundary >= 0:
            breakpoint_indices.append(boundary)

        # 2. Intermediate breakpoint for long conversations (>20 blocks)
        total_blocks = sum(
            len(m.get("content", [])) if isinstance(m.get("content"), list) else 1
            for m in messages
        )
        if total_blocks > 20:
            for i in range(14, len(messages) - 2, 15):
                if len(breakpoint_indices) < max_msg_breakpoints and i not in breakpoint_indices:
                    breakpoint_indices.append(i)

        for idx in breakpoint_indices:
            self._inject_breakpoint(messages, idx)

        return messages

    @staticmethod
    def _estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
        """Estimate total tokens across all messages with image-safe handling.

        - Text blocks: ``len(text) // 4``
        - Image blocks (``type: "image"``): fixed 1568 tokens
        - Tool use/result blocks: ``len(json.dumps(block)) // 4``
        - Plain string content: ``len(content) // 4``
        """
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content) // 4
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        btype = block.get("type", "")
                        if btype == "image":
                            total += _IMAGE_FIXED_TOKENS
                        elif btype == "text":
                            total += len(block.get("text", "")) // 4
                        else:
                            # tool_use, tool_result, etc.
                            total += len(json.dumps(block)) // 4
                    elif isinstance(block, str):
                        total += len(block) // 4
        return total

    @staticmethod
    def _find_static_boundary(messages: list[dict[str, Any]]) -> int:
        """Find the index where stable conversation history ends and dynamic tool results begin.

        Walks backward from the end: skips the latest user message, then skips
        consecutive ``tool`` role results.  Returns the index of the last
        assistant message before those tool results — the optimal breakpoint
        that excludes dynamic tool output from the cached prefix.

        Falls back to ``len(messages) - 2`` when no tool results are present,
        preserving the original second-to-last-message behavior.
        """
        if len(messages) < 2:
            return max(0, len(messages) - 2)

        i = len(messages) - 1
        # Skip last user message
        if messages[i].get("role") == "user":
            i -= 1
        # Skip consecutive tool results
        while i >= 0 and messages[i].get("role") == "tool":
            i -= 1
        # i now points at the last assistant message before tool results
        return max(0, i)

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

    @staticmethod
    def _count_cache_controls(kwargs: dict[str, Any]) -> int:
        """Count all cache_control markers in a request payload."""
        count = 0
        # System blocks
        system = kwargs.get("system")
        if isinstance(system, list):
            for block in system:
                if isinstance(block, dict) and "cache_control" in block:
                    count += 1
        # Tool definitions
        tools = kwargs.get("tools")
        if isinstance(tools, list):
            for tool in tools:
                if isinstance(tool, dict) and "cache_control" in tool:
                    count += 1
        # Message content blocks
        messages = kwargs.get("messages")
        if isinstance(messages, list):
            for msg in messages:
                content = msg.get("content") if isinstance(msg, dict) else None
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and "cache_control" in block:
                            count += 1
        return count

    def _enforce_breakpoint_limit(
        self, kwargs: dict[str, Any], max_breakpoints: int = 4,
    ) -> dict[str, Any]:
        """Strip least-valuable breakpoints if total exceeds *max_breakpoints*.

        Priority (keep highest → strip lowest):
          1. System prompt breakpoint (stable, largest reuse)
          2. Last tool breakpoint (stable, large)
          3. User-provided explicit breakpoints (intentional)
          4. Oldest message breakpoints first (least reuse value)
        """
        if self._count_cache_controls(kwargs) <= max_breakpoints:
            return kwargs

        kwargs = copy.copy(kwargs)

        # Strip message breakpoints oldest-first until within budget
        messages = kwargs.get("messages")
        if isinstance(messages, list):
            messages = [copy.copy(m) for m in messages]
            for i in range(len(messages)):
                if self._count_cache_controls(kwargs) <= max_breakpoints:
                    break
                msg = messages[i]
                content = msg.get("content") if isinstance(msg, dict) else None
                if isinstance(content, list):
                    changed = False
                    new_content = []
                    for block in content:
                        if isinstance(block, dict) and "cache_control" in block:
                            block = {k: v for k, v in block.items() if k != "cache_control"}
                            changed = True
                        new_content.append(block)
                    if changed:
                        messages[i] = {**msg, "content": new_content}
            kwargs["messages"] = messages

        # If still over budget, strip tool breakpoints
        if self._count_cache_controls(kwargs) > max_breakpoints:
            tools = kwargs.get("tools")
            if isinstance(tools, list):
                tools = [
                    {k: v for k, v in t.items() if k != "cache_control"}
                    for t in tools
                ]
                kwargs["tools"] = tools

        return kwargs
