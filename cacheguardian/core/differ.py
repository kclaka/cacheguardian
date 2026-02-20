"""Request diff engine using L1 segment fingerprint comparison."""

from __future__ import annotations

import json
from typing import Any, Optional

from cacheguardian.cache.fingerprint import (
    compute_fingerprint,
    compute_prefix_match_depth,
    find_divergence,
    normalize_json,
)
from cacheguardian.core.optimizer import detect_tool_schema_changes
from cacheguardian.types import CacheBreakWarning, DivergencePoint, Fingerprint


class PromptDiffer:
    """Compares requests to detect cache-breaking changes.

    Uses L1 segment hashes for <1ms detection, then falls back to
    character-level diff only for the specific divergent segment.
    """

    def diff(
        self,
        prev_kwargs: dict[str, Any],
        curr_kwargs: dict[str, Any],
        prev_fingerprint: Fingerprint | None = None,
    ) -> list[CacheBreakWarning]:
        """Compare previous and current request kwargs to detect cache breaks.

        Returns a list of warnings (empty if no breaks detected).
        """
        warnings: list[CacheBreakWarning] = []

        # Model change
        prev_model = prev_kwargs.get("model", "")
        curr_model = curr_kwargs.get("model", "")
        if prev_model and curr_model and prev_model != curr_model:
            warnings.append(CacheBreakWarning(
                reason=f"model changed: '{prev_model}' → '{curr_model}'",
                suggestion="Changing models invalidates the entire cache. Use subagents for different models instead of switching mid-session.",
            ))

        # System prompt change
        prev_system = prev_kwargs.get("system")
        curr_system = curr_kwargs.get("system")
        if prev_system is not None and curr_system is not None:
            if _content_hash(prev_system) != _content_hash(curr_system):
                detail = _compute_detail(prev_system, curr_system)
                warnings.append(CacheBreakWarning(
                    reason="system prompt changed",
                    divergence=DivergencePoint(
                        segment_index=0,
                        segment_label="system",
                        previous_hash=_content_hash(prev_system),
                        new_hash=_content_hash(curr_system),
                        detail=detail,
                    ),
                    suggestion="Use system messages (<system-reminder> in user messages) instead of modifying the system prompt.",
                ))

        # Tool changes (schema, not just order)
        prev_tools = prev_kwargs.get("tools", [])
        curr_tools = curr_kwargs.get("tools", [])
        if prev_tools and curr_tools:
            schema_changes = detect_tool_schema_changes(prev_tools, curr_tools)
            for change in schema_changes:
                warnings.append(CacheBreakWarning(
                    reason=f"tool change: {change}",
                    suggestion="Keep tool definitions stable. Use deferred loading (stubs with defer_loading: true) instead of adding/removing tools.",
                ))

        # Fingerprint-based prefix comparison (fast path)
        if prev_fingerprint is not None:
            curr_fp = compute_fingerprint(
                system=curr_system,
                tools=curr_tools,
                messages=curr_kwargs.get("messages"),
            )
            divergence = find_divergence(prev_fingerprint, curr_fp)
            if divergence and not warnings:
                # We found divergence but didn't catch it above — add a generic warning
                warnings.append(CacheBreakWarning(
                    reason=f"prefix changed at {divergence.segment_label}",
                    divergence=divergence,
                    suggestion="Ensure content before the divergence point is identical across requests.",
                ))

        return warnings

    def estimate_miss_cost(
        self,
        divergence: DivergencePoint | None,
        fingerprint: Fingerprint,
        base_rate_per_mtok: float,
        cached_rate_per_mtok: float,
    ) -> float:
        """Estimate the cost impact of a cache miss.

        Tokens after the divergence point will be charged at base rate instead
        of cached rate.
        """
        if divergence is None:
            return 0.0

        # Estimate: all tokens from divergence point onward are uncached
        # This is an approximation since we don't have exact per-segment token counts
        total_segments = len(fingerprint.segment_hashes)
        if total_segments == 0:
            return 0.0

        uncached_fraction = (total_segments - divergence.segment_index) / total_segments
        uncached_tokens = fingerprint.token_estimate * uncached_fraction

        extra_cost = uncached_tokens * (base_rate_per_mtok - cached_rate_per_mtok) / 1_000_000
        return max(0.0, extra_cost)


def _content_hash(content: Any) -> str:
    """Quick hash for content comparison."""
    from cacheguardian.cache.fingerprint import hash_segment
    return hash_segment(content)


def _compute_detail(prev: Any, curr: Any) -> str:
    """Compute a human-readable diff detail for divergent content."""
    prev_str = _to_str(prev)
    curr_str = _to_str(curr)

    if len(prev_str) > 200:
        prev_str = prev_str[:200] + "..."
    if len(curr_str) > 200:
        curr_str = curr_str[:200] + "..."

    # Find first difference
    min_len = min(len(prev_str), len(curr_str))
    diff_pos = min_len
    for i in range(min_len):
        if prev_str[i] != curr_str[i]:
            diff_pos = i
            break

    context_start = max(0, diff_pos - 20)
    prev_context = prev_str[context_start:diff_pos + 30]
    curr_context = curr_str[context_start:diff_pos + 30]

    return f"Diff at position {diff_pos}: '...{prev_context}...' → '...{curr_context}...'"


def _to_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    return normalize_json(content)
