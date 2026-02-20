"""L1 Cache: Local Python dict for <1ms fingerprint lookups and divergence detection."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from cache_guard.cache.fingerprint import (
    compute_fingerprint,
    compute_prefix_match_depth,
    find_divergence,
)
from cache_guard.types import CacheBreakWarning, DivergencePoint, Fingerprint


class L1Cache:
    """In-process cache storing session fingerprints for instant prefix comparison.

    Keyed by session_id. Each entry stores the most recent fingerprint for that
    session, enabling <1ms divergence detection before the request hits the API.
    """

    def __init__(self) -> None:
        self._store: dict[str, _L1Entry] = {}

    def check(
        self,
        session_id: str,
        system: Any = None,
        tools: list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]] | None = None,
        token_estimate: int = 0,
    ) -> L1CheckResult:
        """Check current request against cached fingerprint for this session.

        Returns an L1CheckResult with hit/miss status and any divergence info.
        This runs in <1ms for even 100k-token prompts because it only compares
        pre-computed segment hashes — never the full content.
        """
        current_fp = compute_fingerprint(
            system=system, tools=tools, messages=messages,
            token_estimate=token_estimate,
        )

        entry = self._store.get(session_id)

        if entry is None:
            # First request in this session — always a "miss" (cache write)
            return L1CheckResult(
                hit=False,
                is_first_request=True,
                fingerprint=current_fp,
                divergence=None,
                prefix_match_depth="N/A — first request",
            )

        prev_fp = entry.fingerprint

        if current_fp.combined == prev_fp.combined:
            # Exact match — L3 provider cache should hit
            return L1CheckResult(
                hit=True,
                is_first_request=False,
                fingerprint=current_fp,
                divergence=None,
                prefix_match_depth="100% — identical to previous request",
            )

        # Divergence detected — find where
        divergence = find_divergence(prev_fp, current_fp)
        matching, total = compute_prefix_match_depth(prev_fp, current_fp)

        depth_pct = (matching / total * 100) if total > 0 else 0
        depth_str = f"{depth_pct:.0f}% — {matching}/{total} segments match (diverged at {divergence.segment_label})" if divergence else f"{depth_pct:.0f}%"

        return L1CheckResult(
            hit=False,
            is_first_request=False,
            fingerprint=current_fp,
            divergence=divergence,
            prefix_match_depth=depth_str,
        )

    def update(self, session_id: str, fingerprint: Fingerprint) -> None:
        """Update the cached fingerprint for a session after a successful API call."""
        self._store[session_id] = _L1Entry(
            fingerprint=fingerprint,
            last_seen=datetime.now(),
        )

    def get_fingerprint(self, session_id: str) -> Fingerprint | None:
        """Get the cached fingerprint for a session, if it exists."""
        entry = self._store.get(session_id)
        return entry.fingerprint if entry else None

    def invalidate(self, session_id: str) -> None:
        """Remove a session's cached fingerprint."""
        self._store.pop(session_id, None)

    def clear(self) -> None:
        """Clear all cached fingerprints."""
        self._store.clear()

    @property
    def session_count(self) -> int:
        return len(self._store)


class _L1Entry:
    """Internal L1 cache entry."""

    __slots__ = ("fingerprint", "last_seen")

    def __init__(self, fingerprint: Fingerprint, last_seen: datetime) -> None:
        self.fingerprint = fingerprint
        self.last_seen = last_seen


class L1CheckResult:
    """Result of an L1 cache check."""

    __slots__ = ("hit", "is_first_request", "fingerprint", "divergence", "prefix_match_depth")

    def __init__(
        self,
        hit: bool,
        is_first_request: bool,
        fingerprint: Fingerprint,
        divergence: DivergencePoint | None,
        prefix_match_depth: str,
    ) -> None:
        self.hit = hit
        self.is_first_request = is_first_request
        self.fingerprint = fingerprint
        self.divergence = divergence
        self.prefix_match_depth = prefix_match_depth

    def to_warning(self) -> CacheBreakWarning | None:
        """Convert a miss with divergence into a CacheBreakWarning."""
        if self.hit or self.is_first_request or not self.divergence:
            return None

        reason = f"prefix changed at {self.divergence.segment_label}"
        suggestion = ""

        label = self.divergence.segment_label
        if label == "system":
            suggestion = "Consider using system messages instead of modifying the system prompt. Move dynamic content (dates, user names) to a <system-reminder> in the next user message."
        elif label == "tools":
            suggestion = "Tool definitions changed. If you added/removed tools, consider using deferred tool loading (stubs) instead. If tool order changed, cache-guard's auto_fix should handle this."
        elif label.startswith("message[") and self.divergence.segment_index < 3:
            suggestion = "Early message changed. Ensure conversation history is append-only."

        return CacheBreakWarning(
            reason=reason,
            divergence=self.divergence,
            suggestion=suggestion,
        )
