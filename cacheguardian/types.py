"""Core type definitions for cacheguardian."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"


class CacheEventType(str, Enum):
    HIT = "hit"
    MISS = "miss"
    WRITE = "write"
    BREAK = "break"
    PROMOTION = "promotion"


@dataclass
class Fingerprint:
    """Rolling hash fingerprint of a prompt's segments."""

    combined: str
    """SHA-256 hash of all segment hashes concatenated."""

    segment_hashes: list[str]
    """Per-segment hashes: [system, tools, msg_0, msg_1, ...]."""

    segment_labels: list[str]
    """Human-readable labels: ['system', 'tools', 'message[0]', ...]."""

    token_estimate: int = 0
    """Approximate token count for the fingerprinted content."""

    segment_token_estimates: list[int] = field(default_factory=list)
    """Per-segment token estimates for accurate divergence cost impact."""


@dataclass
class DivergencePoint:
    """Where two fingerprints diverge."""

    segment_index: int
    """Index of the first divergent segment."""

    segment_label: str
    """Human label of the divergent segment (e.g., 'tools', 'message[3]')."""

    previous_hash: str
    new_hash: str

    detail: str = ""
    """Character-level diff detail for the divergent segment (populated on demand)."""


@dataclass
class CacheMetrics:
    """Per-request cache performance metrics."""

    provider: Provider
    model: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Token counts
    total_input_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    uncached_tokens: int = 0
    output_tokens: int = 0

    # Derived
    cache_hit_rate: float = 0.0

    # Cost
    estimated_cost_actual: float = 0.0
    estimated_cost_without_cache: float = 0.0
    estimated_savings: float = 0.0

    # Cache health
    cache_break_detected: bool = False
    cache_break_reason: Optional[str] = None
    divergence: Optional[DivergencePoint] = None

    # L1 prediction (set before request)
    l1_predicted_hit: Optional[bool] = None


@dataclass
class SessionState:
    """Tracks the state of a conversation session."""

    session_id: str
    provider: Provider
    model: str
    created_at: datetime = field(default_factory=datetime.now)
    last_request_at: Optional[datetime] = None

    # Prefix tracking
    current_fingerprint: Optional[Fingerprint] = None
    system_hash: Optional[str] = None
    tools_hash: Optional[str] = None

    # Cumulative metrics
    request_count: int = 0
    total_input_tokens: int = 0
    total_cached_tokens: int = 0
    total_cache_write_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_actual: float = 0.0
    total_cost_without_cache: float = 0.0
    total_savings: float = 0.0

    # Request timing (for TTL/retention decisions)
    request_timestamps: list[datetime] = field(default_factory=list)

    # Provider-specific
    gemini_cache_name: Optional[str] = None
    openai_cache_key: Optional[str] = None
    anthropic_ttl: str = "5m"

    # Last request kwargs (for diffing)
    last_request_kwargs: Optional[dict[str, Any]] = None

    @property
    def cache_hit_rate(self) -> float:
        total = self.total_cached_tokens + self.total_cache_write_tokens + self.total_input_tokens
        if total == 0:
            return 0.0
        return self.total_cached_tokens / total

    def average_request_interval_seconds(self) -> Optional[float]:
        if len(self.request_timestamps) < 2:
            return None
        intervals = [
            (b - a).total_seconds()
            for a, b in zip(self.request_timestamps[:-1], self.request_timestamps[1:])
        ]
        return sum(intervals) / len(intervals)


@dataclass
class CacheBreakWarning:
    """Warning about a detected cache-breaking change."""

    reason: str
    divergence: Optional[DivergencePoint] = None
    estimated_cost_impact: float = 0.0
    suggestion: str = ""


@dataclass
class CostEstimate:
    """Cost breakdown for a request."""

    input_cost: float = 0.0
    cache_read_cost: float = 0.0
    cache_write_cost: float = 0.0
    storage_cost: float = 0.0  # Gemini only
    output_cost: float = 0.0
    total_cost: float = 0.0
    cost_without_cache: float = 0.0
    savings: float = 0.0


@dataclass
class PromotionDecision:
    """Result of the cache promotion cost-benefit analysis."""

    should_promote: bool
    reason: str
    break_even_n: float
    """Number of requests needed to break even."""
    estimated_n: float
    """Estimated number of requests based on observed frequency."""
    estimated_savings_if_promoted: float = 0.0
    estimated_storage_cost: float = 0.0  # Gemini only


@dataclass
class DryRunResult:
    """Result of a dry-run cache check (no API call)."""

    would_hit_cache: bool
    estimated_cached_tokens: int = 0
    estimated_uncached_tokens: int = 0
    estimated_savings: float = 0.0
    prefix_match_depth: str = ""
    warnings: list[CacheBreakWarning] = field(default_factory=list)
    fingerprint: Optional[Fingerprint] = None
