"""Cache promotion logic: decides when to upgrade to long-lived/explicit caches."""

from __future__ import annotations

from typing import Optional

from cacheguardian.config import CacheGuardConfig, PricingConfig
from cacheguardian.types import PromotionDecision, Provider, SessionState


class CachePromoter:
    """Decides when to promote content to a long-lived or explicit cache.

    Uses the break-even formula:
        N > (C_write + S * T) / (C_input - C_cache_read)

    Where:
        N           = expected number of requests reusing this content
        C_write     = cost to write the cache (token count * write rate)
        S           = storage cost per hour (Gemini only; 0 for Anthropic)
        T           = TTL in hours
        C_input     = standard input token cost per token
        C_cache_read = discounted cache read cost per token
    """

    def __init__(self, config: CacheGuardConfig) -> None:
        self._config = config

    def evaluate_anthropic_1h(
        self, session: SessionState, token_count: int,
    ) -> PromotionDecision:
        """Should we switch from 5m to 1h TTL for Anthropic?

        1h TTL costs 2x base (vs 1.25x for 5m), but avoids cache misses from
        long think times. Worth it when request intervals > 5min.
        """
        pricing = self._config.get_pricing("anthropic", session.model)

        write_cost_5m = token_count * pricing.cache_write_5m / 1_000_000
        write_cost_1h = token_count * pricing.cache_write_1h / 1_000_000
        read_savings = token_count * (pricing.base_input - pricing.cache_read) / 1_000_000

        if read_savings <= 0:
            return PromotionDecision(
                should_promote=False,
                reason="No savings from caching at current pricing.",
                break_even_n=float("inf"),
                estimated_n=0,
            )

        # Break-even: how many reads to justify the higher write cost?
        extra_write_cost = write_cost_1h - write_cost_5m
        break_even_n = extra_write_cost / read_savings if read_savings > 0 else float("inf")

        # Estimate expected N based on session history
        avg_interval = session.average_request_interval_seconds()
        if avg_interval is None or avg_interval <= 0:
            estimated_n = 1.0
        else:
            # How many requests in 1 hour?
            estimated_n = 3600 / avg_interval

        should = estimated_n > break_even_n and avg_interval is not None and avg_interval > 300

        return PromotionDecision(
            should_promote=should,
            reason=f"Avg interval: {avg_interval:.0f}s, estimated {estimated_n:.1f} requests/hr, break-even at {break_even_n:.1f}" if avg_interval else "Not enough data",
            break_even_n=break_even_n,
            estimated_n=estimated_n,
            estimated_savings_if_promoted=read_savings * max(0, estimated_n - break_even_n),
        )

    def evaluate_gemini_explicit(
        self, session: SessionState, token_count: int, ttl_hours: float = 1.0,
    ) -> PromotionDecision:
        """Should we create an explicit CachedContent on Gemini?

        Explicit caches guarantee the discount but incur storage costs.
        """
        pricing = self._config.get_pricing("gemini", session.model)

        write_cost = 0.0  # Gemini doesn't charge for writes, but charges storage
        storage_cost = token_count * pricing.storage_per_hour / 1_000_000 * ttl_hours
        read_savings_per_req = token_count * (pricing.base_input - pricing.cache_read) / 1_000_000

        if read_savings_per_req <= 0:
            return PromotionDecision(
                should_promote=False,
                reason="No savings from caching at current pricing.",
                break_even_n=float("inf"),
                estimated_n=0,
                estimated_storage_cost=storage_cost,
            )

        break_even_n = (write_cost + storage_cost) / read_savings_per_req

        avg_interval = session.average_request_interval_seconds()
        if avg_interval is None or avg_interval <= 0:
            estimated_n = 1.0
        else:
            estimated_n = (ttl_hours * 3600) / avg_interval

        should = estimated_n > break_even_n and token_count >= 1024

        return PromotionDecision(
            should_promote=should,
            reason=f"Storage: ${storage_cost:.4f}/TTL, savings: ${read_savings_per_req:.4f}/req, break-even at {break_even_n:.1f} reqs, estimated {estimated_n:.1f}" if avg_interval else "Not enough data",
            break_even_n=break_even_n,
            estimated_n=estimated_n,
            estimated_savings_if_promoted=read_savings_per_req * max(0, estimated_n - break_even_n),
            estimated_storage_cost=storage_cost,
        )

    def evaluate_openai_retention(
        self, session: SessionState,
    ) -> bool:
        """Should we use prompt_cache_retention='24h' for OpenAI?

        24h retention is free but uses GPU-local storage. Worth it when
        request intervals are > 10 minutes (risk of default cache eviction).
        """
        avg_interval = session.average_request_interval_seconds()
        if avg_interval is None:
            return False
        return avg_interval > 600  # > 10 minutes
