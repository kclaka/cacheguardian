"""Cost calculation and cache metrics for all providers."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from cacheguardian.config import CacheGuardConfig, PricingConfig
from cacheguardian.types import CacheMetrics, CostEstimate, Provider


class MetricsCollector:
    """Extracts and calculates cache metrics from API responses."""

    def __init__(self, config: CacheGuardConfig) -> None:
        self._config = config

    def extract_anthropic(self, response: Any, model: str) -> CacheMetrics:
        """Extract metrics from an Anthropic API response."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return CacheMetrics(provider=Provider.ANTHROPIC, model=model)

        cached = getattr(usage, "cache_read_input_tokens", 0) or 0
        write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0

        total_input = cached + write + input_tokens
        hit_rate = cached / total_input if total_input > 0 else 0.0

        pricing = self._config.get_pricing("anthropic", model)
        cost = self._compute_cost_anthropic(cached, write, input_tokens, output_tokens, pricing)

        return CacheMetrics(
            provider=Provider.ANTHROPIC,
            model=model,
            total_input_tokens=total_input,
            cached_tokens=cached,
            cache_write_tokens=write,
            uncached_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit_rate=hit_rate,
            estimated_cost_actual=cost.total_cost,
            estimated_cost_without_cache=cost.cost_without_cache,
            estimated_savings=cost.savings,
        )

    def extract_openai(self, response: Any, model: str) -> CacheMetrics:
        """Extract metrics from an OpenAI API response."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return CacheMetrics(provider=Provider.OPENAI, model=model)

        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0

        cached = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details:
            cached = getattr(details, "cached_tokens", 0) or 0

        uncached = prompt_tokens - cached
        total_input = prompt_tokens
        hit_rate = cached / total_input if total_input > 0 else 0.0

        pricing = self._config.get_pricing("openai", model)
        cost = self._compute_cost_openai(cached, uncached, output_tokens, pricing)

        return CacheMetrics(
            provider=Provider.OPENAI,
            model=model,
            total_input_tokens=total_input,
            cached_tokens=cached,
            cache_write_tokens=0,  # OpenAI doesn't report writes separately
            uncached_tokens=uncached,
            output_tokens=output_tokens,
            cache_hit_rate=hit_rate,
            estimated_cost_actual=cost.total_cost,
            estimated_cost_without_cache=cost.cost_without_cache,
            estimated_savings=cost.savings,
        )

    def extract_gemini(self, response: Any, model: str) -> CacheMetrics:
        """Extract metrics from a Gemini API response."""
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return CacheMetrics(provider=Provider.GEMINI, model=model)

        total_input = getattr(usage, "prompt_token_count", 0) or 0
        cached = getattr(usage, "cached_content_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0

        uncached = total_input - cached
        hit_rate = cached / total_input if total_input > 0 else 0.0

        pricing = self._config.get_pricing("gemini", model)
        cost = self._compute_cost_gemini(cached, uncached, output_tokens, pricing)

        return CacheMetrics(
            provider=Provider.GEMINI,
            model=model,
            total_input_tokens=total_input,
            cached_tokens=cached,
            cache_write_tokens=0,
            uncached_tokens=uncached,
            output_tokens=output_tokens,
            cache_hit_rate=hit_rate,
            estimated_cost_actual=cost.total_cost,
            estimated_cost_without_cache=cost.cost_without_cache,
            estimated_savings=cost.savings,
        )

    def _compute_cost_anthropic(
        self, cached: int, write: int, uncached: int, output: int, pricing: PricingConfig,
    ) -> CostEstimate:
        cache_read_cost = cached * pricing.cache_read / 1_000_000
        cache_write_cost = write * pricing.cache_write_5m / 1_000_000
        input_cost = uncached * pricing.base_input / 1_000_000
        output_cost = output * pricing.output / 1_000_000

        total = cache_read_cost + cache_write_cost + input_cost + output_cost
        without_cache = (cached + write + uncached) * pricing.base_input / 1_000_000 + output_cost

        return CostEstimate(
            input_cost=input_cost,
            cache_read_cost=cache_read_cost,
            cache_write_cost=cache_write_cost,
            output_cost=output_cost,
            total_cost=total,
            cost_without_cache=without_cache,
            savings=without_cache - total,
        )

    def _compute_cost_openai(
        self, cached: int, uncached: int, output: int, pricing: PricingConfig,
    ) -> CostEstimate:
        cache_read_cost = cached * pricing.cache_read / 1_000_000
        input_cost = uncached * pricing.base_input / 1_000_000
        output_cost = output * pricing.output / 1_000_000

        total = cache_read_cost + input_cost + output_cost
        without_cache = (cached + uncached) * pricing.base_input / 1_000_000 + output_cost

        return CostEstimate(
            input_cost=input_cost,
            cache_read_cost=cache_read_cost,
            output_cost=output_cost,
            total_cost=total,
            cost_without_cache=without_cache,
            savings=without_cache - total,
        )

    def _compute_cost_gemini(
        self, cached: int, uncached: int, output: int, pricing: PricingConfig,
    ) -> CostEstimate:
        cache_read_cost = cached * pricing.cache_read / 1_000_000
        input_cost = uncached * pricing.base_input / 1_000_000
        output_cost = output * pricing.output / 1_000_000
        # Note: storage cost is tracked separately per-session, not per-request

        total = cache_read_cost + input_cost + output_cost
        without_cache = (cached + uncached) * pricing.base_input / 1_000_000 + output_cost

        return CostEstimate(
            input_cost=input_cost,
            cache_read_cost=cache_read_cost,
            output_cost=output_cost,
            total_cost=total,
            cost_without_cache=without_cache,
            savings=without_cache - total,
        )
