"""Model recommendation engine for cache economics optimization."""

from __future__ import annotations

import re
from typing import Optional

from cacheguardian.config import CacheGuardConfig
from cacheguardian.types import ModelRecommendation

# Minimum cacheable tokens per model family (mirrors per-provider _MIN_TOKENS)
_MIN_CACHE_TOKENS: dict[str, dict[str, int]] = {
    "anthropic": {
        "claude-opus-4": 4096,
        "claude-sonnet-4": 1024,
        "claude-haiku-4": 4096,
        "claude-3-5-haiku": 2048,
        "claude-3-haiku": 2048,
    },
    "openai": {
        "gpt-4o": 1024,
        "gpt-4o-mini": 1024,
        "o1": 1024,
        "o3-mini": 1024,
    },
    "gemini": {
        "gemini-2.5-pro": 1024,
        "gemini-2.5-flash": 1024,
    },
}

# Capability tiers (higher = more capable) for downgrade warnings
_CAPABILITY_TIERS: dict[str, dict[str, int]] = {
    "anthropic": {
        "claude-opus-4": 5,
        "claude-sonnet-4": 4,
        "claude-haiku-4": 3,
        "claude-3-5-haiku": 2,
        "claude-3-haiku": 1,
    },
    "openai": {
        "o1": 5,
        "gpt-4o": 4,
        "o3-mini": 3,
        "gpt-4o-mini": 2,
    },
    "gemini": {
        "gemini-2.5-pro": 4,
        "gemini-2.5-flash": 3,
    },
}

# Tier labels for capability notes
_TIER_LABELS: dict[int, str] = {
    5: "frontier",
    4: "high-capability",
    3: "mid-tier",
    2: "lightweight",
    1: "legacy",
}

# Strip date suffixes like -20250514
_DATE_SUFFIX = re.compile(r"-\d{8}$")

# Default output token estimate for cost comparison
_DEFAULT_OUTPUT_ESTIMATE = 500


class ModelAdvisor:
    """Evaluates whether a cheaper model would provide better cache economics."""

    def __init__(self, config: CacheGuardConfig) -> None:
        self._config = config

    @staticmethod
    def _model_family(model: str) -> str:
        """Extract core family: 'claude-sonnet-4-6' -> 'claude-sonnet-4',
        'claude-opus-4-20250514' -> 'claude-opus-4'."""
        # Strip date suffix first (e.g., -20250514)
        cleaned = _DATE_SUFFIX.sub("", model)
        # Strip version digit suffix (e.g., -6 in claude-opus-4-6)
        # Only strip if the second-to-last component is also a digit,
        # indicating a "generation-version" pattern (e.g., 4-6).
        # This avoids stripping the generation number itself (e.g., -4 in claude-opus-4).
        all_parts = cleaned.split("-")
        if (len(all_parts) >= 3
                and all_parts[-1].isdigit() and len(all_parts[-1]) <= 2
                and all_parts[-2].isdigit()):
            return "-".join(all_parts[:-1])
        return cleaned

    @staticmethod
    def _estimate_output_tokens(max_tokens: int | None) -> int:
        """Estimate realistic output tokens from max_tokens parameter.

        Developers often set max_tokens to 4096/8192 as a safety net,
        not as an expected generation length.
        """
        if max_tokens is None:
            return _DEFAULT_OUTPUT_ESTIMATE
        # Cap the estimate at 50% of max_tokens or 500, whichever is larger.
        # Most generations use far less than max_tokens.
        return max(500, min(max_tokens // 2, 2000))

    def _get_min_tokens(self, provider: str, model: str) -> int | None:
        """Look up minimum cache tokens for a model family."""
        provider_thresholds = _MIN_CACHE_TOKENS.get(provider, {})
        family = self._model_family(model)
        if family in provider_thresholds:
            return provider_thresholds[family]
        # Try prefix match for models not in the registry
        for key, threshold in provider_thresholds.items():
            if family.startswith(key) or key.startswith(family):
                return threshold
        return None

    def _get_capability_tier(self, provider: str, model: str) -> int:
        """Look up capability tier for a model."""
        provider_tiers = _CAPABILITY_TIERS.get(provider, {})
        family = self._model_family(model)
        if family in provider_tiers:
            return provider_tiers[family]
        for key, tier in provider_tiers.items():
            if family.startswith(key) or key.startswith(family):
                return tier
        return 0

    def _build_capability_note(
        self, provider: str, current_model: str, recommended_model: str,
    ) -> str:
        """Build a capability warning based on tier difference."""
        current_tier = self._get_capability_tier(provider, current_model)
        rec_tier = self._get_capability_tier(provider, recommended_model)

        if rec_tier >= current_tier or current_tier == 0 or rec_tier == 0:
            return ""

        tier_drop = current_tier - rec_tier
        rec_label = _TIER_LABELS.get(rec_tier, "unknown")
        current_label = _TIER_LABELS.get(current_tier, "unknown")

        if tier_drop >= 2:
            rec_family = self._model_family(recommended_model)
            cur_family = self._model_family(current_model)
            return (
                f"{rec_family} has significantly reduced reasoning capability "
                f"compared to {cur_family}. Verify task complexity before switching."
            )
        else:
            rec_family = self._model_family(recommended_model)
            cur_family = self._model_family(current_model)
            return (
                f"{rec_family} ({rec_label}) is less capable than "
                f"{cur_family} ({current_label})."
            )

    def evaluate(
        self,
        provider: str,
        model: str,
        token_count: int,
        output_tokens: int = 0,
        max_tokens: int | None = None,
    ) -> ModelRecommendation | None:
        """Evaluate whether a different model would provide better cache economics.

        Args:
            provider: Provider name ('anthropic', 'openai', 'gemini').
            model: Current model name (e.g., 'claude-opus-4-6').
            token_count: Estimated input token count.
            output_tokens: Known output tokens (0 = estimate from max_tokens).
            max_tokens: The max_tokens parameter from the request.

        Returns:
            ModelRecommendation if a cheaper alternative exists, None otherwise.
        """
        # Gate: look up current model's threshold
        current_min = self._get_min_tokens(provider, model)
        if current_min is None:
            return None  # Unknown model, can't advise

        # If token count meets or exceeds the threshold, caching already works
        if token_count >= current_min:
            return None

        # Estimate output tokens if not provided
        est_output = output_tokens if output_tokens > 0 else self._estimate_output_tokens(max_tokens)

        # Current cost (no caching possible — all input at base rate)
        current_pricing = self._config.get_pricing(provider, model)
        current_cost = (
            token_count * current_pricing.base_input / 1_000_000
            + est_output * current_pricing.output / 1_000_000
        )

        # Find alternatives: same-provider models where caching would work
        provider_thresholds = _MIN_CACHE_TOKENS.get(provider, {})
        best_rec: ModelRecommendation | None = None
        best_cost = float("inf")

        for alt_family, alt_min in provider_thresholds.items():
            # Skip the current model's family
            if alt_family == self._model_family(model):
                continue
            # Only consider models where caching would actually work
            if token_count < alt_min:
                continue

            # Find the best concrete model name for pricing lookup
            alt_model = self._find_best_model_variant(provider, alt_family)
            if alt_model is None:
                continue

            alt_pricing = self._config.get_pricing(provider, alt_model)
            if alt_pricing.base_input == 0.0 and alt_pricing.cache_read == 0.0:
                continue  # No pricing data

            # Cost with caching on the alternative model
            alt_cost = (
                token_count * alt_pricing.cache_read / 1_000_000
                + est_output * alt_pricing.output / 1_000_000
            )

            if alt_cost < best_cost:
                best_cost = alt_cost
                savings = current_cost - alt_cost
                savings_pct = (savings / current_cost * 100) if current_cost > 0 else 0

                best_rec = ModelRecommendation(
                    recommended_model=alt_model,
                    current_model=model,
                    reason=(
                        f"{token_count:,} tokens < {self._model_family(model)}'s "
                        f"cache min ({current_min:,})"
                    ),
                    current_min_tokens=current_min,
                    recommended_min_tokens=alt_min,
                    estimated_token_count=token_count,
                    current_input_cost_per_mtok=current_pricing.base_input,
                    recommended_cache_read_cost_per_mtok=alt_pricing.cache_read,
                    recommended_input_cost_per_mtok=alt_pricing.base_input,
                    current_cost_per_request=current_cost,
                    recommended_cost_per_request=alt_cost,
                    estimated_savings_per_request=savings,
                    savings_percentage=savings_pct,
                    current_output_cost_per_mtok=current_pricing.output,
                    recommended_output_cost_per_mtok=alt_pricing.output,
                    capability_note=self._build_capability_note(provider, model, alt_model),
                )

        # Only recommend if savings > 10%
        if best_rec is not None and best_rec.savings_percentage <= 10:
            return None

        return best_rec

    def _find_best_model_variant(self, provider: str, family: str) -> str | None:
        """Find the best concrete model name for a family in the pricing tables."""
        from cacheguardian.config import ALL_PRICING

        provider_pricing = ALL_PRICING.get(provider, {})
        # Also check overrides
        overrides = self._config.pricing_overrides.get(provider, {})
        all_models = {**provider_pricing, **overrides}

        # Exact match first
        if family in all_models:
            return family

        # Find models that belong to this family
        candidates = []
        for model_name in all_models:
            if self._model_family(model_name) == family:
                candidates.append(model_name)

        if not candidates:
            return None

        # Prefer the latest version (highest version suffix)
        candidates.sort(reverse=True)
        return candidates[0]
