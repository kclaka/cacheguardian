"""Configuration for cacheguardian."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class PricingConfig:
    """Per-model pricing rates (per million tokens)."""

    base_input: float = 0.0
    cache_read: float = 0.0
    cache_write_5m: float = 0.0
    cache_write_1h: float = 0.0
    output: float = 0.0
    storage_per_hour: float = 0.0  # Gemini only


# Default pricing tables (as of Feb 2026)
ANTHROPIC_PRICING: dict[str, PricingConfig] = {
    "claude-sonnet-4-20250514": PricingConfig(
        base_input=3.00, cache_read=0.30, cache_write_5m=3.75,
        cache_write_1h=6.00, output=15.00,
    ),
    "claude-opus-4-20250514": PricingConfig(
        base_input=15.00, cache_read=1.50, cache_write_5m=18.75,
        cache_write_1h=30.00, output=75.00,
    ),
    "claude-haiku-4-20250514": PricingConfig(
        base_input=0.80, cache_read=0.08, cache_write_5m=1.00,
        cache_write_1h=1.60, output=4.00,
    ),
}

OPENAI_PRICING: dict[str, PricingConfig] = {
    "gpt-4o": PricingConfig(
        base_input=2.50, cache_read=1.25, output=10.00,
    ),
    "gpt-4o-mini": PricingConfig(
        base_input=0.15, cache_read=0.075, output=0.60,
    ),
    "o1": PricingConfig(
        base_input=15.00, cache_read=7.50, output=60.00,
    ),
    "o3-mini": PricingConfig(
        base_input=1.10, cache_read=0.55, output=4.40,
    ),
}

GEMINI_PRICING: dict[str, PricingConfig] = {
    "gemini-2.5-flash": PricingConfig(
        base_input=0.15, cache_read=0.0375, output=0.60,
        storage_per_hour=4.50,
    ),
    "gemini-2.5-pro": PricingConfig(
        base_input=1.25, cache_read=0.3125, output=10.00,
        storage_per_hour=4.50,
    ),
}

ALL_PRICING: dict[str, dict[str, PricingConfig]] = {
    "anthropic": ANTHROPIC_PRICING,
    "openai": OPENAI_PRICING,
    "gemini": GEMINI_PRICING,
}


@dataclass
class CacheGuardConfig:
    """Configuration for cacheguardian middleware."""

    # General
    auto_fix: bool = True
    """Automatically apply safe optimizations (tool sorting, cache_control injection)."""

    strict_mode: bool = False
    """Raise exceptions on cache-breaking changes instead of just warning."""

    log_level: str = "INFO"
    """Logging level: DEBUG, INFO, WARNING, ERROR."""

    # TTL strategy
    ttl_strategy: str = "auto"
    """TTL selection: 'auto' (based on think time), '5m', '1h'."""

    # Privacy
    privacy_mode: bool = False
    """Add timing jitter to prevent cache-timing side-channel attacks."""

    privacy_jitter_ms: tuple[int, int] = (50, 200)
    """Range of jitter in milliseconds (min, max)."""

    # OpenAI routing
    cache_key_fn: Optional[Callable[[Any], str]] = None
    """Function to derive prompt_cache_key for OpenAI (e.g., lambda ctx: f'{ctx.user_id}')."""

    # L2 backend
    l2_backend: Optional[str] = None
    """Redis URL for L2 distributed cache (e.g., 'redis://localhost:6379')."""

    # Pricing overrides
    pricing_overrides: dict[str, dict[str, PricingConfig]] = field(default_factory=dict)
    """Override default pricing: {'anthropic': {'model-name': PricingConfig(...)}}."""

    # Alerts
    min_cache_hit_rate: float = 0.7
    """Alert when session cache hit rate drops below this threshold."""

    def get_pricing(self, provider: str, model: str) -> PricingConfig:
        """Get pricing for a provider/model, checking overrides first."""
        overrides = self.pricing_overrides.get(provider, {})
        if model in overrides:
            return overrides[model]

        defaults = ALL_PRICING.get(provider, {})
        if model in defaults:
            return defaults[model]

        # Fuzzy match: try prefix matching for versioned model names
        for key, pricing in {**defaults, **overrides}.items():
            if model.startswith(key) or key.startswith(model):
                return pricing

        # Return zero pricing if model unknown (metrics still work, costs show 0)
        return PricingConfig()
