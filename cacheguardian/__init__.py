"""cacheguardian: Universal prompt cache optimizer for Anthropic, OpenAI & Gemini.

Usage:
    import cacheguardian

    # Wrap any LLM SDK client
    client = cacheguardian.wrap(anthropic.Anthropic())
    client = cacheguardian.wrap(openai.OpenAI())
    client = cacheguardian.wrap(genai.Client())

    # Async clients work too
    client = cacheguardian.wrap(anthropic.AsyncAnthropic())

    # Dry-run: test cache behavior without API calls
    result = cacheguardian.dry_run(client, model="...", messages=[...])

    # Model recommendation: check if a cheaper model has better cache economics
    rec = cacheguardian.recommend(provider="anthropic", model="claude-opus-4-6", token_count=1500)
"""

from __future__ import annotations

from typing import Any, Optional

from cacheguardian.config import CacheGuardConfig
from cacheguardian.types import DryRunResult, ModelRecommendation

__version__ = "0.3.0"

# Will be populated by middleware module
_wrap_impl = None
_dry_run_impl = None


def wrap(client: Any, **kwargs: Any) -> Any:
    """Wrap an LLM SDK client with cacheguardian middleware.

    Supports: anthropic.Anthropic, anthropic.AsyncAnthropic,
              openai.OpenAI, openai.AsyncOpenAI,
              google.genai.Client

    Args:
        client: An LLM SDK client instance.
        **kwargs: CacheGuardConfig options (auto_fix, ttl_strategy, privacy_mode, etc.)

    Returns:
        A wrapped client that transparently optimizes caching.
    """
    from cacheguardian.middleware.interceptor import wrap_client
    return wrap_client(client, **kwargs)


def dry_run(client: Any, **request_kwargs: Any) -> DryRunResult:
    """Test if a request would hit the cache without making an API call.

    Args:
        client: A cacheguardian wrapped client.
        **request_kwargs: The same kwargs you'd pass to messages.create() etc.

    Returns:
        DryRunResult with predictions about cache behavior.
    """
    from cacheguardian.middleware.interceptor import run_dry_run
    return run_dry_run(client, **request_kwargs)


def recommend(
    client: Any = None,
    *,
    provider: str | None = None,
    model: str,
    token_count: int,
    output_tokens: int = 0,
    max_tokens: int | None = None,
    **config_kwargs: Any,
) -> ModelRecommendation | None:
    """Check if a cheaper model would provide better cache economics.

    Can be used standalone (without a wrapped client) or with a wrapped client
    to inherit its config and pricing overrides.

    Args:
        client: Optional cacheguardian-wrapped client (inherits config).
        provider: Provider name ('anthropic', 'openai', 'gemini').
            Required if client is not provided.
        model: Current model name (e.g., 'claude-opus-4-6').
        token_count: Estimated input token count.
        output_tokens: Known output tokens (0 = estimate from max_tokens).
        max_tokens: The max_tokens parameter for output estimation.
        **config_kwargs: CacheGuardConfig overrides (used when no client).

    Returns:
        ModelRecommendation if a cheaper alternative exists, None otherwise.
    """
    from cacheguardian.core.advisor import ModelAdvisor
    from cacheguardian.middleware.interceptor import _GUARD_ATTR

    if client is not None:
        state = getattr(client, _GUARD_ATTR, None)
        if state is not None:
            config = state.config
            if provider is None:
                provider = state.provider.get_provider().value
        else:
            config = CacheGuardConfig(**config_kwargs)
    else:
        config = CacheGuardConfig(**config_kwargs)

    if provider is None:
        raise ValueError(
            "provider is required when no wrapped client is provided. "
            "Use provider='anthropic', 'openai', or 'gemini'."
        )

    advisor = ModelAdvisor(config)
    return advisor.evaluate(
        provider=provider,
        model=model,
        token_count=token_count,
        output_tokens=output_tokens,
        max_tokens=max_tokens,
    )


def configure(**kwargs: Any) -> CacheGuardConfig:
    """Create a CacheGuardConfig from keyword arguments."""
    return CacheGuardConfig(**kwargs)
