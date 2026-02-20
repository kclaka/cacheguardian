"""cache-guard: Universal prompt cache optimizer for Anthropic, OpenAI & Gemini.

Usage:
    import cache_guard

    # Wrap any LLM SDK client
    client = cache_guard.wrap(anthropic.Anthropic())
    client = cache_guard.wrap(openai.OpenAI())
    client = cache_guard.wrap(genai.Client())

    # Async clients work too
    client = cache_guard.wrap(anthropic.AsyncAnthropic())

    # Dry-run: test cache behavior without API calls
    result = cache_guard.dry_run(client, model="...", messages=[...])
"""

from __future__ import annotations

from typing import Any, Optional

from cache_guard.config import CacheGuardConfig
from cache_guard.types import DryRunResult

__version__ = "0.1.0"

# Will be populated by middleware module
_wrap_impl = None
_dry_run_impl = None


def wrap(client: Any, **kwargs: Any) -> Any:
    """Wrap an LLM SDK client with cache-guard middleware.

    Supports: anthropic.Anthropic, anthropic.AsyncAnthropic,
              openai.OpenAI, openai.AsyncOpenAI,
              google.genai.Client

    Args:
        client: An LLM SDK client instance.
        **kwargs: CacheGuardConfig options (auto_fix, ttl_strategy, privacy_mode, etc.)

    Returns:
        A wrapped client that transparently optimizes caching.
    """
    from cache_guard.middleware.interceptor import wrap_client
    return wrap_client(client, **kwargs)


def dry_run(client: Any, **request_kwargs: Any) -> DryRunResult:
    """Test if a request would hit the cache without making an API call.

    Args:
        client: A cache-guard wrapped client.
        **request_kwargs: The same kwargs you'd pass to messages.create() etc.

    Returns:
        DryRunResult with predictions about cache behavior.
    """
    from cache_guard.middleware.interceptor import run_dry_run
    return run_dry_run(client, **request_kwargs)


def configure(**kwargs: Any) -> CacheGuardConfig:
    """Create a CacheGuardConfig from keyword arguments."""
    return CacheGuardConfig(**kwargs)
