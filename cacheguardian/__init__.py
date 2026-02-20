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
"""

from __future__ import annotations

from typing import Any, Optional

from cacheguardian.config import CacheGuardConfig
from cacheguardian.types import DryRunResult

__version__ = "0.1.0"

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


def configure(**kwargs: Any) -> CacheGuardConfig:
    """Create a CacheGuardConfig from keyword arguments."""
    return CacheGuardConfig(**kwargs)
