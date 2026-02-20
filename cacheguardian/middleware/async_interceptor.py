"""Async middleware interceptor: wraps async SDK clients without blocking the event loop."""

from __future__ import annotations

import asyncio
import random
from typing import Any

from cacheguardian.middleware.interceptor import (
    CacheGuardState,
    _GUARD_ATTR,
    _post_request,
    _pre_request_full,
)
from cacheguardian.types import Provider


def wrap_async_client(client: Any, state: CacheGuardState) -> Any:
    """Wrap an async SDK client with cacheguardian middleware.

    All diff/metrics operations run via asyncio.to_thread() to avoid
    blocking the event loop.
    """
    provider_type = state.provider.get_provider()

    if provider_type == Provider.ANTHROPIC:
        _wrap_anthropic_async(client, state)
    elif provider_type == Provider.OPENAI:
        _wrap_openai_async(client, state)
    elif provider_type == Provider.GEMINI:
        _wrap_gemini_async(client, state)

    setattr(client, _GUARD_ATTR, state)
    return client


def _wrap_anthropic_async(client: Any, state: CacheGuardState) -> None:
    """Wrap AsyncAnthropic().messages.create()."""
    original_create = client.messages.create

    async def guarded_create(**kwargs: Any) -> Any:
        return await _run_async_pipeline(state, kwargs, original_create)

    client.messages.create = guarded_create


def _wrap_openai_async(client: Any, state: CacheGuardState) -> None:
    """Wrap AsyncOpenAI().chat.completions.create()."""
    original_create = client.chat.completions.create

    async def guarded_create(**kwargs: Any) -> Any:
        return await _run_async_pipeline(state, kwargs, original_create)

    client.chat.completions.create = guarded_create


def _wrap_gemini_async(client: Any, state: CacheGuardState) -> None:
    """Wrap async genai.Client().models.generate_content()."""
    # Gemini's async API may vary; wrap generate_content_async if available
    if hasattr(client.models, "generate_content_async"):
        original = client.models.generate_content_async
    else:
        original = client.models.generate_content

    async def guarded_generate(**kwargs: Any) -> Any:
        return await _run_async_pipeline(state, kwargs, original)

    if hasattr(client.models, "generate_content_async"):
        client.models.generate_content_async = guarded_generate
    else:
        client.models.generate_content = guarded_generate


async def _run_async_pipeline(
    state: CacheGuardState,
    kwargs: dict[str, Any],
    original_fn: Any,
) -> Any:
    """Async version of the L1 → transform → L3 → metrics pipeline.

    Pre-request transforms and post-request metrics run in a thread pool
    to avoid blocking the event loop.
    """
    # Pre-request: run diff/transforms off the event loop
    kwargs, session, l1_result = await asyncio.to_thread(
        _pre_request_full, state, kwargs
    )

    # Call the original async SDK method (L3)
    response = await original_fn(**kwargs)

    # Post-request: run metrics/logging off the event loop
    await asyncio.to_thread(
        _post_request, state, kwargs, response, session, l1_result
    )

    # Privacy mode: async jitter
    if state.config.privacy_mode:
        jitter_ms = random.randint(*state.config.privacy_jitter_ms)
        await asyncio.sleep(jitter_ms / 1000)

    return response
