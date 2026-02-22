"""Middleware interceptor: wraps SDK clients and runs the L1→L2→L3 pipeline."""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Optional

from cacheguardian.cache.l1 import L1Cache
from cacheguardian.cache.l2 import L2Cache
from cacheguardian.config import CacheGuardConfig
from cacheguardian.core.differ import PromptDiffer
from cacheguardian.core.logger import (
    log_cache_break,
    log_cache_hit,
    log_cache_miss,
    log_model_recommendation,
    log_session_summary,
    setup_logging,
)
from cacheguardian.core.session import SessionTracker
from cacheguardian.providers.base import CacheProvider
from cacheguardian.types import (
    CacheBreakWarning,
    CacheMetrics,
    DryRunResult,
    Provider,
    SessionState,
)

logger = logging.getLogger("cacheguardian")

# Store for wrapped client state
_GUARD_ATTR = "_cacheguardian_state"


class CacheGuardState:
    """Internal state attached to a wrapped client."""

    def __init__(
        self,
        provider: CacheProvider,
        config: CacheGuardConfig,
        l1: L1Cache,
        l2: L2Cache,
        sessions: SessionTracker,
        differ: PromptDiffer,
    ) -> None:
        self.provider = provider
        self.config = config
        self.l1 = l1
        self.l2 = l2
        self.sessions = sessions
        self.differ = differ


def wrap_client(client: Any, **kwargs: Any) -> Any:
    """Wrap an LLM SDK client with cacheguardian middleware.

    Auto-detects the provider from the client type and wraps the appropriate
    methods (messages.create for Anthropic, chat.completions.create for OpenAI,
    models.generate_content for Gemini).
    """
    config = _build_config(kwargs)
    setup_logging(config.log_level)

    provider_impl = _detect_provider(client, config)
    is_async = _is_async_client(client)

    l1 = L1Cache()
    l2 = L2Cache(config.l2_backend)
    sessions = SessionTracker()
    differ = PromptDiffer()

    state = CacheGuardState(
        provider=provider_impl,
        config=config,
        l1=l1,
        l2=l2,
        sessions=sessions,
        differ=differ,
    )

    if is_async:
        from cacheguardian.middleware.async_interceptor import wrap_async_client
        return wrap_async_client(client, state)

    return _wrap_sync_client(client, state)


def run_dry_run(client: Any, **request_kwargs: Any) -> DryRunResult:
    """Test if a request would hit the cache without making an API call.

    Applies the same transforms as the real pipeline (tool sorting, JSON
    stabilization, cache_control injection) before fingerprinting, so the
    prediction matches what would actually happen at the API level.
    """
    state: CacheGuardState | None = getattr(client, _GUARD_ATTR, None)
    if state is None:
        raise ValueError("Client is not wrapped with cacheguardian. Use cacheguardian.wrap(client) first.")

    # Get session using RAW parts (same as real pipeline)
    raw_parts = state.provider.extract_request_parts(request_kwargs)
    model = raw_parts.get("model", "")

    session = state.sessions.get_or_create(
        provider=state.provider.get_provider(),
        model=model,
        system=raw_parts.get("system"),
        tools=raw_parts.get("tools"),
    )

    # Apply the same transforms as the real pipeline
    import copy
    transformed_kwargs = state.provider.intercept_request(
        copy.copy(request_kwargs), session,
    )
    transformed_parts = state.provider.extract_request_parts(transformed_kwargs)

    # L1 check on transformed data
    l1_result = state.l1.check(
        session_id=session.session_id,
        system=transformed_parts.get("system"),
        tools=transformed_parts.get("tools"),
        messages=transformed_parts.get("messages"),
    )

    # Collect warnings
    warnings: list[CacheBreakWarning] = []
    if not l1_result.is_first_request:
        w = l1_result.to_warning()
        if w:
            warnings.append(w)

    if session.last_request_kwargs:
        diff_warnings = state.differ.diff(
            session.last_request_kwargs, transformed_kwargs,
            prev_fingerprint=state.l1.get_fingerprint(session.session_id),
        )
        warnings.extend(diff_warnings)

    # Estimate savings
    estimated_savings = 0.0
    if l1_result.hit:
        pricing = state.config.get_pricing(
            state.provider.get_provider().value, model,
        )
        estimated_tokens = l1_result.fingerprint.token_estimate
        estimated_savings = estimated_tokens * (pricing.base_input - pricing.cache_read) / 1_000_000

    from cacheguardian.core.logger import log_dry_run
    log_dry_run(l1_result.hit, estimated_savings, warnings)

    # Model recommendation (always evaluate in dry_run — explicit developer action)
    model_rec = None
    if state.config.model_recommendations:
        from cacheguardian.core.advisor import ModelAdvisor

        advisor = ModelAdvisor(state.config)
        provider_name = state.provider.get_provider().value
        max_tokens = request_kwargs.get("max_tokens")
        model_rec = advisor.evaluate(
            provider=provider_name,
            model=model,
            token_count=l1_result.fingerprint.token_estimate,
            max_tokens=max_tokens,
        )
        if model_rec is not None:
            log_model_recommendation(model_rec)

    return DryRunResult(
        would_hit_cache=l1_result.hit,
        estimated_cached_tokens=l1_result.fingerprint.token_estimate if l1_result.hit else 0,
        estimated_uncached_tokens=0 if l1_result.hit else l1_result.fingerprint.token_estimate,
        estimated_savings=estimated_savings,
        prefix_match_depth=l1_result.prefix_match_depth,
        warnings=warnings,
        fingerprint=l1_result.fingerprint,
        model_recommendation=model_rec,
    )


def _wrap_sync_client(client: Any, state: CacheGuardState) -> Any:
    """Wrap a synchronous SDK client."""
    provider_type = state.provider.get_provider()

    if provider_type == Provider.ANTHROPIC:
        _wrap_anthropic_sync(client, state)
    elif provider_type == Provider.OPENAI:
        _wrap_openai_sync(client, state)
    elif provider_type == Provider.GEMINI:
        _wrap_gemini_sync(client, state)

    setattr(client, _GUARD_ATTR, state)
    return client


def _wrap_anthropic_sync(client: Any, state: CacheGuardState) -> None:
    """Wrap anthropic.Anthropic().messages.create()."""
    original_create = client.messages.create

    def guarded_create(**kwargs: Any) -> Any:
        return _run_pipeline(state, kwargs, original_create)

    client.messages.create = guarded_create

    # Also wrap stream if present
    if hasattr(client.messages, "stream"):
        original_stream = client.messages.stream

        def guarded_stream(**kwargs: Any) -> Any:
            # For streaming, we can still optimize the request but metrics come later
            kwargs = _pre_request(state, kwargs)
            return original_stream(**kwargs)

        client.messages.stream = guarded_stream


def _wrap_openai_sync(client: Any, state: CacheGuardState) -> None:
    """Wrap openai.OpenAI().chat.completions.create()."""
    original_create = client.chat.completions.create

    def guarded_create(**kwargs: Any) -> Any:
        return _run_pipeline(state, kwargs, original_create)

    client.chat.completions.create = guarded_create


def _wrap_gemini_sync(client: Any, state: CacheGuardState) -> None:
    """Wrap genai.Client().models.generate_content()."""
    original_generate = client.models.generate_content

    def guarded_generate(**kwargs: Any) -> Any:
        return _run_pipeline(state, kwargs, original_generate)

    client.models.generate_content = guarded_generate


def _run_pipeline(state: CacheGuardState, kwargs: dict[str, Any], original_fn: Any) -> Any:
    """The full L1 → transform → L3 → metrics pipeline."""
    # Pre-request: L1 check + transforms
    kwargs, session, l1_result = _pre_request_full(state, kwargs)

    # Call the original SDK method (L3)
    response = original_fn(**kwargs)

    # Post-request: metrics + logging
    _post_request(state, kwargs, response, session, l1_result)

    # Privacy mode: add timing jitter to mask cache-timing side channel
    if state.config.privacy_mode:
        if state.config.privacy_jitter_mode == "adaptive":
            # Scale jitter proportional to expected TTFT delta.
            # Larger responses have larger absolute TTFT differences,
            # so jitter must scale to stay effective.
            parts = state.provider.extract_request_parts(kwargs)
            msgs = parts.get("messages") or []
            # Rough token estimate for scaling
            est_tokens = sum(
                len(str(m.get("content", ""))) // 4 for m in msgs
            ) if msgs else 500
            # Base: 50-200ms, scale up by sqrt(tokens/1000)
            scale = max(1.0, (est_tokens / 1000) ** 0.5)
            lo = int(state.config.privacy_jitter_ms[0] * scale)
            hi = int(state.config.privacy_jitter_ms[1] * scale)
            jitter_ms = random.randint(lo, hi)
        else:
            jitter_ms = random.randint(*state.config.privacy_jitter_ms)
        time.sleep(jitter_ms / 1000)

    return response


def _pre_request(state: CacheGuardState, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Run pre-request transforms only (for streaming)."""
    parts = state.provider.extract_request_parts(kwargs)
    model = parts.get("model", "")

    session = state.sessions.get_or_create(
        provider=state.provider.get_provider(),
        model=model,
        system=parts.get("system"),
        tools=parts.get("tools"),
    )

    return state.provider.intercept_request(kwargs, session)


def _pre_request_full(state: CacheGuardState, kwargs: dict[str, Any]) -> tuple[dict, SessionState, Any]:
    """Full pre-request: transforms + L1 check + diff.

    Transforms are applied BEFORE the L1 check so that L1 fingerprints the
    same data the API will see (sorted tools, stabilized JSON, cache_control
    on content blocks).  This prevents false MISS reports caused by
    cosmetic differences (e.g. shuffled tool ordering) that the transforms
    would have fixed.
    """
    # 1. Get session using RAW parts (stable session ID derivation)
    raw_parts = state.provider.extract_request_parts(kwargs)
    model = raw_parts.get("model", "")

    session = state.sessions.get_or_create(
        provider=state.provider.get_provider(),
        model=model,
        system=raw_parts.get("system"),
        tools=raw_parts.get("tools"),
    )

    # 2. Apply provider-specific transforms FIRST
    kwargs = state.provider.intercept_request(kwargs, session)

    # 3. L1 cache check on TRANSFORMED data (what the API will see)
    transformed_parts = state.provider.extract_request_parts(kwargs)
    l1_result = state.l1.check(
        session_id=session.session_id,
        system=transformed_parts.get("system"),
        tools=transformed_parts.get("tools"),
        messages=transformed_parts.get("messages"),
    )

    # 4. Diff against previous request (both are transformed)
    #    Skip if L1 reports a hit (prefix extension) — no break to report.
    if session.last_request_kwargs and not l1_result.is_first_request and not l1_result.hit:
        warnings = state.differ.diff(
            session.last_request_kwargs,
            kwargs,
            prev_fingerprint=state.l1.get_fingerprint(session.session_id),
        )
        for w in warnings:
            log_cache_break(w)

        # Also check L1 warning
        l1_warning = l1_result.to_warning()
        if l1_warning and not warnings:
            log_cache_break(l1_warning)

    # 5. Model recommendation (gated on log_level and config flag)
    if (state.config.model_recommendations
            and state.config.log_level.upper() in ("DEBUG", "INFO")):
        max_tokens = kwargs.get("max_tokens")
        _check_model_recommendation(state, model, transformed_parts, max_tokens)

    return kwargs, session, l1_result


def _post_request(
    state: CacheGuardState,
    kwargs: dict[str, Any],
    response: Any,
    session: SessionState,
    l1_result: Any,
) -> None:
    """Post-request: extract metrics, update L1/L2, log."""
    parts = state.provider.extract_request_parts(kwargs)
    model = parts.get("model", "")

    metrics = state.provider.extract_metrics(response, model)

    # Update session
    state.sessions.record_request(
        session,
        input_tokens=metrics.uncached_tokens,
        cached_tokens=metrics.cached_tokens,
        cache_write_tokens=metrics.cache_write_tokens,
        output_tokens=metrics.output_tokens,
        cost_actual=metrics.estimated_cost_actual,
        cost_without_cache=metrics.estimated_cost_without_cache,
    )
    state.sessions.update_hashes(
        session,
        system=parts.get("system"),
        tools=parts.get("tools"),
    )
    session.last_request_kwargs = kwargs

    # Update L1 cache
    state.l1.update(session.session_id, l1_result.fingerprint)

    # Update L2 if available
    if state.l2.available and l1_result.fingerprint:
        state.l2.store_prefix_hash(
            prefix_hash=l1_result.fingerprint.combined,
            provider=state.provider.get_provider().value,
            model=model,
            token_count=metrics.total_input_tokens,
        )

    # Log — suppress noisy MISS logs during cold-start warmup
    if metrics.cache_hit_rate > 0.5:
        log_cache_hit(metrics, session)
    elif session.request_count <= state.config.quiet_early_turns:
        logger.debug(
            "[cacheguardian] Turn %d — cache warming up", session.request_count,
        )
    else:
        reason = ""
        if l1_result and not l1_result.hit and not l1_result.is_first_request:
            reason = f"prefix changed at {l1_result.divergence.segment_label}" if l1_result.divergence else "prefix changed"
        log_cache_miss(metrics, session, reason)

    # Check alert threshold
    if session.request_count > 3 and session.cache_hit_rate < state.config.min_cache_hit_rate:
        logger.warning(
            "[cacheguardian] ALERT: Session cache hit rate %.1f%% is below threshold %.1f%%",
            session.cache_hit_rate * 100,
            state.config.min_cache_hit_rate * 100,
        )


def _check_model_recommendation(
    state: CacheGuardState,
    model: str,
    transformed_parts: dict[str, Any],
    max_tokens: int | None = None,
) -> None:
    """Check if a cheaper model would provide better cache economics."""
    from cacheguardian.core.advisor import ModelAdvisor

    advisor = ModelAdvisor(state.config)
    provider_name = state.provider.get_provider().value

    # Use the L1 fingerprint's token estimate
    messages = transformed_parts.get("messages") or []
    system = transformed_parts.get("system")
    tools = transformed_parts.get("tools")

    # Rough token estimate from content
    token_estimate = 0
    if system:
        token_estimate += len(str(system)) // 4
    if tools:
        token_estimate += len(str(tools)) // 4
    for msg in messages:
        if isinstance(msg, dict):
            token_estimate += len(str(msg.get("content", ""))) // 4
        else:
            # Handle SDK objects (e.g., genai_types.Content)
            token_estimate += len(str(msg)) // 4

    rec = advisor.evaluate(
        provider=provider_name,
        model=model,
        token_count=token_estimate,
        max_tokens=max_tokens,
    )
    if rec is not None:
        log_model_recommendation(rec)


def _detect_provider(client: Any, config: CacheGuardConfig) -> CacheProvider:
    """Auto-detect which provider the client belongs to."""
    client_type = type(client).__module__ + "." + type(client).__qualname__

    if "anthropic" in client_type.lower():
        from cacheguardian.providers.anthropic import AnthropicProvider
        return AnthropicProvider(config)
    elif "openai" in client_type.lower():
        from cacheguardian.providers.openai import OpenAIProvider
        return OpenAIProvider(config)
    elif "google" in client_type.lower() or "genai" in client_type.lower():
        from cacheguardian.providers.gemini import GeminiProvider
        return GeminiProvider(config, gemini_client=client)
    else:
        raise ValueError(
            f"Unknown client type: {client_type}. "
            "Supported: anthropic.Anthropic, openai.OpenAI, google.genai.Client "
            "(and their async variants)."
        )


def _is_async_client(client: Any) -> bool:
    """Check if the client is an async variant."""
    name = type(client).__name__.lower()
    return "async" in name


def _build_config(kwargs: dict[str, Any]) -> CacheGuardConfig:
    """Build CacheGuardConfig from wrap() kwargs."""
    config_fields = {
        f.name for f in CacheGuardConfig.__dataclass_fields__.values()
    }
    config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
    return CacheGuardConfig(**config_kwargs)
