"""Rich-based colored logging for cache-guard."""

from __future__ import annotations

import logging
from typing import Optional

from rich.console import Console
from rich.text import Text

from cache_guard.types import CacheBreakWarning, CacheMetrics, SessionState

console = Console(stderr=True)
logger = logging.getLogger("cache_guard")


def setup_logging(level: str = "INFO") -> None:
    """Configure cache-guard logging."""
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)


def log_cache_hit(metrics: CacheMetrics, session: SessionState) -> None:
    """Log a cache hit with savings."""
    text = Text()
    text.append("[cache-guard] ", style="bold cyan")
    text.append("L1 HIT", style="bold green")
    text.append(f" | Cache hit {metrics.cache_hit_rate:.1%}", style="green")
    text.append(f" | Saved ${metrics.estimated_savings:.4f}", style="bold green")
    text.append(f" | Session total: ${session.total_savings:.2f} saved", style="dim")
    console.print(text)


def log_cache_miss(metrics: CacheMetrics, session: SessionState, reason: str = "") -> None:
    """Log a cache miss with cost impact."""
    text = Text()
    text.append("[cache-guard] ", style="bold cyan")
    text.append("L1 MISS", style="bold yellow")
    if reason:
        text.append(f" — {reason}", style="yellow")
    text.append(f" | Predicted cost: ${metrics.estimated_cost_actual:.4f}", style="yellow")
    text.append(f" | Session total: ${session.total_savings:.2f} saved", style="dim")
    console.print(text)


def log_cache_break(warning: CacheBreakWarning) -> None:
    """Log a cache-breaking change with diff."""
    text = Text()
    text.append("[cache-guard] ", style="bold cyan")
    text.append("CACHE BREAK", style="bold red")
    text.append(f" — {warning.reason}", style="red")
    if warning.estimated_cost_impact > 0:
        text.append(f" — estimated cost: ${warning.estimated_cost_impact:.4f}", style="red")
    console.print(text)

    if warning.suggestion:
        suggestion = Text()
        suggestion.append("  Suggestion: ", style="bold")
        suggestion.append(warning.suggestion, style="italic")
        console.print(suggestion)

    if warning.divergence and warning.divergence.detail:
        console.print(f"  Diverged at: {warning.divergence.segment_label}", style="dim")
        console.print(f"  {warning.divergence.detail}", style="dim red")


def log_session_summary(session: SessionState) -> None:
    """Log end-of-session summary."""
    text = Text()
    text.append("\n[cache-guard] ", style="bold cyan")
    text.append("Session Summary", style="bold underline")
    console.print(text)

    console.print(f"  Provider: {session.provider.value}", style="dim")
    console.print(f"  Model: {session.model}", style="dim")
    console.print(f"  Requests: {session.request_count}")
    console.print(f"  Total input tokens: {session.total_input_tokens:,}")
    console.print(f"  Total cached tokens: {session.total_cached_tokens:,}")
    console.print(f"  Cache hit rate: {session.cache_hit_rate:.1%}")

    savings_style = "bold green" if session.total_savings > 0 else "bold red"
    console.print(f"  Total cost: ${session.total_cost_actual:.4f}", style="dim")
    console.print(f"  Cost without cache: ${session.total_cost_without_cache:.4f}", style="dim")
    console.print(f"  Total savings: ${session.total_savings:.4f}", style=savings_style)


def log_promotion(provider: str, token_count: int, ttl: str) -> None:
    """Log a cache promotion event."""
    text = Text()
    text.append("[cache-guard] ", style="bold cyan")
    text.append("PROMOTED", style="bold magenta")
    text.append(f" — {token_count:,} tokens → {provider} explicit cache (TTL: {ttl})")
    console.print(text)


def log_dry_run(would_hit: bool, savings: float, warnings: list[CacheBreakWarning]) -> None:
    """Log dry-run results."""
    text = Text()
    text.append("[cache-guard] ", style="bold cyan")
    text.append("DRY RUN", style="bold blue")
    if would_hit:
        text.append(" — would HIT cache", style="green")
        text.append(f" | Estimated savings: ${savings:.4f}", style="green")
    else:
        text.append(" — would MISS cache", style="yellow")
    console.print(text)

    for w in warnings:
        console.print(f"  Warning: {w.reason}", style="yellow")
