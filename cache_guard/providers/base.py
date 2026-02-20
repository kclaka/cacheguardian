"""Abstract base class for provider-specific cache optimization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from cache_guard.config import CacheGuardConfig
from cache_guard.types import CacheBreakWarning, CacheMetrics, Provider, SessionState


class CacheProvider(ABC):
    """Base class for provider-specific cache optimization."""

    def __init__(self, config: CacheGuardConfig) -> None:
        self.config = config

    @abstractmethod
    def get_provider(self) -> Provider:
        """Return the provider enum value."""
        ...

    @abstractmethod
    def intercept_request(self, kwargs: dict[str, Any], session: SessionState) -> dict[str, Any]:
        """Transform the request before sending to optimize caching.

        Returns modified request kwargs. The original dict may be mutated.
        """
        ...

    @abstractmethod
    def extract_metrics(self, response: Any, model: str) -> CacheMetrics:
        """Parse the API response to extract cache performance metrics."""
        ...

    @abstractmethod
    def extract_request_parts(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Extract system, tools, messages from provider-specific kwargs.

        Returns a dict with keys: 'system', 'tools', 'messages', 'model'.
        """
        ...

    @abstractmethod
    def get_min_cache_tokens(self, model: str) -> int:
        """Return the minimum token count for caching to apply."""
        ...
