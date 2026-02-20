"""Session tracking for cache-guard."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Optional

from cache_guard.cache.fingerprint import hash_segment
from cache_guard.types import Provider, SessionState


class SessionTracker:
    """Manages session state across API calls.

    Sessions are identified by a combination of provider + model + system prompt + tools.
    Each session tracks prefix fingerprints, cumulative metrics, and request timing.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def get_or_create(
        self,
        provider: Provider,
        model: str,
        system: Any = None,
        tools: list[dict[str, Any]] | None = None,
        session_id: str | None = None,
    ) -> SessionState:
        """Get an existing session or create a new one.

        If no explicit session_id is provided, one is derived from the
        provider + model + system prompt hash.
        """
        if session_id is None:
            session_id = self._derive_session_id(provider, model, system)

        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(
                session_id=session_id,
                provider=provider,
                model=model,
                system_hash=hash_segment(system) if system else None,
                tools_hash=hash_segment(tools) if tools else None,
            )

        return self._sessions[session_id]

    def record_request(
        self,
        session: SessionState,
        input_tokens: int = 0,
        cached_tokens: int = 0,
        cache_write_tokens: int = 0,
        output_tokens: int = 0,
        cost_actual: float = 0.0,
        cost_without_cache: float = 0.0,
    ) -> None:
        """Record metrics from a completed API call."""
        now = datetime.now()
        session.request_count += 1
        session.last_request_at = now
        session.request_timestamps.append(now)

        session.total_input_tokens += input_tokens
        session.total_cached_tokens += cached_tokens
        session.total_cache_write_tokens += cache_write_tokens
        session.total_output_tokens += output_tokens
        session.total_cost_actual += cost_actual
        session.total_cost_without_cache += cost_without_cache
        session.total_savings += cost_without_cache - cost_actual

    def check_model_change(self, session: SessionState, model: str) -> bool:
        """Check if the model changed mid-session."""
        if session.model != model:
            return True
        return False

    def check_system_change(self, session: SessionState, system: Any) -> bool:
        """Check if the system prompt changed mid-session."""
        if system is None:
            return False
        current_hash = hash_segment(system)
        if session.system_hash is not None and session.system_hash != current_hash:
            return True
        return False

    def check_tools_change(self, session: SessionState, tools: list[dict[str, Any]] | None) -> bool:
        """Check if tools changed mid-session (ignoring order, checking schema)."""
        if tools is None:
            return False
        # Sort tools before hashing so order changes don't trigger false positives
        sorted_tools = sorted(tools, key=lambda t: t.get("name", ""))
        current_hash = hash_segment(sorted_tools)
        if session.tools_hash is not None and session.tools_hash != current_hash:
            return True
        return False

    def update_hashes(
        self, session: SessionState, system: Any = None, tools: list[dict[str, Any]] | None = None,
    ) -> None:
        """Update stored hashes after a successful request."""
        if system is not None:
            session.system_hash = hash_segment(system)
        if tools is not None:
            sorted_tools = sorted(tools, key=lambda t: t.get("name", ""))
            session.tools_hash = hash_segment(sorted_tools)

    def get_session(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def remove_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    @property
    def active_sessions(self) -> list[SessionState]:
        return list(self._sessions.values())

    def _derive_session_id(self, provider: Provider, model: str, system: Any) -> str:
        """Derive a session ID from provider + model + system prompt."""
        parts = [provider.value, model]
        if system:
            parts.append(hash_segment(system))
        raw = ":".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]
