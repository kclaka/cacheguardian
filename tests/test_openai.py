"""Tests for OpenAI provider: cache_key, retention, reordering, thresholds."""

import pytest
from cache_guard.config import CacheGuardConfig
from cache_guard.providers.openai import OpenAIProvider
from cache_guard.types import Provider, SessionState
from datetime import datetime, timedelta


def _make_session(interval_seconds: float = 60, num_requests: int = 5, **kwargs) -> SessionState:
    defaults = dict(
        session_id="test",
        provider=Provider.OPENAI,
        model="gpt-4o",
    )
    defaults.update(kwargs)
    session = SessionState(**defaults)
    now = datetime.now()
    session.request_timestamps = [
        now - timedelta(seconds=interval_seconds * (num_requests - i))
        for i in range(num_requests)
    ]
    session.request_count = num_requests
    return session


class TestOpenAIProvider:
    def setup_method(self):
        self.config = CacheGuardConfig(auto_fix=True)
        self.provider = OpenAIProvider(self.config)

    def test_tools_sorted(self):
        kwargs = {
            "model": "gpt-4o",
            "tools": [
                {"name": "zebra", "function": {}},
                {"name": "alpha", "function": {}},
            ],
            "messages": [{"role": "user", "content": "hi"}],
        }
        session = _make_session()
        result = self.provider.intercept_request(kwargs, session)
        assert result["tools"][0]["name"] == "alpha"

    def test_system_messages_reordered_first(self):
        kwargs = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "Be helpful"},
            ],
        }
        session = _make_session()
        result = self.provider.intercept_request(kwargs, session)
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][1]["role"] == "user"

    def test_auto_cache_key(self):
        """Should auto-derive a prompt_cache_key."""
        kwargs = {
            "model": "gpt-4o",
            "messages": [{"role": "system", "content": "Be helpful"}],
        }
        session = _make_session(system_hash="abc123")
        result = self.provider.intercept_request(kwargs, session)
        assert "prompt_cache_key" in result
        assert result["prompt_cache_key"].startswith("cg_")

    def test_no_overwrite_existing_cache_key(self):
        kwargs = {
            "model": "gpt-4o",
            "prompt_cache_key": "my-key",
            "messages": [{"role": "user", "content": "hi"}],
        }
        session = _make_session()
        result = self.provider.intercept_request(kwargs, session)
        assert result["prompt_cache_key"] == "my-key"

    def test_custom_cache_key_fn(self):
        config = CacheGuardConfig(
            auto_fix=True,
            cache_key_fn=lambda ctx: f"user_{ctx.session_id}",
        )
        provider = OpenAIProvider(config)
        kwargs = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
        }
        session = _make_session()
        result = provider.intercept_request(kwargs, session)
        assert result["prompt_cache_key"] == f"user_{session.session_id}"

    def test_24h_retention_for_slow_sessions(self):
        kwargs = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
        }
        # Session with 15-minute intervals
        session = _make_session(interval_seconds=900)
        result = self.provider.intercept_request(kwargs, session)
        assert result.get("prompt_cache_retention") == "24h"

    def test_no_retention_for_fast_sessions(self):
        kwargs = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
        }
        session = _make_session(interval_seconds=30)
        result = self.provider.intercept_request(kwargs, session)
        assert "prompt_cache_retention" not in result

    def test_auto_fix_disabled(self):
        config = CacheGuardConfig(auto_fix=False)
        provider = OpenAIProvider(config)
        kwargs = {
            "model": "gpt-4o",
            "tools": [{"name": "z"}, {"name": "a"}],
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "rule"},
            ],
        }
        session = _make_session()
        result = provider.intercept_request(kwargs, session)
        # Not sorted
        assert result["tools"][0]["name"] == "z"
        # Not reordered
        assert result["messages"][0]["role"] == "user"

    def test_extract_request_parts(self):
        kwargs = {
            "model": "gpt-4o",
            "tools": [{"name": "search"}],
            "messages": [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "hi"},
            ],
        }
        parts = self.provider.extract_request_parts(kwargs)
        assert parts["model"] == "gpt-4o"
        assert parts["system"] == "Be helpful"
        assert len(parts["tools"]) == 1

    def test_get_min_cache_tokens(self):
        assert self.provider.get_min_cache_tokens("gpt-4o") == 1024
        assert self.provider.get_min_cache_tokens("any-model") == 1024

    def test_get_provider(self):
        assert self.provider.get_provider() == Provider.OPENAI
