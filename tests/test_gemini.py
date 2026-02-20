"""Tests for Gemini provider: promotion, TTL, safety lock, storage tracking."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from cacheguardian.config import CacheGuardConfig, PricingConfig
from cacheguardian.persistence.cache_registry import CacheRegistry
from cacheguardian.providers.gemini import GeminiProvider
from cacheguardian.types import Provider, SessionState


def _make_session(interval_seconds: float = 60, num_requests: int = 10, **kwargs) -> SessionState:
    defaults = dict(
        session_id="test",
        provider=Provider.GEMINI,
        model="gemini-2.5-flash",
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


class TestGeminiProvider:
    def setup_method(self):
        self.config = CacheGuardConfig(auto_fix=True)
        self.registry = CacheRegistry.__new__(CacheRegistry)
        self.registry._data = {}
        self.registry._path = None
        self.registry._save = lambda: None  # disable disk writes in tests
        self.provider = GeminiProvider(self.config, registry=self.registry)

    def test_tools_sorted(self):
        kwargs = {
            "model": "gemini-2.5-flash",
            "tools": [{"name": "z"}, {"name": "a"}],
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        }
        session = _make_session()
        result = self.provider.intercept_request(kwargs, session)
        assert result["tools"][0]["name"] == "a"

    def test_reuse_existing_cache(self):
        """If session has a gemini_cache_name, it should be used."""
        kwargs = {
            "model": "gemini-2.5-flash",
            "config": {},
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        }
        session = _make_session(gemini_cache_name="cache/abc123")
        result = self.provider.intercept_request(kwargs, session)
        assert result["config"]["cached_content"] == "cache/abc123"

    def test_no_promotion_without_client(self):
        """Without a gemini_client, no promotion should happen."""
        provider = GeminiProvider(self.config, gemini_client=None, registry=self.registry)
        kwargs = {
            "model": "gemini-2.5-flash",
            "system_instruction": "Be helpful " * 1000,
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        }
        session = _make_session(num_requests=10)
        result = provider.intercept_request(kwargs, session)
        assert session.gemini_cache_name is None

    def test_extract_request_parts(self):
        kwargs = {
            "model": "gemini-2.5-flash",
            "system_instruction": "Be helpful",
            "tools": [{"name": "search"}],
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        }
        parts = self.provider.extract_request_parts(kwargs)
        assert parts["model"] == "gemini-2.5-flash"
        assert parts["system"] == "Be helpful"
        assert parts["messages"] is not None

    def test_get_min_cache_tokens(self):
        assert self.provider.get_min_cache_tokens("gemini-2.5-flash") == 1024
        assert self.provider.get_min_cache_tokens("gemini-2.0-flash") == 4096

    def test_get_provider(self):
        assert self.provider.get_provider() == Provider.GEMINI

    def test_auto_fix_disabled(self):
        config = CacheGuardConfig(auto_fix=False)
        provider = GeminiProvider(config, registry=self.registry)
        kwargs = {
            "model": "gemini-2.5-flash",
            "tools": [{"name": "z"}, {"name": "a"}],
            "contents": [],
        }
        session = _make_session()
        result = provider.intercept_request(kwargs, session)
        assert result["tools"][0]["name"] == "z"  # not sorted


class TestCacheRegistry:
    def setup_method(self):
        self.registry = CacheRegistry.__new__(CacheRegistry)
        self.registry._data = {}
        self.registry._path = None
        self.registry._save = lambda: None

    def test_add_and_get(self):
        self.registry.add(
            cache_name="cache/test1",
            model="gemini-2.5-flash",
            session_id="s1",
            ttl_seconds=3600,
        )
        assert self.registry.count == 1
        entries = self.registry.get_all()
        assert entries[0]["cache_name"] == "cache/test1"

    def test_remove(self):
        self.registry.add("cache/test1", "model", "s1", 3600)
        self.registry.remove("cache/test1")
        assert self.registry.count == 0

    def test_get_by_session(self):
        self.registry.add("cache/a", "model", "s1", 3600)
        self.registry.add("cache/b", "model", "s2", 3600)
        entries = self.registry.get_by_session("s1")
        assert len(entries) == 1
        assert entries[0]["cache_name"] == "cache/a"

    def test_stale_detection(self):
        self.registry.add("cache/old", "model", "s1", 3600)
        # Manually set last_accessed to 3 hours ago
        self.registry._data["cache/old"]["last_accessed"] = (
            datetime.now() - timedelta(hours=3)
        ).isoformat()

        stale = self.registry.get_stale_entries(max_age_hours=2.0)
        assert len(stale) == 1
        assert stale[0]["cache_name"] == "cache/old"

    def test_touch_updates_time(self):
        self.registry.add("cache/test", "model", "s1", 3600)
        # Set old time
        self.registry._data["cache/test"]["last_accessed"] = (
            datetime.now() - timedelta(hours=5)
        ).isoformat()

        self.registry.touch("cache/test")

        stale = self.registry.get_stale_entries(max_age_hours=2.0)
        assert len(stale) == 0  # no longer stale after touch
