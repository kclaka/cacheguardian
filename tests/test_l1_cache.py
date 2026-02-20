"""Tests for L1 cache: fingerprint cache, divergence detection, warnings."""

import pytest
from cache_guard.cache.l1 import L1Cache


class TestL1Cache:
    def setup_method(self):
        self.l1 = L1Cache()

    def test_first_request_is_miss(self):
        result = self.l1.check(
            session_id="s1",
            system="You are helpful",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert not result.hit
        assert result.is_first_request
        assert result.divergence is None

    def test_identical_request_is_hit(self):
        kwargs = dict(
            session_id="s1",
            system="You are helpful",
            messages=[{"role": "user", "content": "hi"}],
        )
        # First request
        first = self.l1.check(**kwargs)
        self.l1.update("s1", first.fingerprint)

        # Identical second request
        second = self.l1.check(**kwargs)
        assert second.hit
        assert not second.is_first_request
        assert second.divergence is None
        assert "100%" in second.prefix_match_depth

    def test_system_change_detected(self):
        # First request
        r1 = self.l1.check(
            session_id="s1",
            system="You are helpful",
            messages=[{"role": "user", "content": "hi"}],
        )
        self.l1.update("s1", r1.fingerprint)

        # System prompt changed
        r2 = self.l1.check(
            session_id="s1",
            system="You are VERY helpful",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert not r2.hit
        assert r2.divergence is not None
        assert r2.divergence.segment_label == "system"

    def test_message_appended_detected(self):
        # First request
        r1 = self.l1.check(
            session_id="s1",
            system="Be helpful",
            messages=[{"role": "user", "content": "hi"}],
        )
        self.l1.update("s1", r1.fingerprint)

        # New message appended
        r2 = self.l1.check(
            session_id="s1",
            system="Be helpful",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        )
        assert not r2.hit
        assert r2.divergence is not None
        assert "message[1]" in r2.divergence.segment_label

    def test_tool_change_detected(self):
        tools_v1 = [{"name": "search", "description": "v1"}]
        tools_v2 = [{"name": "search", "description": "v2"}]

        r1 = self.l1.check(session_id="s1", tools=tools_v1, messages=[])
        self.l1.update("s1", r1.fingerprint)

        r2 = self.l1.check(session_id="s1", tools=tools_v2, messages=[])
        assert not r2.hit
        assert r2.divergence is not None
        assert r2.divergence.segment_label == "tools"

    def test_separate_sessions_independent(self):
        r1 = self.l1.check(session_id="s1", system="A", messages=[])
        self.l1.update("s1", r1.fingerprint)

        r2 = self.l1.check(session_id="s2", system="B", messages=[])
        assert r2.is_first_request

    def test_invalidate_session(self):
        r1 = self.l1.check(session_id="s1", system="A", messages=[])
        self.l1.update("s1", r1.fingerprint)

        self.l1.invalidate("s1")

        r2 = self.l1.check(session_id="s1", system="A", messages=[])
        assert r2.is_first_request

    def test_clear_all(self):
        r1 = self.l1.check(session_id="s1", system="A", messages=[])
        self.l1.update("s1", r1.fingerprint)
        assert self.l1.session_count == 1

        self.l1.clear()
        assert self.l1.session_count == 0


class TestL1CacheWarnings:
    def setup_method(self):
        self.l1 = L1Cache()

    def test_first_request_no_warning(self):
        result = self.l1.check(session_id="s1", system="A", messages=[])
        assert result.to_warning() is None

    def test_hit_no_warning(self):
        r1 = self.l1.check(session_id="s1", system="A", messages=[])
        self.l1.update("s1", r1.fingerprint)
        r2 = self.l1.check(session_id="s1", system="A", messages=[])
        assert r2.to_warning() is None

    def test_system_change_warning(self):
        r1 = self.l1.check(session_id="s1", system="A", messages=[])
        self.l1.update("s1", r1.fingerprint)

        r2 = self.l1.check(session_id="s1", system="B", messages=[])
        warning = r2.to_warning()
        assert warning is not None
        assert "system" in warning.reason
        assert "system message" in warning.suggestion.lower() or "system-reminder" in warning.suggestion.lower()

    def test_tools_change_warning(self):
        r1 = self.l1.check(session_id="s1", tools=[{"name": "a"}], messages=[])
        self.l1.update("s1", r1.fingerprint)

        r2 = self.l1.check(session_id="s1", tools=[{"name": "b"}], messages=[])
        warning = r2.to_warning()
        assert warning is not None
        assert "tools" in warning.reason
        assert "deferred" in warning.suggestion.lower() or "stub" in warning.suggestion.lower()
