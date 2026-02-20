"""Tests for session tracking."""

import pytest
from cache_guard.core.session import SessionTracker
from cache_guard.types import Provider


class TestSessionTracker:
    def setup_method(self):
        self.tracker = SessionTracker()

    def test_create_session(self):
        session = self.tracker.get_or_create(
            provider=Provider.ANTHROPIC,
            model="claude-sonnet-4",
            system="You are helpful",
        )
        assert session.provider == Provider.ANTHROPIC
        assert session.model == "claude-sonnet-4"
        assert session.request_count == 0

    def test_same_params_same_session(self):
        s1 = self.tracker.get_or_create(
            provider=Provider.ANTHROPIC,
            model="claude-sonnet-4",
            system="You are helpful",
        )
        s2 = self.tracker.get_or_create(
            provider=Provider.ANTHROPIC,
            model="claude-sonnet-4",
            system="You are helpful",
        )
        assert s1.session_id == s2.session_id

    def test_different_system_different_session(self):
        s1 = self.tracker.get_or_create(
            provider=Provider.ANTHROPIC,
            model="claude-sonnet-4",
            system="System A",
        )
        s2 = self.tracker.get_or_create(
            provider=Provider.ANTHROPIC,
            model="claude-sonnet-4",
            system="System B",
        )
        assert s1.session_id != s2.session_id

    def test_explicit_session_id(self):
        session = self.tracker.get_or_create(
            provider=Provider.OPENAI,
            model="gpt-4o",
            session_id="my-custom-id",
        )
        assert session.session_id == "my-custom-id"

    def test_record_request(self):
        session = self.tracker.get_or_create(
            provider=Provider.ANTHROPIC,
            model="claude-sonnet-4",
        )
        self.tracker.record_request(
            session,
            input_tokens=1000,
            cached_tokens=5000,
            output_tokens=500,
            cost_actual=0.01,
            cost_without_cache=0.05,
        )
        assert session.request_count == 1
        assert session.total_input_tokens == 1000
        assert session.total_cached_tokens == 5000
        assert session.total_savings == 0.04

    def test_cache_hit_rate(self):
        session = self.tracker.get_or_create(
            provider=Provider.ANTHROPIC,
            model="claude-sonnet-4",
        )
        self.tracker.record_request(
            session,
            input_tokens=2000,
            cached_tokens=8000,
        )
        assert session.cache_hit_rate == 0.8

    def test_average_request_interval(self):
        session = self.tracker.get_or_create(
            provider=Provider.ANTHROPIC,
            model="claude-sonnet-4",
        )
        # No requests yet
        assert session.average_request_interval_seconds() is None

        # One request
        self.tracker.record_request(session)
        assert session.average_request_interval_seconds() is None

        # Two requests
        self.tracker.record_request(session)
        interval = session.average_request_interval_seconds()
        assert interval is not None
        assert interval >= 0

    def test_check_model_change(self):
        session = self.tracker.get_or_create(
            provider=Provider.ANTHROPIC,
            model="claude-sonnet-4",
        )
        assert not self.tracker.check_model_change(session, "claude-sonnet-4")
        assert self.tracker.check_model_change(session, "claude-opus-4")

    def test_check_system_change(self):
        session = self.tracker.get_or_create(
            provider=Provider.ANTHROPIC,
            model="claude-sonnet-4",
            system="Original",
        )
        assert not self.tracker.check_system_change(session, "Original")
        assert self.tracker.check_system_change(session, "Modified")

    def test_check_tools_change_ignores_order(self):
        tools = [{"name": "b"}, {"name": "a"}]
        session = self.tracker.get_or_create(
            provider=Provider.ANTHROPIC,
            model="claude-sonnet-4",
            tools=tools,
        )
        self.tracker.update_hashes(session, tools=tools)

        # Same tools, different order — should NOT be a change
        reordered = [{"name": "a"}, {"name": "b"}]
        assert not self.tracker.check_tools_change(session, reordered)

        # Actually different tools — should be a change
        different = [{"name": "a"}, {"name": "c"}]
        assert self.tracker.check_tools_change(session, different)
