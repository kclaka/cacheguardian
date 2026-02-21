"""Tests for Anthropic provider: cache_control, breakpoints, tool sorting, TTL."""

import pytest
from cacheguardian.config import CacheGuardConfig
from cacheguardian.providers.anthropic import AnthropicProvider
from cacheguardian.types import Provider, SessionState


def _make_session(**kwargs) -> SessionState:
    defaults = dict(
        session_id="test",
        provider=Provider.ANTHROPIC,
        model="claude-sonnet-4",
    )
    defaults.update(kwargs)
    return SessionState(**defaults)


class TestAnthropicProvider:
    def setup_method(self):
        self.config = CacheGuardConfig(auto_fix=True)
        self.provider = AnthropicProvider(self.config)

    def test_system_cache_control_injected(self):
        """Should add cache_control to the system prompt content block."""
        kwargs = {
            "model": "claude-sonnet-4",
            "system": "You are helpful",
            "messages": [{"role": "user", "content": "hi"}],
        }
        session = _make_session()
        result = self.provider.intercept_request(kwargs, session)
        # System should be converted to content blocks with cache_control
        assert isinstance(result["system"], list)
        assert result["system"][-1]["cache_control"] == {"type": "ephemeral"}
        assert result["system"][-1]["text"] == "You are helpful"

    def test_tools_cache_control_injected(self):
        """Should add cache_control to the last tool definition."""
        kwargs = {
            "model": "claude-sonnet-4",
            "tools": [
                {"name": "alpha", "input_schema": {}},
                {"name": "beta", "input_schema": {}},
            ],
            "messages": [],
        }
        session = _make_session()
        result = self.provider.intercept_request(kwargs, session)
        # Last tool should have cache_control
        assert "cache_control" in result["tools"][-1]
        assert result["tools"][-1]["cache_control"]["type"] == "ephemeral"
        # First tool should NOT have cache_control
        assert "cache_control" not in result["tools"][0]

    def test_no_overwrite_existing_cache_control(self):
        """Should not overwrite existing cache_control on system blocks."""
        kwargs = {
            "model": "claude-sonnet-4",
            "system": [{"type": "text", "text": "Be helpful", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            "messages": [],
        }
        session = _make_session()
        result = self.provider.intercept_request(kwargs, session)
        assert result["system"][-1]["cache_control"]["ttl"] == "1h"

    def test_tools_sorted(self):
        """Tools should be sorted alphabetically by name."""
        kwargs = {
            "model": "claude-sonnet-4",
            "tools": [
                {"name": "zebra", "input_schema": {"properties": {}}},
                {"name": "alpha", "input_schema": {"properties": {}}},
            ],
            "messages": [],
        }
        session = _make_session()
        result = self.provider.intercept_request(kwargs, session)
        assert result["tools"][0]["name"] == "alpha"
        assert result["tools"][1]["name"] == "zebra"

    def test_tools_json_stabilized(self):
        """Tool schemas should have keys sorted."""
        kwargs = {
            "model": "claude-sonnet-4",
            "tools": [
                {"name": "test", "input_schema": {"z_prop": {}, "a_prop": {}}},
            ],
            "messages": [],
        }
        session = _make_session()
        result = self.provider.intercept_request(kwargs, session)
        keys = list(result["tools"][0]["input_schema"].keys())
        assert keys == sorted(keys)

    def test_auto_fix_disabled(self):
        """No modifications when auto_fix=False."""
        config = CacheGuardConfig(auto_fix=False)
        provider = AnthropicProvider(config)
        kwargs = {
            "model": "claude-sonnet-4",
            "tools": [{"name": "z"}, {"name": "a"}],
            "messages": [],
        }
        session = _make_session()
        result = provider.intercept_request(kwargs, session)
        assert "cache_control" not in result
        assert result["tools"][0]["name"] == "z"  # not sorted

    def test_extract_request_parts(self):
        kwargs = {
            "model": "claude-sonnet-4",
            "system": "Be helpful",
            "tools": [{"name": "search"}],
            "messages": [{"role": "user", "content": "hi"}],
        }
        parts = self.provider.extract_request_parts(kwargs)
        assert parts["model"] == "claude-sonnet-4"
        assert parts["system"] == "Be helpful"
        assert len(parts["tools"]) == 1
        assert len(parts["messages"]) == 1

    def test_min_cache_tokens(self):
        assert self.provider.get_min_cache_tokens("claude-sonnet-4-20250514") == 1024
        assert self.provider.get_min_cache_tokens("claude-opus-4-20250514") == 4096
        assert self.provider.get_min_cache_tokens("unknown-model") == 1024

    def test_get_provider(self):
        assert self.provider.get_provider() == Provider.ANTHROPIC


class TestIntermediateBreakpoints:
    def setup_method(self):
        self.config = CacheGuardConfig(auto_fix=True)
        self.provider = AnthropicProvider(self.config)

    def test_conversation_prefix_breakpoint(self):
        """Should add cache_control to second-to-last message (conversation prefix)."""
        messages = [
            {"role": "user", "content": f"message {i}"}
            for i in range(10)
        ]
        kwargs = {"model": "claude-sonnet-4", "messages": messages}
        session = _make_session()
        result = self.provider.intercept_request(kwargs, session)
        # Second-to-last message (index 8) should have cache_control
        second_last = result["messages"][8]
        assert isinstance(second_last["content"], list)
        assert "cache_control" in second_last["content"][-1]
        # Last message should NOT have cache_control
        last = result["messages"][9]
        if isinstance(last.get("content"), list):
            assert "cache_control" not in last["content"][-1]

    def test_breakpoints_for_long_conversations(self):
        """> 20 blocks: intermediate breakpoints should be added."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": f"message {i}"}]}
            for i in range(30)
        ]
        kwargs = {"model": "claude-sonnet-4", "messages": messages}
        session = _make_session()
        result = self.provider.intercept_request(kwargs, session)
        # Should have at least one intermediate breakpoint
        has_breakpoint = False
        for msg in result["messages"]:
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        has_breakpoint = True
                        break
        assert has_breakpoint
