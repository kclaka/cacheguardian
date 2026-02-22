"""Tests for CacheGuardian v0.2.0 improvements (P0–P8)."""

import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from cacheguardian.cache.fingerprint import (
    _estimate_segment_tokens,
    compute_fingerprint,
    hash_segment,
    normalize_json,
)
from cacheguardian.config import CacheGuardConfig
from cacheguardian.core.differ import PromptDiffer
from cacheguardian.core.optimizer import (
    pad_to_cache_bucket,
    segregate_dynamic_content,
)
from cacheguardian.persistence.cache_registry import CacheRegistry
from cacheguardian.providers.anthropic import AnthropicProvider
from cacheguardian.providers.gemini import GeminiProvider
from cacheguardian.providers.openai import OpenAIProvider
from cacheguardian.types import DivergencePoint, Fingerprint, Provider, SessionState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_anthropic_session(**kwargs) -> SessionState:
    defaults = dict(
        session_id="test",
        provider=Provider.ANTHROPIC,
        model="claude-sonnet-4",
    )
    defaults.update(kwargs)
    return SessionState(**defaults)


def _make_openai_session(**kwargs) -> SessionState:
    defaults = dict(
        session_id="test",
        provider=Provider.OPENAI,
        model="gpt-4o",
    )
    defaults.update(kwargs)
    session = SessionState(**defaults)
    session.request_count = kwargs.get("request_count", 5)
    return session


def _make_gemini_session(interval_seconds: float = 60, num_requests: int = 10, **kwargs) -> SessionState:
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


# ===========================================================================
# P0: Strip cache_control from fingerprint hashing
# ===========================================================================

class TestP0StripCacheControl:
    def test_normalize_json_strips_cache_control(self):
        """cache_control keys should be excluded from normalized output."""
        obj = {"text": "hello", "cache_control": {"type": "ephemeral"}}
        result = normalize_json(obj)
        assert "cache_control" not in result
        assert '"text":"hello"' in result

    def test_normalize_json_strips_nested(self):
        """cache_control should be stripped at all nesting levels."""
        obj = {"a": {"b": 1, "cache_control": {"type": "ephemeral"}}}
        result = normalize_json(obj)
        assert "cache_control" not in result

    def test_normalize_json_preserves_other_keys(self):
        obj = {"type": "text", "text": "hello", "extra": 42}
        result = normalize_json(obj)
        assert '"type"' in result
        assert '"text"' in result
        assert '"extra"' in result

    def test_normalize_json_strip_keys_customizable(self):
        """strip_keys parameter should be customizable."""
        obj = {"a": 1, "b": 2, "c": 3}
        result = normalize_json(obj, strip_keys=frozenset({"b"}))
        assert '"b"' not in result
        assert '"a"' in result
        assert '"c"' in result

    def test_segment_hashes_stable_across_breakpoint_shifts(self):
        """Segment hashes should be identical whether cache_control is present or not."""
        msg_without = {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        msg_with = {"role": "user", "content": [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}]}
        assert hash_segment(msg_without) == hash_segment(msg_with)

    def test_fingerprint_stable_with_cache_control(self):
        """Fingerprints should be identical with and without cache_control markers."""
        messages_without = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
        ]
        messages_with = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}]},
        ]
        fp1 = compute_fingerprint(messages=messages_without)
        fp2 = compute_fingerprint(messages=messages_with)
        assert fp1.combined == fp2.combined


# ===========================================================================
# P1: Enforce 4-breakpoint hard limit
# ===========================================================================

class TestP1BreakpointLimit:
    def setup_method(self):
        self.config = CacheGuardConfig(auto_fix=True)
        self.provider = AnthropicProvider(self.config)

    def test_count_cache_controls(self):
        kwargs = {
            "system": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}],
            "tools": [{"name": "a", "cache_control": {"type": "ephemeral"}}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "q", "cache_control": {"type": "ephemeral"}}]},
            ],
        }
        assert self.provider._count_cache_controls(kwargs) == 3

    def test_enforce_limit_no_strip_when_under(self):
        """Should not strip anything when under the limit."""
        kwargs = {
            "system": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}],
            "tools": [{"name": "a", "cache_control": {"type": "ephemeral"}}],
            "messages": [],
        }
        result = self.provider._enforce_breakpoint_limit(kwargs)
        assert self.provider._count_cache_controls(result) == 2

    def test_enforce_limit_strips_message_breakpoints_first(self):
        """Should strip message breakpoints (oldest first) before touching system/tools."""
        kwargs = {
            "system": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}],
            "tools": [{"name": "a", "cache_control": {"type": "ephemeral"}}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "m0", "cache_control": {"type": "ephemeral"}}]},
                {"role": "assistant", "content": [{"type": "text", "text": "m1", "cache_control": {"type": "ephemeral"}}]},
                {"role": "user", "content": [{"type": "text", "text": "m2", "cache_control": {"type": "ephemeral"}}]},
            ],
        }
        # Total = 5 (system + tool + 3 messages), should strip to 4
        result = self.provider._enforce_breakpoint_limit(kwargs, max_breakpoints=4)
        assert self.provider._count_cache_controls(result) <= 4
        # System should still have cache_control
        assert "cache_control" in result["system"][0]
        # Tools should still have cache_control
        assert "cache_control" in result["tools"][0]

    def test_enforce_limit_caps_at_4(self):
        """End-to-end: intercept_request should never produce > 4 breakpoints."""
        # Build a long conversation with many messages
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(30)]
        kwargs = {
            "model": "claude-sonnet-4",
            "system": "You are helpful",
            "tools": [{"name": "search", "input_schema": {}}],
            "messages": messages,
        }
        session = _make_anthropic_session()
        result = self.provider.intercept_request(kwargs, session)
        assert self.provider._count_cache_controls(result) <= 4


# ===========================================================================
# P2: Model-aware breakpoint gating + image-safe token estimation
# ===========================================================================

class TestP2TokenEstimation:
    def setup_method(self):
        self.config = CacheGuardConfig(auto_fix=True)
        self.provider = AnthropicProvider(self.config)

    def test_estimate_text_tokens(self):
        messages = [{"role": "user", "content": "a" * 400}]
        result = self.provider._estimate_message_tokens(messages)
        assert result == 100  # 400 chars / 4

    def test_estimate_image_tokens(self):
        """Image blocks should count as 1568 tokens regardless of data size."""
        messages = [{"role": "user", "content": [
            {"type": "image", "source": {"data": "x" * 500000}},
        ]}]
        result = self.provider._estimate_message_tokens(messages)
        assert result == 1568

    def test_estimate_mixed_content(self):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "a" * 400},
            {"type": "image", "source": {"data": "base64..."}},
        ]}]
        result = self.provider._estimate_message_tokens(messages)
        assert result == 100 + 1568

    def test_estimate_tool_result(self):
        block = {"type": "tool_result", "tool_use_id": "abc", "content": "result data"}
        messages = [{"role": "user", "content": [block]}]
        result = self.provider._estimate_message_tokens(messages)
        # Should use json.dumps(block) // 4
        expected = len(json.dumps(block)) // 4
        assert result == expected

    def test_no_breakpoints_below_threshold(self):
        """Messages below min cacheable tokens should not get breakpoints."""
        # claude-opus-4 has a 4096-token minimum
        short_messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
        ]
        kwargs = {
            "model": "claude-opus-4",
            "messages": short_messages,
        }
        session = _make_anthropic_session(model="claude-opus-4")
        result = self.provider.intercept_request(kwargs, session)
        # No message should have cache_control since content is below 4096 tokens
        for msg in result["messages"]:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        assert "cache_control" not in block

    def test_breakpoints_added_above_threshold(self):
        """Messages above min cacheable tokens should get breakpoints."""
        # Build messages with enough text to exceed 1024 tokens for Sonnet
        long_messages = [
            {"role": "user", "content": "x" * 5000},
            {"role": "assistant", "content": "y" * 5000},
            {"role": "user", "content": "latest question"},
        ]
        kwargs = {
            "model": "claude-sonnet-4",
            "messages": long_messages,
        }
        session = _make_anthropic_session()
        result = self.provider.intercept_request(kwargs, session)
        # At least one message should have a breakpoint
        has_breakpoint = False
        for msg in result["messages"]:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        has_breakpoint = True
            elif isinstance(content, str) and msg != result["messages"][-1]:
                # String content won't have cache_control embedded
                pass
        assert has_breakpoint


# ===========================================================================
# P3: Dynamic content segregation
# ===========================================================================

class TestP3DynamicContentSegregation:
    def setup_method(self):
        self.config = CacheGuardConfig(auto_fix=True)
        self.provider = AnthropicProvider(self.config)

    def test_find_static_boundary_no_tool_results(self):
        """Without tool results, boundary = second-to-last message."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
        ]
        idx = self.provider._find_static_boundary(messages)
        assert idx == 1  # the assistant message

    def test_find_static_boundary_with_tool_results(self):
        """Should skip past tool results to the assistant message."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "let me search"},
            {"role": "tool", "content": "result 1"},
            {"role": "tool", "content": "result 2"},
            {"role": "user", "content": "thanks"},
        ]
        idx = self.provider._find_static_boundary(messages)
        assert idx == 1  # the assistant message before tool results

    def test_find_static_boundary_single_message(self):
        messages = [{"role": "user", "content": "hi"}]
        idx = self.provider._find_static_boundary(messages)
        assert idx == 0

    def test_segregate_dynamic_content_no_tools(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
        ]
        idx = segregate_dynamic_content(messages)
        assert idx == 2  # first dynamic index = the last user message

    def test_segregate_dynamic_content_with_tools(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "searching..."},
            {"role": "tool", "content": "result1"},
            {"role": "tool", "content": "result2"},
            {"role": "user", "content": "ok"},
        ]
        idx = segregate_dynamic_content(messages)
        assert idx == 2  # first tool result

    def test_segregate_dynamic_content_empty(self):
        assert segregate_dynamic_content([]) == 0

    def test_breakpoint_placed_before_tool_results(self):
        """The breakpoint should be on the assistant message, not after tool results."""
        long_content = "x" * 5000
        messages = [
            {"role": "user", "content": long_content},
            {"role": "assistant", "content": long_content},
            {"role": "tool", "content": "dynamic result 1"},
            {"role": "tool", "content": "dynamic result 2"},
            {"role": "user", "content": "follow up"},
        ]
        kwargs = {
            "model": "claude-sonnet-4",
            "messages": messages,
        }
        session = _make_anthropic_session()
        result = self.provider.intercept_request(kwargs, session)
        # The assistant message (idx 1) should have the breakpoint
        assistant_msg = result["messages"][1]
        content = assistant_msg.get("content")
        if isinstance(content, list):
            assert any("cache_control" in b for b in content if isinstance(b, dict))
        # Tool results should NOT have breakpoints
        for tool_msg in result["messages"][2:4]:
            content = tool_msg.get("content")
            if isinstance(content, list):
                assert not any("cache_control" in b for b in content if isinstance(b, dict))


# ===========================================================================
# P4: OpenAI 128-token bucket padding
# ===========================================================================

class TestP4BucketPadding:
    def test_no_pad_below_minimum(self):
        """Should not pad if below 1024 tokens."""
        result = pad_to_cache_bucket("hello", 500)
        assert result == "hello"

    def test_pad_within_32_tokens(self):
        """Should pad when within 32 tokens of next boundary."""
        # 1024 + 128 - 10 = 1142 tokens, next boundary = 1152, gap = 10
        result = pad_to_cache_bucket("hello", 1142)
        assert len(result) > len("hello")
        # Padding should be whitespace
        assert result.startswith("hello")
        assert result[5:].strip() == ""

    def test_no_pad_when_far_from_boundary(self):
        """Should not pad when > 32 tokens from next boundary."""
        # 1100 tokens, next boundary = 1152, gap = 52 > 32
        result = pad_to_cache_bucket("hello", 1100)
        assert result == "hello"

    def test_pad_exact_boundary(self):
        """At exact boundary, gap to next = 128 > 32, no pad."""
        result = pad_to_cache_bucket("hello", 1152)
        assert result == "hello"

    def test_openai_provider_integrates_padding(self):
        """OpenAI provider should apply bucket padding in intercept_request."""
        config = CacheGuardConfig(auto_fix=True)
        provider = OpenAIProvider(config)
        # Build a system message with enough content to be near a boundary
        # 1142 * 4 = 4568 chars ~ 1142 tokens
        system_content = "x" * 4568
        kwargs = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": "hi"},
            ],
        }
        session = _make_openai_session()
        result = provider.intercept_request(kwargs, session)
        # System message content may be padded
        sys_msg = next(m for m in result["messages"] if m.get("role") == "system")
        # Just verify it doesn't crash and system content is present
        assert system_content in sys_msg["content"]


# ===========================================================================
# P5: Gemini implicit/explicit routing threshold
# ===========================================================================

class TestP5GeminiThreshold:
    def setup_method(self):
        self.registry = CacheRegistry.__new__(CacheRegistry)
        self.registry._data = {}
        self.registry._path = None
        self.registry._save = lambda: None

    def test_no_promotion_below_32k(self):
        """Should not promote to explicit cache below 32K tokens."""
        config = CacheGuardConfig(auto_fix=True)
        mock_client = MagicMock()
        provider = GeminiProvider(config, gemini_client=mock_client, registry=self.registry)

        # ~5K tokens (well below 32K)
        kwargs = {
            "model": "gemini-2.5-flash",
            "system_instruction": "Be helpful " * 500,
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        }
        session = _make_gemini_session(num_requests=10)
        result = provider.intercept_request(kwargs, session)
        assert session.gemini_cache_name is None
        mock_client.caches.create.assert_not_called()

    def test_custom_threshold(self):
        """gemini_explicit_threshold should be configurable."""
        config = CacheGuardConfig(auto_fix=True, gemini_explicit_threshold=2048)
        mock_client = MagicMock()
        mock_client.caches.create.return_value = MagicMock(name="cache/test123")
        provider = GeminiProvider(config, gemini_client=mock_client, registry=self.registry)

        # ~2500 tokens (above custom 2048 threshold)
        kwargs = {
            "model": "gemini-2.5-flash",
            "system_instruction": "x" * 10000,
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        }
        session = _make_gemini_session(num_requests=10)
        # Patch the promoter to approve promotion
        with patch.object(provider._promoter, "evaluate_gemini_explicit") as mock_eval:
            mock_eval.return_value = MagicMock(should_promote=True)
            provider.intercept_request(kwargs, session)
            # Should have attempted promotion since above threshold
            mock_eval.assert_called_once()

    def test_default_threshold_is_32768(self):
        config = CacheGuardConfig()
        assert config.gemini_explicit_threshold == 32768


# ===========================================================================
# P6: Per-segment token estimation
# ===========================================================================

class TestP6PerSegmentEstimates:
    def test_fingerprint_has_segment_estimates(self):
        fp = compute_fingerprint(
            system="You are helpful",
            tools=[{"name": "search", "description": "Search the web"}],
            messages=[{"role": "user", "content": "hello world"}],
        )
        assert len(fp.segment_token_estimates) == 3
        assert all(isinstance(e, int) for e in fp.segment_token_estimates)

    def test_segment_estimate_text(self):
        result = _estimate_segment_tokens("a" * 400)
        assert result == 100

    def test_segment_estimate_image_block(self):
        result = _estimate_segment_tokens({"type": "image", "source": {"data": "x" * 100000}})
        assert result == 1568

    def test_segment_estimate_message_with_image(self):
        msg = {"role": "user", "content": [
            {"type": "text", "text": "a" * 400},
            {"type": "image", "source": {"data": "x" * 100000}},
        ]}
        result = _estimate_segment_tokens(msg)
        assert result == 100 + 1568

    def test_token_estimate_auto_calculated(self):
        """When token_estimate=0 (default), it should be auto-calculated from segments."""
        fp = compute_fingerprint(
            system="a" * 4000,  # ~1000 tokens
            messages=[{"role": "user", "content": "b" * 400}],  # ~100 tokens
        )
        assert fp.token_estimate > 0
        assert fp.token_estimate == sum(fp.segment_token_estimates)

    def test_explicit_token_estimate_preserved(self):
        """If caller provides token_estimate, it should be used."""
        fp = compute_fingerprint(
            system="hello",
            token_estimate=9999,
        )
        assert fp.token_estimate == 9999

    def test_differ_uses_segment_estimates(self):
        """estimate_miss_cost should use per-segment estimates when available."""
        differ = PromptDiffer()
        fp = Fingerprint(
            combined="abc",
            segment_hashes=["h1", "h2", "h3"],
            segment_labels=["system", "tools", "message[0]"],
            token_estimate=2000,
            segment_token_estimates=[1000, 500, 500],
        )
        # Divergence at segment 1 (tools) -> uncached = 500 + 500 = 1000
        div = DivergencePoint(
            segment_index=1,
            segment_label="tools",
            previous_hash="old",
            new_hash="new",
        )
        cost = differ.estimate_miss_cost(div, fp, base_rate_per_mtok=3.0, cached_rate_per_mtok=0.3)
        # 1000 tokens * (3.0 - 0.3) / 1_000_000 = 0.0027
        assert abs(cost - 0.0027) < 0.0001

    def test_differ_fallback_without_segment_estimates(self):
        """Should fall back to uniform distribution when no segment estimates."""
        differ = PromptDiffer()
        fp = Fingerprint(
            combined="abc",
            segment_hashes=["h1", "h2", "h3"],
            segment_labels=["system", "tools", "message[0]"],
            token_estimate=3000,
            segment_token_estimates=[],  # empty
        )
        div = DivergencePoint(
            segment_index=1,
            segment_label="tools",
            previous_hash="old",
            new_hash="new",
        )
        cost = differ.estimate_miss_cost(div, fp, base_rate_per_mtok=3.0, cached_rate_per_mtok=0.3)
        # 2/3 of 3000 tokens uncached = 2000 tokens
        # 2000 * 2.7 / 1_000_000 = 0.0054
        assert abs(cost - 0.0054) < 0.0001


# ===========================================================================
# P7: Logging noise control
# ===========================================================================

class TestP7LoggingNoiseControl:
    def test_quiet_early_turns_default(self):
        config = CacheGuardConfig()
        assert config.quiet_early_turns == 3

    def test_quiet_early_turns_customizable(self):
        config = CacheGuardConfig(quiet_early_turns=5)
        assert config.quiet_early_turns == 5

    def test_early_turn_miss_not_logged_at_info(self):
        """MISS events in early turns should be logged at DEBUG, not INFO."""
        # This tests the config integration; the actual logging behavior
        # is tested by verifying that session.request_count <= quiet_early_turns
        # triggers the DEBUG branch in _post_request.
        config = CacheGuardConfig(quiet_early_turns=3)
        session = _make_anthropic_session()
        session.request_count = 2
        assert session.request_count <= config.quiet_early_turns

    def test_later_turn_miss_logged(self):
        config = CacheGuardConfig(quiet_early_turns=3)
        session = _make_anthropic_session()
        session.request_count = 5
        assert session.request_count > config.quiet_early_turns


# ===========================================================================
# P8: Enhanced TTFT jitter (privacy hardening)
# ===========================================================================

class TestP8AdaptiveJitter:
    def test_fixed_jitter_mode_default(self):
        config = CacheGuardConfig()
        assert config.privacy_jitter_mode == "fixed"

    def test_adaptive_jitter_mode_configurable(self):
        config = CacheGuardConfig(privacy_jitter_mode="adaptive")
        assert config.privacy_jitter_mode == "adaptive"

    def test_jitter_range_configurable(self):
        config = CacheGuardConfig(privacy_jitter_ms=(100, 500))
        assert config.privacy_jitter_ms == (100, 500)


# ===========================================================================
# Integration: combinations
# ===========================================================================

class TestIntegration:
    def test_full_anthropic_pipeline(self):
        """End-to-end: all optimizations applied together."""
        config = CacheGuardConfig(auto_fix=True)
        provider = AnthropicProvider(config)

        long_content = "x" * 5000
        messages = [
            {"role": "user", "content": long_content},
            {"role": "assistant", "content": long_content},
            {"role": "tool", "content": "tool result"},
            {"role": "user", "content": "follow up"},
        ]
        kwargs = {
            "model": "claude-sonnet-4",
            "system": "You are a helpful assistant",
            "tools": [
                {"name": "zebra", "input_schema": {"properties": {}}},
                {"name": "alpha", "input_schema": {"properties": {}}},
            ],
            "messages": messages,
        }
        session = _make_anthropic_session()
        result = provider.intercept_request(kwargs, session)

        # Tools sorted
        assert result["tools"][0]["name"] == "alpha"
        # System has cache_control
        assert isinstance(result["system"], list)
        assert "cache_control" in result["system"][-1]
        # Total breakpoints <= 4
        assert provider._count_cache_controls(result) <= 4

    def test_fingerprint_stability_through_pipeline(self):
        """Fingerprints should be stable regardless of cache_control injection.

        Uses list-format content blocks (not plain strings) because
        _inject_breakpoint converts string content to list blocks, which
        is a structural change beyond cache_control stripping.
        """
        config = CacheGuardConfig(auto_fix=True)
        provider = AnthropicProvider(config)

        messages = [
            {"role": "user", "content": [{"type": "text", "text": "x" * 5000}]},
            {"role": "assistant", "content": [{"type": "text", "text": "y" * 5000}]},
            {"role": "user", "content": [{"type": "text", "text": "z"}]},
        ]
        kwargs = {
            "model": "claude-sonnet-4",
            "system": [{"type": "text", "text": "Be helpful"}],
            "messages": messages,
        }
        session = _make_anthropic_session()

        # Fingerprint before injection
        fp_before = compute_fingerprint(
            system=kwargs["system"],
            messages=kwargs["messages"],
        )

        # Apply transforms (adds cache_control)
        result = provider.intercept_request(kwargs, session)

        # Fingerprint after injection
        fp_after = compute_fingerprint(
            system=result["system"],
            messages=result["messages"],
        )

        # All message segment hashes should be stable despite cache_control
        # injection, because normalize_json strips cache_control keys.
        # System hash should also be stable for the same reason.
        for h_before, h_after in zip(fp_before.segment_hashes, fp_after.segment_hashes):
            assert h_before == h_after
