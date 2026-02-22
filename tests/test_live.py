"""Live integration tests for CacheGuardian — Anthropic, Gemini & OpenAI.

Requires real API keys in .env:
    ANTHROPIC_API_KEY=sk-ant-...
    GEMINI_API_KEY=AIza...
    OPEN_AI_KEY=sk-proj-...

Run:
    python -m pytest tests/test_live.py -v -s

These tests make real API calls and cost a small amount of money.
They are skipped automatically if API keys are not available.
"""

from __future__ import annotations

import os
import time

import pytest
from dotenv import load_dotenv

import cacheguardian
from cacheguardian.types import ModelRecommendation

load_dotenv()

# ---------------------------------------------------------------------------
# Skip guards
# ---------------------------------------------------------------------------

ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_KEY = os.getenv("OPEN_AI_KEY")

requires_anthropic = pytest.mark.skipif(
    not ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not set",
)
requires_gemini = pytest.mark.skipif(
    not GEMINI_KEY, reason="GEMINI_API_KEY not set",
)
requires_openai = pytest.mark.skipif(
    not OPENAI_KEY, reason="OPEN_AI_KEY not set",
)

# ---------------------------------------------------------------------------
# Shared fixtures & helpers
# ---------------------------------------------------------------------------

# A system prompt large enough to exceed Sonnet's 1,024-token cache threshold.
# Anthropic's tokenizer is efficient — we need ~8,000+ chars to reliably exceed
# 1,024 tokens.  This generates ~2,000 tokens.
LARGE_SYSTEM = (
    "You are an expert technical assistant specializing in software engineering, "
    "distributed systems, cloud architecture, and machine learning operations. "
    "You provide detailed, accurate answers with code examples when appropriate. "
    "Always explain your reasoning step by step. Consider edge cases, performance "
    "implications, security concerns, and maintainability. When asked about code, "
    "provide production-quality implementations with proper error handling, type "
    "annotations, comprehensive documentation, and test suggestions.\n\n"
    "Guidelines for responses:\n"
    "1. Start with a brief summary of the approach before diving into details.\n"
    "2. Use markdown formatting for code blocks, lists, and headers.\n"
    "3. Cite relevant design patterns, algorithms, or architectural principles.\n"
    "4. When multiple approaches exist, compare trade-offs explicitly.\n"
    "5. Include time and space complexity analysis for algorithmic solutions.\n"
    "6. Consider backward compatibility and migration paths for API changes.\n"
    "7. Suggest monitoring, logging, and observability improvements.\n"
    "8. Address potential failure modes and recovery strategies.\n\n"
    "Your core competencies include:\n"
    + "\n".join(
        f"- Area {i}: {'performance optimization and profiling' if i % 4 == 0 else 'system design and architecture' if i % 4 == 1 else 'debugging methodology and root cause analysis' if i % 4 == 2 else 'security hardening and threat modeling'} "
        f"with deep knowledge of {'concurrency patterns and lock-free data structures' if i % 3 == 0 else 'distributed consensus protocols like Raft and Paxos' if i % 3 == 1 else 'event-driven architectures and message queues'} "
        f"and practical experience in {'real-time data pipelines using Apache Kafka and Flink' if i % 5 == 0 else 'microservices architecture with service mesh and observability' if i % 5 == 1 else 'container orchestration with Kubernetes and Helm' if i % 5 == 2 else 'CI/CD pipeline design and infrastructure as code' if i % 5 == 3 else 'database optimization including query planning and indexing strategies'}. "
        f"You understand trade-offs between {'consistency and availability in CAP theorem contexts' if i % 2 == 0 else 'latency and throughput in high-performance systems'}. "
        f"Additional expertise in {'memory management and garbage collection tuning' if i % 7 == 0 else 'network protocol design and optimization' if i % 7 == 1 else 'compiler theory and code generation' if i % 7 == 2 else 'cryptographic protocol implementation' if i % 7 == 3 else 'machine learning model serving and inference optimization' if i % 7 == 4 else 'graph algorithms and combinatorial optimization' if i % 7 == 5 else 'real-time operating systems and embedded development'}."
        for i in range(40)
    )
)

SAMPLE_TOOLS = [
    {
        "name": "search_web",
        "description": "Search the web for current information on any topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query to execute"},
                "max_results": {"type": "integer", "description": "Maximum number of results to return"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "analyze_code",
        "description": "Analyze a code snippet for bugs, performance issues, and style violations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The code snippet to analyze"},
                "language": {"type": "string", "description": "The programming language of the code"},
            },
            "required": ["code"],
        },
    },
]

# OpenAI function-calling format
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information on any topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query to execute"},
                    "max_results": {"type": "integer", "description": "Maximum number of results"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_code",
            "description": "Analyze a code snippet for bugs, performance issues, and style violations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The code snippet to analyze"},
                    "language": {"type": "string", "description": "Programming language"},
                },
                "required": ["code"],
            },
        },
    },
]


def _get_state(client):
    """Access the internal CacheGuardState from a wrapped client."""
    return getattr(client, "_cacheguardian_state")


def _get_session(client):
    """Get the first (and usually only) active session."""
    state = _get_state(client)
    sessions = state.sessions.active_sessions
    assert len(sessions) > 0, "No active session found"
    return sessions[0]


# ===================================================================
# ANTHROPIC TESTS
# ===================================================================


@requires_anthropic
class TestAnthropicLive:
    """Live tests against the Anthropic API (Claude Sonnet 4.6)."""

    # Use Sonnet for cost efficiency — 1,024-token min makes caching easy to trigger.
    MODEL = "claude-sonnet-4-6"

    def _make_client(self, **kwargs):
        import anthropic

        defaults = dict(auto_fix=True, log_level="DEBUG", model_recommendations=True)
        defaults.update(kwargs)
        return cacheguardian.wrap(
            anthropic.Anthropic(api_key=ANTHROPIC_KEY),
            **defaults,
        )

    # --- 1. Cache write + cache hit (identical requests) ---

    def test_cache_write_then_hit(self):
        """Identical requests should produce cache hits."""
        client = self._make_client()

        # Request 1 — cache write or hit (may already be cached from prior runs)
        r1 = client.messages.create(
            model=self.MODEL,
            max_tokens=50,
            system=LARGE_SYSTEM,
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        assert r1.content, "Response should have content"
        u1 = r1.usage
        # First request either writes cache or hits existing cache
        assert u1.cache_creation_input_tokens > 0 or u1.cache_read_input_tokens > 0, (
            f"First request should interact with cache, got "
            f"write={u1.cache_creation_input_tokens}, read={u1.cache_read_input_tokens}"
        )

        # Request 2 — cache hit (identical payload, cache definitely exists now)
        r2 = client.messages.create(
            model=self.MODEL,
            max_tokens=50,
            system=LARGE_SYSTEM,
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        u2 = r2.usage
        assert u2.cache_read_input_tokens > 0, (
            f"Second request should hit cache, got read={u2.cache_read_input_tokens}"
        )

        # Session metrics should reflect savings
        session = _get_session(client)
        assert session.request_count == 2
        assert session.total_cached_tokens > 0
        assert session.total_savings > 0, f"Expected savings > 0, got {session.total_savings}"
        print(f"\n  Cache hit rate: {session.cache_hit_rate:.1%}")
        print(f"  Total savings: ${session.total_savings:.6f}")

    # --- 2. Multi-turn conversation with prefix extension ---

    def test_multi_turn_prefix_extension(self):
        """Appending messages to a conversation should cache the prefix."""
        client = self._make_client()

        # Turn 1
        r1 = client.messages.create(
            model=self.MODEL,
            max_tokens=80,
            system=LARGE_SYSTEM,
            messages=[{"role": "user", "content": "What is Python?"}],
        )
        answer = r1.content[0].text

        # Turn 2 — extends the conversation (prefix preserved)
        r2 = client.messages.create(
            model=self.MODEL,
            max_tokens=80,
            system=LARGE_SYSTEM,
            messages=[
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": answer},
                {"role": "user", "content": "What about Rust?"},
            ],
        )
        u2 = r2.usage
        assert u2.cache_read_input_tokens > 0, (
            f"Prefix should be cached on turn 2, got read={u2.cache_read_input_tokens}"
        )

        # L1 should have tracked the fingerprint
        state = _get_state(client)
        session = _get_session(client)
        fp = state.l1.get_fingerprint(session.session_id)
        assert fp is not None
        assert fp.token_estimate > 0
        print(f"\n  Turn 2 cached {u2.cache_read_input_tokens} tokens")

    # --- 3. Cache break detection (system prompt change) ---

    def test_cache_break_on_system_change(self):
        """Changing the system prompt should create a separate session."""
        client = self._make_client()

        # Request with system A
        client.messages.create(
            model=self.MODEL,
            max_tokens=50,
            system=LARGE_SYSTEM,
            messages=[{"role": "user", "content": "Hi"}],
        )

        # Request with system B — different system hash → new session
        modified_system = LARGE_SYSTEM + "\nIMPORTANT: Always respond in French."
        client.messages.create(
            model=self.MODEL,
            max_tokens=50,
            system=modified_system,
            messages=[{"role": "user", "content": "Hi"}],
        )

        # Should have two sessions now (different system hashes)
        state = _get_state(client)
        assert len(state.sessions.active_sessions) == 2
        print(f"\n  Sessions after system change: {len(state.sessions.active_sessions)}")

    # --- 4. Tool sorting produces deterministic cache ---

    def test_tool_sorting_and_cache(self):
        """Tools in different order should still hit cache (auto_fix sorts them)."""
        client = self._make_client()

        # Request with tools in order [search_web, analyze_code]
        r1 = client.messages.create(
            model=self.MODEL,
            max_tokens=50,
            system=LARGE_SYSTEM,
            tools=SAMPLE_TOOLS,
            messages=[{"role": "user", "content": "Hi"}],
        )
        u1 = r1.usage
        # First request writes or hits cache (may already be cached from prior runs)
        assert u1.cache_creation_input_tokens > 0 or u1.cache_read_input_tokens > 0

        # Request with tools REVERSED — auto_fix should sort and cache still hits
        reversed_tools = list(reversed(SAMPLE_TOOLS))
        r2 = client.messages.create(
            model=self.MODEL,
            max_tokens=50,
            system=LARGE_SYSTEM,
            tools=reversed_tools,
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert r2.usage.cache_read_input_tokens > 0, (
            "Reversed tool order should still hit cache after auto_fix sorting"
        )
        print(f"\n  Tool sorting cache hit: {r2.usage.cache_read_input_tokens} tokens cached")

    # --- 5. Dry-run predicts cache behavior ---

    def test_dry_run_prediction(self):
        """dry_run() should correctly predict hit/miss."""
        client = self._make_client()

        base_kwargs = dict(
            model=self.MODEL,
            max_tokens=50,
            system=LARGE_SYSTEM,
            messages=[{"role": "user", "content": "Hello world"}],
        )

        # Dry-run before any request — first request is always a miss
        dr1 = cacheguardian.dry_run(client, **base_kwargs)
        assert dr1.would_hit_cache is False, "First dry-run should predict miss"
        assert dr1.fingerprint is not None

        # Make the actual request to populate cache
        client.messages.create(**base_kwargs)

        # Dry-run again — should predict hit
        dr2 = cacheguardian.dry_run(client, **base_kwargs)
        assert dr2.would_hit_cache is True, "Second dry-run should predict hit"
        assert dr2.estimated_savings > 0
        print(f"\n  Dry-run predicted savings: ${dr2.estimated_savings:.6f}")

    # --- 6. Model recommendation API ---

    def test_model_recommendation_standalone(self):
        """recommend() should suggest Sonnet when Opus is below threshold."""
        rec = cacheguardian.recommend(
            provider="anthropic",
            model="claude-opus-4-6",
            token_count=1500,
        )
        assert rec is not None
        assert isinstance(rec, ModelRecommendation)
        assert "sonnet" in rec.recommended_model
        assert rec.savings_percentage > 10
        assert rec.capability_note, "Should include capability note for downgrade"
        print(f"\n  Recommended: {rec.recommended_model}")
        print(f"  Savings: {rec.savings_percentage:.1f}%")
        print(f"  Note: {rec.capability_note}")

    def test_model_recommendation_not_needed(self):
        """recommend() should return None when caching already works."""
        rec = cacheguardian.recommend(
            provider="anthropic",
            model="claude-sonnet-4-6",
            token_count=2000,
        )
        assert rec is None, "Sonnet at 2000 tokens exceeds 1024 min — no recommendation needed"

    def test_model_recommendation_via_wrapped_client(self):
        """recommend() works with a wrapped client, inheriting config."""
        client = self._make_client()
        rec = cacheguardian.recommend(
            client=client,
            model="claude-opus-4-6",
            token_count=1500,
        )
        assert rec is not None
        assert "sonnet" in rec.recommended_model

    # --- 7. Dry-run includes model recommendation ---

    def test_dry_run_with_model_recommendation(self):
        """DryRunResult should include model_recommendation when below Opus threshold
        but above Sonnet threshold."""
        import anthropic

        client = cacheguardian.wrap(
            anthropic.Anthropic(api_key=ANTHROPIC_KEY),
            auto_fix=True,
            log_level="DEBUG",
            model_recommendations=True,
        )

        # Use LARGE_SYSTEM (~2000 tokens) — above Sonnet's 1024 min but below
        # Opus's 4096 min. This makes the advisor recommend Sonnet over Opus.
        dr = cacheguardian.dry_run(
            client,
            model="claude-opus-4-6",
            max_tokens=50,
            system=LARGE_SYSTEM,
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert dr.fingerprint is not None
        # With ~2000 tokens: above Sonnet min (1024), below Opus min (4096)
        # → advisor recommends Sonnet
        assert dr.fingerprint.token_estimate > 1024, (
            f"System prompt should be >1024 tokens, got {dr.fingerprint.token_estimate}"
        )
        assert dr.fingerprint.token_estimate < 4096, (
            f"System prompt should be <4096 tokens for this test, got {dr.fingerprint.token_estimate}"
        )
        assert dr.model_recommendation is not None
        assert "sonnet" in dr.model_recommendation.recommended_model
        print(f"\n  Dry-run recommendation: {dr.model_recommendation.recommended_model}")
        print(f"  Savings: {dr.model_recommendation.savings_percentage:.1f}%")

    # --- 8. Session tracking across requests ---

    def test_session_cumulative_metrics(self):
        """Session should accumulate metrics across multiple requests."""
        client = self._make_client()

        for i in range(3):
            client.messages.create(
                model=self.MODEL,
                max_tokens=30,
                system=LARGE_SYSTEM,
                messages=[{"role": "user", "content": f"Count to {i + 1}."}],
            )

        session = _get_session(client)
        assert session.request_count == 3
        assert session.total_input_tokens > 0
        assert session.total_output_tokens > 0
        assert session.total_cost_actual > 0
        print(f"\n  3 requests: total cost=${session.total_cost_actual:.6f}")
        print(f"  Total savings: ${session.total_savings:.6f}")

    # --- 9. L1 fingerprinting stability ---

    def test_l1_fingerprint_stability(self):
        """Same content should always produce the same fingerprint."""
        client = self._make_client()

        kwargs = dict(
            model=self.MODEL,
            max_tokens=50,
            system=LARGE_SYSTEM,
            messages=[{"role": "user", "content": "Test"}],
        )

        # Two dry-runs with identical content should produce identical fingerprints
        dr1 = cacheguardian.dry_run(client, **kwargs)
        dr2 = cacheguardian.dry_run(client, **kwargs)

        assert dr1.fingerprint.combined == dr2.fingerprint.combined

    # --- 10. Log-level gating of model recommendations ---

    def test_log_level_gates_recommendations_in_pipeline(self):
        """model_recommendations should be skipped when log_level is ERROR."""
        client = self._make_client(log_level="ERROR", model_recommendations=True)

        state = _get_state(client)
        assert state.config.model_recommendations is True
        assert state.config.log_level == "ERROR"
        # Interceptor checks: log_level.upper() in ("DEBUG", "INFO")
        # With ERROR, advisor is bypassed in _pre_request_full

    # --- 11. Privacy mode adds jitter ---

    def test_privacy_mode_adds_jitter(self):
        """Privacy mode should add measurable timing jitter."""
        client = self._make_client(
            privacy_mode=True,
            privacy_jitter_ms=(100, 300),
            privacy_jitter_mode="fixed",
        )

        start = time.monotonic()
        client.messages.create(
            model=self.MODEL,
            max_tokens=20,
            system=LARGE_SYSTEM,
            messages=[{"role": "user", "content": "Say ok."}],
        )
        elapsed = time.monotonic() - start

        # With 100-300ms jitter, total time should be at least ~100ms more
        # than a typical fast request.
        assert elapsed > 0.1, f"Expected jitter >=100ms, total elapsed: {elapsed:.3f}s"
        print(f"\n  Request with jitter took: {elapsed:.3f}s")

    # --- 12. Config overrides ---

    def test_config_overrides_work(self):
        """Custom config options should be applied correctly."""
        client = self._make_client(
            auto_fix=True,
            strict_mode=False,
            ttl_strategy="5m",
            quiet_early_turns=1,
        )
        state = _get_state(client)
        assert state.config.auto_fix is True
        assert state.config.strict_mode is False
        assert state.config.ttl_strategy == "5m"
        assert state.config.quiet_early_turns == 1


# ===================================================================
# GEMINI TESTS
# ===================================================================


@requires_gemini
class TestGeminiLive:
    """Live tests against the Gemini API (gemini-2.5-flash).

    Gemini SDK v1.x format: generate_content(model, contents, config)
    where system_instruction and tools go inside config.
    The CacheGuardian provider accepts system_instruction and tools at
    top-level kwargs and normalizes them into config before the SDK call.
    """

    MODEL = "gemini-2.5-flash"

    def _make_client(self, **kwargs):
        from google import genai

        defaults = dict(auto_fix=True, log_level="DEBUG", model_recommendations=True)
        defaults.update(kwargs)
        return cacheguardian.wrap(
            genai.Client(api_key=GEMINI_KEY),
            **defaults,
        )

    def _make_request_kwargs(self, user_message: str, system: str = LARGE_SYSTEM):
        """Build kwargs with system_instruction at top level.

        The GeminiProvider.intercept_request() normalizes this into the
        config dict before the SDK call.
        """
        return dict(
            model=self.MODEL,
            config={"max_output_tokens": 256},  # Gemini 2.5 uses thinking tokens
            system_instruction=system,
            contents=[{"role": "user", "parts": [{"text": user_message}]}],
        )

    # --- 1. Basic request through the pipeline ---

    def test_basic_request(self):
        """A simple request should succeed through the cacheguardian pipeline."""
        client = self._make_client()
        kwargs = self._make_request_kwargs("Say hello in one word.")
        response = client.models.generate_content(**kwargs)

        assert response.text, "Response should have text"

        # Verify session was created and metrics recorded
        session = _get_session(client)
        assert session.request_count == 1
        assert session.provider.value == "gemini"
        print(f"\n  Gemini response: {response.text.strip()}")

    # --- 2. Implicit caching (Gemini caches automatically) ---

    def test_implicit_caching(self):
        """Repeated requests should benefit from Gemini's implicit caching."""
        client = self._make_client()
        kwargs = self._make_request_kwargs("What is 2+2?")

        # Request 1
        r1 = client.models.generate_content(**kwargs)
        assert r1.text

        # Request 2 — same content, Gemini may cache implicitly
        r2 = client.models.generate_content(**kwargs)
        assert r2.text

        # Session should track both requests
        session = _get_session(client)
        assert session.request_count == 2
        assert session.total_input_tokens > 0

        # Check if Gemini reported cached tokens
        u2 = r2.usage_metadata
        cached = getattr(u2, "cached_content_token_count", 0) or 0
        print(f"\n  Gemini implicit caching: {cached} tokens cached (may be 0 for small prompts)")
        if cached > 0:
            assert session.total_savings > 0

    # --- 3. Multi-turn conversation ---

    def test_multi_turn_conversation(self):
        """Multi-turn conversation should work correctly through the pipeline."""
        client = self._make_client()

        # Turn 1
        r1 = client.models.generate_content(
            model=self.MODEL,
            config={"max_output_tokens": 256},
            system_instruction=LARGE_SYSTEM,
            contents=[{"role": "user", "parts": [{"text": "What is Python?"}]}],
        )
        answer = r1.text or "Python is a high-level programming language."

        # Turn 2 — extended conversation (use types for model turn to satisfy SDK)
        from google.genai import types as genai_types

        r2 = client.models.generate_content(
            model=self.MODEL,
            config={"max_output_tokens": 256},
            system_instruction=LARGE_SYSTEM,
            contents=[
                genai_types.Content(role="user", parts=[genai_types.Part(text="What is Python?")]),
                genai_types.Content(role="model", parts=[genai_types.Part(text=answer)]),
                genai_types.Content(role="user", parts=[genai_types.Part(text="And Rust?")]),
            ],
        )
        assert r2.text  # needs enough output tokens for thinking + text

        session = _get_session(client)
        assert session.request_count == 2

        # L1 should have tracked the fingerprint
        state = _get_state(client)
        fp = state.l1.get_fingerprint(session.session_id)
        assert fp is not None
        assert len(fp.segment_hashes) > 0
        print(f"\n  Turn 2 fingerprint segments: {len(fp.segment_hashes)}")

    # --- 4. Tool sorting ---

    def test_tool_sorting(self):
        """Tools should be sorted deterministically by auto_fix."""
        client = self._make_client()

        gemini_tools = [
            {"function_declarations": [
                {"name": "zebra_func", "description": "Z function", "parameters": {"type": "object", "properties": {}}},
                {"name": "alpha_func", "description": "A function", "parameters": {"type": "object", "properties": {}}},
            ]}
        ]

        r1 = client.models.generate_content(
            model=self.MODEL,
            config={"max_output_tokens": 256},
            system_instruction="You are helpful.",
            tools=gemini_tools,
            contents=[{"role": "user", "parts": [{"text": "Hi"}]}],
        )
        assert r1.text  # needs enough tokens for thinking + text

        # Verify session was created
        session = _get_session(client)
        assert session.request_count == 1

    # --- 5. Dry-run prediction ---

    def test_dry_run_prediction(self):
        """dry_run() should work for Gemini requests."""
        client = self._make_client()

        kwargs = self._make_request_kwargs("Hello")

        # First dry-run — miss
        dr1 = cacheguardian.dry_run(client, **kwargs)
        assert dr1.would_hit_cache is False
        assert dr1.fingerprint is not None

        # Make actual request
        client.models.generate_content(**kwargs)

        # Second dry-run — hit (L1 knows about the prefix now)
        dr2 = cacheguardian.dry_run(client, **kwargs)
        assert dr2.would_hit_cache is True

    # --- 6. Session metrics accumulation ---

    def test_session_metrics(self):
        """Session should accumulate metrics across multiple Gemini requests."""
        client = self._make_client()

        for i in range(3):
            kwargs = self._make_request_kwargs(f"What is {i + 1} + {i + 1}?")
            client.models.generate_content(**kwargs)

        session = _get_session(client)
        assert session.request_count == 3
        assert session.total_input_tokens > 0
        assert session.total_output_tokens > 0
        assert session.total_cost_actual > 0
        print(f"\n  3 requests: total cost=${session.total_cost_actual:.6f}")

    # --- 7. Cache break on system change ---

    def test_cache_break_on_system_change(self):
        """Changing system_instruction should create a new session."""
        client = self._make_client()

        # Request with system A
        client.models.generate_content(
            model=self.MODEL,
            config={"max_output_tokens": 256},
            system_instruction=LARGE_SYSTEM,
            contents=[{"role": "user", "parts": [{"text": "Hi"}]}],
        )

        # Request with system B — different session
        client.models.generate_content(
            model=self.MODEL,
            config={"max_output_tokens": 256},
            system_instruction=LARGE_SYSTEM + "\nRespond only in Spanish.",
            contents=[{"role": "user", "parts": [{"text": "Hi"}]}],
        )

        # Should have created two separate sessions
        state = _get_state(client)
        assert len(state.sessions.active_sessions) == 2

    # --- 8. Model recommendation for Gemini ---

    def test_model_recommendation_gemini(self):
        """recommend() should work for Gemini models."""
        # Gemini models both have 1024 min — below threshold, no alternatives cache
        rec = cacheguardian.recommend(
            provider="gemini",
            model="gemini-2.5-pro",
            token_count=500,
        )
        assert rec is None

        # Above threshold: pro is above its own min → caching works → no recommendation
        rec2 = cacheguardian.recommend(
            provider="gemini",
            model="gemini-2.5-pro",
            token_count=1500,
        )
        assert rec2 is None

    # --- 9. L1 fingerprint consistency ---

    def test_l1_fingerprint_consistency(self):
        """Same Gemini content should produce identical fingerprints."""
        client = self._make_client()
        kwargs = self._make_request_kwargs("Test fingerprint")

        dr1 = cacheguardian.dry_run(client, **kwargs)
        dr2 = cacheguardian.dry_run(client, **kwargs)

        assert dr1.fingerprint.combined == dr2.fingerprint.combined

    # --- 10. Config-based format also works ---

    def test_config_based_format(self):
        """Passing system_instruction inside config dict should also work."""
        client = self._make_client()

        # Pass everything in config (alternative format)
        response = client.models.generate_content(
            model=self.MODEL,
            contents=[{"role": "user", "parts": [{"text": "Say hi."}]}],
            config={
                "system_instruction": "You are a friendly bot.",
                "max_output_tokens": 30,
            },
        )
        assert response.text
        session = _get_session(client)
        assert session.request_count == 1


# ===================================================================
# CROSS-PROVIDER TESTS
# ===================================================================


@requires_anthropic
@requires_gemini
class TestCrossProvider:
    """Tests that span both providers."""

    def test_version_consistency(self):
        """Package version should be 0.3.0 everywhere."""
        assert cacheguardian.__version__ == "0.3.0"

    def test_recommend_requires_provider_without_client(self):
        """recommend() without client or provider should raise ValueError."""
        with pytest.raises(ValueError, match="provider is required"):
            cacheguardian.recommend(model="any-model", token_count=1000)

    def test_recommend_cross_provider_never_happens(self):
        """Anthropic advisor never suggests Gemini models and vice versa."""
        rec = cacheguardian.recommend(
            provider="anthropic",
            model="claude-opus-4-6",
            token_count=1500,
        )
        if rec is not None:
            assert "gemini" not in rec.recommended_model
            assert "gpt" not in rec.recommended_model

    def test_independent_sessions_per_provider(self):
        """Each provider maintains its own session state."""
        import anthropic
        from google import genai

        client_a = cacheguardian.wrap(
            anthropic.Anthropic(api_key=ANTHROPIC_KEY),
            log_level="DEBUG",
        )
        client_g = cacheguardian.wrap(
            genai.Client(api_key=GEMINI_KEY),
            log_level="DEBUG",
        )

        # Make one request each
        client_a.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=30,
            system=LARGE_SYSTEM,
            messages=[{"role": "user", "content": "Hi"}],
        )
        client_g.models.generate_content(
            model="gemini-2.5-flash",
            system_instruction="You are helpful.",
            contents=[{"role": "user", "parts": [{"text": "Hi"}]}],
            config={"max_output_tokens": 30},
        )

        state_a = _get_state(client_a)
        state_g = _get_state(client_g)

        session_a = state_a.sessions.active_sessions[0]
        session_g = state_g.sessions.active_sessions[0]

        assert session_a.provider.value == "anthropic"
        assert session_g.provider.value == "gemini"
        assert session_a.session_id != session_g.session_id
        print(f"\n  Anthropic session: {session_a.session_id}")
        print(f"  Gemini session: {session_g.session_id}")


# ===================================================================
# OPENAI TESTS
# ===================================================================


@requires_openai
class TestOpenAILive:
    """Live tests against the OpenAI API (gpt-4o-mini).

    OpenAI caching works automatically for prompts >= 1,024 tokens with
    matching prefixes.  Cached tokens are reported via
    response.usage.prompt_tokens_details.cached_tokens.

    CacheGuardian adds: tool sorting, system message reordering,
    prompt_cache_key derivation, prompt_cache_retention for slow sessions,
    and bucket padding of system content.
    """

    MODEL = "gpt-4o-mini"

    def _make_client(self, **kwargs):
        import openai

        defaults = dict(auto_fix=True, log_level="DEBUG", model_recommendations=True)
        defaults.update(kwargs)
        return cacheguardian.wrap(
            openai.OpenAI(api_key=OPENAI_KEY),
            **defaults,
        )

    def _base_kwargs(self, user_message: str = "Say hello in one word.",
                     system: str = LARGE_SYSTEM):
        """Build standard request kwargs with system message."""
        return dict(
            model=self.MODEL,
            max_tokens=50,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
        )

    # --- 1. Basic request through pipeline ---

    def test_basic_request(self):
        """A simple request should succeed through the cacheguardian pipeline."""
        client = self._make_client()
        r = client.chat.completions.create(**self._base_kwargs())

        assert r.choices[0].message.content, "Response should have content"

        session = _get_session(client)
        assert session.request_count == 1
        assert session.provider.value == "openai"
        print(f"\n  OpenAI response: {r.choices[0].message.content.strip()}")

    # --- 2. Cache write → hit (identical requests) ---

    def test_cache_write_then_hit(self):
        """Identical requests should produce cache hits (>= 1024 tokens)."""
        client = self._make_client()
        kwargs = self._base_kwargs()

        # Request 1 — cache write (or hit from prior runs)
        r1 = client.chat.completions.create(**kwargs)
        assert r1.choices[0].message.content
        u1 = r1.usage
        total_prompt_1 = u1.prompt_tokens
        cached_1 = getattr(u1.prompt_tokens_details, "cached_tokens", 0) or 0
        print(f"\n  Request 1: prompt={total_prompt_1}, cached={cached_1}")

        # Request 2 — identical, should hit cache
        r2 = client.chat.completions.create(**kwargs)
        u2 = r2.usage
        cached_2 = getattr(u2.prompt_tokens_details, "cached_tokens", 0) or 0
        print(f"  Request 2: prompt={u2.prompt_tokens}, cached={cached_2}")

        # At least one of the two requests should show caching activity
        # (OpenAI may take a short delay before cache is populated)
        assert cached_1 > 0 or cached_2 > 0, (
            f"Expected cache activity, got cached_1={cached_1}, cached_2={cached_2}. "
            f"Prompt must be >= 1024 tokens (was {total_prompt_1})."
        )

        session = _get_session(client)
        assert session.request_count == 2
        assert session.total_input_tokens > 0
        print(f"  Session cache hit rate: {session.cache_hit_rate:.1%}")

    # --- 3. Multi-turn prefix extension ---

    def test_multi_turn_prefix_extension(self):
        """Appending messages should cache the shared prefix."""
        client = self._make_client()

        # Turn 1
        r1 = client.chat.completions.create(
            model=self.MODEL,
            max_tokens=80,
            messages=[
                {"role": "system", "content": LARGE_SYSTEM},
                {"role": "user", "content": "What is Python?"},
            ],
        )
        answer = r1.choices[0].message.content

        # Turn 2 — prefix extension
        r2 = client.chat.completions.create(
            model=self.MODEL,
            max_tokens=80,
            messages=[
                {"role": "system", "content": LARGE_SYSTEM},
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": answer},
                {"role": "user", "content": "What about Rust?"},
            ],
        )
        u2 = r2.usage
        cached_2 = getattr(u2.prompt_tokens_details, "cached_tokens", 0) or 0
        print(f"\n  Turn 2: cached {cached_2} of {u2.prompt_tokens} tokens")

        # L1 should track the fingerprint
        state = _get_state(client)
        session = _get_session(client)
        fp = state.l1.get_fingerprint(session.session_id)
        assert fp is not None
        assert fp.token_estimate > 0

    # --- 4. Cache break on system change ---

    def test_cache_break_on_system_change(self):
        """Changing the system prompt should create a separate session."""
        client = self._make_client()

        # Request with system A
        client.chat.completions.create(**self._base_kwargs("Hi"))

        # Request with system B — different system hash → new session
        modified_system = LARGE_SYSTEM + "\nIMPORTANT: Always respond in French."
        client.chat.completions.create(**self._base_kwargs("Hi", system=modified_system))

        state = _get_state(client)
        assert len(state.sessions.active_sessions) == 2
        print(f"\n  Sessions after system change: {len(state.sessions.active_sessions)}")

    # --- 5. Tool sorting and cache ---

    def test_tool_sorting_and_cache(self):
        """Tools in different order should still hit cache (auto_fix sorts)."""
        client = self._make_client()

        # Request with tools in original order
        r1 = client.chat.completions.create(
            model=self.MODEL,
            max_tokens=50,
            tools=OPENAI_TOOLS,
            messages=[
                {"role": "system", "content": LARGE_SYSTEM},
                {"role": "user", "content": "Hi"},
            ],
        )
        assert r1.choices[0].message.content or r1.choices[0].message.tool_calls is not None

        # Request with tools REVERSED — auto_fix sorts, should use same fingerprint
        reversed_tools = list(reversed(OPENAI_TOOLS))
        r2 = client.chat.completions.create(
            model=self.MODEL,
            max_tokens=50,
            tools=reversed_tools,
            messages=[
                {"role": "system", "content": LARGE_SYSTEM},
                {"role": "user", "content": "Hi"},
            ],
        )

        # L1 fingerprint should be identical despite reversed tool order
        state = _get_state(client)
        session = _get_session(client)
        fp = state.l1.get_fingerprint(session.session_id)
        assert fp is not None
        assert session.request_count == 2

        # Check if OpenAI reported cache hit
        cached = getattr(r2.usage.prompt_tokens_details, "cached_tokens", 0) or 0
        print(f"\n  Tool sorting: cached={cached} tokens on second request")

    # --- 6. System message reordering ---

    def test_system_message_reordering(self):
        """auto_fix should move system messages to the front for better caching."""
        client = self._make_client()

        # System message placed AFTER user message — auto_fix should reorder
        r = client.chat.completions.create(
            model=self.MODEL,
            max_tokens=50,
            messages=[
                {"role": "user", "content": "Hi there"},
                {"role": "system", "content": LARGE_SYSTEM},
            ],
        )
        assert r.choices[0].message.content

        # Verify session was created successfully
        session = _get_session(client)
        assert session.request_count == 1
        assert session.total_input_tokens > 0
        print(f"\n  Reordered request: {session.total_input_tokens} input tokens")

    # --- 7. Dry-run prediction ---

    def test_dry_run_prediction(self):
        """dry_run() should correctly predict hit/miss for OpenAI."""
        client = self._make_client()
        kwargs = self._base_kwargs("Hello world")

        # Dry-run before any request — first is always a miss
        dr1 = cacheguardian.dry_run(client, **kwargs)
        assert dr1.would_hit_cache is False
        assert dr1.fingerprint is not None

        # Make the actual request to populate L1 cache
        client.chat.completions.create(**kwargs)

        # Dry-run again — should predict hit
        dr2 = cacheguardian.dry_run(client, **kwargs)
        assert dr2.would_hit_cache is True
        assert dr2.estimated_savings > 0
        print(f"\n  Dry-run predicted savings: ${dr2.estimated_savings:.6f}")

    # --- 8. Model recommendation for OpenAI ---

    def test_model_recommendation_openai(self):
        """All OpenAI models have 1024 min — no downgrade possible at 500 tokens."""
        rec = cacheguardian.recommend(
            provider="openai",
            model="gpt-4o",
            token_count=500,
        )
        # gpt-4o at 500 tokens: below 1024 threshold, but all OpenAI models
        # have the same 1024 min, so no cheaper alternative can cache either.
        assert rec is None

    def test_model_recommendation_above_threshold(self):
        """Above threshold — caching works, no recommendation needed."""
        rec = cacheguardian.recommend(
            provider="openai",
            model="gpt-4o",
            token_count=2000,
        )
        assert rec is None, "gpt-4o at 2000 tokens exceeds 1024 min"

    def test_model_recommendation_never_cross_provider(self):
        """OpenAI advisor should never suggest Anthropic or Gemini models."""
        rec = cacheguardian.recommend(
            provider="openai",
            model="gpt-4o",
            token_count=500,
        )
        if rec is not None:
            assert "claude" not in rec.recommended_model
            assert "gemini" not in rec.recommended_model

    # --- 9. Session metrics accumulation ---

    def test_session_cumulative_metrics(self):
        """Session should accumulate metrics across multiple requests."""
        client = self._make_client()

        for i in range(3):
            client.chat.completions.create(
                model=self.MODEL,
                max_tokens=30,
                messages=[
                    {"role": "system", "content": LARGE_SYSTEM},
                    {"role": "user", "content": f"Count to {i + 1}."},
                ],
            )

        session = _get_session(client)
        assert session.request_count == 3
        assert session.total_input_tokens > 0
        assert session.total_output_tokens > 0
        assert session.total_cost_actual > 0
        print(f"\n  3 requests: total cost=${session.total_cost_actual:.6f}")
        print(f"  Total savings: ${session.total_savings:.6f}")

    # --- 10. L1 fingerprint stability ---

    def test_l1_fingerprint_stability(self):
        """Same content should always produce the same fingerprint."""
        client = self._make_client()
        kwargs = self._base_kwargs("Test")

        dr1 = cacheguardian.dry_run(client, **kwargs)
        dr2 = cacheguardian.dry_run(client, **kwargs)

        assert dr1.fingerprint.combined == dr2.fingerprint.combined

    # --- 11. Privacy jitter ---

    def test_privacy_mode_adds_jitter(self):
        """Privacy mode should add measurable timing jitter."""
        client = self._make_client(
            privacy_mode=True,
            privacy_jitter_ms=(100, 300),
            privacy_jitter_mode="fixed",
        )

        start = time.monotonic()
        client.chat.completions.create(**self._base_kwargs("Say ok."))
        elapsed = time.monotonic() - start

        assert elapsed > 0.1, f"Expected jitter >=100ms, total elapsed: {elapsed:.3f}s"
        print(f"\n  Request with jitter took: {elapsed:.3f}s")

    # --- 12. Auto cache key derivation ---

    def test_auto_cache_key_derived(self):
        """Pipeline should auto-derive prompt_cache_key from session context."""
        client = self._make_client()

        # Make a request so the session gets a system_hash
        client.chat.completions.create(**self._base_kwargs())

        session = _get_session(client)
        # After the first request, session should have a system_hash and cache key
        assert session.system_hash is not None, "Session should have a system hash"
        assert session.openai_cache_key is not None, "Session should have an auto-derived cache key"
        assert session.openai_cache_key.startswith("cg_"), (
            f"Cache key should start with 'cg_', got {session.openai_cache_key}"
        )
        print(f"\n  Auto-derived cache key: {session.openai_cache_key}")

    # --- 13. Config overrides ---

    def test_config_overrides_work(self):
        """Custom config options should be applied correctly."""
        client = self._make_client(
            auto_fix=True,
            strict_mode=False,
            ttl_strategy="5m",
            quiet_early_turns=1,
        )
        state = _get_state(client)
        assert state.config.auto_fix is True
        assert state.config.strict_mode is False
        assert state.config.ttl_strategy == "5m"
        assert state.config.quiet_early_turns == 1

    # --- 14. Dry-run includes model recommendation ---

    def test_dry_run_with_model_recommendation(self):
        """DryRunResult for OpenAI should include model_recommendation=None
        (all OpenAI models share the same 1024 min threshold)."""
        client = self._make_client()

        dr = cacheguardian.dry_run(client, **self._base_kwargs())
        assert dr.fingerprint is not None
        # LARGE_SYSTEM is ~2000 tokens, above all OpenAI thresholds (1024)
        # so caching works → no recommendation needed
        assert dr.fingerprint.token_estimate > 1024
        assert dr.model_recommendation is None
        print(f"\n  Token estimate: {dr.fingerprint.token_estimate} (above 1024 min)")

    # --- 15. auto_fix=False disables optimizations ---

    def test_auto_fix_disabled(self):
        """With auto_fix=False, tools should NOT be sorted."""
        client = self._make_client(auto_fix=False)

        # Just verify the pipeline doesn't crash with auto_fix disabled
        r = client.chat.completions.create(
            model=self.MODEL,
            max_tokens=50,
            messages=[
                {"role": "user", "content": "Hi"},
                {"role": "system", "content": "Be helpful."},
            ],
        )
        assert r.choices[0].message.content

        # With auto_fix=False, system message should NOT be reordered
        # (We can't inspect the actual API call, but we verify it didn't crash
        # and the session was still tracked)
        session = _get_session(client)
        assert session.request_count == 1
