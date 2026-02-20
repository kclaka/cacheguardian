"""Tests for cache promotion (break-even formula, promotion decisions)."""

import pytest
from datetime import datetime, timedelta
from cache_guard.config import CacheGuardConfig, PricingConfig
from cache_guard.core.promoter import CachePromoter
from cache_guard.types import Provider, SessionState


def _make_session(
    provider: Provider = Provider.ANTHROPIC,
    model: str = "claude-sonnet-4",
    interval_seconds: float = 60,
    num_requests: int = 10,
) -> SessionState:
    """Create a session with simulated request history."""
    session = SessionState(
        session_id="test",
        provider=provider,
        model=model,
        request_count=num_requests,
    )
    now = datetime.now()
    session.request_timestamps = [
        now - timedelta(seconds=interval_seconds * (num_requests - i))
        for i in range(num_requests)
    ]
    return session


class TestAnthropicPromotion:
    def setup_method(self):
        self.config = CacheGuardConfig(
            pricing_overrides={
                "anthropic": {
                    "claude-sonnet-4": PricingConfig(
                        base_input=3.00,
                        cache_read=0.30,
                        cache_write_5m=3.75,
                        cache_write_1h=6.00,
                        output=15.00,
                    ),
                }
            }
        )
        self.promoter = CachePromoter(self.config)

    def test_fast_session_no_promotion(self):
        """Requests every 30 seconds — 5m TTL is fine."""
        session = _make_session(interval_seconds=30)
        decision = self.promoter.evaluate_anthropic_1h(session, token_count=50000)
        assert not decision.should_promote

    def test_slow_session_promotes(self):
        """Requests every 10 minutes — 1h TTL is better."""
        session = _make_session(interval_seconds=600)
        decision = self.promoter.evaluate_anthropic_1h(session, token_count=50000)
        assert decision.should_promote

    def test_break_even_n_positive(self):
        session = _make_session(interval_seconds=600)
        decision = self.promoter.evaluate_anthropic_1h(session, token_count=50000)
        assert decision.break_even_n > 0

    def test_no_history(self):
        session = SessionState(
            session_id="test",
            provider=Provider.ANTHROPIC,
            model="claude-sonnet-4",
        )
        decision = self.promoter.evaluate_anthropic_1h(session, token_count=50000)
        assert not decision.should_promote


class TestGeminiPromotion:
    def setup_method(self):
        self.config = CacheGuardConfig(
            pricing_overrides={
                "gemini": {
                    "gemini-2.5-flash": PricingConfig(
                        base_input=0.15,
                        cache_read=0.0375,
                        output=0.60,
                        storage_per_hour=4.50,
                    ),
                }
            }
        )
        self.promoter = CachePromoter(self.config)

    def test_frequent_requests_promotes(self):
        """Frequent requests with large content — explicit cache makes sense."""
        session = _make_session(
            provider=Provider.GEMINI,
            model="gemini-2.5-flash",
            interval_seconds=30,
            num_requests=20,
        )
        decision = self.promoter.evaluate_gemini_explicit(session, token_count=50000)
        assert decision.should_promote
        assert decision.estimated_savings_if_promoted > 0

    def test_infrequent_requests_no_promotion(self):
        """Infrequent requests — storage cost outweighs savings."""
        session = _make_session(
            provider=Provider.GEMINI,
            model="gemini-2.5-flash",
            interval_seconds=3600,
            num_requests=5,
        )
        decision = self.promoter.evaluate_gemini_explicit(session, token_count=5000)
        assert not decision.should_promote

    def test_small_content_no_promotion(self):
        """Content below 1024 tokens — not eligible."""
        session = _make_session(
            provider=Provider.GEMINI,
            model="gemini-2.5-flash",
            interval_seconds=30,
        )
        decision = self.promoter.evaluate_gemini_explicit(session, token_count=500)
        assert not decision.should_promote

    def test_storage_cost_tracked(self):
        session = _make_session(
            provider=Provider.GEMINI,
            model="gemini-2.5-flash",
            interval_seconds=60,
        )
        decision = self.promoter.evaluate_gemini_explicit(
            session, token_count=50000, ttl_hours=2.0,
        )
        assert decision.estimated_storage_cost > 0


class TestOpenAIRetention:
    def setup_method(self):
        self.config = CacheGuardConfig()
        self.promoter = CachePromoter(self.config)

    def test_fast_session_no_24h(self):
        session = _make_session(
            provider=Provider.OPENAI,
            model="gpt-4o",
            interval_seconds=30,
        )
        assert not self.promoter.evaluate_openai_retention(session)

    def test_slow_session_24h(self):
        session = _make_session(
            provider=Provider.OPENAI,
            model="gpt-4o",
            interval_seconds=900,  # 15 minutes
        )
        assert self.promoter.evaluate_openai_retention(session)

    def test_no_history(self):
        session = SessionState(
            session_id="test",
            provider=Provider.OPENAI,
            model="gpt-4o",
        )
        assert not self.promoter.evaluate_openai_retention(session)
