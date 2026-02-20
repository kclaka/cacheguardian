"""Tests for cost calculations across all 3 providers."""

import pytest
from unittest.mock import MagicMock

from cacheguardian.config import CacheGuardConfig, PricingConfig
from cacheguardian.core.metrics import MetricsCollector
from cacheguardian.types import Provider


class TestAnthropicMetrics:
    def setup_method(self):
        self.config = CacheGuardConfig(
            pricing_overrides={
                "anthropic": {
                    "claude-sonnet-4": PricingConfig(
                        base_input=3.00,
                        cache_read=0.30,
                        cache_write_5m=3.75,
                        output=15.00,
                    ),
                }
            }
        )
        self.collector = MetricsCollector(self.config)

    def test_full_cache_hit(self):
        """All tokens cached — maximum savings."""
        response = MagicMock()
        response.usage.cache_read_input_tokens = 10000
        response.usage.cache_creation_input_tokens = 0
        response.usage.input_tokens = 0
        response.usage.output_tokens = 500

        metrics = self.collector.extract_anthropic(response, "claude-sonnet-4")

        assert metrics.cached_tokens == 10000
        assert metrics.cache_write_tokens == 0
        assert metrics.uncached_tokens == 0
        assert metrics.cache_hit_rate == 1.0
        assert metrics.estimated_savings > 0

    def test_no_cache(self):
        """No caching — all tokens at base rate."""
        response = MagicMock()
        response.usage.cache_read_input_tokens = 0
        response.usage.cache_creation_input_tokens = 0
        response.usage.input_tokens = 10000
        response.usage.output_tokens = 500

        metrics = self.collector.extract_anthropic(response, "claude-sonnet-4")

        assert metrics.cache_hit_rate == 0.0
        assert metrics.estimated_savings == 0.0

    def test_cache_write(self):
        """First request writes to cache — write premium applied."""
        response = MagicMock()
        response.usage.cache_read_input_tokens = 0
        response.usage.cache_creation_input_tokens = 10000
        response.usage.input_tokens = 0
        response.usage.output_tokens = 500

        metrics = self.collector.extract_anthropic(response, "claude-sonnet-4")

        assert metrics.cache_write_tokens == 10000
        # Write costs more than base, so savings should be negative
        assert metrics.estimated_cost_actual > 0

    def test_partial_hit(self):
        """Some tokens cached, some not."""
        response = MagicMock()
        response.usage.cache_read_input_tokens = 8000
        response.usage.cache_creation_input_tokens = 0
        response.usage.input_tokens = 2000
        response.usage.output_tokens = 500

        metrics = self.collector.extract_anthropic(response, "claude-sonnet-4")

        assert metrics.cache_hit_rate == 0.8
        assert metrics.estimated_savings > 0

    def test_savings_calculation(self):
        """Verify savings = cost_without_cache - cost_actual."""
        response = MagicMock()
        response.usage.cache_read_input_tokens = 50000
        response.usage.cache_creation_input_tokens = 0
        response.usage.input_tokens = 0
        response.usage.output_tokens = 1000

        metrics = self.collector.extract_anthropic(response, "claude-sonnet-4")

        expected_without = 50000 * 3.00 / 1_000_000 + 1000 * 15.00 / 1_000_000
        expected_actual = 50000 * 0.30 / 1_000_000 + 1000 * 15.00 / 1_000_000
        expected_savings = expected_without - expected_actual

        assert abs(metrics.estimated_savings - expected_savings) < 0.0001


class TestOpenAIMetrics:
    def setup_method(self):
        self.config = CacheGuardConfig(
            pricing_overrides={
                "openai": {
                    "gpt-4o": PricingConfig(
                        base_input=2.50,
                        cache_read=1.25,
                        output=10.00,
                    ),
                }
            }
        )
        self.collector = MetricsCollector(self.config)

    def test_full_cache_hit(self):
        response = MagicMock()
        response.usage.prompt_tokens = 10000
        response.usage.completion_tokens = 500
        response.usage.prompt_tokens_details.cached_tokens = 10000

        metrics = self.collector.extract_openai(response, "gpt-4o")

        assert metrics.cached_tokens == 10000
        assert metrics.uncached_tokens == 0
        assert metrics.cache_hit_rate == 1.0
        assert metrics.estimated_savings > 0

    def test_no_cache(self):
        response = MagicMock()
        response.usage.prompt_tokens = 10000
        response.usage.completion_tokens = 500
        response.usage.prompt_tokens_details.cached_tokens = 0

        metrics = self.collector.extract_openai(response, "gpt-4o")

        assert metrics.cache_hit_rate == 0.0
        assert metrics.estimated_savings == 0.0

    def test_no_prompt_details(self):
        """Handle missing prompt_tokens_details gracefully."""
        response = MagicMock()
        response.usage.prompt_tokens = 10000
        response.usage.completion_tokens = 500
        response.usage.prompt_tokens_details = None

        metrics = self.collector.extract_openai(response, "gpt-4o")
        assert metrics.cached_tokens == 0


class TestGeminiMetrics:
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
        self.collector = MetricsCollector(self.config)

    def test_full_cache_hit(self):
        response = MagicMock()
        response.usage_metadata.prompt_token_count = 10000
        response.usage_metadata.cached_content_token_count = 10000
        response.usage_metadata.candidates_token_count = 500

        metrics = self.collector.extract_gemini(response, "gemini-2.5-flash")

        assert metrics.cached_tokens == 10000
        assert metrics.cache_hit_rate == 1.0
        assert metrics.estimated_savings > 0

    def test_no_cache(self):
        response = MagicMock()
        response.usage_metadata.prompt_token_count = 10000
        response.usage_metadata.cached_content_token_count = 0
        response.usage_metadata.candidates_token_count = 500

        metrics = self.collector.extract_gemini(response, "gemini-2.5-flash")

        assert metrics.cache_hit_rate == 0.0
