"""Tests for fingerprint engine: normalization, rolling hash, segment comparison."""

import pytest
from cache_guard.cache.fingerprint import (
    compute_fingerprint,
    compute_prefix_match_depth,
    find_divergence,
    hash_segment,
    normalize_json,
    normalize_whitespace,
)


class TestNormalization:
    def test_normalize_json_sorts_keys(self):
        result = normalize_json({"b": 1, "a": 2})
        assert result == '{"a":2,"b":1}'

    def test_normalize_json_nested(self):
        result = normalize_json({"z": {"b": 1, "a": 2}, "a": 3})
        assert result == '{"a":3,"z":{"a":2,"b":1}}'

    def test_normalize_json_list(self):
        result = normalize_json([3, 1, 2])
        assert result == "[3,1,2]"

    def test_normalize_json_string(self):
        result = normalize_json("hello")
        assert result == '"hello"'

    def test_normalize_json_none(self):
        assert normalize_json(None) == "null"

    def test_normalize_json_bool(self):
        assert normalize_json(True) == "true"
        assert normalize_json(False) == "false"

    def test_normalize_whitespace(self):
        assert normalize_whitespace("  hello   world  ") == "hello world"
        assert normalize_whitespace("a\n\tb") == "a b"


class TestHashSegment:
    def test_same_content_same_hash(self):
        assert hash_segment("hello") == hash_segment("hello")

    def test_different_content_different_hash(self):
        assert hash_segment("hello") != hash_segment("world")

    def test_dict_key_order_irrelevant(self):
        """Non-deterministic dict ordering should produce the same hash."""
        assert hash_segment({"b": 1, "a": 2}) == hash_segment({"a": 2, "b": 1})

    def test_whitespace_normalization(self):
        assert hash_segment("hello  world") == hash_segment("hello world")

    def test_list_content(self):
        h1 = hash_segment([{"name": "tool_a"}, {"name": "tool_b"}])
        h2 = hash_segment([{"name": "tool_a"}, {"name": "tool_b"}])
        assert h1 == h2

    def test_list_order_matters(self):
        h1 = hash_segment([{"name": "tool_a"}, {"name": "tool_b"}])
        h2 = hash_segment([{"name": "tool_b"}, {"name": "tool_a"}])
        assert h1 != h2


class TestComputeFingerprint:
    def test_basic_fingerprint(self):
        fp = compute_fingerprint(
            system="You are helpful",
            tools=[{"name": "search", "description": "Search"}],
            messages=[{"role": "user", "content": "hello"}],
        )
        assert fp.combined
        assert len(fp.segment_hashes) == 3
        assert fp.segment_labels == ["system", "tools", "message[0]"]

    def test_identical_inputs_same_fingerprint(self):
        kwargs = dict(
            system="Be helpful",
            tools=[{"name": "calc"}],
            messages=[{"role": "user", "content": "hi"}],
        )
        fp1 = compute_fingerprint(**kwargs)
        fp2 = compute_fingerprint(**kwargs)
        assert fp1.combined == fp2.combined
        assert fp1.segment_hashes == fp2.segment_hashes

    def test_different_system_different_fingerprint(self):
        fp1 = compute_fingerprint(system="A", messages=[])
        fp2 = compute_fingerprint(system="B", messages=[])
        assert fp1.combined != fp2.combined

    def test_no_system(self):
        fp = compute_fingerprint(messages=[{"role": "user", "content": "hi"}])
        assert len(fp.segment_hashes) == 1
        assert fp.segment_labels == ["message[0]"]

    def test_no_tools(self):
        fp = compute_fingerprint(system="test", messages=[])
        assert len(fp.segment_hashes) == 1
        assert fp.segment_labels == ["system"]

    def test_token_estimate_stored(self):
        fp = compute_fingerprint(system="test", token_estimate=5000)
        assert fp.token_estimate == 5000


class TestFindDivergence:
    def test_identical_no_divergence(self):
        fp = compute_fingerprint(system="A", messages=[{"role": "user", "content": "hi"}])
        assert find_divergence(fp, fp) is None

    def test_system_changed(self):
        fp1 = compute_fingerprint(system="A", messages=[{"role": "user", "content": "hi"}])
        fp2 = compute_fingerprint(system="B", messages=[{"role": "user", "content": "hi"}])
        div = find_divergence(fp1, fp2)
        assert div is not None
        assert div.segment_index == 0
        assert div.segment_label == "system"

    def test_message_changed(self):
        fp1 = compute_fingerprint(
            system="same",
            messages=[{"role": "user", "content": "hello"}],
        )
        fp2 = compute_fingerprint(
            system="same",
            messages=[{"role": "user", "content": "goodbye"}],
        )
        div = find_divergence(fp1, fp2)
        assert div is not None
        assert div.segment_index == 1
        assert div.segment_label == "message[0]"

    def test_new_message_appended(self):
        fp1 = compute_fingerprint(
            system="same",
            messages=[{"role": "user", "content": "hi"}],
        )
        fp2 = compute_fingerprint(
            system="same",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        )
        # First two segments match, divergence at the new message
        div = find_divergence(fp1, fp2)
        assert div is not None
        assert div.segment_index == 2
        assert div.segment_label == "message[1]"


class TestPrefixMatchDepth:
    def test_full_match(self):
        fp = compute_fingerprint(system="A", messages=[{"role": "user", "content": "hi"}])
        matching, total = compute_prefix_match_depth(fp, fp)
        assert matching == 2
        assert total == 2

    def test_no_match(self):
        fp1 = compute_fingerprint(system="A")
        fp2 = compute_fingerprint(system="B")
        matching, total = compute_prefix_match_depth(fp1, fp2)
        assert matching == 0
        assert total == 1

    def test_partial_match(self):
        fp1 = compute_fingerprint(
            system="same",
            messages=[{"role": "user", "content": "hello"}],
        )
        fp2 = compute_fingerprint(
            system="same",
            messages=[{"role": "user", "content": "goodbye"}],
        )
        matching, total = compute_prefix_match_depth(fp1, fp2)
        assert matching == 1
        assert total == 2
