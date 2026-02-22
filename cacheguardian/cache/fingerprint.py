"""Normalization and rolling hash fingerprint generation for prompt segments."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from cacheguardian.types import DivergencePoint, Fingerprint

# Keys that carry caching metadata but not semantic content.
# Stripping these from hashes prevents false MISS reports when
# cache_control breakpoints shift position between turns.
_METADATA_KEYS: frozenset[str] = frozenset({"cache_control"})


def normalize_json(
    obj: Any,
    *,
    strip_keys: frozenset[str] = _METADATA_KEYS,
) -> str:
    """Recursively sort JSON keys and produce a stable string representation.

    This eliminates non-deterministic ordering from dicts, which is the #1
    cause of silent cache misses in languages/runtimes with unstable dict ordering.

    Keys listed in *strip_keys* (default: ``{"cache_control"}``) are excluded
    from the output so that caching metadata injected by the provider layer
    does not change segment hashes.
    """
    if isinstance(obj, dict):
        sorted_items = sorted(
            (k, normalize_json(v, strip_keys=strip_keys))
            for k, v in obj.items()
            if k not in strip_keys
        )
        return "{" + ",".join(f'"{k}":{v}' for k, v in sorted_items) + "}"
    elif isinstance(obj, (list, tuple)):
        return "[" + ",".join(normalize_json(item, strip_keys=strip_keys) for item in obj) + "]"
    elif isinstance(obj, str):
        return json.dumps(obj)
    elif isinstance(obj, bool):
        return "true" if obj else "false"
    elif obj is None:
        return "null"
    else:
        return json.dumps(obj)


def normalize_whitespace(text: str) -> str:
    """Strip and collapse whitespace for stable hashing."""
    return re.sub(r"\s+", " ", text.strip())


def hash_segment(content: Any) -> str:
    """Hash a single prompt segment (system prompt, tool block, message, etc.)."""
    if isinstance(content, str):
        normalized = normalize_whitespace(content)
    elif isinstance(content, (dict, list)):
        normalized = normalize_json(content)
    else:
        normalized = str(content)

    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def extract_segments(
    system: Any = None,
    tools: list[dict[str, Any]] | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> list[tuple[str, Any]]:
    """Extract labeled segments from request kwargs.

    Returns list of (label, content) tuples in prefix order:
    system → tools → message[0] → message[1] → ...
    """
    segments: list[tuple[str, Any]] = []

    if system is not None:
        segments.append(("system", system))

    if tools:
        segments.append(("tools", tools))

    if messages:
        for i, msg in enumerate(messages):
            segments.append((f"message[{i}]", msg))

    return segments


def _estimate_segment_tokens(content: Any) -> int:
    """Estimate token count for a single segment with image-safe handling.

    - Strings: ``len // 4``
    - Dicts/lists with ``type: "image"`` blocks: 1568 fixed tokens per image
    - Other dicts/lists: ``len(json.dumps(...)) // 4``
    """
    if isinstance(content, str):
        return len(content) // 4
    if isinstance(content, dict):
        if content.get("type") == "image":
            return 1568
        # Message dict — inspect content field
        inner = content.get("content", "")
        if isinstance(inner, str):
            return len(inner) // 4
        if isinstance(inner, list):
            total = 0
            for block in inner:
                if isinstance(block, dict):
                    if block.get("type") == "image":
                        total += 1568
                    elif block.get("type") == "text":
                        total += len(block.get("text", "")) // 4
                    else:
                        total += len(json.dumps(block)) // 4
                elif isinstance(block, str):
                    total += len(block) // 4
            return total
        return len(json.dumps(content)) // 4
    if isinstance(content, list):
        return len(json.dumps(content)) // 4
    return len(str(content)) // 4


def compute_fingerprint(
    system: Any = None,
    tools: list[dict[str, Any]] | None = None,
    messages: list[dict[str, Any]] | None = None,
    token_estimate: int = 0,
) -> Fingerprint:
    """Compute a rolling hash fingerprint for the given prompt segments.

    Hashes each segment independently (system, tools, each message) so that
    divergence detection can identify the exact segment that changed.
    Also computes per-segment token estimates for accurate cost calculations.
    """
    segments = extract_segments(system=system, tools=tools, messages=messages)

    segment_hashes = []
    segment_labels = []
    segment_token_estimates = []
    for label, content in segments:
        segment_hashes.append(hash_segment(content))
        segment_labels.append(label)
        segment_token_estimates.append(_estimate_segment_tokens(content))

    # Combined fingerprint: hash of all segment hashes concatenated
    combined_input = ":".join(segment_hashes)
    combined = hashlib.sha256(combined_input.encode("utf-8")).hexdigest()[:16]

    # Use computed total if caller didn't provide one
    computed_total = sum(segment_token_estimates)
    if token_estimate == 0:
        token_estimate = computed_total

    return Fingerprint(
        combined=combined,
        segment_hashes=segment_hashes,
        segment_labels=segment_labels,
        token_estimate=token_estimate,
        segment_token_estimates=segment_token_estimates,
    )


def find_divergence(
    prev: Fingerprint,
    curr: Fingerprint,
) -> DivergencePoint | None:
    """Find the first segment where two fingerprints diverge.

    Returns None if they are identical.
    """
    max_len = max(len(prev.segment_hashes), len(curr.segment_hashes))

    for i in range(max_len):
        prev_hash = prev.segment_hashes[i] if i < len(prev.segment_hashes) else "<missing>"
        curr_hash = curr.segment_hashes[i] if i < len(curr.segment_hashes) else "<missing>"

        if prev_hash != curr_hash:
            label = (
                curr.segment_labels[i]
                if i < len(curr.segment_labels)
                else prev.segment_labels[i]
                if i < len(prev.segment_labels)
                else f"segment[{i}]"
            )
            return DivergencePoint(
                segment_index=i,
                segment_label=label,
                previous_hash=prev_hash,
                new_hash=curr_hash,
            )

    return None


def compute_prefix_match_depth(prev: Fingerprint, curr: Fingerprint) -> tuple[int, int]:
    """Return (matching_segments, total_segments) for prefix match depth."""
    total = max(len(prev.segment_hashes), len(curr.segment_hashes))
    matching = 0
    for i in range(min(len(prev.segment_hashes), len(curr.segment_hashes))):
        if prev.segment_hashes[i] == curr.segment_hashes[i]:
            matching += 1
        else:
            break
    return matching, total
