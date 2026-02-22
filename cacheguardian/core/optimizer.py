"""Shared optimization transforms for cache-friendly request construction."""

from __future__ import annotations

import json
import re
from typing import Any


def sort_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort tools alphabetically by name for deterministic ordering.

    This is the single most impactful auto-fix: non-deterministic tool ordering
    is the #1 cause of silent cache misses across all providers.
    """
    return sorted(tools, key=lambda t: t.get("name", ""))


def stabilize_json_keys(obj: Any) -> Any:
    """Recursively sort all dict keys for stable serialization.

    Prevents cache misses caused by non-deterministic key ordering in tool
    definitions, especially from Swift/Go/other languages with random map ordering.
    """
    if isinstance(obj, dict):
        return {k: stabilize_json_keys(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        return [stabilize_json_keys(item) for item in obj]
    return obj


def reorder_static_first(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure static content appears before dynamic content in messages.

    For OpenAI: system messages should be first, followed by stable context,
    then dynamic/changing messages.

    Note: We only reorder system messages to the front. We do NOT reorder
    user/assistant messages as that would change semantics.
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]
    other_msgs = [m for m in messages if m.get("role") != "system"]
    return system_msgs + other_msgs


class SystemPromptTemplate:
    """Separates static and dynamic parts of a system prompt.

    Usage:
        template = SystemPromptTemplate(
            "You are a helpful assistant. The current date is {date}. User: {user_name}."
        )
        static = template.static_part
        # "You are a helpful assistant. The current date is {date}. User: {user_name}."
        # (full template used as system prompt — stays stable)

        reminder = template.render_dynamic(date="2026-02-20", user_name="Alice")
        # "Updated context: date=2026-02-20, user_name=Alice"
        # (injected as a system-reminder message, not into the system prompt)
    """

    # Matches {variable_name} patterns
    _VAR_PATTERN = re.compile(r"\{(\w+)\}")

    def __init__(self, template: str) -> None:
        self.template = template
        self.variable_names = self._VAR_PATTERN.findall(template)

    @property
    def static_part(self) -> str:
        """The template string itself — used as the system prompt (never changes)."""
        return self.template

    @property
    def has_variables(self) -> bool:
        return len(self.variable_names) > 0

    def render_dynamic(self, **kwargs: str) -> str:
        """Render the dynamic parts as a system-reminder message.

        Instead of interpolating into the system prompt (which breaks the cache),
        we emit a separate message with the current values.
        """
        parts = []
        for name in self.variable_names:
            if name in kwargs:
                parts.append(f"{name}={kwargs[name]}")
        return f"Updated context: {', '.join(parts)}" if parts else ""


def pad_to_cache_bucket(
    system_content: str,
    total_prefix_tokens: int,
    bucket_size: int = 128,
    min_tokens: int = 1024,
) -> str:
    """Pad system content to the next OpenAI 128-token cache bucket boundary.

    OpenAI caches in strict *bucket_size*-token increments after the initial
    *min_tokens* minimum.  If the prompt is within ~32 tokens of the next
    boundary, appending whitespace is nearly free and recovers the full
    cache discount for those tokens.

    Returns the (possibly padded) system content string.
    """
    if total_prefix_tokens < min_tokens:
        return system_content  # below threshold, no benefit
    next_boundary = ((total_prefix_tokens // bucket_size) + 1) * bucket_size
    gap = next_boundary - total_prefix_tokens
    if gap <= 32:  # within 32 tokens of next boundary — pad
        padding = " " * (gap * 4)  # ~4 chars per token
        return system_content + padding
    return system_content


def segregate_dynamic_content(messages: list[dict[str, Any]]) -> int:
    """Return the index of the first dynamic (tool result) message in the most recent turn.

    Walks backward from the end of *messages*, skipping the latest user
    message, then skipping consecutive ``tool`` role results.  Returns the
    index immediately *after* the last stable message — i.e., the first
    dynamic tool result index.

    If there are no trailing tool results, returns ``len(messages) - 1``
    (the last user message itself is the boundary).
    """
    if not messages:
        return 0
    i = len(messages) - 1
    # Skip last user message
    if i >= 0 and messages[i].get("role") == "user":
        i -= 1
    # Skip consecutive tool results
    while i >= 0 and messages[i].get("role") == "tool":
        i -= 1
    # The first dynamic message starts at i + 1
    return i + 1


def detect_tool_schema_changes(
    prev_tools: list[dict[str, Any]],
    curr_tools: list[dict[str, Any]],
) -> list[str]:
    """Detect tool schema mutations (not just ordering).

    Returns a list of human-readable change descriptions.
    Catches: new tools added, tools removed, parameter changes.
    """
    changes = []

    prev_by_name = {t.get("name", ""): t for t in prev_tools}
    curr_by_name = {t.get("name", ""): t for t in curr_tools}

    # Added tools
    for name in curr_by_name:
        if name not in prev_by_name:
            changes.append(f"tool added: '{name}'")

    # Removed tools
    for name in prev_by_name:
        if name not in curr_by_name:
            changes.append(f"tool removed: '{name}'")

    # Schema changes
    for name in prev_by_name:
        if name in curr_by_name:
            prev_json = json.dumps(stabilize_json_keys(prev_by_name[name]), sort_keys=True)
            curr_json = json.dumps(stabilize_json_keys(curr_by_name[name]), sort_keys=True)
            if prev_json != curr_json:
                # Find what changed
                prev_params = set(_extract_param_names(prev_by_name[name]))
                curr_params = set(_extract_param_names(curr_by_name[name]))
                added = curr_params - prev_params
                removed = prev_params - curr_params
                detail = ""
                if added:
                    detail += f" (added params: {', '.join(sorted(added))})"
                if removed:
                    detail += f" (removed params: {', '.join(sorted(removed))})"
                if not detail:
                    detail = " (schema modified)"
                changes.append(f"tool schema changed: '{name}'{detail}")

    return changes


def _extract_param_names(tool: dict[str, Any]) -> list[str]:
    """Extract parameter names from a tool definition."""
    params = []
    # Anthropic-style: input_schema.properties
    schema = tool.get("input_schema", {})
    if not schema:
        # OpenAI-style: function.parameters.properties
        func = tool.get("function", {})
        schema = func.get("parameters", {})
    props = schema.get("properties", {})
    return list(props.keys())
