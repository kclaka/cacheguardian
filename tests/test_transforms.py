"""Tests for optimizer transforms: tool sorting, schema diffing, JSON stabilization."""

import pytest
from cache_guard.core.optimizer import (
    SystemPromptTemplate,
    detect_tool_schema_changes,
    reorder_static_first,
    sort_tools,
    stabilize_json_keys,
)


class TestSortTools:
    def test_sorts_by_name(self):
        tools = [
            {"name": "zebra", "description": "Z"},
            {"name": "alpha", "description": "A"},
            {"name": "middle", "description": "M"},
        ]
        result = sort_tools(tools)
        assert [t["name"] for t in result] == ["alpha", "middle", "zebra"]

    def test_empty_list(self):
        assert sort_tools([]) == []

    def test_already_sorted(self):
        tools = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        assert sort_tools(tools) == tools

    def test_preserves_all_fields(self):
        tools = [
            {"name": "b", "description": "desc_b", "input_schema": {"type": "object"}},
            {"name": "a", "description": "desc_a", "input_schema": {"type": "string"}},
        ]
        result = sort_tools(tools)
        assert result[0]["name"] == "a"
        assert result[0]["description"] == "desc_a"
        assert result[0]["input_schema"] == {"type": "string"}

    def test_deterministic_across_calls(self):
        """Same input always produces same output (no randomness)."""
        tools = [{"name": "c"}, {"name": "a"}, {"name": "b"}]
        results = [sort_tools(tools) for _ in range(10)]
        assert all(r == results[0] for r in results)


class TestStabilizeJsonKeys:
    def test_sorts_dict_keys(self):
        result = stabilize_json_keys({"z": 1, "a": 2, "m": 3})
        assert list(result.keys()) == ["a", "m", "z"]

    def test_nested_dicts(self):
        result = stabilize_json_keys({"b": {"z": 1, "a": 2}, "a": 3})
        assert list(result.keys()) == ["a", "b"]
        assert list(result["b"].keys()) == ["a", "z"]

    def test_list_with_dicts(self):
        result = stabilize_json_keys([{"b": 1, "a": 2}])
        assert list(result[0].keys()) == ["a", "b"]

    def test_primitive_passthrough(self):
        assert stabilize_json_keys("hello") == "hello"
        assert stabilize_json_keys(42) == 42
        assert stabilize_json_keys(None) is None


class TestReorderStaticFirst:
    def test_system_messages_first(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "you are helpful"},
            {"role": "assistant", "content": "hello"},
        ]
        result = reorder_static_first(messages)
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"

    def test_no_system_unchanged(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = reorder_static_first(messages)
        assert result == messages

    def test_multiple_system_messages(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "rule 1"},
            {"role": "system", "content": "rule 2"},
        ]
        result = reorder_static_first(messages)
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "system"
        assert result[2]["role"] == "user"


class TestSystemPromptTemplate:
    def test_detects_variables(self):
        tpl = SystemPromptTemplate("Today is {date}. User: {user_name}")
        assert tpl.has_variables
        assert "date" in tpl.variable_names
        assert "user_name" in tpl.variable_names

    def test_no_variables(self):
        tpl = SystemPromptTemplate("You are a helpful assistant.")
        assert not tpl.has_variables
        assert tpl.variable_names == []

    def test_static_part_unchanged(self):
        template_str = "Today is {date}. Be helpful."
        tpl = SystemPromptTemplate(template_str)
        assert tpl.static_part == template_str

    def test_render_dynamic(self):
        tpl = SystemPromptTemplate("Today is {date}. User: {user_name}")
        result = tpl.render_dynamic(date="2026-02-20", user_name="Alice")
        assert "date=2026-02-20" in result
        assert "user_name=Alice" in result

    def test_render_dynamic_partial(self):
        tpl = SystemPromptTemplate("Today is {date}. User: {user_name}")
        result = tpl.render_dynamic(date="2026-02-20")
        assert "date=2026-02-20" in result
        assert "user_name" not in result


class TestDetectToolSchemaChanges:
    def test_no_changes(self):
        tools = [{"name": "search", "input_schema": {"properties": {"q": {"type": "string"}}}}]
        assert detect_tool_schema_changes(tools, tools) == []

    def test_tool_added(self):
        prev = [{"name": "search"}]
        curr = [{"name": "search"}, {"name": "calculate"}]
        changes = detect_tool_schema_changes(prev, curr)
        assert any("added" in c and "calculate" in c for c in changes)

    def test_tool_removed(self):
        prev = [{"name": "search"}, {"name": "calculate"}]
        curr = [{"name": "search"}]
        changes = detect_tool_schema_changes(prev, curr)
        assert any("removed" in c and "calculate" in c for c in changes)

    def test_param_added(self):
        prev = [{"name": "search", "input_schema": {"properties": {"q": {"type": "string"}}}}]
        curr = [{"name": "search", "input_schema": {"properties": {"q": {"type": "string"}, "format": {"type": "string"}}}}]
        changes = detect_tool_schema_changes(prev, curr)
        assert any("schema changed" in c and "format" in c for c in changes)

    def test_param_removed(self):
        prev = [{"name": "search", "input_schema": {"properties": {"q": {}, "format": {}}}}]
        curr = [{"name": "search", "input_schema": {"properties": {"q": {}}}}]
        changes = detect_tool_schema_changes(prev, curr)
        assert any("schema changed" in c and "format" in c for c in changes)

    def test_openai_style_tools(self):
        """OpenAI uses function.parameters.properties."""
        prev = [{"name": "search", "function": {"parameters": {"properties": {"q": {}}}}}]
        curr = [{"name": "search", "function": {"parameters": {"properties": {"q": {}, "limit": {}}}}}]
        changes = detect_tool_schema_changes(prev, curr)
        assert any("schema changed" in c for c in changes)
