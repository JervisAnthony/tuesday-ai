"""Tests for explicit TUESDAY tool registration and lookup."""

import ast
from pathlib import Path

import pytest

from tuesday.tools import (
    BaseTool,
    DuplicateToolError,
    ToolInvocation,
    ToolNotFoundError,
    ToolRegistry,
    ToolRegistryError,
    ToolResult,
)


class StubTool(BaseTool):
    """Configurable test tool that records execution calls."""

    def __init__(self, name: object) -> None:
        self._name = name
        self.execute_calls = 0

    @property
    def name(self) -> str:
        return self._name  # type: ignore[return-value]

    @property
    def description(self) -> str:
        return "A registry test tool."

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        self.execute_calls += 1
        self._validate_invocation(invocation)
        return ToolResult(
            tool_name=self.name,
            invocation_id=invocation.invocation_id,
            output=None,
        )


def test_empty_registries_have_no_names_and_are_independent() -> None:
    first = ToolRegistry()
    second = ToolRegistry()

    assert first.names == ()
    assert second.names == ()


def test_registered_tool_is_retrieved_by_exact_identity() -> None:
    registry = ToolRegistry()
    tool = StubTool("calendar.search")

    registry.register(tool)

    assert registry.names == ("calendar.search",)
    assert registry.get("calendar.search") is tool


def test_multiple_tools_preserve_registration_order() -> None:
    registry = ToolRegistry()
    registry.register(StubTool("calendar.search"))
    registry.register(StubTool("email.read"))

    assert registry.names == ("calendar.search", "email.read")


@pytest.mark.parametrize("invalid_tool", [None, object(), "email.read", 1])
def test_non_tools_are_rejected_without_changing_state(
    invalid_tool: object,
) -> None:
    registry = ToolRegistry()

    with pytest.raises(TypeError, match="tool must be a BaseTool instance"):
        registry.register(invalid_tool)  # type: ignore[arg-type]

    assert registry.names == ()


def test_non_string_tool_name_is_rejected_without_changing_state() -> None:
    registry = ToolRegistry()

    with pytest.raises(TypeError, match="Tool name must be a string"):
        registry.register(StubTool(7))

    assert registry.names == ()


@pytest.mark.parametrize("name", ["", " ", "\t", "\n"])
def test_empty_tool_name_is_rejected_without_changing_state(name: str) -> None:
    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Tool name must not be empty"):
        registry.register(StubTool(name))

    assert registry.names == ()


@pytest.mark.parametrize("name", [" calendar.search", "calendar.search "])
def test_surrounding_whitespace_is_rejected_without_changing_state(
    name: str,
) -> None:
    registry = ToolRegistry()

    with pytest.raises(
        ValueError,
        match="Tool name must not have surrounding whitespace",
    ):
        registry.register(StubTool(name))

    assert registry.names == ()


def test_same_tool_instance_cannot_be_registered_twice() -> None:
    registry = ToolRegistry()
    tool = StubTool("email.read")
    registry.register(tool)

    with pytest.raises(
        DuplicateToolError,
        match="A tool named 'email.read' is already registered",
    ):
        registry.register(tool)

    assert registry.names == ("email.read",)
    assert registry.get("email.read") is tool


def test_duplicate_name_does_not_replace_original_tool() -> None:
    registry = ToolRegistry()
    original = StubTool("email.read")
    registry.register(original)

    with pytest.raises(DuplicateToolError, match="email.read"):
        registry.register(StubTool("email.read"))

    assert registry.names == ("email.read",)
    assert registry.get("email.read") is original


def test_tool_names_are_exact_and_case_sensitive() -> None:
    registry = ToolRegistry()
    lowercase_tool = StubTool("calendar.search")
    titlecase_tool = StubTool("Calendar.Search")
    registry.register(lowercase_tool)
    registry.register(titlecase_tool)

    assert registry.get("calendar.search") is lowercase_tool
    assert registry.get("Calendar.Search") is titlecase_tool
    for unregistered_name in ("CALENDAR.SEARCH", "calendar.Search"):
        with pytest.raises(ToolNotFoundError, match=unregistered_name):
            registry.get(unregistered_name)


@pytest.mark.parametrize("name", ["missing", "", " calendar.search "])
def test_missing_string_name_raises_tool_not_found(name: str) -> None:
    registry = ToolRegistry()

    with pytest.raises(ToolNotFoundError, match="No tool named"):
        registry.get(name)


@pytest.mark.parametrize("name", [None, object(), 1, True])
def test_non_string_lookup_name_is_rejected(name: object) -> None:
    registry = ToolRegistry()

    with pytest.raises(TypeError, match="Tool lookup name must be a string"):
        registry.get(name)  # type: ignore[arg-type]


def test_names_are_immutable_independent_snapshots() -> None:
    registry = ToolRegistry()
    registry.register(StubTool("calendar.search"))
    earlier_names = registry.names

    extended_names = earlier_names + ("caller.change",)
    registry.register(StubTool("email.read"))

    assert isinstance(earlier_names, tuple)
    assert earlier_names == ("calendar.search",)
    assert extended_names == ("calendar.search", "caller.change")
    assert registry.names == ("calendar.search", "email.read")


def test_registry_instances_are_isolated() -> None:
    first = ToolRegistry()
    second = ToolRegistry()
    tool = StubTool("calendar.search")

    first.register(tool)

    assert first.get("calendar.search") is tool
    assert second.names == ()
    with pytest.raises(ToolNotFoundError):
        second.get("calendar.search")


def test_registration_and_lookup_never_execute_tool() -> None:
    registry = ToolRegistry()
    tool = StubTool("calendar.search")

    registry.register(tool)
    assert registry.get(tool.name) is tool

    assert tool.execute_calls == 0


def test_registry_errors_have_tool_specific_hierarchy() -> None:
    assert issubclass(DuplicateToolError, ToolRegistryError)
    assert issubclass(ToolNotFoundError, ToolRegistryError)
    assert issubclass(ToolRegistryError, RuntimeError)
    assert not issubclass(ToolNotFoundError, KeyError)


def test_registry_symbols_are_publicly_exported() -> None:
    import tuesday.tools as tools

    assert tools.ToolRegistry is ToolRegistry
    assert tools.ToolRegistryError is ToolRegistryError
    assert tools.DuplicateToolError is DuplicateToolError
    assert tools.ToolNotFoundError is ToolNotFoundError


def test_registry_module_has_only_allowed_production_imports() -> None:
    source_path = (
        Path(__file__).parents[2] / "src" / "tuesday" / "tools" / "registry.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )

    assert imports == {"tuesday.tools.base"}
