"""Tests for deterministic single-tool execution."""

import ast
import asyncio
import inspect
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest

import tuesday.tools as tools
from tuesday.tools import (
    BaseTool,
    DeterministicToolExecutor,
    InvalidToolResultError,
    ToolExecutionError,
    ToolInvocation,
    ToolNotFoundError,
    ToolRegistry,
    ToolResult,
)


class RecordingTool(BaseTool):
    """Configurable tool that records every received invocation."""

    def __init__(
        self,
        name: str,
        *,
        result: object = None,
        error: BaseException | None = None,
    ) -> None:
        self._name = name
        self.result = result
        self.error = error
        self.execute_calls = 0
        self.invocations: list[ToolInvocation] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Record deterministic executor calls."

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        self.execute_calls += 1
        self.invocations.append(invocation)
        if self.error is not None:
            raise self.error
        return cast(ToolResult, self.result)


class TrackingRegistry(ToolRegistry):
    """Registry subclass that records lookup attempts."""

    def __init__(self) -> None:
        super().__init__()
        self.get_calls = 0

    def get(self, name: str) -> BaseTool:
        self.get_calls += 1
        return super().get(name)


def run_execute(
    executor: DeterministicToolExecutor,
    invocation: ToolInvocation,
) -> ToolResult:
    return asyncio.run(executor.execute(invocation))


def configured_execution(
    name: str = "calendar.search",
    *,
    output: object = None,
) -> tuple[
    DeterministicToolExecutor,
    RecordingTool,
    ToolInvocation,
    ToolResult,
]:
    invocation = ToolInvocation(tool_name=name)
    result = ToolResult(
        tool_name=name,
        invocation_id=invocation.invocation_id,
        output=output,
    )
    tool = RecordingTool(name, result=result)
    registry = ToolRegistry()
    registry.register(tool)
    return DeterministicToolExecutor(registry), tool, invocation, result


def test_constructor_accepts_and_preserves_exact_registry() -> None:
    registry = ToolRegistry()

    executor = DeterministicToolExecutor(registry)

    assert executor._registry is registry


def test_constructor_accepts_registry_subclass() -> None:
    registry = TrackingRegistry()

    assert DeterministicToolExecutor(registry)._registry is registry


@pytest.mark.parametrize("invalid_registry", [None, object(), {}, "registry"])
def test_constructor_rejects_non_registry(invalid_registry: object) -> None:
    with pytest.raises(
        TypeError,
        match="registry must be a ToolRegistry instance",
    ):
        DeterministicToolExecutor(invalid_registry)  # type: ignore[arg-type]


def test_executor_is_slotted_with_only_registry_state() -> None:
    executor = DeterministicToolExecutor(ToolRegistry())

    assert DeterministicToolExecutor.__slots__ == ("_registry",)
    assert not hasattr(executor, "__dict__")


def test_execute_contract_is_asynchronous_with_exact_parameters() -> None:
    assert inspect.iscoroutinefunction(DeterministicToolExecutor.execute)
    assert tuple(inspect.signature(DeterministicToolExecutor.execute).parameters) == (
        "self",
        "invocation",
    )


@pytest.mark.parametrize("invalid_invocation", [None, object(), "invalid", 123])
def test_invalid_invocation_fails_before_lookup(
    invalid_invocation: object,
) -> None:
    registry = TrackingRegistry()
    executor = DeterministicToolExecutor(registry)

    with pytest.raises(TypeError, match="invocation must be a ToolInvocation"):
        asyncio.run(
            executor.execute(invalid_invocation)  # type: ignore[arg-type]
        )

    assert registry.get_calls == 0


def test_success_executes_once_with_exact_invocation_and_result_identity() -> None:
    executor, tool, invocation, result = configured_execution()

    returned = run_execute(executor, invocation)

    assert tool.execute_calls == 1
    assert tool.invocations == [invocation]
    assert tool.invocations[0] is invocation
    assert returned is result


def test_nested_output_is_preserved_on_original_result() -> None:
    output = {"events": [{"title": "Standup", "attendees": ["Ada", "Lin"]}]}
    executor, _, invocation, result = configured_execution(output=output)

    returned = run_execute(executor, invocation)

    assert returned is result
    assert returned.output is result.output
    assert returned.output["events"][0]["attendees"] == ("Ada", "Lin")  # type: ignore[index]


def test_case_mismatched_name_is_not_resolved_or_executed() -> None:
    registry = ToolRegistry()
    tool = RecordingTool("calendar.search")
    registry.register(tool)
    executor = DeterministicToolExecutor(registry)

    with pytest.raises(ToolNotFoundError):
        run_execute(executor, ToolInvocation(tool_name="Calendar.Search"))

    assert tool.execute_calls == 0


def test_missing_tool_propagates_and_no_other_tool_executes() -> None:
    registry = ToolRegistry()
    unrelated = RecordingTool("email.read")
    registry.register(unrelated)
    executor = DeterministicToolExecutor(registry)

    with pytest.raises(ToolNotFoundError) as caught:
        run_execute(executor, ToolInvocation(tool_name="calendar.search"))

    assert type(caught.value) is ToolNotFoundError
    assert unrelated.execute_calls == 0


@pytest.mark.parametrize(
    "error",
    [ToolExecutionError("failure"), RuntimeError("unexpected")],
)
def test_tool_exception_propagates_by_identity_without_retry(
    error: BaseException,
) -> None:
    registry = ToolRegistry()
    tool = RecordingTool("calendar.search", error=error)
    registry.register(tool)
    executor = DeterministicToolExecutor(registry)

    with pytest.raises(type(error)) as caught:
        run_execute(executor, ToolInvocation(tool_name=tool.name))

    assert caught.value is error
    assert tool.execute_calls == 1


@pytest.mark.parametrize("invalid_result", [None, object(), "invalid"])
def test_invalid_result_type_fails_explicitly_after_one_execution(
    invalid_result: object,
) -> None:
    invocation = ToolInvocation(tool_name="calendar.search")
    tool = RecordingTool(invocation.tool_name, result=invalid_result)
    registry = ToolRegistry()
    registry.register(tool)

    with pytest.raises(
        InvalidToolResultError,
        match="Tool execution must return a ToolResult",
    ):
        run_execute(DeterministicToolExecutor(registry), invocation)

    assert tool.execute_calls == 1


def test_result_tool_name_mismatch_is_rejected_without_rewriting() -> None:
    invocation = ToolInvocation(tool_name="calendar.search")
    result = ToolResult(
        tool_name="calendar.other",
        invocation_id=invocation.invocation_id,
        output=None,
    )
    tool = RecordingTool(invocation.tool_name, result=result)
    registry = ToolRegistry()
    registry.register(tool)

    with pytest.raises(
        InvalidToolResultError,
        match="Tool result name does not match the invocation",
    ):
        run_execute(DeterministicToolExecutor(registry), invocation)

    assert tool.result is result
    assert result.tool_name == "calendar.other"
    assert tool.execute_calls == 1


def test_result_invocation_id_mismatch_is_rejected_without_rewriting() -> None:
    invocation = ToolInvocation(tool_name="calendar.search")
    mismatched_id = uuid4()
    result = ToolResult(
        tool_name=invocation.tool_name,
        invocation_id=mismatched_id,
        output=None,
    )
    tool = RecordingTool(invocation.tool_name, result=result)
    registry = ToolRegistry()
    registry.register(tool)

    with pytest.raises(
        InvalidToolResultError,
        match="Tool result invocation_id does not match the invocation",
    ):
        run_execute(DeterministicToolExecutor(registry), invocation)

    assert result.invocation_id is mismatched_id
    assert tool.execute_calls == 1


def test_execution_preserves_registry_names_and_tool_identity() -> None:
    executor, tool, invocation, _ = configured_execution()
    registry = executor._registry
    names_before = registry.names

    run_execute(executor, invocation)

    assert registry.names == names_before
    assert registry.get(tool.name) is tool


def test_only_explicitly_selected_tool_executes() -> None:
    invocation = ToolInvocation(tool_name="calendar.search")
    selected_result = ToolResult(
        tool_name=invocation.tool_name,
        invocation_id=invocation.invocation_id,
        output="selected",
    )
    selected = RecordingTool(invocation.tool_name, result=selected_result)
    other = RecordingTool("email.read")
    registry = ToolRegistry()
    registry.register(selected)
    registry.register(other)

    returned = run_execute(DeterministicToolExecutor(registry), invocation)

    assert returned is selected_result
    assert selected.execute_calls == 1
    assert other.execute_calls == 0


def test_repeated_explicit_calls_each_execute_once_without_history() -> None:
    registry = ToolRegistry()
    tool = RecordingTool("calendar.search")
    registry.register(tool)
    executor = DeterministicToolExecutor(registry)
    first = ToolInvocation(tool_name=tool.name)
    second = ToolInvocation(tool_name=tool.name)

    tool.result = ToolResult(tool.name, first.invocation_id, "first")
    assert run_execute(executor, first) is tool.result
    tool.result = ToolResult(tool.name, second.invocation_id, "second")
    assert run_execute(executor, second) is tool.result

    assert tool.execute_calls == 2
    assert tool.invocations == [first, second]
    assert not hasattr(executor, "_history")


def test_executor_errors_and_types_are_publicly_exported() -> None:
    assert tools.DeterministicToolExecutor is DeterministicToolExecutor
    assert tools.InvalidToolResultError is InvalidToolResultError
    assert issubclass(InvalidToolResultError, ValueError)
    assert not issubclass(InvalidToolResultError, ToolExecutionError)


def test_executor_module_has_only_allowed_production_imports() -> None:
    source_path = (
        Path(__file__).parents[2] / "src" / "tuesday" / "tools" / "executor.py"
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

    assert imports == {"tuesday.tools.base", "tuesday.tools.registry"}
