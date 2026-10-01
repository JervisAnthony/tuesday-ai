"""Tests for the deterministic basic calculator tool."""

import ast
import asyncio
import inspect
from pathlib import Path

import pytest

import tuesday.tools as tools
from tuesday.tools import (
    BaseTool,
    BasicCalculatorTool,
    DeterministicToolExecutor,
    GuardedToolExecutor,
    StaticToolAuthorizationPolicy,
    ToolAuthorizationDeniedError,
    ToolAuthorizationOutcome,
    ToolExecutionError,
    ToolInvocation,
    ToolRegistry,
    ToolResult,
)

ARGUMENT_ERROR = (
    "Calculator invocation arguments must be exactly: operation, left, right."
)
OPERATION_ERROR = (
    "Calculator operation must be one of: add, subtract, multiply, divide."
)
RANGE_ERROR = "Calculator result is outside the supported finite numeric range."


def invocation_for(
    operation: object = "add",
    left: object = 2,
    right: object = 3,
    *,
    tool_name: str = "calculator.basic",
) -> ToolInvocation:
    return ToolInvocation(
        tool_name=tool_name,
        arguments={
            "operation": operation,
            "left": left,
            "right": right,
        },  # type: ignore[dict-item]
    )


def execute(
    tool: BasicCalculatorTool,
    invocation: ToolInvocation,
) -> ToolResult:
    return asyncio.run(tool.execute(invocation))


def test_class_shape_and_stable_metadata() -> None:
    tool = BasicCalculatorTool()

    assert issubclass(BasicCalculatorTool, BaseTool)
    assert BasicCalculatorTool.__slots__ == ()
    assert not hasattr(tool, "__dict__")
    assert tool.name == "calculator.basic"
    assert tool.name == "calculator.basic"
    assert tool.description == (
        "Perform one basic arithmetic operation on two finite numbers."
    )
    assert tool.description == (
        "Perform one basic arithmetic operation on two finite numbers."
    )


def test_execute_is_async_with_exact_parameters() -> None:
    assert inspect.iscoroutinefunction(BasicCalculatorTool.execute)
    assert tuple(inspect.signature(BasicCalculatorTool.execute).parameters) == (
        "self",
        "invocation",
    )


@pytest.mark.parametrize(
    "invalid_invocation",
    [None, object(), "calculator.basic", 123, {}],
)
def test_wrong_invocation_type_uses_base_validation(
    invalid_invocation: object,
) -> None:
    tool = BasicCalculatorTool()

    with pytest.raises(TypeError, match="invocation must be a ToolInvocation"):
        asyncio.run(
            tool.execute(invalid_invocation)  # type: ignore[arg-type]
        )


def test_wrong_tool_target_uses_base_validation() -> None:
    tool = BasicCalculatorTool()

    with pytest.raises(ValueError, match="must target this tool"):
        execute(tool, invocation_for(tool_name="calculator.other"))


@pytest.mark.parametrize(
    "arguments",
    [
        {"left": 1, "right": 2},
        {"operation": "add", "right": 2},
        {"operation": "add", "left": 1},
        {},
        {"operation": "add", "left": 1, "right": 2, "precision": 2},
        {"operation": "add", "left": 1, "right": 2, "round": True},
        {"operation": "add", "left": 1, "right": 2, "foo": None},
    ],
)
def test_arguments_must_have_exact_required_keys(
    arguments: dict[str, object],
) -> None:
    invocation = ToolInvocation(
        tool_name="calculator.basic",
        arguments=arguments,  # type: ignore[arg-type]
    )

    with pytest.raises(ToolExecutionError, match=ARGUMENT_ERROR):
        execute(BasicCalculatorTool(), invocation)


@pytest.mark.parametrize("invalid_operation", [None, True, 1, 1.5, {}, ()])
def test_operation_must_be_string(invalid_operation: object) -> None:
    with pytest.raises(
        ToolExecutionError,
        match="Calculator operation must be a string",
    ):
        execute(BasicCalculatorTool(), invocation_for(invalid_operation))


@pytest.mark.parametrize(
    "invalid_operation",
    ["ADD", "Add", " add", "add ", "+", "power", "modulo", "divide_by"],
)
def test_operation_must_be_exact_supported_token(
    invalid_operation: str,
) -> None:
    with pytest.raises(ToolExecutionError, match=OPERATION_ERROR):
        execute(BasicCalculatorTool(), invocation_for(invalid_operation))


@pytest.mark.parametrize(
    ("operand_name", "invalid_operand"),
    [
        (operand_name, invalid_operand)
        for operand_name in ("left", "right")
        for invalid_operand in (True, False, None, "1", "1.5", {}, (), [])
    ],
)
def test_operands_require_strict_int_or_float(
    operand_name: str,
    invalid_operand: object,
) -> None:
    operands = {"left": 2, "right": 3, operand_name: invalid_operand}
    invocation = invocation_for(
        left=operands["left"],
        right=operands["right"],
    )

    with pytest.raises(
        ToolExecutionError,
        match=f"Calculator {operand_name} operand must be an int or float",
    ):
        execute(BasicCalculatorTool(), invocation)


@pytest.mark.parametrize(
    ("operation", "left", "right", "expected", "expected_type"),
    [
        ("add", 2, 3, 5, int),
        ("add", -2, 3, 1, int),
        ("add", 2.5, 3, 5.5, float),
        ("add", -2.5, -1.5, -4.0, float),
        ("subtract", 5, 3, 2, int),
        ("subtract", 3, 5, -2, int),
        ("subtract", 5.5, 2, 3.5, float),
        ("multiply", 4, 3, 12, int),
        ("multiply", -4, 3, -12, int),
        ("multiply", 2.5, 4, 10.0, float),
        ("multiply", 0, 100, 0, int),
        ("divide", 6, 3, 2.0, float),
        ("divide", 5, 2, 2.5, float),
        ("divide", -6, 3, -2.0, float),
        ("divide", 0, 5, 0.0, float),
    ],
)
def test_supported_arithmetic_and_result_correlation(
    operation: str,
    left: int | float,
    right: int | float,
    expected: int | float,
    expected_type: type[int] | type[float],
) -> None:
    tool = BasicCalculatorTool()
    invocation = invocation_for(operation, left, right)

    result = execute(tool, invocation)

    assert isinstance(result, ToolResult)
    assert result.tool_name == tool.name
    assert result.invocation_id is invocation.invocation_id
    assert result.output == expected
    assert type(result.output) is expected_type


@pytest.mark.parametrize("zero", [0, 0.0, -0.0])
def test_division_by_zero_is_translated_to_tool_error(
    zero: int | float,
) -> None:
    with pytest.raises(
        ToolExecutionError,
        match="Calculator cannot divide by zero",
    ) as caught:
        execute(BasicCalculatorTool(), invocation_for("divide", 5, zero))

    assert not isinstance(caught.value, ZeroDivisionError)


def test_non_finite_float_result_is_rejected_as_tool_error() -> None:
    with pytest.raises(ToolExecutionError, match=RANGE_ERROR):
        execute(
            BasicCalculatorTool(),
            invocation_for("multiply", 1e308, 1e308),
        )


def test_large_integer_division_overflow_is_translated_to_tool_error() -> None:
    with pytest.raises(ToolExecutionError, match=RANGE_ERROR) as caught:
        execute(
            BasicCalculatorTool(),
            invocation_for("divide", 10**400, 1),
        )

    assert isinstance(caught.value.__cause__, OverflowError)


def test_repeated_execution_returns_fresh_equal_results_without_state() -> None:
    tool = BasicCalculatorTool()
    invocation = invocation_for("multiply", 6, 7)

    first = execute(tool, invocation)
    second = execute(tool, invocation)

    assert first == second
    assert first is not second
    assert not hasattr(tool, "history")
    assert not hasattr(tool, "counter")
    assert not hasattr(tool, "last_result")
    assert not hasattr(tool, "last_invocation")
    assert not hasattr(tool, "cache")


def test_registry_preserves_exact_calculator_identity() -> None:
    tool = BasicCalculatorTool()
    registry = ToolRegistry()

    registry.register(tool)

    assert registry.names == ("calculator.basic",)
    assert registry.get("calculator.basic") is tool


def test_deterministic_executor_calculates_42() -> None:
    tool = BasicCalculatorTool()
    registry = ToolRegistry()
    registry.register(tool)
    executor = DeterministicToolExecutor(registry)
    invocation = invocation_for("add", 20, 22)

    result = asyncio.run(executor.execute(invocation))

    assert result.output == 42
    assert result.tool_name == tool.name
    assert result.invocation_id is invocation.invocation_id


def test_real_guarded_allow_chain_multiplies_6_by_7() -> None:
    tool = BasicCalculatorTool()
    registry = ToolRegistry()
    registry.register(tool)
    policy = StaticToolAuthorizationPolicy(
        {tool.name: ToolAuthorizationOutcome.ALLOW}
    )
    guarded = GuardedToolExecutor(
        policy,
        DeterministicToolExecutor(registry),
    )
    invocation = invocation_for("multiply", 6, 7)

    result = asyncio.run(guarded.execute(invocation))

    assert result.output == 42
    assert result.tool_name == tool.name
    assert result.invocation_id is invocation.invocation_id


def test_guarded_default_denial_blocks_calculator() -> None:
    tool = BasicCalculatorTool()
    registry = ToolRegistry()
    registry.register(tool)
    guarded = GuardedToolExecutor(
        StaticToolAuthorizationPolicy({}),
        DeterministicToolExecutor(registry),
    )
    invocation = invocation_for("add", 20, 22)

    with pytest.raises(ToolAuthorizationDeniedError) as caught:
        asyncio.run(guarded.execute(invocation))

    assert caught.value.decision.tool_name == tool.name
    assert caught.value.decision.invocation_id is invocation.invocation_id


def test_calculator_is_publicly_exported() -> None:
    assert tools.BasicCalculatorTool is BasicCalculatorTool


def test_calculator_module_has_only_allowed_dependencies() -> None:
    source_path = (
        Path(__file__).parents[2]
        / "src"
        / "tuesday"
        / "tools"
        / "calculator.py"
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

    assert imports == {"math", "tuesday.tools.base"}
