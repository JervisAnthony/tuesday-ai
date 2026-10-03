"""Tests for successful execution to provider-neutral result translation."""

import ast
import inspect
from dataclasses import FrozenInstanceError, fields
from unittest.mock import Mock
from uuid import UUID

import pytest

import tuesday.bridges as bridges
import tuesday.bridges.model_results as model_results
from tuesday.bridges import ModelToolExecution, ModelToolResultBridge
from tuesday.language_models import LanguageModelToolCall, LanguageModelToolResult
from tuesday.tools.base import ToolInvocation, ToolResult, ToolValue


def make_execution(output: ToolValue = 42) -> ModelToolExecution:
    call = LanguageModelToolCall("provider-specific:call_1", "unknown.tool", {})
    invocation = ToolInvocation(call.name, call.arguments, UUID(int=1))
    result = ToolResult(call.name, invocation.invocation_id, output)
    return ModelToolExecution(call, invocation, result)


@pytest.mark.parametrize("output", [None, "answer", 42, 1.5, True])
def test_bridge_preserves_model_correlation_and_scalar_output(
    output: ToolValue,
) -> None:
    execution = make_execution(output)
    result = ModelToolResultBridge().to_model_result(execution)
    assert isinstance(result, LanguageModelToolResult)
    assert result.call_id == execution.tool_call.call_id
    assert result.call_id != str(execution.invocation.invocation_id)
    assert result.output == output
    assert type(result.output) is type(output)
    assert tuple(field.name for field in fields(result)) == ("call_id", "output")


def test_nested_snapshot_and_source_objects_remain_unchanged() -> None:
    execution = make_execution(
        {
            "answer": 42,
            "metadata": {"ok": True, "items": [1, 2, None]},
        }
    )
    original = execution
    call, invocation, tool_result = (
        execution.tool_call,
        execution.invocation,
        execution.result,
    )
    output = tool_result.output
    metadata = output["metadata"]
    result = ModelToolResultBridge().to_model_result(execution)
    assert execution is original
    assert execution.tool_call is call
    assert execution.invocation is invocation
    assert execution.result is tool_result
    assert tool_result.output is output
    assert tool_result.output["metadata"] is metadata
    assert call.call_id == "provider-specific:call_1"
    assert invocation.invocation_id == UUID(int=1)
    assert result.output == output
    assert result.output is not output
    assert result.output["metadata"] is not metadata
    assert result.output["metadata"]["items"] is not metadata["items"]
    for mapping in (output, metadata, result.output, result.output["metadata"]):
        with pytest.raises(TypeError):
            mapping["changed"] = True
    for obj, field in (
        (execution, "result"),
        (call, "call_id"),
        (invocation, "invocation_id"),
        (tool_result, "output"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(obj, field, None)


def test_tuple_output_gets_independent_recursive_snapshot() -> None:
    execution = make_execution((1, {"items": [None, False]}))
    result = ModelToolResultBridge().to_model_result(execution)
    assert result.output == execution.result.output
    assert result.output is not execution.result.output
    assert result.output[1] is not execution.result.output[1]
    with pytest.raises(TypeError):
        result.output[0] = 2


@pytest.mark.parametrize(
    "invalid", [None, {}, object(), "execution", ToolResult("tool", UUID(int=1), None)]
)
def test_bridge_requires_actual_execution(
    invalid: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = Mock()
    monkeypatch.setattr(model_results, "LanguageModelToolResult", target)
    with pytest.raises(TypeError) as caught:
        ModelToolResultBridge().to_model_result(invalid)
    assert str(caught.value) == "execution must be a ModelToolExecution."
    target.assert_not_called()


def test_bridge_delegates_snapshot_to_target_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = make_execution({"value": 42})
    expected = LanguageModelToolResult(
        execution.tool_call.call_id, execution.result.output
    )
    target = Mock(return_value=expected)
    monkeypatch.setattr(model_results, "LanguageModelToolResult", target)
    result = ModelToolResultBridge().to_model_result(execution)
    assert result is expected
    target.assert_called_once_with(
        call_id=execution.tool_call.call_id,
        output=execution.result.output,
    )
    assert target.call_args.kwargs["output"] is execution.result.output


def test_bridge_is_stateless_synchronous_and_public() -> None:
    bridge = ModelToolResultBridge()
    assert ModelToolResultBridge.__slots__ == ()
    assert not hasattr(bridge, "__dict__")
    assert not inspect.iscoroutinefunction(ModelToolResultBridge.to_model_result)
    assert tuple(
        inspect.signature(ModelToolResultBridge.to_model_result).parameters
    ) == (
        "self",
        "execution",
    )
    assert bridges.ModelToolResultBridge is ModelToolResultBridge
    assert bridges.__all__ == [
        "GuardedModelToolExecutor",
        "ModelToolExecution",
        "ModelToolInvocationBridge",
        "ModelToolResultBridge",
    ]
    first = bridge.to_model_result(make_execution(1))
    second = bridge.to_model_result(make_execution(2))
    assert first.output == 1
    assert second.output == 2
    assert first is not second


def test_bridge_has_only_result_translation_dependencies() -> None:
    tree = ast.parse(inspect.getsource(model_results))
    imports = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imports == {
        "tuesday.bridges.model_execution",
        "tuesday.language_models.tools",
    }
