"""Tests for model intent to invocation candidate conversion."""

import ast
import inspect
from dataclasses import FrozenInstanceError, fields
from unittest.mock import Mock
from uuid import UUID

import pytest

import tuesday.bridges as bridges
import tuesday.bridges.model_tools as model_tools
from tuesday.bridges import ModelToolInvocationBridge
from tuesday.language_models.tools import LanguageModelToolCall
from tuesday.tools.base import ToolInvocation

INVOCATION_ID = UUID("11111111-1111-1111-1111-111111111111")


def test_basic_conversion() -> None:
    call = LanguageModelToolCall(
        "call_abc123",
        "calculator.basic",
        {"operation": "multiply", "left": 6, "right": 7},
    )
    factory = Mock(return_value=INVOCATION_ID)
    invocation = ModelToolInvocationBridge(
        invocation_id_factory=factory,
    ).to_invocation(call)

    assert isinstance(invocation, ToolInvocation)
    assert invocation.tool_name == call.name
    assert invocation.arguments == call.arguments
    assert invocation.arguments is not call.arguments
    assert invocation.invocation_id is INVOCATION_ID
    assert tuple(field.name for field in fields(invocation)) == (
        "tool_name",
        "arguments",
        "invocation_id",
    )
    factory.assert_called_once_with()


@pytest.mark.parametrize(
    "call_id",
    [
        "call_abc123",
        "toolu_01XYZ",
        "opaque:not-a-uuid",
        str(UUID(int=2)),
    ],
)
def test_call_id_remains_separate(call_id: str) -> None:
    call = LanguageModelToolCall(call_id, "unknown.tool", {})
    invocation = ModelToolInvocationBridge(
        invocation_id_factory=lambda: INVOCATION_ID,
    ).to_invocation(call)
    assert call.call_id == call_id
    assert invocation.invocation_id is INVOCATION_ID
    assert str(invocation.invocation_id) != call_id


@pytest.mark.parametrize(
    "name",
    [
        "calculator.basic",
        "Calculator.Basic",
        "unknown.tool",
        "Tool-Name_v2",
    ],
)
def test_exact_names_and_empty_arguments(name: str) -> None:
    invocation = ModelToolInvocationBridge().to_invocation(
        LanguageModelToolCall("call_1", name, {}),
    )
    assert invocation.tool_name == name
    assert invocation.arguments == {}


def test_recursive_snapshot_preserves_source_and_data() -> None:
    call = LanguageModelToolCall(
        "call_1",
        "unknown.tool",
        {
            "payload": {
                "": 1,
                " nested ": 2,
                "items": [None, True, 3, 1.5, "s", {"ok": False}],
            },
        },
    )
    original = call
    arguments = call.arguments
    payload = arguments["payload"]
    invocation = ModelToolInvocationBridge().to_invocation(call)

    assert call is original
    assert call.call_id == "call_1"
    assert call.name == "unknown.tool"
    assert call.arguments is arguments
    assert call.arguments["payload"] is payload
    assert invocation.arguments == arguments
    assert invocation.arguments is not arguments
    target_payload = invocation.arguments["payload"]
    assert target_payload is not payload
    assert target_payload["items"] is not payload["items"]
    assert target_payload["items"][-1] is not payload["items"][-1]
    for mapping in (
        arguments,
        payload,
        invocation.arguments,
        target_payload,
        payload["items"][-1],
        target_payload["items"][-1],
    ):
        with pytest.raises(TypeError):
            mapping["changed"] = True
    with pytest.raises(FrozenInstanceError):
        call.call_id = "changed"
    with pytest.raises(FrozenInstanceError):
        invocation.tool_name = "changed"


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        {},
        object(),
        "call",
        {"call_id": "call_1", "name": "tool", "arguments": {}},
    ],
)
def test_source_type_validation_precedes_factory(invalid: object) -> None:
    factory = Mock(return_value=INVOCATION_ID)
    bridge = ModelToolInvocationBridge(invocation_id_factory=factory)
    with pytest.raises(TypeError) as caught:
        bridge.to_invocation(invalid)
    assert str(caught.value) == "tool_call must be a LanguageModelToolCall."
    factory.assert_not_called()


def test_factory_called_once_per_conversion_without_retained_calls() -> None:
    second_id = UUID(int=2)
    factory = Mock(side_effect=[INVOCATION_ID, second_id])
    bridge = ModelToolInvocationBridge(invocation_id_factory=factory)
    factory.assert_not_called()
    call = LanguageModelToolCall("call_1", "unknown.tool", {})
    first = bridge.to_invocation(call)
    factory.assert_called_once_with()
    second = bridge.to_invocation(call)
    assert factory.call_count == 2
    assert first is not second
    assert first.invocation_id is INVOCATION_ID
    assert second.invocation_id is second_id


@pytest.mark.parametrize("invalid", [None, 42, "uuid4", object()])
def test_factory_must_be_callable(invalid: object) -> None:
    with pytest.raises(TypeError) as caught:
        ModelToolInvocationBridge(invocation_id_factory=invalid)
    assert str(caught.value) == "invocation_id_factory must be callable."


@pytest.mark.parametrize("invalid", [None, "uuid", str(INVOCATION_ID), 123, object()])
def test_factory_output_must_be_uuid(invalid: object) -> None:
    factory = Mock(return_value=invalid)
    with pytest.raises(TypeError) as caught:
        ModelToolInvocationBridge(invocation_id_factory=factory).to_invocation(
            LanguageModelToolCall("call_1", "unknown.tool", {}),
        )
    assert str(caught.value) == "invocation_id_factory must return a UUID."
    factory.assert_called_once_with()


def test_factory_exception_propagates_unchanged() -> None:
    error = RuntimeError("UUID generation failed")
    factory = Mock(side_effect=error)
    with pytest.raises(RuntimeError) as caught:
        ModelToolInvocationBridge(invocation_id_factory=factory).to_invocation(
            LanguageModelToolCall("call_1", "unknown.tool", {}),
        )
    assert caught.value is error
    factory.assert_called_once_with()


@pytest.mark.parametrize("key", ["", " ", " left", "left "])
def test_target_argument_name_validation_remains_authoritative(key: str) -> None:
    call = LanguageModelToolCall("call_1", "unknown.tool", {key: 1})
    factory = Mock(return_value=INVOCATION_ID)
    with pytest.raises(ValueError) as direct:
        ToolInvocation(call.name, call.arguments, INVOCATION_ID)
    with pytest.raises(ValueError) as bridged:
        ModelToolInvocationBridge(invocation_id_factory=factory).to_invocation(call)
    assert type(bridged.value) is type(direct.value)
    assert bridged.value.args == direct.value.args
    assert call.arguments == {key: 1}
    factory.assert_called_once_with()


def test_target_exception_propagates_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    error = TypeError("target contract failure")
    target = Mock(side_effect=error)
    monkeypatch.setattr(model_tools, "ToolInvocation", target)
    call = LanguageModelToolCall("call_1", "unknown.tool", {})
    with pytest.raises(TypeError) as caught:
        ModelToolInvocationBridge(
            invocation_id_factory=lambda: INVOCATION_ID,
        ).to_invocation(call)
    assert caught.value is error
    target.assert_called_once_with(
        tool_name=call.name,
        arguments=call.arguments,
        invocation_id=INVOCATION_ID,
    )


def test_default_factory_returns_uuid() -> None:
    invocation = ModelToolInvocationBridge().to_invocation(
        LanguageModelToolCall("call_1", "unknown.tool", {}),
    )
    assert isinstance(invocation.invocation_id, UUID)


def test_public_surface_is_small_and_synchronous() -> None:
    assert bridges.ModelToolInvocationBridge is ModelToolInvocationBridge
    assert bridges.__all__ == ["ModelToolInvocationBridge"]
    assert not inspect.iscoroutinefunction(ModelToolInvocationBridge.to_invocation)
    assert tuple(
        inspect.signature(ModelToolInvocationBridge.to_invocation).parameters
    ) == (
        "self",
        "tool_call",
    )
    assert ModelToolInvocationBridge.__slots__ == ("_invocation_id_factory",)
    assert not hasattr(ModelToolInvocationBridge(), "__dict__")


def test_bridge_has_only_explicit_boundary_dependencies() -> None:
    tree = ast.parse(inspect.getsource(model_tools))
    imports = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imports == {
        "collections.abc",
        "uuid",
        "tuesday.language_models.tools",
        "tuesday.tools.base",
    }
