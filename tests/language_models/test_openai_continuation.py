"""Deterministic contracts and payload tests for explicit stateless replay."""

import ast
import asyncio
import inspect
import json
from dataclasses import FrozenInstanceError, fields
from math import inf, nan
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openai import OpenAIError
from openai.types.responses import ResponseFunctionToolCall

import tuesday.language_models as language_models
import tuesday.language_models.openai_continuation as contracts
from tuesday.config import LanguageModelSettings
from tuesday.domain import MessageRole
from tuesday.language_models import (
    LanguageModelMessage,
    LanguageModelProviderError,
    LanguageModelRequest,
    LanguageModelResponse,
    LanguageModelToolCall,
    LanguageModelToolDefinition,
    LanguageModelToolResult,
    OpenAIContinuationState,
    OpenAIGenerationResult,
    OpenAILanguageModelProvider,
)


class Item:
    """Public serializer fake with deliberately separate internal metadata."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        for key, value in payload.items():
            setattr(self, key, value)
        self.private_metadata = "must not be replayed"
        self.dumps = []

    def model_dump(self, **kwargs):
        self.dumps.append(kwargs)
        return self.payload


def function(call_id="call_1"):
    return Item(
        {
            "type": "function_call",
            "id": "item_" + call_id,
            "call_id": call_id,
            "name": "calculator.basic",
            "arguments": "{}",
        }
    )


def raw(*items, text=None):
    return SimpleNamespace(output=items, output_text=text, model="resolved-model")


def request():
    return LanguageModelRequest(
        (LanguageModelMessage(MessageRole.USER, "Hello"),),
        (
            LanguageModelToolDefinition(
                "calculator.basic", "Arithmetic", {"type": "object"}
            ),
        ),
    )


def state(**changes):
    values = dict(
        request=request(),
        model="configured-model",
        temperature=None,
        replay_items_json=('{"type":"function_call","call_id":"call_1"}',),
        pending_call_ids=("call_1",),
    )
    values.update(changes)
    return OpenAIContinuationState(**values)


def provider(*responses, temperature=None):
    create = AsyncMock(side_effect=list(responses))
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    settings = LanguageModelSettings(
        "openai", "configured-model", "fake-key", temperature=temperature
    )
    return OpenAILanguageModelProvider(settings, client=client), create


def test_contract_fields_identity_and_immutability():
    saved = state()
    assert tuple(field.name for field in fields(saved)) == (
        "request",
        "model",
        "temperature",
        "replay_items_json",
        "pending_call_ids",
    )
    response = LanguageModelResponse(
        None,
        "openai",
        "resolved-model",
        (LanguageModelToolCall("call_1", "calculator.basic", {}),),
    )
    result = OpenAIGenerationResult(response, saved)
    assert result.response is response
    assert result.continuation is saved
    assert tuple(field.name for field in fields(result)) == ("response", "continuation")
    for obj in (saved, result):
        assert not hasattr(obj, "__dict__")
        for field in fields(obj):
            with pytest.raises(FrozenInstanceError):
                setattr(obj, field.name, None)
    original = request()
    assert state(request=original).request is original


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("request", None, TypeError),
        ("request", {}, TypeError),
        ("model", 1, TypeError),
        ("model", "", ValueError),
        ("model", " ", ValueError),
        ("model", " model", ValueError),
        ("model", "model ", ValueError),
        ("temperature", True, TypeError),
        ("temperature", "0.2", TypeError),
        ("temperature", object(), TypeError),
        ("temperature", nan, ValueError),
        ("temperature", inf, ValueError),
        ("temperature", -inf, ValueError),
        ("temperature", -0.1, ValueError),
        ("temperature", 2.1, ValueError),
        ("replay_items_json", [], TypeError),
        ("replay_items_json", (), ValueError),
        ("replay_items_json", (1,), TypeError),
        ("replay_items_json", ("broken",), ValueError),
        ("replay_items_json", ("[]",), ValueError),
        ("replay_items_json", ("null",), ValueError),
        ("replay_items_json", ('"text"',), ValueError),
        ("replay_items_json", ('{"x":NaN}',), ValueError),
        ("replay_items_json", ('{"x":1e999}',), ValueError),
        ("pending_call_ids", [], TypeError),
        ("pending_call_ids", (), ValueError),
        ("pending_call_ids", (None,), TypeError),
        ("pending_call_ids", ("",), ValueError),
        ("pending_call_ids", (" ",), ValueError),
        ("pending_call_ids", (" call",), ValueError),
        ("pending_call_ids", ("call ",), ValueError),
        ("pending_call_ids", ("call", "call"), ValueError),
    ],
)
def test_state_validation(field, value, error):
    with pytest.raises(error):
        state(**{field: value})


@pytest.mark.parametrize("temperature", [None, 0, 2, 0.2])
def test_valid_state_preserves_primitives(temperature):
    saved = state(temperature=temperature, pending_call_ids=("Call", "call"))
    assert saved.temperature is temperature
    assert saved.pending_call_ids == ("Call", "call")


@pytest.mark.parametrize(
    "case", ["response", "continuation", "provider", "missing", "extra", "order"]
)
def test_generation_result_validation(case):
    saved = state(pending_call_ids=("a", "b"))
    response = LanguageModelResponse(
        None,
        "openai",
        "model",
        (
            LanguageModelToolCall("a", "tool", {}),
            LanguageModelToolCall("b", "tool", {}),
        ),
    )
    if case == "response":
        response = object()
    elif case == "continuation":
        saved = object()
    elif case == "provider":
        response = LanguageModelResponse("text", "other", "model")
    elif case == "missing":
        saved = None
    elif case == "extra":
        response = LanguageModelResponse("text", "openai", "model")
    else:
        saved = state(pending_call_ids=("b", "a"))
    with pytest.raises(
        TypeError if case in ("response", "continuation") else ValueError
    ):
        OpenAIGenerationResult(response, saved)


@pytest.mark.parametrize("temperature", [None, 0.2])
def test_opt_in_text_payload_equals_ordinary_generate(temperature):
    adapter, create = provider(
        raw(text="answer"), raw(text="answer"), temperature=temperature
    )
    original = request()
    ordinary = asyncio.run(adapter.generate(original))
    explicit = asyncio.run(adapter.generate_with_continuation(original))
    assert explicit.response == ordinary
    assert explicit.continuation is None
    assert create.call_args_list[0] == create.call_args_list[1]
    assert create.await_count == 2
    payload = create.call_args.kwargs
    assert payload["store"] is payload["stream"] is False
    assert set(payload) == {"model", "input", "store", "stream", "tools"} | (
        {"temperature"} if temperature is not None else set()
    )


def test_all_items_are_public_json_snapshots():
    reasoning = Item(
        {"type": "reasoning", "encrypted_content": "opaque-秘密", "summary": []}
    )
    message = Item({"type": "message", "content": [{"text": "hello"}]})
    call = function()
    adapter, create = provider(
        raw(reasoning, message, call, text="also text"), temperature=0.2
    )
    original = request()
    result = asyncio.run(adapter.generate_with_continuation(original))
    saved = result.continuation
    assert saved.request is original
    assert saved.model == "configured-model"
    assert result.response.model == "resolved-model"
    assert saved.temperature == 0.2
    assert saved.pending_call_ids == ("call_1",)
    expected = [reasoning.payload.copy(), message.payload.copy(), call.payload.copy()]
    assert [json.loads(item) for item in saved.replay_items_json] == expected
    assert "秘密" in saved.replay_items_json[0]
    for item in (reasoning, message, call):
        assert item.dumps == [{"mode": "json", "exclude_none": True}]
    reasoning.payload["encrypted_content"] = "changed"
    message.payload["content"][0]["text"] = "changed"
    assert json.loads(saved.replay_items_json[0])["encrypted_content"] == "opaque-秘密"
    assert json.loads(saved.replay_items_json[1])["content"][0]["text"] == "hello"
    create.assert_awaited_once()


@pytest.mark.parametrize(
    "output,encoded",
    [
        (None, "null"),
        ("hello", '"hello"'),
        (42, "42"),
        (1.5, "1.5"),
        (True, "true"),
        (False, "false"),
        ({"answer": 42}, '{"answer":42}'),
        ({"nested": {"items": [1, None, True]}}, '{"nested":{"items":[1,null,true]}}'),
        ("秘密", '"秘密"'),
    ],
)
def test_output_serialization_and_exact_replay_payload(output, encoded):
    adapter, create = provider(raw(text="done"))
    saved = state()
    result = asyncio.run(
        adapter.continue_with_tool_results(
            saved, (LanguageModelToolResult("call_1", output),)
        )
    )
    assert result.continuation is None
    create.assert_awaited_once_with(
        model=saved.model,
        input=[
            {"role": "user", "content": "Hello"},
            json.loads(saved.replay_items_json[0]),
            {"type": "function_call_output", "call_id": "call_1", "output": encoded},
        ],
        store=False,
        stream=False,
        tools=[
            {
                "type": "function",
                "name": "calculator.basic",
                "description": "Arithmetic",
                "parameters": {"type": "object"},
            }
        ],
    )


def test_explicit_multi_step_accumulation_and_settings_snapshot():
    first = [
        Item({"type": "reasoning", "encrypted_content": "one"}),
        function("call_1"),
    ]
    second = [
        Item({"type": "reasoning", "encrypted_content": "two"}),
        function("call_2"),
    ]
    adapter, create = provider(
        raw(*first), raw(*second), raw(text="done"), temperature=0.2
    )
    original = request()
    first_state = asyncio.run(adapter.generate_with_continuation(original)).continuation
    snapshots = first_state.replay_items_json
    adapter._settings = LanguageModelSettings(
        "openai", "changed-model", "fake", temperature=1
    )
    second_state = asyncio.run(
        adapter.continue_with_tool_results(
            first_state, (LanguageModelToolResult("call_1", 42),)
        )
    ).continuation
    expected = (
        [item.payload for item in first]
        + [
            {"type": "function_call_output", "call_id": "call_1", "output": "42"},
        ]
        + [item.payload for item in second]
    )
    assert [json.loads(item) for item in second_state.replay_items_json] == expected
    assert first_state.replay_items_json is snapshots
    assert second_state is not first_state
    assert second_state.request is original
    assert second_state.pending_call_ids == ("call_2",)
    final = asyncio.run(
        adapter.continue_with_tool_results(
            second_state, (LanguageModelToolResult("call_2", None),)
        )
    )
    assert final.continuation is None
    assert create.await_count == 3
    payload = create.call_args.kwargs
    assert payload["input"] == [{"role": "user", "content": "Hello"}] + expected + [
        {"type": "function_call_output", "call_id": "call_2", "output": "null"},
    ]
    assert payload["model"] == "configured-model"
    assert payload["temperature"] == 0.2
    assert payload["store"] is payload["stream"] is False
    assert "previous_response_id" not in payload
    assert "include" not in payload


def test_multiple_pending_results_keep_exact_order():
    adapter, create = provider(raw(function("a"), function("b")), raw(text="done"))
    saved = asyncio.run(adapter.generate_with_continuation(request())).continuation
    assert saved.pending_call_ids == ("a", "b")
    asyncio.run(
        adapter.continue_with_tool_results(
            saved,
            (
                LanguageModelToolResult("a", 1),
                LanguageModelToolResult("b", 2),
            ),
        )
    )
    assert [item["call_id"] for item in create.call_args.kwargs["input"][-2:]] == [
        "a",
        "b",
    ]
    assert create.await_count == 2


@pytest.mark.parametrize(
    "ids", [("a",), ("a", "b", "c"), ("b", "a"), ("a", "a"), ("A", "b")]
)
def test_wrong_result_sets_fail_before_provider_access(ids):
    adapter, create = provider()
    with pytest.raises(ValueError, match="must exactly match pending calls"):
        asyncio.run(
            adapter.continue_with_tool_results(
                state(pending_call_ids=("a", "b")),
                tuple(LanguageModelToolResult(call_id, None) for call_id in ids),
            )
        )
    create.assert_not_awaited()


@pytest.mark.parametrize(
    "invalid,error,message",
    [
        ([], TypeError, "OpenAI continuation tool results must be a tuple."),
        (None, TypeError, "OpenAI continuation tool results must be a tuple."),
        ((), ValueError, "OpenAI continuation requires at least one tool result."),
        (
            (object(),),
            TypeError,
            "OpenAI continuation tool results must be "
            "LanguageModelToolResult instances.",
        ),
    ],
)
def test_invalid_results(invalid, error, message):
    adapter, create = provider()
    with pytest.raises(error) as caught:
        asyncio.run(adapter.continue_with_tool_results(state(), invalid))
    assert str(caught.value) == message
    create.assert_not_awaited()


@pytest.mark.parametrize("invalid", [None, {}, object(), "request"])
def test_actual_method_source_types(invalid):
    adapter, create = provider()
    with pytest.raises(TypeError, match="request must be a LanguageModelRequest"):
        asyncio.run(adapter.generate_with_continuation(invalid))
    with pytest.raises(TypeError, match="state must be an OpenAIContinuationState"):
        asyncio.run(adapter.continue_with_tool_results(invalid, ()))
    create.assert_not_awaited()


@pytest.mark.parametrize(
    "method", ["generate_with_continuation", "continue_with_tool_results"]
)
def test_provider_error_is_chained_without_retry(method):
    error = OpenAIError("private detail")
    adapter, create = provider(error)
    args = (
        (request(),)
        if method.startswith("generate")
        else (state(), (LanguageModelToolResult("call_1", 42),))
    )
    with pytest.raises(LanguageModelProviderError) as caught:
        asyncio.run(getattr(adapter, method)(*args))
    assert str(caught.value) == "OpenAI provider request failed."
    assert caught.value.__cause__ is error
    create.assert_awaited_once()


@pytest.mark.parametrize(
    "mode", ["missing", "noncallable", "raises", "list", "scalar", "object", "nan"]
)
def test_snapshot_failures_are_safe_and_opt_in(mode):
    item = function()
    if mode == "missing":
        item = SimpleNamespace(
            type=item.type,
            call_id=item.call_id,
            name=item.name,
            arguments=item.arguments,
        )
    elif mode == "noncallable":
        item.model_dump = None
    elif mode == "raises":

        def fail(**kwargs):
            raise RuntimeError("secret")

        item.model_dump = fail
    else:
        item.payload = {
            "list": [],
            "scalar": 42,
            "object": {"x": object()},
            "nan": {"x": nan},
        }[mode]
    adapter, create = provider(raw(item), raw(item))
    assert asyncio.run(adapter.generate(request())).tool_calls[0].call_id == "call_1"
    with pytest.raises(LanguageModelProviderError) as caught:
        asyncio.run(adapter.generate_with_continuation(request()))
    assert (
        str(caught.value) == "OpenAI provider returned output that cannot be replayed."
    )
    assert caught.value.__cause__ is not None
    assert create.await_count == 2


def test_real_sdk_public_serializer_is_supported():
    item = ResponseFunctionToolCall(
        type="function_call", name="calculator.basic", arguments="{}", call_id="call_1"
    )
    adapter, _ = provider(raw(item))
    saved = asyncio.run(adapter.generate_with_continuation(request())).continuation
    assert json.loads(saved.replay_items_json[0]) == item.model_dump(
        mode="json", exclude_none=True
    )


def test_contract_dependencies_and_public_exports():
    tree = ast.parse(inspect.getsource(contracts))
    imports = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imports == {"dataclasses", "json", "math", "tuesday.language_models.base"}
    assert contracts.__all__ == ["OpenAIContinuationState", "OpenAIGenerationResult"]
    assert language_models.OpenAIContinuationState is OpenAIContinuationState
    assert language_models.OpenAIGenerationResult is OpenAIGenerationResult
