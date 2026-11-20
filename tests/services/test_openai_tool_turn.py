"""Contracts and offline integration tests for one bounded guarded tool turn."""

import ast
import asyncio
import inspect
import json
from dataclasses import FrozenInstanceError, fields, replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest
from openai.types.responses import ResponseFunctionToolCall

import tuesday.language_models.openai as provider_module
import tuesday.services as services
import tuesday.services.openai_tool_turn as turn_module
from tuesday.bridges import (
    GuardedModelToolExecutor,
    ModelToolExecution,
    ModelToolInvocationBridge,
    ModelToolResultBridge,
)
from tuesday.config import LanguageModelSettings
from tuesday.domain import MessageRole
from tuesday.language_models import (
    LanguageModelMessage,
    LanguageModelRequest,
    LanguageModelResponse,
    LanguageModelToolCall,
    LanguageModelToolDefinition,
    LanguageModelToolResult,
    OpenAIContinuationState,
    OpenAIGenerationResult,
    OpenAILanguageModelProvider,
)
from tuesday.services import (
    MultipleModelToolCallsUnsupportedError,
    OpenAIToolTurnResult,
    OpenAIToolTurnService,
)
from tuesday.tools import (
    BasicCalculatorTool,
    DeterministicToolExecutor,
    GuardedToolExecutor,
    StaticToolAuthorizationPolicy,
    ToolAuthorizationDeniedError,
    ToolAuthorizationOutcome,
    ToolConfirmationRequiredError,
    ToolExecutionError,
    ToolInvocation,
    ToolNotFoundError,
    ToolRegistry,
    ToolResult,
)

INVOCATION_ID = UUID(int=37)
ARGUMENTS = {"operation": "multiply", "left": 6, "right": 7}


def request():
    return LanguageModelRequest(
        (LanguageModelMessage(MessageRole.USER, "What is 6 times 7?"),),
        (
            LanguageModelToolDefinition(
                "calculator.basic", "Arithmetic", {"type": "object"}
            ),
        ),
    )


def function(call_id="call_calc_1", name="calculator.basic"):
    return ResponseFunctionToolCall(
        type="function_call",
        id="item_" + call_id,
        name=name,
        arguments=json.dumps(ARGUMENTS),
        call_id=call_id,
    )


def raw(*items, text=None):
    return SimpleNamespace(output=items, output_text=text, model="resolved-model")


def generation(*calls, text=None):
    response = LanguageModelResponse(text, "openai", "resolved-model", calls)
    state = (
        OpenAIContinuationState(
            request(),
            "configured-model",
            None,
            ("{}",),
            tuple(call.call_id for call in calls),
        )
        if calls
        else None
    )
    return OpenAIGenerationResult(response, state)


def execution_for(call):
    invocation = ToolInvocation(call.name, call.arguments, INVOCATION_ID)
    return ModelToolExecution(
        call, invocation, ToolResult(call.name, invocation.invocation_id, 42)
    )


def turn_values():
    call = LanguageModelToolCall("call_calc_1", "calculator.basic", ARGUMENTS)
    execution = execution_for(call)
    return dict(
        initial=generation(call, text="I will calculate that."),
        execution=execution,
        model_result=LanguageModelToolResult(call.call_id, execution.result.output),
        continued=generation(text="The result is 42."),
    )


@pytest.fixture
def stack(monkeypatch):
    """Use actual production components with spies at every existing boundary."""
    create = AsyncMock(side_effect=[raw(function()), raw(text="The result is 42.")])
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    provider = OpenAILanguageModelProvider(
        LanguageModelSettings("openai", "configured-model", "fake-key"),
        client=client,
    )
    registry = ToolRegistry()
    tool = BasicCalculatorTool()
    registry.register(tool)
    policy = StaticToolAuthorizationPolicy({tool.name: ToolAuthorizationOutcome.ALLOW})
    factory = Mock(return_value=INVOCATION_ID)
    bridge = ModelToolInvocationBridge(invocation_id_factory=factory)
    executor = GuardedModelToolExecutor(
        bridge, GuardedToolExecutor(policy, DeterministicToolExecutor(registry))
    )
    result_bridge = ModelToolResultBridge()
    authorize_method = StaticToolAuthorizationPolicy.authorize
    events = Mock()
    spies = {}
    for name, target, method, asynchronous in (
        ("initial", provider, "generate_with_continuation", True),
        ("continued", provider, "continue_with_tool_results", True),
        ("execution", GuardedModelToolExecutor, "execute", True),
        ("invocation", ModelToolInvocationBridge, "to_invocation", False),
        ("authorize", StaticToolAuthorizationPolicy, "authorize", True),
        ("lookup", registry, "get", False),
        ("tool", BasicCalculatorTool, "execute", True),
        ("result", ModelToolResultBridge, "to_model_result", False),
    ):
        bound = {
            "execution": executor,
            "invocation": bridge,
            "authorize": policy,
            "tool": tool,
            "result": result_bridge,
        }.get(name, target)
        spy_type = AsyncMock if asynchronous else Mock
        spy = spy_type(wraps=getattr(bound, method))
        monkeypatch.setattr(target, method, spy)
        events.attach_mock(spy, name)
        spies[name] = spy
    ordinary = AsyncMock(side_effect=AssertionError("ordinary generate is forbidden"))
    monkeypatch.setattr(provider, "generate", ordinary)
    return SimpleNamespace(
        provider=provider,
        executor=executor,
        result_bridge=result_bridge,
        policy=policy,
        registry=registry,
        create=create,
        factory=factory,
        events=events,
        spies=spies,
        ordinary=ordinary,
        authorize_method=authorize_method,
        service=OpenAIToolTurnService(provider, executor, result_bridge),
    )


def run(stack, model_request=None):
    return asyncio.run(
        stack.service.run_once(request() if model_request is None else model_request)
    )


def assert_stopped(stack, expected):
    assert [call[0] for call in stack.events.mock_calls] == expected
    stack.ordinary.assert_not_awaited()


def test_result_is_frozen_slotted_and_preserves_all_stage_identities():
    values = turn_values()
    result = OpenAIToolTurnResult(**values)
    assert tuple(field.name for field in fields(result)) == (
        "initial",
        "execution",
        "model_result",
        "continued",
    )
    assert not hasattr(result, "__dict__")
    for name, value in values.items():
        assert getattr(result, name) is value
        with pytest.raises(FrozenInstanceError):
            setattr(result, name, None)
    assert result.execution.tool_call is result.initial.response.tool_calls[0]
    assert result.initial.continuation is values["initial"].continuation


def test_text_only_result():
    initial = generation(text="Hello")
    result = OpenAIToolTurnResult(initial, None, None, None)
    assert result.initial is initial
    assert result.initial.continuation is None
    assert result.execution is result.model_result is result.continued is None


@pytest.mark.parametrize("field", ["initial", "execution", "model_result", "continued"])
@pytest.mark.parametrize("invalid", [{}, object(), "invalid"])
def test_result_rejects_wrong_stage_types(field, invalid):
    values = turn_values()
    values[field] = invalid
    with pytest.raises(TypeError, match=field + " must be"):
        OpenAIToolTurnResult(**values)


@pytest.mark.parametrize("field", ["initial", "execution", "model_result", "continued"])
def test_result_rejects_missing_single_tool_stages(field):
    values = turn_values()
    values[field] = None
    with pytest.raises(ValueError if field == "continued" else TypeError):
        OpenAIToolTurnResult(**values)


@pytest.mark.parametrize("field", ["execution", "model_result", "continued"])
def test_text_only_result_rejects_unexpected_stages(field):
    values = dict.fromkeys(("execution", "model_result", "continued"))
    values[field] = turn_values()[field]
    with pytest.raises(ValueError, match="Text-only tool turn"):
        OpenAIToolTurnResult(initial=generation(text="Hello"), **values)


@pytest.mark.parametrize("mismatch", ["identity", "call_id", "output", "multiple"])
def test_result_rejects_correlation_mismatches_and_multiple_calls(mismatch):
    values = turn_values()
    call = values["execution"].tool_call
    if mismatch == "identity":
        # Equal values still fail: the original source object must be retained.
        values["execution"] = execution_for(replace(call))
    elif mismatch == "call_id":
        values["model_result"] = LanguageModelToolResult("other", 42)
    elif mismatch == "output":
        values["model_result"] = LanguageModelToolResult(call.call_id, 43)
    else:
        values["initial"] = generation(call, replace(call, call_id="other"))
    with pytest.raises(ValueError):
        OpenAIToolTurnResult(**values)


def test_constructor_identity_slots_and_async_surface(stack):
    assert stack.service._provider is stack.provider
    assert stack.service._executor is stack.executor
    assert stack.service._result_bridge is stack.result_bridge
    assert OpenAIToolTurnService.__slots__ == (
        "_provider",
        "_executor",
        "_result_bridge",
    )
    assert not hasattr(stack.service, "__dict__")
    assert inspect.iscoroutinefunction(OpenAIToolTurnService.run_once)
    assert tuple(inspect.signature(OpenAIToolTurnService.run_once).parameters) == (
        "self",
        "request",
    )
    assert tuple(inspect.signature(OpenAIToolTurnService).parameters) == (
        "provider",
        "executor",
        "result_bridge",
    )


@pytest.mark.parametrize("dependency", ["provider", "executor", "result_bridge"])
@pytest.mark.parametrize(
    "invalid",
    [
        None,
        {},
        object(),
        SimpleNamespace(
            generate_with_continuation=AsyncMock(),
            execute=AsyncMock(),
            to_model_result=Mock(),
        ),
    ],
)
def test_constructor_requires_actual_dependency_types(stack, dependency, invalid):
    values = dict(
        provider=stack.provider,
        executor=stack.executor,
        result_bridge=stack.result_bridge,
    )
    values[dependency] = invalid
    with pytest.raises(TypeError, match=dependency + " must be"):
        OpenAIToolTurnService(**values)
    assert_stopped(stack, [])


@pytest.mark.parametrize("invalid", [None, {}, object(), "request"])
def test_invalid_request_fails_before_dependencies(stack, invalid):
    with pytest.raises(TypeError) as caught:
        asyncio.run(stack.service.run_once(invalid))
    assert str(caught.value) == "request must be a LanguageModelRequest."
    stack.create.assert_not_awaited()
    assert_stopped(stack, [])


@pytest.mark.parametrize(
    "invalid",
    [None, {}, object(), "result", LanguageModelResponse("Hello", "openai", "model")],
)
def test_invalid_initial_result_fails_before_execution(stack, invalid):
    stack.spies["initial"].return_value = invalid
    with pytest.raises(TypeError, match="initial must be an OpenAIGenerationResult"):
        run(stack)
    assert_stopped(stack, ["initial"])
    stack.factory.assert_not_called()


@pytest.mark.parametrize("stubbed", [False, True])
def test_text_only_path_calls_only_initial_generation(stack, stubbed):
    stack.create.side_effect = [raw(text="Hello")]
    expected = generation(text="Hello")
    if stubbed:
        stack.spies["initial"].return_value = expected
    original = request()
    result = run(stack, original)
    stack.spies["initial"].assert_awaited_once_with(original)
    if stubbed:
        assert result.initial is expected
        stack.create.assert_not_awaited()
    else:
        stack.create.assert_awaited_once()
    assert result.initial.response.content == "Hello"
    assert result.execution is result.model_result is result.continued is None
    assert_stopped(stack, ["initial"])
    stack.factory.assert_not_called()


def test_service_retains_exact_objects_returned_by_each_boundary(stack):
    values = turn_values()
    stack.spies["initial"].return_value = values["initial"]
    stack.spies["execution"].return_value = values["execution"]
    stack.spies["result"].return_value = values["model_result"]
    stack.spies["continued"].return_value = values["continued"]
    result = run(stack)
    for name, value in values.items():
        assert getattr(result, name) is value
    assert_stopped(stack, ["initial", "execution", "result", "continued"])


@pytest.mark.parametrize("mixed", [False, True])
def test_real_calculator_turn_and_exact_guarded_call_order(stack, mixed):
    item = function()
    content = "I will calculate that." if mixed else None
    stack.create.side_effect = [raw(item, text=content), raw(text="The result is 42.")]
    original = request()
    result = run(stack, original)
    assert result.initial.response.content == content
    assert result.continued.response.content == "The result is 42."
    assert result.continued.continuation is None
    call = result.initial.response.tool_calls[0]
    assert result.execution.tool_call is call
    assert result.execution.result.output == result.model_result.output == 42
    assert result.model_result.call_id == call.call_id == "call_calc_1"
    assert result.execution.invocation.invocation_id is INVOCATION_ID
    assert str(INVOCATION_ID) != result.model_result.call_id
    stack.factory.assert_called_once_with()
    stack.spies["initial"].assert_awaited_once_with(original)
    stack.spies["execution"].assert_awaited_once_with(call)
    stack.spies["result"].assert_called_once_with(result.execution)
    stack.spies["continued"].assert_awaited_once_with(
        result.initial.continuation, (result.model_result,)
    )
    assert stack.spies["execution"].call_args.args[0] is call
    assert stack.spies["result"].call_args.args[0] is result.execution
    state, results = stack.spies["continued"].call_args.args
    assert state is result.initial.continuation
    assert isinstance(results, tuple) and results[0] is result.model_result
    assert stack.create.await_count == 2
    first, second = [call.kwargs for call in stack.create.call_args_list]
    assert first["input"] == [{"role": "user", "content": "What is 6 times 7?"}]
    assert second["input"] == first["input"] + [
        item.model_dump(mode="json", exclude_none=True),
        {"type": "function_call_output", "call_id": "call_calc_1", "output": "42"},
    ]
    for payload in (first, second):
        assert payload["store"] is payload["stream"] is False
        assert "previous_response_id" not in payload
    assert first["tools"] == second["tools"]
    assert_stopped(
        stack,
        [
            "initial",
            "execution",
            "invocation",
            "authorize",
            "lookup",
            "tool",
            "result",
            "continued",
        ],
    )


def test_multiple_initial_calls_rejected_before_invocation_or_authorization(stack):
    stack.create.side_effect = [raw(function("call_a"), function("call_b"))]
    with pytest.raises(MultipleModelToolCallsUnsupportedError) as caught:
        run(stack)
    assert str(caught.value) == "OpenAI tool turn supports at most one model tool call."
    stack.create.assert_awaited_once()
    stack.factory.assert_not_called()
    assert_stopped(stack, ["initial"])


@pytest.mark.parametrize(
    "outcome,error_type",
    [
        (ToolAuthorizationOutcome.DENY, ToolAuthorizationDeniedError),
        (ToolAuthorizationOutcome.REQUIRE_CONFIRMATION, ToolConfirmationRequiredError),
        (None, ToolAuthorizationDeniedError),
    ],
)
def test_blocked_and_unconfigured_tools_never_lookup_or_continue(
    stack, outcome, error_type
):
    policy = StaticToolAuthorizationPolicy(
        {} if outcome is None else {"calculator.basic": outcome}
    )
    decisions = []

    async def authorize(invocation):
        decision = await stack.authorize_method(policy, invocation)
        decisions.append(decision)
        return decision

    stack.spies["authorize"].side_effect = authorize
    with pytest.raises(error_type) as caught:
        run(stack)
    assert caught.value.decision is decisions[0]
    assert caught.value.decision.invocation_id is INVOCATION_ID
    assert caught.value.decision.outcome is (
        ToolAuthorizationOutcome.DENY if outcome is None else outcome
    )
    stack.create.assert_awaited_once()
    stack.factory.assert_called_once_with()
    assert_stopped(stack, ["initial", "execution", "invocation", "authorize"])


def test_explicitly_allowed_unknown_tool_authorizes_before_failed_lookup(stack):
    stack.create.side_effect = [raw(function(name="unknown.tool"))]
    policy = StaticToolAuthorizationPolicy(
        {"unknown.tool": ToolAuthorizationOutcome.ALLOW}
    )

    async def authorize(invocation):
        return await stack.authorize_method(policy, invocation)

    stack.spies["authorize"].side_effect = authorize
    with pytest.raises(ToolNotFoundError):
        run(stack)
    stack.create.assert_awaited_once()
    stack.spies["lookup"].assert_called_once_with("unknown.tool")
    assert_stopped(stack, ["initial", "execution", "invocation", "authorize", "lookup"])


@pytest.mark.parametrize(
    "error", [ToolExecutionError("tool failed"), RuntimeError("tool failed")]
)
def test_authorized_tool_failure_propagates_exact_exception_once(stack, error):
    stack.spies["tool"].side_effect = error
    with pytest.raises(type(error)) as caught:
        run(stack)
    assert caught.value is error
    stack.create.assert_awaited_once()
    stack.spies["tool"].assert_awaited_once()
    assert_stopped(
        stack, ["initial", "execution", "invocation", "authorize", "lookup", "tool"]
    )


@pytest.mark.parametrize("count", [1, 2])
def test_further_continued_tool_calls_remain_visible_without_execution(stack, count):
    further = [function("call_" + str(index)) for index in range(count)]
    stack.create.side_effect = [raw(function()), raw(*further, text="More work")]
    result = run(stack)
    assert result.execution.tool_call is result.initial.response.tool_calls[0]
    assert result.model_result.call_id == "call_calc_1"
    assert result.continued.response.content == "More work"
    assert (
        tuple(call.call_id for call in result.continued.response.tool_calls)
        == (result.continued.continuation.pending_call_ids)
        == tuple(item.call_id for item in further)
    )
    assert result.continued.continuation.request is result.initial.continuation.request
    assert stack.create.await_count == 2
    stack.spies["tool"].assert_awaited_once()
    stack.spies["execution"].assert_awaited_once()
    stack.spies["continued"].assert_awaited_once()


@pytest.mark.parametrize("invalid", [None, {}, object(), "execution", "equal call"])
def test_invalid_or_unrelated_execution_stops_before_result_bridge(stack, invalid):
    call = LanguageModelToolCall("call_calc_1", "calculator.basic", ARGUMENTS)
    stack.spies["initial"].return_value = generation(call)
    if invalid == "equal call":
        invalid = execution_for(replace(call))
    stack.spies["execution"].return_value = invalid
    with pytest.raises(
        ValueError if isinstance(invalid, ModelToolExecution) else TypeError
    ):
        run(stack)
    assert_stopped(stack, ["initial", "execution"])


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        {},
        object(),
        "result",
        LanguageModelToolResult("wrong_id", 42),
        LanguageModelToolResult("call_calc_1", 43),
    ],
)
def test_invalid_model_result_stops_before_continuation(stack, invalid):
    stack.spies["result"].return_value = invalid
    with pytest.raises(
        ValueError if isinstance(invalid, LanguageModelToolResult) else TypeError
    ):
        run(stack)
    stack.create.assert_awaited_once()
    assert_stopped(
        stack,
        ["initial", "execution", "invocation", "authorize", "lookup", "tool", "result"],
    )


@pytest.mark.parametrize(
    "invalid",
    [None, {}, object(), "result", LanguageModelResponse("Hello", "openai", "model")],
)
def test_invalid_continued_result_fails_after_execution_without_rollback(
    stack, invalid
):
    # Execution already succeeded: validation cannot undo any tool side effects.
    stack.spies["continued"].return_value = invalid
    with pytest.raises(TypeError, match="continued must be an OpenAIGenerationResult"):
        run(stack)
    stack.spies["tool"].assert_awaited_once()
    stack.spies["continued"].assert_awaited_once()
    stack.create.assert_awaited_once()


@pytest.mark.parametrize("stage", ["initial", "execution", "result", "continued"])
def test_dependency_failures_propagate_unchanged_without_retry(stack, stage):
    error = RuntimeError(stage + " failed")
    stack.spies[stage].side_effect = error
    with pytest.raises(RuntimeError) as caught:
        run(stack)
    assert caught.value is error
    expected = [
        "initial",
        "execution",
        "invocation",
        "authorize",
        "lookup",
        "tool",
        "result",
        "continued",
    ]
    assert_stopped(stack, expected[: expected.index(stage) + 1])
    stack.spies[stage].assert_called_once()


def imports_of(module):
    return {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(ast.parse(inspect.getsource(module)))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }


def test_service_dependencies_are_limited_and_provider_cannot_execute_tools():
    assert imports_of(turn_module) == {
        "dataclasses",
        "tuesday.bridges",
        "tuesday.language_models",
    }
    assert not any(
        name.startswith(("tuesday.bridges", "tuesday.tools", "tuesday.services"))
        for name in imports_of(provider_module)
    )
    tree = ast.parse(inspect.getsource(OpenAIToolTurnService))
    assert not any(
        isinstance(node, (ast.While, ast.For, ast.AsyncFor)) for node in ast.walk(tree)
    )
    called = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert sorted(called) == sorted(
        [
            "generate_with_continuation",
            "execute",
            "to_model_result",
            "continue_with_tool_results",
        ]
    )
    assert "run_once" not in called


def test_exact_public_exports_preserve_stateful_service():
    assert services.__all__ == [
        "MultipleModelToolCallsUnsupportedError",
        "OpenAIToolTurnResult",
        "OpenAIToolTurnService",
        "StatefulConversationService",
    ]
    for name in services.__all__[:3]:
        assert getattr(services, name) is getattr(turn_module, name)
    assert (
        services.StatefulConversationService.__module__
        == "tuesday.services.conversation"
    )
