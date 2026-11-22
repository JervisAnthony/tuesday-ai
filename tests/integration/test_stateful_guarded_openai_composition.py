"""Offline integration of caller-owned history and bounded guarded tool turns."""

import ast
import asyncio
import inspect
import json
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import pytest
from openai.types.responses import ResponseFunctionToolCall

import tuesday.composition as composition_module
from tuesday.agents import (
    AgentRegistry,
    ConversationalAgent,
    GuardedOpenAIConversationalAgent,
    IncompleteGuardedToolTurnError,
    ModelBackedConversationalAgent,
    UnsupportedModelToolCallError,
)
from tuesday.bridges import (
    GuardedModelToolExecutor,
    ModelToolInvocationBridge,
    ModelToolResultBridge,
)
from tuesday.composition import (
    create_default_orchestrator,
    create_guarded_openai_orchestrator,
    create_model_backed_orchestrator,
    create_stateful_guarded_openai_service,
)
from tuesday.config import LanguageModelSettings
from tuesday.conversations import (
    BaseConversationRepository,
    ConversationRepositoryError,
    InMemoryConversationRepository,
)
from tuesday.domain import (
    ConversationContext,
    MessageRole,
    TuesdayRequest,
    TuesdayResponse,
)
from tuesday.language_models import (
    LanguageModelProviderError,
    LanguageModelToolDefinition,
    OpenAILanguageModelProvider,
)
from tuesday.orchestration import TuesdayOrchestrator
from tuesday.preparation import DirectiveRequestPreparer, RequestPreparationError
from tuesday.prompting import (
    DEFAULT_CONVERSATIONAL_SYSTEM_PROMPT,
    ConversationalPromptRenderer,
)
from tuesday.routing import DeterministicRouter, NoRouteFoundError
from tuesday.services.conversation import StatefulConversationService
from tuesday.services.openai_tool_turn import (
    MultipleModelToolCallsUnsupportedError,
    OpenAIToolTurnService,
    UnadvertisedModelToolCallError,
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
    ToolNotFoundError,
    ToolRegistry,
)

INVOCATION_ID = UUID(int=41)
TOOLS = (LanguageModelToolDefinition("calculator.basic", "Arithmetic", {}),)


def function(call_id="call_stateful_41", name="calculator.basic"):
    return ResponseFunctionToolCall(
        type="function_call",
        id="item_" + call_id,
        call_id=call_id,
        name=name,
        arguments=json.dumps({"operation": "multiply", "left": 6, "right": 7}),
    )


def raw(*items, text=None):
    return SimpleNamespace(output=items, output_text=text, model="resolved-model")


def handle(stack, request):
    return asyncio.run(stack.stateful.handle(request))


def message_values(context):
    return [(message.role, message.content) for message in context.messages]


def assert_correlated(response, request):
    assert isinstance(response, TuesdayResponse)
    assert response.conversation_id == request.conversation_id
    assert response.request_id == request.request_id
    assert isinstance(response.response_id, UUID)


@pytest.fixture
def outcome():
    return ToolAuthorizationOutcome.ALLOW


@pytest.fixture
def stack(monkeypatch, outcome):
    """Real application and guarded components with a fake Responses client."""
    create = AsyncMock(side_effect=[raw(function()), raw(text="The result is 42.")])
    provider = OpenAILanguageModelProvider(
        LanguageModelSettings("openai", "configured-model", "fake-key"),
        client=SimpleNamespace(responses=SimpleNamespace(create=create)),
    )
    registry = ToolRegistry()
    calculator = BasicCalculatorTool()
    registry.register(calculator)
    policy = StaticToolAuthorizationPolicy(
        {
            "calculator.basic": outcome,
            "unknown.tool": ToolAuthorizationOutcome.ALLOW,
        }
    )
    factory = Mock(return_value=INVOCATION_ID)
    invocation_bridge = ModelToolInvocationBridge(invocation_id_factory=factory)
    executor = GuardedModelToolExecutor(
        invocation_bridge,
        GuardedToolExecutor(policy, DeterministicToolExecutor(registry)),
    )
    result_bridge = ModelToolResultBridge()
    tool_turn = OpenAIToolTurnService(provider, executor, result_bridge)
    repository = InMemoryConversationRepository()
    read = repository.get_context
    events = Mock()
    spies = {}
    for name, target, bound, method, asynchronous in (
        ("get", InMemoryConversationRepository, repository, "get_context", True),
        ("append", InMemoryConversationRepository, repository, "append_messages", True),
        ("turn", OpenAIToolTurnService, tool_turn, "run_once", True),
        ("initial", provider, provider, "generate_with_continuation", True),
        ("continued", provider, provider, "continue_with_tool_results", True),
        ("execution", GuardedModelToolExecutor, executor, "execute", True),
        (
            "invocation",
            ModelToolInvocationBridge,
            invocation_bridge,
            "to_invocation",
            False,
        ),
        ("authorize", StaticToolAuthorizationPolicy, policy, "authorize", True),
        ("lookup", registry, registry, "get", False),
        ("tool", BasicCalculatorTool, calculator, "execute", True),
        ("result", ModelToolResultBridge, result_bridge, "to_model_result", False),
    ):
        spy = (AsyncMock if asynchronous else Mock)(wraps=getattr(bound, method))
        monkeypatch.setattr(target, method, spy)
        events.attach_mock(spy, name)
        spies[name] = spy
    stateful = create_stateful_guarded_openai_service(tool_turn, TOOLS, repository)
    return SimpleNamespace(
        stateful=stateful,
        tool_turn=tool_turn,
        repository=repository,
        read=read,
        provider=provider,
        create=create,
        factory=factory,
        spies=spies,
        events=events,
    )


def establish_history(stack):
    """Complete a real first turn, then observe only the subsequent attempt."""
    request = TuesdayRequest("/chat Earlier successful request")
    stack.create.side_effect = [raw(text="Earlier successful reply")]
    handle(stack, request)
    before = asyncio.run(stack.read(request.conversation_id))
    stack.events.reset_mock()
    stack.create.reset_mock()
    return request, before


def test_factory_contract_identity_and_side_effect_free_construction(stack):
    renderer = ConversationalPromptRenderer("Custom stateful composition prompt")
    service = create_stateful_guarded_openai_service(
        stack.tool_turn, TOOLS, stack.repository, renderer=renderer
    )
    assert type(service) is StatefulConversationService
    assert service._repository is stack.repository
    assert isinstance(stack.repository, BaseConversationRepository)
    runtime = service._orchestrator
    assert type(runtime) is TuesdayOrchestrator
    assert type(runtime._registry) is AgentRegistry
    assert runtime._registry.names == ("conversation",)
    assert len(runtime._registry) == 1
    agent = runtime._registry.get("conversation")
    assert type(agent) is GuardedOpenAIConversationalAgent
    assert agent._tool_turn_service is stack.tool_turn
    assert agent._tools is TOOLS
    assert agent._renderer is renderer
    assert type(runtime._router) is DeterministicRouter
    assert runtime._router.routes == (
        ("chat", "conversation"),
        ("conversation", "conversation"),
    )
    assert type(runtime._request_preparer) is DirectiveRequestPreparer
    assert stack.events.mock_calls == []
    stack.create.assert_not_awaited()
    parameters = inspect.signature(create_stateful_guarded_openai_service).parameters
    assert tuple(parameters) == ("tool_turn_service", "tools", "repository", "renderer")
    assert parameters["renderer"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["renderer"].default is None
    assert (
        inspect.signature(create_stateful_guarded_openai_service).return_annotation
        is StatefulConversationService
    )


def test_fresh_graphs_retain_shared_explicit_dependencies(stack):
    first = stack.stateful
    second = create_stateful_guarded_openai_service(
        stack.tool_turn, TOOLS, stack.repository
    )
    assert first is not second
    assert first._repository is second._repository is stack.repository
    left, right = first._orchestrator, second._orchestrator
    assert left is not right
    for name in ("_registry", "_router", "_request_preparer"):
        assert getattr(left, name) is not getattr(right, name)
    a, b = left._registry.get("conversation"), right._registry.get("conversation")
    assert a is not b
    assert a._tool_turn_service is b._tool_turn_service is stack.tool_turn
    assert a._tools is b._tools is TOOLS
    assert isinstance(a._renderer, ConversationalPromptRenderer)
    assert a._renderer is not b._renderer
    assert stack.events.mock_calls == []


@pytest.mark.parametrize(
    "invalid,error_type",
    [
        ("service-none", TypeError),
        ("service-object", TypeError),
        ("tools-list", TypeError),
        ("tools-empty", ValueError),
        ("tools-invalid", TypeError),
        ("tools-duplicate", ValueError),
        ("renderer", TypeError),
        ("repository-none", TypeError),
        ("repository-object", TypeError),
    ],
)
def test_invalid_dependencies_use_existing_validation_without_execution(
    stack, invalid, error_type
):
    values = dict(
        tool_turn_service=stack.tool_turn,
        tools=TOOLS,
        repository=stack.repository,
        renderer=None,
    )
    if invalid.startswith("service"):
        values["tool_turn_service"] = None if invalid == "service-none" else object()
    elif invalid.startswith("repository"):
        values["repository"] = None if invalid == "repository-none" else object()
    elif invalid == "renderer":
        values["renderer"] = object()
    else:
        values["tools"] = {
            "tools-list": list(TOOLS),
            "tools-empty": (),
            "tools-invalid": (object(),),
            "tools-duplicate": (TOOLS[0], replace(TOOLS[0])),
        }[invalid]
    with pytest.raises(error_type):
        create_stateful_guarded_openai_service(**values)
    assert stack.events.mock_calls == []
    stack.create.assert_not_awaited()


@pytest.mark.parametrize("custom", [False, True])
def test_two_text_turns_reuse_original_history_and_preserve_ids(stack, custom):
    renderer = ConversationalPromptRenderer("  Custom stateful prompt\nline two  ")
    if custom:
        stack.stateful = create_stateful_guarded_openai_service(
            stack.tool_turn, TOOLS, stack.repository, renderer=renderer
        )
    stack.create.side_effect = [
        raw(text="  Hello.\nWelcome.  "),
        raw(text="Earlier greeting."),
    ]
    first = TuesdayRequest("/chat Hello TUESDAY")
    second = TuesdayRequest(
        "/conversation What did I say earlier?", first.conversation_id
    )
    originals = (replace(first), replace(second))
    reply_one = handle(stack, first)
    snapshot = asyncio.run(stack.read(first.conversation_id))
    assert message_values(snapshot) == [
        (MessageRole.USER, first.content),
        (MessageRole.ASSISTANT, reply_one.content),
    ]
    reply_two = handle(stack, second)
    for reply, request in ((reply_one, first), (reply_two, second)):
        assert_correlated(reply, request)
    assert (first, second) == originals
    assert reply_one.response_id != reply_two.response_id
    assert reply_one.content == "  Hello.\nWelcome.  "
    assert reply_two.content == "Earlier greeting."
    system = renderer.system_prompt if custom else DEFAULT_CONVERSATIONAL_SYSTEM_PROMPT
    first_payload, second_payload = [
        call.kwargs for call in stack.create.call_args_list
    ]
    assert first_payload["input"] == [
        {"role": "system", "content": system},
        {"role": "user", "content": "Hello TUESDAY"},
    ]
    assert second_payload["input"] == [
        {"role": "system", "content": system},
        {"role": "user", "content": first.content},
        {"role": "assistant", "content": reply_one.content},
        {"role": "user", "content": "What did I say earlier?"},
    ]
    saved = asyncio.run(stack.read(first.conversation_id))
    assert message_values(saved) == [
        (MessageRole.USER, first.content),
        (MessageRole.ASSISTANT, reply_one.content),
        (MessageRole.USER, second.content),
        (MessageRole.ASSISTANT, reply_two.content),
    ]
    assert saved.messages[:2] == snapshot.messages
    assert len(snapshot.messages) == 2
    assert [call[0] for call in stack.events.mock_calls] == [
        "get",
        "turn",
        "initial",
        "append",
        "get",
        "turn",
        "initial",
        "append",
    ]
    assert stack.spies["get"].await_count == stack.spies["append"].await_count == 2
    for call in stack.spies["turn"].call_args_list:
        assert call.args[0].tools is TOOLS
    stack.spies["tool"].assert_not_awaited()
    stack.spies["continued"].assert_not_awaited()


def test_calculator_turn_then_text_turn_reuses_only_completed_conversation(stack):
    item = function()
    stack.create.side_effect = [
        raw(item, text="I will calculate that."),
        raw(text="The result is 42."),
        raw(text="The earlier result was 42."),
    ]
    first = TuesdayRequest("/chat What is 6 times 7?")
    reply_one = handle(stack, first)
    second = TuesdayRequest("/conversation What was the result?", first.conversation_id)
    reply_two = handle(stack, second)
    assert_correlated(reply_one, first)
    assert_correlated(reply_two, second)
    assert reply_one.content == "The result is 42."
    assert reply_two.content == "The earlier result was 42."
    assert stack.create.await_count == 3
    stack.spies["tool"].assert_awaited_once()
    stack.spies["authorize"].assert_awaited_once()
    stack.spies["result"].assert_called_once()
    stack.spies["continued"].assert_awaited_once()
    execution = stack.spies["result"].call_args.args[0]
    assert execution.result.output == 42
    assert execution.tool_call.call_id == "call_stateful_41"
    assert execution.invocation.invocation_id is INVOCATION_ID
    assert str(INVOCATION_ID) != execution.tool_call.call_id
    state, results = stack.spies["continued"].call_args.args
    assert state.request is stack.spies["turn"].call_args_list[0].args[0]
    assert results[0].call_id == "call_stateful_41"
    assert results[0].output == 42
    initial, continued, last = [call.kwargs for call in stack.create.call_args_list]
    assert continued["input"] == initial["input"] + [
        item.model_dump(mode="json", exclude_none=True),
        {"type": "function_call_output", "call_id": "call_stateful_41", "output": "42"},
    ]
    assert last["input"] == [
        {"role": "system", "content": DEFAULT_CONVERSATIONAL_SYSTEM_PROMPT},
        {"role": "user", "content": first.content},
        {"role": "assistant", "content": reply_one.content},
        {"role": "user", "content": "What was the result?"},
    ]
    for payload in (initial, continued, last):
        assert payload["store"] is payload["stream"] is False
        assert "previous_response_id" not in payload
        assert [tool["name"] for tool in payload["tools"]] == ["calculator.basic"]
    saved = asyncio.run(stack.read(first.conversation_id))
    assert message_values(saved) == [
        (MessageRole.USER, first.content),
        (MessageRole.ASSISTANT, reply_one.content),
        (MessageRole.USER, second.content),
        (MessageRole.ASSISTANT, reply_two.content),
    ]
    assert [call[0] for call in stack.events.mock_calls] == [
        "get",
        "turn",
        "initial",
        "execution",
        "invocation",
        "authorize",
        "lookup",
        "tool",
        "result",
        "continued",
        "append",
        "get",
        "turn",
        "initial",
        "append",
    ]


def test_interleaved_conversation_ids_keep_histories_separate(stack):
    conversation_a, conversation_b = uuid4(), uuid4()
    requests = (
        TuesdayRequest("/chat A first question", conversation_a),
        TuesdayRequest("/chat B first question", conversation_b),
        TuesdayRequest("/conversation A second question", conversation_a),
        TuesdayRequest("/conversation B second question", conversation_b),
    )
    answers = ("A first answer", "B first answer", "A second answer", "B second answer")
    stack.create.side_effect = [raw(text=answer) for answer in answers]
    for request, answer in zip(requests, answers, strict=True):
        response = handle(stack, request)
        assert_correlated(response, request)
        assert response.content == answer
    payloads = [call.kwargs["input"] for call in stack.create.call_args_list]
    system = {"role": "system", "content": DEFAULT_CONVERSATIONAL_SYSTEM_PROMPT}
    assert payloads[0] == [system, {"role": "user", "content": "A first question"}]
    assert payloads[1] == [system, {"role": "user", "content": "B first question"}]
    for first_index, second_index in ((0, 2), (1, 3)):
        assert payloads[second_index] == [
            system,
            {"role": "user", "content": requests[first_index].content},
            {"role": "assistant", "content": answers[first_index]},
            {
                "role": "user",
                "content": requests[second_index].content.split(" ", 1)[1],
            },
        ]
        saved = asyncio.run(stack.read(requests[first_index].conversation_id))
        assert message_values(saved) == [
            (MessageRole.USER, requests[first_index].content),
            (MessageRole.ASSISTANT, answers[first_index]),
            (MessageRole.USER, requests[second_index].content),
            (MessageRole.ASSISTANT, answers[second_index]),
        ]
    assert stack.spies["append"].await_count == 4
    stack.spies["tool"].assert_not_awaited()


@pytest.mark.parametrize(
    "failure,outcome,error_type",
    [
        ("deny", ToolAuthorizationOutcome.DENY, ToolAuthorizationDeniedError),
        (
            "confirmation",
            ToolAuthorizationOutcome.REQUIRE_CONFIRMATION,
            ToolConfirmationRequiredError,
        ),
        (
            "unadvertised",
            ToolAuthorizationOutcome.ALLOW,
            UnadvertisedModelToolCallError,
        ),
        (
            "multiple",
            ToolAuthorizationOutcome.ALLOW,
            MultipleModelToolCallsUnsupportedError,
        ),
        ("incomplete", ToolAuthorizationOutcome.ALLOW, IncompleteGuardedToolTurnError),
        (
            "incomplete-mixed",
            ToolAuthorizationOutcome.ALLOW,
            IncompleteGuardedToolTurnError,
        ),
        ("provider", ToolAuthorizationOutcome.ALLOW, LanguageModelProviderError),
        ("lookup", ToolAuthorizationOutcome.ALLOW, ToolNotFoundError),
        ("tool", ToolAuthorizationOutcome.ALLOW, ToolExecutionError),
        ("result", ToolAuthorizationOutcome.ALLOW, RuntimeError),
        ("continued", ToolAuthorizationOutcome.ALLOW, LanguageModelProviderError),
    ],
)
def test_failed_later_turn_preserves_existing_successful_history(
    stack, failure, error_type
):
    first, before = establish_history(stack)
    request = TuesdayRequest(
        "/conversation Attempt another operation", first.conversation_id
    )
    stack.create.side_effect = [raw(function()), raw(text="Completed result")]
    error = None
    if failure == "unadvertised":
        stack.create.side_effect = [raw(function(name="unknown.tool"))]
    elif failure == "multiple":
        stack.create.side_effect = [raw(function(), function("call_second"))]
    elif failure.startswith("incomplete"):
        stack.create.side_effect = [
            raw(function()),
            raw(
                function("call_more"),
                text="Partial response" if failure.endswith("mixed") else None,
            ),
        ]
    elif failure == "lookup":
        tools = (LanguageModelToolDefinition("unknown.tool", "Unknown", {}),)
        stack.stateful = create_stateful_guarded_openai_service(
            stack.tool_turn, tools, stack.repository
        )
        stack.create.side_effect = [raw(function(name="unknown.tool"))]
    elif failure in ("provider", "tool", "result", "continued"):
        error = error_type(failure + " failed")
        stage = "initial" if failure == "provider" else failure
        stack.spies[stage].side_effect = error
    with pytest.raises(error_type) as caught:
        handle(stack, request)
    if error is not None:
        assert caught.value is error
    after = asyncio.run(stack.read(first.conversation_id))
    assert after.messages is before.messages
    assert message_values(after) == [
        (MessageRole.USER, first.content),
        (MessageRole.ASSISTANT, "Earlier successful reply"),
    ]
    stack.spies["get"].assert_awaited_once_with(first.conversation_id)
    stack.spies["append"].assert_not_awaited()
    stack.spies["turn"].assert_awaited_once()
    if failure in ("unadvertised", "multiple", "provider"):
        stack.factory.assert_not_called()
        stack.spies["authorize"].assert_not_awaited()
    if failure.startswith("incomplete") or failure in ("result", "continued"):
        stack.spies["tool"].assert_awaited_once()
        assert stack.spies["result"].call_args.args[0].result.output == 42
    else:
        assert stack.spies["tool"].await_count == (1 if failure == "tool" else 0)
    if failure.startswith("incomplete") or failure == "continued":
        stack.spies["continued"].assert_awaited_once()
    else:
        stack.spies["continued"].assert_not_awaited()
    assert stack.create.await_count == (
        0 if failure == "provider" else 2 if failure.startswith("incomplete") else 1
    )


@pytest.mark.parametrize(
    "failure", ["read-error", "invalid-context", "mismatched-context"]
)
def test_repository_load_failures_stop_before_model_execution(stack, failure):
    first, before = establish_history(stack)
    request = TuesdayRequest("/chat New request", first.conversation_id)
    error = ConversationRepositoryError("repository read failed")
    if failure == "read-error":
        stack.spies["get"].side_effect = error
        error_type = ConversationRepositoryError
    elif failure == "invalid-context":
        stack.spies["get"].return_value = object()
        error_type = TypeError
    else:
        stack.spies["get"].return_value = ConversationContext()
        error_type = ValueError
    with pytest.raises(error_type) as caught:
        handle(stack, request)
    if failure == "read-error":
        assert caught.value is error
    assert asyncio.run(stack.read(first.conversation_id)).messages is before.messages
    stack.spies["get"].assert_awaited_once_with(first.conversation_id)
    stack.spies["turn"].assert_not_awaited()
    stack.create.assert_not_awaited()
    stack.spies["tool"].assert_not_awaited()
    stack.spies["append"].assert_not_awaited()


@pytest.mark.parametrize("tool_turn", [False, True])
def test_append_error_propagates_after_completion_without_repeating_execution(
    stack, tool_turn
):
    first, before = establish_history(stack)
    request = TuesdayRequest("/chat What is 6 times 7?", first.conversation_id)
    stack.create.side_effect = (
        [raw(function()), raw(text="The result is 42.")]
        if tool_turn
        else [raw(text="Completed text answer")]
    )
    error = ConversationRepositoryError("repository append failed")
    stack.spies["append"].side_effect = error
    with pytest.raises(ConversationRepositoryError) as caught:
        handle(stack, request)
    assert caught.value is error
    assert asyncio.run(stack.read(first.conversation_id)).messages is before.messages
    stack.spies["get"].assert_awaited_once()
    stack.spies["turn"].assert_awaited_once()
    stack.spies["append"].assert_awaited_once()
    conversation_id, attempted = stack.spies["append"].call_args.args
    assert conversation_id == request.conversation_id
    assert [(message.role, message.content) for message in attempted] == [
        (MessageRole.USER, request.content),
        (
            MessageRole.ASSISTANT,
            "The result is 42." if tool_turn else "Completed text answer",
        ),
    ]
    assert stack.create.await_count == (2 if tool_turn else 1)
    assert stack.spies["tool"].await_count == (1 if tool_turn else 0)
    assert stack.spies["continued"].await_count == (1 if tool_turn else 0)
    if tool_turn:
        assert stack.spies["result"].call_args.args[0].result.output == 42
    assert stack.events.mock_calls[-1][0] == "append"


@pytest.mark.parametrize(
    "content,error_type",
    [
        ("/unknown Hello", NoRouteFoundError),
        ("Hello", NoRouteFoundError),
        ("/Chat Hello", NoRouteFoundError),
        ("/chat", RequestPreparationError),
        ("/conversation", RequestPreparationError),
    ],
)
def test_invalid_routes_preserve_history_without_execution(stack, content, error_type):
    first, before = establish_history(stack)
    request = TuesdayRequest(content, first.conversation_id)
    with pytest.raises(error_type):
        handle(stack, request)
    assert asyncio.run(stack.read(first.conversation_id)).messages is before.messages
    stack.spies["turn"].assert_not_awaited()
    stack.create.assert_not_awaited()
    stack.spies["tool"].assert_not_awaited()
    stack.spies["continued"].assert_not_awaited()
    stack.spies["append"].assert_not_awaited()


@pytest.mark.parametrize("stage", ["orchestrator", "service"])
def test_factory_propagates_existing_construction_exceptions_unchanged(
    stack, monkeypatch, stage
):
    error = ValueError(stage + " construction failed")
    compose = Mock(wraps=composition_module.create_guarded_openai_orchestrator)
    construct = Mock(wraps=StatefulConversationService)
    failing = compose if stage == "orchestrator" else construct
    failing.side_effect = error
    monkeypatch.setattr(
        composition_module, "create_guarded_openai_orchestrator", compose
    )
    monkeypatch.setattr(composition_module, "StatefulConversationService", construct)
    with pytest.raises(ValueError) as caught:
        create_stateful_guarded_openai_service(stack.tool_turn, TOOLS, stack.repository)
    assert caught.value is error
    compose.assert_called_once_with(stack.tool_turn, TOOLS, renderer=None)
    if stage == "service":
        construct.assert_called_once()
        assert type(construct.call_args.args[0]) is TuesdayOrchestrator
        assert construct.call_args.args[1] is stack.repository
    else:
        construct.assert_not_called()
    assert stack.events.mock_calls == []
    stack.create.assert_not_awaited()


def test_all_existing_orchestrator_factories_remain_unwrapped_and_explicit(stack):
    default = create_default_orchestrator()
    model_backed = create_model_backed_orchestrator(stack.provider)
    guarded = create_guarded_openai_orchestrator(stack.tool_turn, TOOLS)
    for runtime in (default, model_backed, guarded):
        assert type(runtime) is TuesdayOrchestrator
    assert type(default._registry.get("conversation")) is ConversationalAgent
    assert (
        type(model_backed._registry.get("conversation"))
        is ModelBackedConversationalAgent
    )
    assert (
        type(guarded._registry.get("conversation")) is GuardedOpenAIConversationalAgent
    )
    request = TuesdayRequest("/chat Hello")
    context = ConversationContext(request.conversation_id)
    assert (
        asyncio.run(default.handle(request, context)).content
        == "TUESDAY received: Hello"
    )
    stack.create.assert_not_awaited()
    stack.create.side_effect = [raw(function())]
    with pytest.raises(UnsupportedModelToolCallError):
        asyncio.run(model_backed.handle(request, context))
    stack.spies["turn"].assert_not_awaited()
    stack.spies["get"].assert_not_awaited()
    stack.spies["append"].assert_not_awaited()
    assert tuple(inspect.signature(create_default_orchestrator).parameters) == ()
    assert tuple(inspect.signature(create_model_backed_orchestrator).parameters) == (
        "provider",
        "renderer",
    )
    assert tuple(inspect.signature(create_guarded_openai_orchestrator).parameters) == (
        "tool_turn_service",
        "tools",
        "renderer",
    )


def test_public_factory_only_reuses_existing_composition_and_stateful_service():
    assert (
        composition_module.create_stateful_guarded_openai_service
        is create_stateful_guarded_openai_service
    )
    assert composition_module.__all__ == [
        "create_default_orchestrator",
        "create_guarded_openai_orchestrator",
        "create_model_backed_orchestrator",
        "create_stateful_guarded_openai_service",
    ]
    tree = ast.parse(inspect.getsource(create_stateful_guarded_openai_service))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert [node.func.id for node in calls] == [
        "create_guarded_openai_orchestrator",
        "StatefulConversationService",
    ]
    assert not any(
        isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.Await))
        for node in ast.walk(tree)
    )


@pytest.mark.parametrize(
    "statement",
    [
        "import tuesday.composition; import tuesday.services; "
        "import tuesday.conversations",
        "import tuesday.services; import tuesday.conversations; "
        "import tuesday.composition",
        "import tuesday.agents; import tuesday.composition; import tuesday.services",
        "import tuesday.conversations; import tuesday.agents; "
        "import tuesday.composition",
        "from tuesday.composition import create_stateful_guarded_openai_service",
    ],
)
def test_clean_interpreter_imports(statement):
    result = subprocess.run(
        [sys.executable, "-c", statement], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
