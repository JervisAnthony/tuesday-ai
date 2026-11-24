"""Offline integration of caller-owned history and bounded guarded tool turns."""

import ast
import asyncio
import inspect
import json
import sqlite3
import subprocess
import sys
from contextlib import closing
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
    SQLiteConversationRepository,
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


@pytest.fixture
def sqlite_stack(stack, tmp_path):
    """Reuse the existing guarded stack with an explicitly selected durable backend."""
    stack.database_path = tmp_path / "history.sqlite3"
    stack.repository = SQLiteConversationRepository(stack.database_path)
    stack.read = stack.repository.get_context
    stack.stateful = create_stateful_guarded_openai_service(
        stack.tool_turn, TOOLS, stack.repository
    )
    return stack


@pytest.mark.parametrize("limit", [None, 0, 2])
def test_sqlite_recreated_services_recover_full_history_with_opt_in_prompt_limits(
    sqlite_stack, limit
):
    stack = sqlite_stack
    renderer = ConversationalPromptRenderer(
        "Custom durable system prompt", max_history_messages=limit
    )
    conversation_id = uuid4()
    expected = []
    stack.create.side_effect = [
        raw(text=f"  Answer {index}\nline two  ") for index in range(4)
    ]
    for index in range(4):
        repository = SQLiteConversationRepository(stack.database_path)
        stack.stateful = create_stateful_guarded_openai_service(
            stack.tool_turn, TOOLS, repository, renderer=renderer
        )
        assert stack.stateful._repository is repository
        agent = stack.stateful._orchestrator._registry.get("conversation")
        assert agent._renderer is renderer
        recovered = asyncio.run(repository.get_context(conversation_id))
        assert message_values(recovered) == expected
        request = TuesdayRequest(f"/conversation Question {index}", conversation_id)
        response = handle(stack, request)
        assert_correlated(response, request)
        assert response.content == f"  Answer {index}\nline two  "
        prior = expected if limit is None else [] if limit == 0 else expected[-2:]
        assert stack.create.call_args.kwargs["input"] == [
            {"role": "system", "content": renderer.system_prompt},
            *({"role": role.value, "content": content} for role, content in prior),
            {"role": "user", "content": f"Question {index}"},
        ]
        expected.extend(
            [(MessageRole.USER, request.content),
             (MessageRole.ASSISTANT, response.content)]
        )
        reopened = SQLiteConversationRepository(stack.database_path)
        saved = asyncio.run(reopened.get_context(conversation_id))
        assert message_values(saved) == expected
        assert len(saved.messages) == 2 * (index + 1)
    assert len(saved.messages) == 8
    assert stack.create.await_count == 4
    stack.spies["tool"].assert_not_awaited()
    stack.spies["continued"].assert_not_awaited()


def test_sqlite_repository_works_with_existing_deterministic_stateful_service(
    sqlite_stack,
):
    repository = sqlite_stack.repository
    service = StatefulConversationService(create_default_orchestrator(), repository)
    request = TuesdayRequest("/chat Hello durable TUESDAY")
    response = asyncio.run(service.handle(request))
    assert_correlated(response, request)
    saved = asyncio.run(repository.get_context(request.conversation_id))
    assert message_values(saved) == [
        (MessageRole.USER, request.content), (MessageRole.ASSISTANT, response.content)
    ]
    sqlite_stack.create.assert_not_awaited()


def test_sqlite_calculator_completion_survives_repository_and_service_restart(
    sqlite_stack,
):
    stack = sqlite_stack
    item = function("call_sqlite_43")
    stack.create.side_effect = [raw(item), raw(text="The result is 42.")]
    first = TuesdayRequest("/chat What is 6 times 7?")
    response = handle(stack, first)
    assert_correlated(response, first)
    assert response.content == "The result is 42."
    for name in ("execution", "authorize", "tool", "continued"):
        stack.spies[name].assert_awaited_once()
    stack.factory.assert_called_once_with()
    execution = stack.spies["result"].call_args.args[0]
    assert execution.result.output == 42
    assert execution.tool_call.call_id == "call_sqlite_43"
    assert execution.invocation.invocation_id is INVOCATION_ID
    assert str(INVOCATION_ID) != execution.tool_call.call_id
    stack.spies["authorize"].assert_awaited_once_with(execution.invocation)
    state, results = stack.spies["continued"].call_args.args
    assert state.request is stack.spies["turn"].call_args.args[0]
    assert state.request.tools is TOOLS
    assert results[0].call_id == item.call_id
    assert results[0].output == 42
    initial, continued = [call.kwargs for call in stack.create.call_args_list]
    assert [tool["name"] for tool in initial["tools"]] == ["calculator.basic"]
    assert continued["tools"] == initial["tools"]
    assert continued["input"] == initial["input"] + [
        item.model_dump(mode="json", exclude_none=True),
        {"type": "function_call_output", "call_id": item.call_id, "output": "42"},
    ]
    before = asyncio.run(stack.read(first.conversation_id))
    assert message_values(before) == [
        (MessageRole.USER, first.content), (MessageRole.ASSISTANT, response.content)
    ]
    recreated = SQLiteConversationRepository(stack.database_path)
    stack.stateful = create_stateful_guarded_openai_service(
        stack.tool_turn, TOOLS, recreated
    )
    second = TuesdayRequest("/conversation What was the result?", first.conversation_id)
    stack.create.side_effect = [raw(text="It was 42.")]
    reply = handle(stack, second)
    assert_correlated(reply, second)
    assert reply.content == "It was 42."
    assert stack.create.call_args.kwargs["input"] == [
        {"role": "system", "content": DEFAULT_CONVERSATIONAL_SYSTEM_PROMPT},
        {"role": "user", "content": first.content},
        {"role": "assistant", "content": response.content},
        {"role": "user", "content": "What was the result?"},
    ]
    saved = asyncio.run(recreated.get_context(first.conversation_id))
    assert saved.messages[:2] == before.messages
    assert message_values(saved) == message_values(before) + [
        (MessageRole.USER, second.content), (MessageRole.ASSISTANT, reply.content)
    ]
    assert len(saved.messages) == 4
    assert stack.create.await_count == 3
    stack.spies["tool"].assert_awaited_once()
    stack.spies["continued"].assert_awaited_once()


@pytest.mark.parametrize(
    "failure,outcome,error_type",
    [
        ("deny", ToolAuthorizationOutcome.DENY, ToolAuthorizationDeniedError),
        (
            "confirmation", ToolAuthorizationOutcome.REQUIRE_CONFIRMATION,
            ToolConfirmationRequiredError,
        ),
        (
            "unadvertised", ToolAuthorizationOutcome.ALLOW,
            UnadvertisedModelToolCallError,
        ),
        (
            "multiple", ToolAuthorizationOutcome.ALLOW,
            MultipleModelToolCallsUnsupportedError,
        ),
        ("incomplete", ToolAuthorizationOutcome.ALLOW, IncompleteGuardedToolTurnError),
    ],
)
def test_sqlite_guarded_failures_preserve_only_previously_committed_history(
    sqlite_stack, failure, error_type
):
    stack = sqlite_stack
    first, before = establish_history(stack)
    if failure == "unadvertised":
        stack.create.side_effect = [raw(function(name="unknown.tool"))]
    elif failure == "multiple":
        stack.create.side_effect = [raw(function(), function("call_second"))]
    elif failure == "incomplete":
        stack.create.side_effect = [raw(function()), raw(function("call_more"))]
    else:
        stack.create.side_effect = [raw(function())]
    request = TuesdayRequest("/chat Attempt another operation", first.conversation_id)
    with pytest.raises(error_type):
        handle(stack, request)
    reopened = SQLiteConversationRepository(stack.database_path)
    assert asyncio.run(reopened.get_context(first.conversation_id)) == before
    assert stack.create.await_count == (2 if failure == "incomplete" else 1)
    if failure == "incomplete":
        stack.spies["tool"].assert_awaited_once()
        stack.spies["continued"].assert_awaited_once()
    else:
        stack.spies["tool"].assert_not_awaited()
        stack.spies["continued"].assert_not_awaited()
    if failure in ("unadvertised", "multiple"):
        stack.spies["authorize"].assert_not_awaited()


def test_sqlite_append_failure_after_calculator_rolls_back_without_execution_retry(
    sqlite_stack,
):
    stack = sqlite_stack
    first, before = establish_history(stack)
    with closing(sqlite3.connect(stack.database_path)) as connection:
        connection.execute("""
            CREATE TRIGGER reject_completed_reply BEFORE INSERT
            ON tuesday_conversation_messages WHEN NEW.content = 'The result is 42.'
            BEGIN SELECT RAISE(ABORT, 'Cannot append completed reply'); END
        """)
    stack.create.side_effect = [
        raw(function("call_sqlite_43")), raw(text="The result is 42.")
    ]
    request = TuesdayRequest("/chat What is 6 times 7?", first.conversation_id)
    with pytest.raises(ConversationRepositoryError) as caught:
        handle(stack, request)
    assert isinstance(caught.value.__cause__, sqlite3.IntegrityError)
    stack.spies["tool"].assert_awaited_once()
    stack.spies["authorize"].assert_awaited_once()
    stack.spies["continued"].assert_awaited_once()
    assert stack.create.await_count == 2
    assert stack.spies["result"].call_args.args[0].result.output == 42
    reopened = SQLiteConversationRepository(stack.database_path)
    assert asyncio.run(reopened.get_context(first.conversation_id)) == before


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


@pytest.mark.parametrize("limit", [None, 0, 1, 2])
def test_interleaved_conversation_ids_keep_histories_separate(stack, limit):
    renderer = ConversationalPromptRenderer(max_history_messages=limit)
    stack.stateful = create_stateful_guarded_openai_service(
        stack.tool_turn, TOOLS, stack.repository, renderer=renderer
    )
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
        prior = [
            {"role": "user", "content": requests[first_index].content},
            {"role": "assistant", "content": answers[first_index]},
        ]
        expected_history = {None: prior, 0: [], 1: [prior[1]], 2: prior}[limit]
        assert payloads[second_index] == [
            system,
            *expected_history,
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
@pytest.mark.parametrize("limit", [None, 0, 1, 2])
def test_failed_later_turn_preserves_existing_successful_history(
    stack, failure, error_type, limit
):
    renderer = ConversationalPromptRenderer(max_history_messages=limit)
    stack.stateful = create_stateful_guarded_openai_service(
        stack.tool_turn, TOOLS, stack.repository, renderer=renderer
    )
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
            stack.tool_turn, tools, stack.repository, renderer=renderer
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


@pytest.mark.parametrize("limit", [None, 0, 2])
def test_eight_text_turns_bound_provider_input_but_retain_all_history(
    stack, monkeypatch, limit
):
    renderer = ConversationalPromptRenderer(
        "  Stateful bounded system\nline two  ", max_history_messages=limit
    )
    stack.stateful = create_stateful_guarded_openai_service(
        stack.tool_turn, TOOLS, stack.repository, renderer=renderer
    )
    agent = stack.stateful._orchestrator._registry.get("conversation")
    assert agent._renderer is renderer
    agent_handle = AsyncMock(wraps=agent.handle)
    monkeypatch.setattr(GuardedOpenAIConversationalAgent, "handle", agent_handle)
    conversation_id = uuid4()
    requests = [
        TuesdayRequest(
            f"{'/chat' if index % 2 == 0 else '/conversation'} Question {index}",
            conversation_id,
        )
        for index in range(8)
    ]
    answers = [f"  Answer {index}\nline two  " for index in range(8)]
    originals = [replace(request) for request in requests]
    stack.create.side_effect = [raw(text=answer) for answer in answers]
    expected_stored = []
    expected_provider_history = []
    previous = asyncio.run(stack.read(conversation_id))
    responses = []
    for index, (request, answer) in enumerate(zip(requests, answers, strict=True)):
        response = handle(stack, request)
        responses.append(response)
        assert_correlated(response, request)
        assert response.content == answer
        prepared, loaded_context = agent_handle.call_args.args
        assert prepared.content == f"Question {index}"
        assert prepared.request_id == request.request_id
        assert prepared.conversation_id == conversation_id
        assert loaded_context.messages is previous.messages
        assert message_values(loaded_context) == expected_stored
        payload = stack.create.call_args.kwargs
        assert payload["input"] == [
            {"role": "system", "content": renderer.system_prompt},
            *expected_provider_history,
            {"role": "user", "content": f"Question {index}"},
        ]
        pair = [(MessageRole.USER, request.content), (MessageRole.ASSISTANT, answer)]
        expected_stored.extend(pair)
        provider_pair = [{"role": role.value, "content": text} for role, text in pair]
        if limit is None:
            expected_provider_history.extend(provider_pair)
        elif limit == 2:
            expected_provider_history = provider_pair
        saved = asyncio.run(stack.read(conversation_id))
        assert message_values(saved) == expected_stored
        assert len(saved.messages) == (index + 1) * 2
        assert all(
            old is retained
            for old, retained in zip(
                previous.messages, saved.messages[:-2], strict=True
            )
        )
        previous = saved
    assert requests == originals
    assert len({response.response_id for response in responses}) == 8
    assert len(previous.messages) == 16
    assert stack.spies["get"].await_count == 8
    for call in stack.spies["get"].await_args_list:
        assert call.args == (conversation_id,)
    assert stack.spies["append"].await_count == stack.create.await_count == 8
    for call in stack.spies["turn"].call_args_list:
        assert call.args[0].tools is TOOLS
    stack.spies["tool"].assert_not_awaited()
    stack.spies["continued"].assert_not_awaited()


def test_bounded_calculator_turn_preserves_scope_execution_and_full_history(stack):
    renderer = ConversationalPromptRenderer(max_history_messages=2)
    stack.stateful = create_stateful_guarded_openai_service(
        stack.tool_turn, TOOLS, stack.repository, renderer=renderer
    )
    first, _ = establish_history(stack)
    second = TuesdayRequest("/conversation Second question", first.conversation_id)
    stack.create.side_effect = [raw(text="Second reply")]
    handle(stack, second)
    before = asyncio.run(stack.read(first.conversation_id))
    assert len(before.messages) == 4
    stack.events.reset_mock()
    stack.create.reset_mock()
    item = function("call_bounded_42")
    stack.create.side_effect = [
        raw(item, text="I will calculate that."),
        raw(text="The result is 42."),
    ]
    request = TuesdayRequest("/chat What is 6 times 7?", first.conversation_id)

    response = handle(stack, request)

    assert_correlated(response, request)
    assert response.content == "The result is 42."
    for name in (
        "get",
        "turn",
        "initial",
        "execution",
        "authorize",
        "tool",
        "continued",
        "append",
    ):
        stack.spies[name].assert_awaited_once()
    stack.spies["invocation"].assert_called_once()
    stack.spies["result"].assert_called_once()
    stack.factory.assert_called_once_with()
    execution = stack.spies["result"].call_args.args[0]
    assert execution.tool_call is stack.spies["execution"].call_args.args[0]
    assert execution.tool_call.call_id == item.call_id
    assert execution.invocation.invocation_id is INVOCATION_ID
    assert str(execution.invocation.invocation_id) != item.call_id
    stack.spies["authorize"].assert_awaited_once_with(execution.invocation)
    assert execution.result.output == 42
    model_request = stack.spies["turn"].call_args.args[0]
    assert model_request is stack.spies["initial"].call_args.args[0]
    assert model_request.tools is TOOLS
    state, results = stack.spies["continued"].call_args.args
    assert state.request is model_request
    assert len(results) == 1
    assert results[0].call_id == item.call_id
    assert results[0].output == 42
    assert stack.create.await_count == 2
    initial, continued = [call.kwargs for call in stack.create.call_args_list]
    assert initial["input"] == [
        {"role": "system", "content": renderer.system_prompt},
        {"role": "user", "content": second.content},
        {"role": "assistant", "content": "Second reply"},
        {"role": "user", "content": "What is 6 times 7?"},
    ]
    assert initial["tools"] == continued["tools"] == [
        {
            "type": "function",
            "name": "calculator.basic",
            "description": "Arithmetic",
            "parameters": {},
        }
    ]
    assert continued["input"] == initial["input"] + [
        item.model_dump(mode="json", exclude_none=True),
        {"type": "function_call_output", "call_id": item.call_id, "output": "42"},
    ]
    for payload in (initial, continued):
        assert payload["store"] is payload["stream"] is False
        assert "previous_response_id" not in payload
    saved = asyncio.run(stack.read(request.conversation_id))
    assert saved.messages[:4] == before.messages
    assert all(
        old is retained
        for old, retained in zip(before.messages, saved.messages[:4], strict=True)
    )
    assert message_values(saved) == message_values(before) + [
        (MessageRole.USER, request.content),
        (MessageRole.ASSISTANT, response.content),
    ]


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
