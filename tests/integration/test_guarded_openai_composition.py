"""Offline integration tests for explicit guarded OpenAI application wiring."""

import ast
import asyncio
import inspect
import json
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID

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
)
from tuesday.config import LanguageModelSettings
from tuesday.conversations import InMemoryConversationRepository
from tuesday.domain import (
    ConversationContext,
    ConversationMessage,
    MessageRole,
    TuesdayRequest,
    TuesdayResponse,
)
from tuesday.language_models import (
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
from tuesday.services import (
    MultipleModelToolCallsUnsupportedError,
    StatefulConversationService,
    UnadvertisedModelToolCallError,
)
from tuesday.services.openai_tool_turn import OpenAIToolTurnService
from tuesday.tools import (
    BasicCalculatorTool,
    DeterministicToolExecutor,
    GuardedToolExecutor,
    StaticToolAuthorizationPolicy,
    ToolAuthorizationDeniedError,
    ToolAuthorizationOutcome,
    ToolConfirmationRequiredError,
    ToolRegistry,
)

INVOCATION_ID = UUID(int=40)
TOOLS = (LanguageModelToolDefinition("calculator.basic", "Arithmetic", {}),)


def function(call_id="call_factory_40", name="calculator.basic"):
    return ResponseFunctionToolCall(
        type="function_call",
        id="item_" + call_id,
        call_id=call_id,
        name=name,
        arguments=json.dumps({"operation": "multiply", "left": 6, "right": 7}),
    )


def raw(*items, text=None):
    return SimpleNamespace(output=items, output_text=text, model="resolved-model")


def interaction(content="/chat What is 6 times 7?", messages=()):
    request = TuesdayRequest(content)
    return request, ConversationContext(request.conversation_id, messages)


def run(runtime, content="/chat What is 6 times 7?", messages=()):
    request, context = interaction(content, messages)
    response = asyncio.run(runtime.handle(request, context))
    return request, context, response


@pytest.fixture
def outcome():
    return ToolAuthorizationOutcome.ALLOW


@pytest.fixture
def stack(monkeypatch, outcome):
    """Real production stack; only Responses network access is replaced."""
    create = AsyncMock(side_effect=[raw(function()), raw(text="The result is 42.")])
    provider = OpenAILanguageModelProvider(
        LanguageModelSettings("openai", "configured-model", "fake-key"),
        client=SimpleNamespace(responses=SimpleNamespace(create=create)),
    )
    registry = ToolRegistry()
    calculator = BasicCalculatorTool()
    registry.register(calculator)
    policy = StaticToolAuthorizationPolicy({"calculator.basic": outcome})
    authorize_method = StaticToolAuthorizationPolicy.authorize
    factory = Mock(return_value=INVOCATION_ID)
    invocation_bridge = ModelToolInvocationBridge(invocation_id_factory=factory)
    executor = GuardedModelToolExecutor(
        invocation_bridge,
        GuardedToolExecutor(policy, DeterministicToolExecutor(registry)),
    )
    result_bridge = ModelToolResultBridge()
    service = OpenAIToolTurnService(provider, executor, result_bridge)
    spies = {}
    for name, target, bound, method, asynchronous in (
        ("turn", OpenAIToolTurnService, service, "run_once", True),
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
        spies[name] = spy
    return SimpleNamespace(
        service=service,
        provider=provider,
        create=create,
        factory=factory,
        spies=spies,
        authorize_method=authorize_method,
    )


def test_factory_builds_exact_runtime_with_injected_dependencies(stack):
    runtime = create_guarded_openai_orchestrator(stack.service, TOOLS)
    assert type(runtime) is TuesdayOrchestrator
    assert type(runtime._registry) is AgentRegistry
    assert type(runtime._router) is DeterministicRouter
    assert type(runtime._request_preparer) is DirectiveRequestPreparer
    assert len(runtime._registry) == 1
    assert runtime._registry.names == ("conversation",)
    agent = runtime._registry.get("conversation")
    assert type(agent) is GuardedOpenAIConversationalAgent
    assert agent._tool_turn_service is stack.service
    assert agent._tools is TOOLS
    assert runtime._router.routes == (
        ("chat", "conversation"),
        ("conversation", "conversation"),
    )
    stack.spies["turn"].assert_not_awaited()
    stack.create.assert_not_awaited()
    parameters = inspect.signature(create_guarded_openai_orchestrator).parameters
    assert tuple(parameters) == ("tool_turn_service", "tools", "renderer")
    assert parameters["renderer"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["renderer"].default is None


@pytest.mark.parametrize("custom", [False, True])
def test_renderer_selection_reaches_real_model_input(stack, custom):
    renderer = ConversationalPromptRenderer("Custom guarded composition prompt")
    runtime = create_guarded_openai_orchestrator(
        stack.service, TOOLS, **({"renderer": renderer} if custom else {})
    )
    agent = runtime._registry.get("conversation")
    assert isinstance(agent._renderer, ConversationalPromptRenderer)
    if custom:
        assert agent._renderer is renderer
    stack.create.side_effect = [raw(text="Hello from TUESDAY.")]
    _, _, response = run(runtime, "/chat Hello TUESDAY")
    assert response.content == "Hello from TUESDAY."
    assert stack.create.call_args.kwargs["input"][0] == {
        "role": "system",
        "content": renderer.system_prompt
        if custom
        else DEFAULT_CONVERSATIONAL_SYSTEM_PROMPT,
    }


@pytest.mark.parametrize(
    "invalid",
    [
        "none-service",
        "object-service",
        "dict-service",
        "none-tools",
        "list-tools",
        "empty-tools",
        "invalid-entry",
        "duplicate-names",
        "invalid-renderer",
    ],
)
def test_invalid_dependencies_delegate_to_agent_before_composition(
    stack, monkeypatch, invalid
):
    values = dict(tool_turn_service=stack.service, tools=TOOLS, renderer=None)
    if invalid.endswith("service"):
        values["tool_turn_service"] = {
            "none-service": None,
            "object-service": object(),
            "dict-service": {},
        }[invalid]
    elif invalid == "invalid-renderer":
        values["renderer"] = object()
    else:
        values["tools"] = {
            "none-tools": None,
            "list-tools": list(TOOLS),
            "empty-tools": (),
            "invalid-entry": (TOOLS[0], object()),
            "duplicate-names": (TOOLS[0], replace(TOOLS[0])),
        }[invalid]
    with pytest.raises((TypeError, ValueError)) as expected:
        GuardedOpenAIConversationalAgent(**values)
    compose = Mock(wraps=composition_module._compose_conversational_orchestrator)
    monkeypatch.setattr(
        composition_module, "_compose_conversational_orchestrator", compose
    )
    with pytest.raises(type(expected.value)) as caught:
        create_guarded_openai_orchestrator(**values)
    assert caught.value.args == expected.value.args
    compose.assert_not_called()
    stack.create.assert_not_awaited()


def test_factory_propagates_exact_constructor_error_without_retry(stack, monkeypatch):
    error = ValueError("constructor rejected configuration")
    constructor = Mock(side_effect=error)
    compose = Mock()
    monkeypatch.setattr(
        composition_module, "GuardedOpenAIConversationalAgent", constructor
    )
    monkeypatch.setattr(
        composition_module, "_compose_conversational_orchestrator", compose
    )
    with pytest.raises(ValueError) as caught:
        create_guarded_openai_orchestrator(stack.service, TOOLS)
    assert caught.value is error
    constructor.assert_called_once_with(stack.service, TOOLS, renderer=None)
    compose.assert_not_called()


def test_tool_definitions_keep_identity_schema_and_order_without_discovery(stack):
    tools = (
        LanguageModelToolDefinition("other.tool", "Other", {"type": "object"}),
        LanguageModelToolDefinition("OTHER.TOOL", "Case-distinct", {"type": "object"}),
        replace(TOOLS[0], name="Calculator.Basic"),
    )
    schemas = tuple(tool.parameters for tool in tools)
    runtime = create_guarded_openai_orchestrator(stack.service, tools)
    configured = runtime._registry.get("conversation")._tools
    assert configured is tools
    assert all(
        actual is expected for actual, expected in zip(configured, tools, strict=True)
    )
    assert all(
        tool.parameters is schema
        for tool, schema in zip(configured, schemas, strict=True)
    )
    stack.create.side_effect = [raw(text="Only explicitly configured tools")]
    run(runtime)
    assert stack.spies["initial"].call_args.args[0].tools is tools
    assert [item["name"] for item in stack.create.call_args.kwargs["tools"]] == [
        "other.tool",
        "OTHER.TOOL",
        "Calculator.Basic",
    ]
    stack.spies["tool"].assert_not_awaited()


def test_factory_calls_create_fresh_runtime_graphs(stack):
    first = create_guarded_openai_orchestrator(stack.service, TOOLS)
    second = create_guarded_openai_orchestrator(stack.service, TOOLS)
    assert first is not second
    for name in ("_registry", "_router", "_request_preparer"):
        assert getattr(first, name) is not getattr(second, name)
    first_agent = first._registry.get("conversation")
    second_agent = second._registry.get("conversation")
    assert first_agent is not second_agent
    assert (
        first_agent._tool_turn_service
        is second_agent._tool_turn_service
        is stack.service
    )
    assert first_agent._tools is second_agent._tools is TOOLS
    assert first_agent._renderer is not second_agent._renderer


@pytest.mark.parametrize("directive", ["/chat", "/conversation"])
def test_text_turn_preserves_history_whitespace_and_correlation(stack, directive):
    runtime = create_guarded_openai_orchestrator(stack.service, TOOLS)
    prior = (
        ConversationMessage(MessageRole.USER, "  Earlier question\nline two  "),
        ConversationMessage(MessageRole.ASSISTANT, "  Earlier answer\nline two  "),
    )
    request, context = interaction(directive + "   Hello TUESDAY\nsecond line  ", prior)
    before_request, before_context = replace(request), replace(context)
    stack.create.side_effect = [raw(text="  Hello from TUESDAY.\nWelcome.  ")]
    response = asyncio.run(runtime.handle(request, context))
    assert isinstance(response, TuesdayResponse)
    assert response.content == "  Hello from TUESDAY.\nWelcome.  "
    assert response.conversation_id == request.conversation_id
    assert response.request_id == request.request_id
    assert isinstance(response.response_id, UUID)
    assert request == before_request and context == before_context
    assert context.messages is prior
    stack.create.assert_awaited_once()
    assert stack.create.call_args.kwargs["input"] == [
        {"role": "system", "content": DEFAULT_CONVERSATIONAL_SYSTEM_PROMPT},
        *(
            {"role": message.role.value, "content": message.content}
            for message in prior
        ),
        {"role": "user", "content": "Hello TUESDAY\nsecond line  "},
    ]
    model_request = stack.spies["initial"].call_args.args[0]
    assert model_request.tools is TOOLS
    assert [item["name"] for item in stack.create.call_args.kwargs["tools"]] == [
        "calculator.basic"
    ]
    stack.spies["turn"].assert_awaited_once_with(model_request)
    for name in ("execution", "authorize", "tool", "result", "continued"):
        stack.spies[name].assert_not_called()


@pytest.mark.parametrize("directive", ["/chat", "/conversation"])
def test_real_guarded_calculator_turn_through_factory(stack, monkeypatch, directive):
    runtime = create_guarded_openai_orchestrator(stack.service, TOOLS)
    agent = runtime._registry.get("conversation")
    handle = AsyncMock(wraps=agent.handle)
    monkeypatch.setattr(GuardedOpenAIConversationalAgent, "handle", handle)
    item = function()
    stack.create.side_effect = [
        raw(item, text="I will calculate that."),
        raw(text="The result is 42."),
    ]
    request, context, response = run(runtime, directive + " What is 6 times 7?")
    handle.assert_awaited_once()
    prepared, passed_context = handle.call_args.args
    assert prepared is not request
    assert prepared.content == "What is 6 times 7?"
    assert prepared.request_id == request.request_id == response.request_id
    assert (
        prepared.conversation_id == request.conversation_id == response.conversation_id
    )
    assert passed_context is context
    assert response.content == "The result is 42."
    assert request.content.startswith(directive)
    for name in ("turn", "initial", "execution", "authorize", "tool", "continued"):
        stack.spies[name].assert_awaited_once()
    stack.spies["invocation"].assert_called_once()
    stack.spies["result"].assert_called_once()
    stack.factory.assert_called_once_with()
    execution = stack.spies["result"].call_args.args[0]
    assert execution.result.output == 42
    assert execution.tool_call.call_id == "call_factory_40"
    assert execution.invocation.invocation_id is INVOCATION_ID
    assert str(INVOCATION_ID) != execution.tool_call.call_id
    model_request = stack.spies["turn"].call_args.args[0]
    assert model_request is stack.spies["initial"].call_args.args[0]
    assert model_request.tools is TOOLS
    state, results = stack.spies["continued"].call_args.args
    assert state.request is model_request
    assert results[0].call_id == "call_factory_40"
    assert results[0].output == 42
    assert stack.create.await_count == 2
    first, second = [call.kwargs for call in stack.create.call_args_list]
    assert first["input"][-1] == {"role": "user", "content": prepared.content}
    assert (
        first["tools"]
        == second["tools"]
        == [
            {
                "type": "function",
                "name": "calculator.basic",
                "description": "Arithmetic",
                "parameters": {},
            }
        ]
    )
    assert second["input"] == first["input"] + [
        item.model_dump(mode="json", exclude_none=True),
        {"type": "function_call_output", "call_id": "call_factory_40", "output": "42"},
    ]
    for payload in (first, second):
        assert payload["store"] is payload["stream"] is False
        assert "previous_response_id" not in payload


@pytest.mark.parametrize("content", ["/unknown Hello", "Hello", "/Chat Hello"])
def test_invalid_routes_never_reach_provider_or_tools(stack, content):
    runtime = create_guarded_openai_orchestrator(stack.service, TOOLS)
    with pytest.raises(NoRouteFoundError):
        run(runtime, content)
    stack.spies["turn"].assert_not_awaited()
    stack.create.assert_not_awaited()
    stack.spies["tool"].assert_not_awaited()
    stack.spies["continued"].assert_not_awaited()


@pytest.mark.parametrize("content", ["/chat", "/conversation", "/chat   "])
def test_missing_content_fails_in_existing_preparer(stack, content):
    runtime = create_guarded_openai_orchestrator(stack.service, TOOLS)
    with pytest.raises(
        RequestPreparationError, match="content after the routing directive"
    ):
        run(runtime, content)
    stack.spies["turn"].assert_not_awaited()
    stack.create.assert_not_awaited()
    stack.spies["tool"].assert_not_awaited()


def test_context_mismatch_fails_before_provider_and_tools(stack):
    runtime = create_guarded_openai_orchestrator(stack.service, TOOLS)
    request, _ = interaction()
    with pytest.raises(ValueError, match="same conversation"):
        asyncio.run(runtime.handle(request, ConversationContext()))
    stack.spies["turn"].assert_not_awaited()
    stack.create.assert_not_awaited()
    stack.spies["tool"].assert_not_awaited()


@pytest.mark.parametrize(
    "outcome,error_type",
    [
        (ToolAuthorizationOutcome.DENY, ToolAuthorizationDeniedError),
        (ToolAuthorizationOutcome.REQUIRE_CONFIRMATION, ToolConfirmationRequiredError),
    ],
)
def test_advertised_tools_still_require_guarded_authorization(stack, error_type):
    runtime = create_guarded_openai_orchestrator(stack.service, TOOLS)
    with pytest.raises(error_type):
        run(runtime)
    stack.spies["authorize"].assert_awaited_once()
    stack.create.assert_awaited_once()
    stack.spies["tool"].assert_not_awaited()
    stack.spies["lookup"].assert_not_called()
    stack.spies["result"].assert_not_called()
    stack.spies["continued"].assert_not_awaited()


@pytest.mark.parametrize("failure", ["unadvertised", "multiple"])
def test_initial_safety_failures_stop_before_execution(stack, failure):
    runtime = create_guarded_openai_orchestrator(stack.service, TOOLS)
    items = (
        [function(name="unknown.tool")]
        if failure == "unadvertised"
        else [function(), function("call_second")]
    )
    stack.create.side_effect = [raw(*items)]
    error_type = (
        UnadvertisedModelToolCallError
        if failure == "unadvertised"
        else MultipleModelToolCallsUnsupportedError
    )
    with pytest.raises(error_type):
        run(runtime)
    stack.create.assert_awaited_once()
    stack.factory.assert_not_called()
    for name in ("execution", "invocation", "authorize", "tool", "result", "continued"):
        stack.spies[name].assert_not_called()


@pytest.mark.parametrize("mixed", [False, True])
def test_incomplete_continuation_propagates_without_additional_execution(stack, mixed):
    runtime = create_guarded_openai_orchestrator(stack.service, TOOLS)
    stack.create.side_effect = [
        raw(function()),
        raw(function("call_more"), text="I need another operation." if mixed else None),
    ]
    with pytest.raises(IncompleteGuardedToolTurnError) as caught:
        run(runtime)
    assert str(caught.value) == (
        "Guarded OpenAI conversational turn has unresolved model tool calls."
    )
    stack.spies["turn"].assert_awaited_once()
    stack.spies["tool"].assert_awaited_once()
    stack.spies["result"].assert_called_once()
    stack.spies["continued"].assert_awaited_once()
    assert stack.create.await_count == 2


@pytest.mark.parametrize("mode", ["text", "tool", "denied", "incomplete"])
def test_stateful_service_persists_only_successful_completed_pairs(stack, mode):
    runtime = create_guarded_openai_orchestrator(stack.service, TOOLS)
    if mode == "text":
        stack.create.side_effect = [raw(text="Hello from TUESDAY.")]
    elif mode == "incomplete":
        stack.create.side_effect = [raw(function()), raw(function("call_more"))]
    elif mode == "denied":
        # Use a real denying policy through the existing guarded path.
        policy = StaticToolAuthorizationPolicy(
            {"calculator.basic": ToolAuthorizationOutcome.DENY}
        )

        async def deny(invocation):
            return await stack.authorize_method(policy, invocation)

        stack.spies["authorize"].side_effect = deny
    repository = InMemoryConversationRepository()
    service = StatefulConversationService(runtime, repository)
    request, _ = interaction()
    if mode in ("denied", "incomplete"):
        error_type = (
            ToolAuthorizationDeniedError
            if mode == "denied"
            else IncompleteGuardedToolTurnError
        )
        with pytest.raises(error_type):
            asyncio.run(service.handle(request))
    else:
        response = asyncio.run(service.handle(request))
        assert response.content == (
            "Hello from TUESDAY." if mode == "text" else "The result is 42."
        )
    saved = asyncio.run(repository.get_context(request.conversation_id))
    if mode in ("denied", "incomplete"):
        assert saved.messages == ()
    else:
        assert [(message.role, message.content) for message in saved.messages] == [
            (MessageRole.USER, request.content),
            (MessageRole.ASSISTANT, response.content),
        ]
    if mode == "incomplete":
        stack.spies["tool"].assert_awaited_once()
        assert stack.spies["result"].call_args.args[0].result.output == 42


def test_original_factories_remain_explicit_and_unchanged(stack):
    default = create_default_orchestrator()
    assert type(default._registry.get("conversation")) is ConversationalAgent
    assert run(default, "/chat Hello")[2].content == "TUESDAY received: Hello"
    stack.create.assert_not_awaited()
    old = create_model_backed_orchestrator(stack.provider)
    assert type(old._registry.get("conversation")) is ModelBackedConversationalAgent
    stack.create.side_effect = [raw(text="Text-only answer"), raw(function())]
    assert run(old, "/chat Hello")[2].content == "Text-only answer"
    assert "tools" not in stack.create.call_args.kwargs
    with pytest.raises(UnsupportedModelToolCallError):
        run(old)
    stack.spies["turn"].assert_not_awaited()
    stack.spies["tool"].assert_not_awaited()
    assert tuple(inspect.signature(create_default_orchestrator).parameters) == ()
    assert tuple(inspect.signature(create_model_backed_orchestrator).parameters) == (
        "provider",
        "renderer",
    )


def test_new_factory_is_public_and_only_delegates_to_agent_and_shared_helper():
    assert (
        composition_module.create_guarded_openai_orchestrator
        is create_guarded_openai_orchestrator
    )
    assert composition_module.__all__ == [
        "create_default_orchestrator",
        "create_guarded_openai_orchestrator",
        "create_model_backed_orchestrator",
    ]
    tree = ast.parse(inspect.getsource(create_guarded_openai_orchestrator))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert [node.func.id for node in calls] == [
        "GuardedOpenAIConversationalAgent",
        "_compose_conversational_orchestrator",
    ]
    assert not any(
        isinstance(node, (ast.If, ast.For, ast.While, ast.Try))
        for node in ast.walk(tree)
    )
    imports = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(ast.parse(inspect.getsource(composition_module)))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imports == {
        "tuesday.agents",
        "tuesday.language_models.base",
        "tuesday.language_models.tools",
        "tuesday.orchestration",
        "tuesday.preparation",
        "tuesday.prompting",
        "tuesday.routing",
        "tuesday.services.openai_tool_turn",
    }


@pytest.mark.parametrize(
    "statement",
    [
        "import tuesday.composition; import tuesday.agents; import tuesday.services",
        "import tuesday.agents; import tuesday.composition; import tuesday.services",
        "import tuesday.services; import tuesday.composition; import tuesday.agents",
        "from tuesday.composition import create_guarded_openai_orchestrator",
    ],
)
def test_clean_process_imports(statement):
    result = subprocess.run(
        [sys.executable, "-c", statement], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
