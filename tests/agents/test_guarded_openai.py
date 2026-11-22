"""Offline contracts and runtime integration for the opt-in guarded agent."""

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

import tuesday.agents as agents
import tuesday.agents.guarded_openai as agent_module
from tuesday.agents import (
    AgentRegistry,
    BaseAgent,
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
    LanguageModelProviderError,
    LanguageModelRequest,
    LanguageModelToolDefinition,
    LanguageModelToolResult,
    OpenAILanguageModelProvider,
)
from tuesday.orchestration import TuesdayOrchestrator
from tuesday.preparation import DirectiveRequestPreparer
from tuesday.prompting import (
    DEFAULT_CONVERSATIONAL_SYSTEM_PROMPT,
    ConversationalPromptRenderer,
)
from tuesday.routing import DeterministicRouter
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
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistry,
)

INVOCATION_ID = UUID(int=39)
ARGUMENTS = {"operation": "multiply", "left": 6, "right": 7}
TOOLS = (LanguageModelToolDefinition("calculator.basic", "Arithmetic", {}),)


def function(call_id="call_calc_39", name="calculator.basic"):
    return ResponseFunctionToolCall(
        type="function_call",
        id="item_" + call_id,
        call_id=call_id,
        name=name,
        arguments=json.dumps(ARGUMENTS),
    )


def raw(*items, text=None):
    return SimpleNamespace(output=items, output_text=text, model="resolved-model")


def interaction(content="What is 6 times 7?", messages=()):
    request = TuesdayRequest(content)
    return request, ConversationContext(request.conversation_id, messages)


def run(stack, request=None, context=None):
    if request is None:
        request, context = interaction()
    return asyncio.run(stack.agent.handle(request, context))


def orchestrator(agent):
    registry = AgentRegistry()
    registry.register(agent)
    return TuesdayOrchestrator(
        DeterministicRouter({"chat": "conversation", "conversation": "conversation"}),
        registry,
        DirectiveRequestPreparer(),
    )


@pytest.fixture
def stack(monkeypatch):
    """Real guarded stack with a fake Responses client and observable boundaries."""
    create = AsyncMock(side_effect=[raw(function()), raw(text="The result is 42.")])
    provider = OpenAILanguageModelProvider(
        LanguageModelSettings("openai", "configured-model", "fake-key"),
        client=SimpleNamespace(responses=SimpleNamespace(create=create)),
    )
    tool = BasicCalculatorTool()
    registry = ToolRegistry()
    registry.register(tool)
    policy = StaticToolAuthorizationPolicy(
        {"calculator.basic": ToolAuthorizationOutcome.ALLOW}
    )
    authorize_method = StaticToolAuthorizationPolicy.authorize
    factory = Mock(return_value=INVOCATION_ID)
    executor = GuardedModelToolExecutor(
        ModelToolInvocationBridge(invocation_id_factory=factory),
        GuardedToolExecutor(policy, DeterministicToolExecutor(registry)),
    )
    bridge = ModelToolResultBridge()
    service = OpenAIToolTurnService(provider, executor, bridge)
    renderer = ConversationalPromptRenderer()
    render_method = ConversationalPromptRenderer.render
    spies = {}
    for name, target, bound, method, asynchronous in (
        ("service", OpenAIToolTurnService, service, "run_once", True),
        ("render", ConversationalPromptRenderer, renderer, "render", False),
        ("initial", provider, provider, "generate_with_continuation", True),
        ("continued", provider, provider, "continue_with_tool_results", True),
        ("execution", GuardedModelToolExecutor, executor, "execute", True),
        ("authorize", StaticToolAuthorizationPolicy, policy, "authorize", True),
        ("lookup", registry, registry, "get", False),
        ("tool", BasicCalculatorTool, tool, "execute", True),
        ("result", ModelToolResultBridge, bridge, "to_model_result", False),
    ):
        spy = (AsyncMock if asynchronous else Mock)(wraps=getattr(bound, method))
        monkeypatch.setattr(target, method, spy)
        spies[name] = spy
    return SimpleNamespace(
        agent=GuardedOpenAIConversationalAgent(service, TOOLS, renderer=renderer),
        service=service,
        provider=provider,
        renderer=renderer,
        create=create,
        factory=factory,
        spies=spies,
        render_method=render_method,
        authorize_method=authorize_method,
    )


def test_constructor_identity_slots_and_contract(stack):
    agent = stack.agent
    assert GuardedOpenAIConversationalAgent.__bases__ == (BaseAgent,)
    assert agent._tool_turn_service is stack.service
    assert agent._tools is TOOLS
    assert agent._renderer is stack.renderer
    assert agent.__slots__ == ("_tool_turn_service", "_tools", "_renderer")
    assert BaseAgent.__slots__ == ()
    assert not hasattr(agent, "__dict__")
    with pytest.raises(AttributeError):
        agent.extra = "unexpected state"
    assert agent.name == "conversation" == ConversationalAgent().name
    for word in ("OpenAI", "conversational", "bounded", "guarded"):
        assert word in agent.description
    assert inspect.iscoroutinefunction(agent.handle)
    assert tuple(inspect.signature(agent.handle).parameters) == ("request", "context")
    parameters = inspect.signature(GuardedOpenAIConversationalAgent).parameters
    assert tuple(parameters) == ("tool_turn_service", "tools", "renderer")
    assert parameters["renderer"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["renderer"].default is None


@pytest.mark.parametrize("invalid", [None, {}, object(), "service", SimpleNamespace()])
def test_constructor_requires_actual_service(stack, invalid):
    with pytest.raises(TypeError, match="tool_turn_service must be"):
        GuardedOpenAIConversationalAgent(invalid, TOOLS)


@pytest.mark.parametrize("invalid", [None, [], list(TOOLS), {}, "tools", object()])
def test_constructor_requires_tool_tuple(stack, invalid):
    with pytest.raises(TypeError, match="tools must be a tuple"):
        GuardedOpenAIConversationalAgent(stack.service, invalid)


@pytest.mark.parametrize(
    "invalid", [(object(),), (TOOLS[0], None), ("calculator.basic",)]
)
def test_constructor_requires_definitions(stack, invalid):
    with pytest.raises(TypeError, match="LanguageModelToolDefinition instances"):
        GuardedOpenAIConversationalAgent(stack.service, invalid)


@pytest.mark.parametrize(
    "tools", [(), (TOOLS[0], TOOLS[0]), (TOOLS[0], replace(TOOLS[0]))]
)
def test_constructor_rejects_empty_or_duplicate_tools(stack, tools):
    with pytest.raises(ValueError):
        GuardedOpenAIConversationalAgent(stack.service, tools)


@pytest.mark.parametrize("invalid", [object(), {}, "renderer"])
def test_constructor_validates_before_assigning_dependencies(stack, invalid):
    candidate = object.__new__(GuardedOpenAIConversationalAgent)
    with pytest.raises(TypeError, match="renderer must be"):
        candidate.__init__(stack.service, TOOLS, renderer=invalid)
    for name in candidate.__slots__:
        assert not hasattr(candidate, name)


def test_default_renderer_and_case_distinct_definition_order(stack):
    tools = (replace(TOOLS[0], name="Calculator.Basic"), TOOLS[0])
    first = GuardedOpenAIConversationalAgent(stack.service, tools)
    second = GuardedOpenAIConversationalAgent(stack.service, tools)
    assert isinstance(first._renderer, ConversationalPromptRenderer)
    assert first._renderer is not second._renderer
    assert first._tools is tools
    assert first._tools[0] is tools[0]
    assert first._tools[1] is TOOLS[0]


@pytest.mark.parametrize("field", ["request", "context"])
@pytest.mark.parametrize("invalid", [None, {}, object(), "invalid"])
def test_invalid_interaction_stops_before_rendering(stack, field, invalid):
    request, context = interaction()
    with pytest.raises(TypeError, match=field + " must be"):
        asyncio.run(
            stack.agent.handle(
                invalid if field == "request" else request,
                invalid if field == "context" else context,
            )
        )
    stack.spies["render"].assert_not_called()
    stack.spies["service"].assert_not_awaited()


def test_context_mismatch_uses_base_validation_before_rendering(stack, monkeypatch):
    validate = Mock(wraps=BaseAgent._validate_context)
    monkeypatch.setattr(BaseAgent, "_validate_context", staticmethod(validate))
    request, _ = interaction()
    context = ConversationContext()
    with pytest.raises(ValueError, match="same conversation"):
        run(stack, request, context)
    validate.assert_called_once_with(request, context)
    stack.spies["render"].assert_not_called()
    stack.spies["service"].assert_not_awaited()


@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("custom_prompt", [False, True])
def test_rendered_prompt_and_tool_identities_are_preserved(
    stack, history, custom_prompt
):
    renderer = (
        ConversationalPromptRenderer("  Custom system\nprompt  ")
        if custom_prompt
        else stack.renderer
    )
    stack.agent = GuardedOpenAIConversationalAgent(
        stack.service, TOOLS, renderer=renderer
    )
    prior = (
        (
            ConversationMessage(MessageRole.USER, "  Earlier question\nline 2  "),
            ConversationMessage(MessageRole.ASSISTANT, "Earlier answer"),
            ConversationMessage(MessageRole.SYSTEM, "Earlier system"),
        )
        if history
        else ()
    )
    request, context = interaction("  Current question\nline 2  ", prior)
    original_request, original_context = replace(request), replace(context)
    rendered = stack.render_method(renderer, request, context)
    stack.spies["render"].return_value = rendered
    stack.create.side_effect = [raw(text="Hello from TUESDAY")]
    response = run(stack, request, context)
    stack.spies["render"].assert_called_once_with(request, context)
    stack.spies["service"].assert_awaited_once()
    model_request = stack.spies["service"].call_args.args[0]
    assert isinstance(model_request, LanguageModelRequest)
    assert model_request is not rendered
    assert model_request.messages is rendered.messages
    assert model_request.tools is TOOLS
    assert rendered.tools == ()
    assert [(message.role, message.content) for message in model_request.messages] == [
        (MessageRole.SYSTEM, renderer.system_prompt),
        *((message.role, message.content) for message in prior),
        (MessageRole.USER, request.content),
    ]
    if not custom_prompt:
        assert renderer.system_prompt == DEFAULT_CONVERSATIONAL_SYSTEM_PROMPT
    assert request == original_request and context == original_context
    assert context.messages is prior
    assert response.content == "Hello from TUESDAY"
    assert response.conversation_id == request.conversation_id
    assert response.request_id == request.request_id
    assert isinstance(response.response_id, UUID)
    stack.spies["execution"].assert_not_awaited()
    stack.spies["continued"].assert_not_awaited()
    stack.create.assert_awaited_once()


@pytest.mark.parametrize("invalid", [None, {}, object(), "request"])
def test_invalid_rendered_request_stops_before_service(stack, invalid):
    stack.spies["render"].return_value = invalid
    with pytest.raises(TypeError, match="renderer must return a LanguageModelRequest"):
        run(stack)
    stack.spies["render"].assert_called_once()
    stack.spies["service"].assert_not_awaited()


def test_preconfigured_renderer_tools_are_rejected_without_merging(stack):
    request, context = interaction()
    rendered = replace(
        stack.render_method(stack.renderer, request, context), tools=TOOLS
    )
    stack.spies["render"].return_value = rendered
    with pytest.raises(ValueError) as caught:
        run(stack, request, context)
    assert (
        str(caught.value)
        == "Conversational renderer must not preconfigure model tools."
    )
    assert rendered.tools is TOOLS
    stack.spies["service"].assert_not_awaited()


@pytest.mark.parametrize("content", ["Hello from TUESDAY", "  Answer\nsecond line  "])
def test_text_only_content_and_independent_response_ids(stack, content):
    stack.create.side_effect = [raw(text=content), raw(text=content)]
    request, context = interaction()
    first = run(stack, request, context)
    second = run(stack, request, context)
    assert first.content == second.content == content
    assert first.response_id != second.response_id
    assert stack.spies["service"].await_count == stack.create.await_count == 2
    stack.spies["tool"].assert_not_awaited()
    stack.spies["continued"].assert_not_awaited()


@pytest.mark.parametrize("mixed", [False, True])
def test_real_calculator_turn_returns_only_terminal_text(stack, mixed):
    item = function()
    stack.create.side_effect = [
        raw(item, text="I will calculate that." if mixed else None),
        raw(text="  The result is 42.\nDone.  "),
    ]
    request, context = interaction()
    response = run(stack, request, context)
    assert isinstance(response, TuesdayResponse)
    assert response.content == "  The result is 42.\nDone.  "
    assert response.conversation_id == request.conversation_id
    assert response.request_id == request.request_id
    stack.spies["service"].assert_awaited_once()
    for name in ("initial", "execution", "authorize", "tool", "continued"):
        stack.spies[name].assert_awaited_once()
    stack.spies["result"].assert_called_once()
    stack.factory.assert_called_once_with()
    execution = stack.spies["result"].call_args.args[0]
    assert execution.result.output == 42
    assert execution.invocation.invocation_id is INVOCATION_ID
    assert str(INVOCATION_ID) != execution.tool_call.call_id == "call_calc_39"
    state, results = stack.spies["continued"].call_args.args
    assert state.request is stack.spies["service"].call_args.args[0]
    assert state.request is stack.spies["initial"].call_args.args[0]
    assert results == (LanguageModelToolResult("call_calc_39", 42),)
    assert stack.create.await_count == 2
    first, second = [call.kwargs for call in stack.create.call_args_list]
    assert first["input"][-1] == {"role": "user", "content": request.content}
    assert second["input"] == first["input"] + [
        item.model_dump(mode="json", exclude_none=True),
        {"type": "function_call_output", "call_id": "call_calc_39", "output": "42"},
    ]
    assert first["tools"] == second["tools"]
    assert [tool["name"] for tool in first["tools"]] == ["calculator.basic"]
    for payload in (first, second):
        assert payload["store"] is payload["stream"] is False
        assert "previous_response_id" not in payload


@pytest.mark.parametrize("mixed", [False, True])
def test_unresolved_continuation_fails_after_one_execution(stack, mixed, monkeypatch):
    stack.create.side_effect = [
        raw(function()),
        raw(function("call_next"), text="I need another operation." if mixed else None),
    ]
    response_constructor = Mock(side_effect=AssertionError("No partial response"))
    monkeypatch.setattr(agent_module, "TuesdayResponse", response_constructor)
    with pytest.raises(IncompleteGuardedToolTurnError) as caught:
        run(stack)
    assert str(caught.value) == (
        "Guarded OpenAI conversational turn has unresolved model tool calls."
    )
    assert isinstance(caught.value, RuntimeError)
    response_constructor.assert_not_called()
    stack.spies["service"].assert_awaited_once()
    stack.spies["execution"].assert_awaited_once()
    stack.spies["tool"].assert_awaited_once()
    stack.spies["result"].assert_called_once()
    stack.spies["continued"].assert_awaited_once()
    assert stack.create.await_count == 2


@pytest.mark.parametrize("failure", ["unadvertised", "multiple"])
def test_initial_call_rejection_preserves_service_boundary(stack, failure):
    items = (
        [function(name="unknown.tool")]
        if failure == "unadvertised"
        else [function(), function("call_other")]
    )
    stack.create.side_effect = [raw(*items)]
    error_type = (
        UnadvertisedModelToolCallError
        if failure == "unadvertised"
        else MultipleModelToolCallsUnsupportedError
    )
    with pytest.raises(error_type):
        run(stack)
    assert stack.spies["service"].call_args.args[0].tools is TOOLS
    stack.spies["service"].assert_awaited_once()
    stack.create.assert_awaited_once()
    stack.factory.assert_not_called()
    for name in ("execution", "authorize", "lookup", "tool", "result", "continued"):
        stack.spies[name].assert_not_called()


@pytest.mark.parametrize(
    "outcome,error_type",
    [
        (ToolAuthorizationOutcome.DENY, ToolAuthorizationDeniedError),
        (ToolAuthorizationOutcome.REQUIRE_CONFIRMATION, ToolConfirmationRequiredError),
        (None, ToolAuthorizationDeniedError),
    ],
)
def test_advertised_tools_still_require_authorization(stack, outcome, error_type):
    policy = StaticToolAuthorizationPolicy(
        {} if outcome is None else {"calculator.basic": outcome}
    )

    async def authorize(invocation):
        return await stack.authorize_method(policy, invocation)

    stack.spies["authorize"].side_effect = authorize
    with pytest.raises(error_type):
        run(stack)
    stack.spies["authorize"].assert_awaited_once()
    stack.spies["service"].assert_awaited_once()
    stack.create.assert_awaited_once()
    for name in ("lookup", "tool", "result", "continued"):
        stack.spies[name].assert_not_called()


@pytest.mark.parametrize("invalid", [None, {}, object(), "result"])
def test_service_result_type_is_validated(stack, invalid):
    stack.spies["service"].return_value = invalid
    with pytest.raises(TypeError, match="must return an OpenAIToolTurnResult"):
        run(stack)
    stack.spies["service"].assert_awaited_once()
    stack.create.assert_not_awaited()


def test_advertised_and_allowed_unknown_tool_preserves_lookup_failure(stack):
    tools = (LanguageModelToolDefinition("unknown.tool", "Unknown", {}),)
    stack.agent = GuardedOpenAIConversationalAgent(stack.service, tools)
    stack.create.side_effect = [raw(function(name="unknown.tool"))]
    policy = StaticToolAuthorizationPolicy(
        {"unknown.tool": ToolAuthorizationOutcome.ALLOW}
    )

    async def authorize(invocation):
        return await stack.authorize_method(policy, invocation)

    stack.spies["authorize"].side_effect = authorize
    with pytest.raises(ToolNotFoundError):
        run(stack)
    stack.spies["authorize"].assert_awaited_once()
    stack.spies["lookup"].assert_called_once_with("unknown.tool")
    stack.spies["tool"].assert_not_awaited()
    stack.spies["result"].assert_not_called()
    stack.spies["continued"].assert_not_awaited()


@pytest.mark.parametrize("stage", ["service", "initial", "tool", "result", "continued"])
def test_failures_propagate_exact_objects_without_retry(stack, stage):
    error = (
        LanguageModelProviderError("provider failed")
        if stage == "initial"
        else ToolExecutionError("tool failed")
        if stage == "tool"
        else RuntimeError(stage + " failed")
    )
    stack.spies[stage].side_effect = error
    with pytest.raises(type(error)) as caught:
        run(stack)
    assert caught.value is error
    stack.spies[stage].assert_called_once()
    stack.spies["service"].assert_awaited_once()
    assert stack.spies["tool"].await_count <= 1
    assert stack.spies["continued"].await_count <= 1
    assert stack.create.await_count <= 1


def test_mismatched_continuation_request_fails_before_execution(stack):
    generate = stack.provider.generate_with_continuation

    async def unrelated(model_request):
        initial = await generate(model_request)
        return replace(
            initial,
            continuation=replace(initial.continuation, request=replace(model_request)),
        )

    stack.provider.generate_with_continuation = AsyncMock(side_effect=unrelated)
    with pytest.raises(ValueError, match="must match the original tool turn request"):
        run(stack)
    stack.spies["service"].assert_awaited_once()
    stack.factory.assert_not_called()
    stack.spies["authorize"].assert_not_awaited()
    stack.spies["continued"].assert_not_awaited()


def test_invalid_tool_result_correlation_is_not_a_success(stack):
    stack.spies["result"].return_value = LanguageModelToolResult("unrelated", 42)
    with pytest.raises(ValueError, match="call_id must match"):
        run(stack)
    stack.spies["service"].assert_awaited_once()
    stack.spies["tool"].assert_awaited_once()
    stack.spies["continued"].assert_not_awaited()


@pytest.mark.parametrize("directive", ["/chat", "/conversation"])
def test_manual_orchestration_preserves_prepared_content_and_ids(stack, directive):
    request, context = interaction(directive + " What is 6 times 7?")
    response = asyncio.run(orchestrator(stack.agent).handle(request, context))
    prepared = stack.spies["render"].call_args.args[0]
    assert prepared is not request
    assert prepared.content == "What is 6 times 7?"
    assert prepared.request_id == response.request_id == request.request_id
    assert (
        prepared.conversation_id == response.conversation_id == request.conversation_id
    )
    model_request = stack.spies["service"].call_args.args[0]
    assert model_request.messages[-1].content == prepared.content
    assert response.content == "The result is 42."
    assert request.content.startswith(directive)
    assert stack.create.await_count == 2
    stack.spies["tool"].assert_awaited_once()


@pytest.mark.parametrize("mode", ["text", "tool", "incomplete", "denied"])
def test_stateful_service_appends_only_completed_turns(stack, mode):
    if mode == "text":
        stack.create.side_effect = [raw(text="Hello from TUESDAY")]
    elif mode == "incomplete":
        stack.create.side_effect = [raw(function()), raw(function("call_next"))]
    elif mode == "denied":
        policy = StaticToolAuthorizationPolicy({})

        async def authorize(invocation):
            return await stack.authorize_method(policy, invocation)

        stack.spies["authorize"].side_effect = authorize
    repository = InMemoryConversationRepository()
    service = StatefulConversationService(orchestrator(stack.agent), repository)
    request, _ = interaction("/chat What is 6 times 7?")
    prior = (ConversationMessage(MessageRole.USER, "Prior question"),)
    asyncio.run(repository.append_messages(request.conversation_id, prior))
    if mode in ("incomplete", "denied"):
        error_type = (
            IncompleteGuardedToolTurnError
            if mode == "incomplete"
            else ToolAuthorizationDeniedError
        )
        with pytest.raises(error_type):
            asyncio.run(service.handle(request))
    else:
        response = asyncio.run(service.handle(request))
        assert response.content == (
            "Hello from TUESDAY" if mode == "text" else "The result is 42."
        )
    saved = asyncio.run(repository.get_context(request.conversation_id))
    assert saved.messages[0] is prior[0]
    if mode in ("incomplete", "denied"):
        assert saved.messages == prior
    else:
        assert [(message.role, message.content) for message in saved.messages[1:]] == [
            (MessageRole.USER, request.content),
            (MessageRole.ASSISTANT, response.content),
        ]
    if mode == "incomplete":
        # History was not appended, but the first execution remains completed.
        stack.spies["tool"].assert_awaited_once()
        assert stack.spies["result"].call_args.args[0].result.output == 42


def test_existing_factories_and_text_only_agent_remain_unchanged(stack):
    default = create_default_orchestrator()
    assert type(default._registry.get("conversation")) is ConversationalAgent
    old = create_model_backed_orchestrator(stack.provider)
    assert type(old._registry.get("conversation")) is ModelBackedConversationalAgent
    request, context = interaction("/chat Hello")
    assert (
        asyncio.run(default.handle(request, context)).content
        == "TUESDAY received: Hello"
    )
    stack.create.side_effect = [raw(function())]
    with pytest.raises(UnsupportedModelToolCallError):
        asyncio.run(old.handle(request, context))
    stack.spies["service"].assert_not_awaited()
    stack.spies["tool"].assert_not_awaited()
    stack.create.assert_awaited_once()


def test_exact_public_exports_and_architectural_boundaries():
    assert agents.__all__ == [
        "AgentNotFoundError",
        "AgentRegistrationError",
        "AgentRegistry",
        "BaseAgent",
        "ConversationalAgent",
        "GuardedOpenAIConversationalAgent",
        "IncompleteGuardedToolTurnError",
        "ModelBackedConversationalAgent",
        "UnsupportedModelToolCallError",
    ]
    assert agent_module.__all__ == [
        "GuardedOpenAIConversationalAgent",
        "IncompleteGuardedToolTurnError",
    ]
    for name in agent_module.__all__:
        assert getattr(agents, name) is getattr(agent_module, name)
    tree = ast.parse(inspect.getsource(agent_module))
    assert {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } == {
        "tuesday.agents.base",
        "tuesday.domain",
        "tuesday.language_models",
        "tuesday.prompting",
        "tuesday.services.openai_tool_turn",
    }
    handle = ast.parse(inspect.getsource(agent_module))
    assert not any(
        isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.Try))
        for node in ast.walk(handle)
    )
    methods = [
        node.func.attr
        for node in ast.walk(handle)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert sorted(methods) == ["_validate_context", "render", "run_once"]


@pytest.mark.parametrize(
    "modules",
    [
        ("agents", "services", "composition"),
        ("services", "agents", "composition"),
        ("composition", "services", "agents"),
    ],
)
def test_clean_process_import_orders(modules):
    result = subprocess.run(
        [sys.executable, "-c", "; ".join("import tuesday." + name for name in modules)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
