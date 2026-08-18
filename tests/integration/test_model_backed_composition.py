"""Integration tests for TUESDAY's model-backed application composition."""

import ast
import asyncio
import inspect
from uuid import UUID, uuid4

import pytest

import tuesday.composition as composition_module
from tuesday.agents import AgentRegistry, ModelBackedConversationalAgent
from tuesday.composition import (
    create_default_orchestrator,
    create_model_backed_orchestrator,
)
from tuesday.domain import (
    ConversationContext,
    ConversationMessage,
    MessageRole,
    TuesdayRequest,
    TuesdayResponse,
)
from tuesday.language_models import (
    BaseLanguageModelProvider,
    LanguageModelMessage,
    LanguageModelProviderError,
    LanguageModelRequest,
    LanguageModelResponse,
)
from tuesday.orchestration import TuesdayOrchestrator
from tuesday.preparation import (
    DirectiveRequestPreparer,
    RequestPreparationError,
)
from tuesday.prompting import ConversationalPromptRenderer
from tuesday.routing import DeterministicRouter, NoRouteFoundError


class RecordingProvider(BaseLanguageModelProvider):
    """Deterministic provider stub for application-composition tests."""

    def __init__(self, content: str = "Model response") -> None:
        self.content = content
        self.requests: list[LanguageModelRequest] = []

    @property
    def name(self) -> str:
        return "stub"

    async def generate(
        self,
        request: LanguageModelRequest,
    ) -> LanguageModelResponse:
        self.requests.append(request)
        return LanguageModelResponse(
            content=self.content,
            provider=self.name,
            model="stub-model",
        )


class RaisingProvider(BaseLanguageModelProvider):
    """Provider stub that records one request and raises."""

    def __init__(self, error: LanguageModelProviderError) -> None:
        self.error = error
        self.requests: list[LanguageModelRequest] = []

    @property
    def name(self) -> str:
        return "raising"

    async def generate(
        self,
        request: LanguageModelRequest,
    ) -> LanguageModelResponse:
        self.requests.append(request)
        raise self.error


def run_interaction(
    orchestrator: TuesdayOrchestrator,
    content: str,
    messages: tuple[ConversationMessage, ...] = (),
) -> tuple[TuesdayRequest, ConversationContext, TuesdayResponse]:
    conversation_id = uuid4()
    request = TuesdayRequest(content=content, conversation_id=conversation_id)
    context = ConversationContext(
        conversation_id=conversation_id,
        messages=messages,
    )
    response = asyncio.run(orchestrator.handle(request, context))
    return request, context, response


def test_factory_returns_orchestrator_with_model_backed_components() -> None:
    provider = RecordingProvider()
    orchestrator = create_model_backed_orchestrator(provider)

    assert isinstance(orchestrator, TuesdayOrchestrator)
    assert isinstance(orchestrator._router, DeterministicRouter)
    assert isinstance(orchestrator._registry, AgentRegistry)
    assert isinstance(orchestrator._request_preparer, DirectiveRequestPreparer)
    assert orchestrator._registry.names == ("conversation",)

    agent = orchestrator._registry.get("conversation")
    assert isinstance(agent, ModelBackedConversationalAgent)
    assert agent._provider is provider


def test_model_backed_composition_has_exact_existing_route_policy() -> None:
    orchestrator = create_model_backed_orchestrator(RecordingProvider())

    assert orchestrator._router.routes == (
        ("chat", "conversation"),
        ("conversation", "conversation"),
    )


@pytest.mark.parametrize("directive", ["/chat", "/conversation"])
def test_model_backed_directives_complete_end_to_end_interaction(
    directive: str,
) -> None:
    provider = RecordingProvider("Generated answer")
    request, _, response = run_interaction(
        create_model_backed_orchestrator(provider),
        f"{directive} Hello",
    )

    assert response.content == "Generated answer"
    assert response.conversation_id == request.conversation_id
    assert response.request_id == request.request_id
    assert isinstance(response.response_id, UUID)
    assert len(provider.requests) == 1


def test_routing_directive_is_removed_before_model_rendering() -> None:
    provider = RecordingProvider()

    run_interaction(
        create_model_backed_orchestrator(provider),
        "/chat Hello TUESDAY",
    )

    model_request = provider.requests[0]
    assert model_request.messages[-1] == LanguageModelMessage(
        role=MessageRole.USER,
        content="Hello TUESDAY",
    )
    assert "/chat" not in model_request.messages[-1].content


def test_prior_history_reaches_provider_in_exact_order_and_roles() -> None:
    history = (
        ConversationMessage(MessageRole.USER, "Earlier question"),
        ConversationMessage(MessageRole.ASSISTANT, "Earlier answer"),
        ConversationMessage(MessageRole.SYSTEM, "Earlier system context"),
    )
    provider = RecordingProvider()

    run_interaction(
        create_model_backed_orchestrator(provider),
        "/conversation Current question",
        history,
    )

    messages = provider.requests[0].messages
    assert [(message.role, message.content) for message in messages[1:]] == [
        (MessageRole.USER, "Earlier question"),
        (MessageRole.ASSISTANT, "Earlier answer"),
        (MessageRole.SYSTEM, "Earlier system context"),
        (MessageRole.USER, "Current question"),
    ]


def test_custom_renderer_is_wired_into_model_backed_agent() -> None:
    provider = RecordingProvider()
    renderer = ConversationalPromptRenderer(
        system_prompt="Custom integration system prompt"
    )

    orchestrator = create_model_backed_orchestrator(
        provider,
        renderer=renderer,
    )
    run_interaction(orchestrator, "/chat Hello")

    agent = orchestrator._registry.get("conversation")
    assert agent._renderer is renderer
    assert provider.requests[0].messages[0].content == (
        "Custom integration system prompt"
    )


def test_source_request_and_context_remain_unchanged_end_to_end() -> None:
    provider = RecordingProvider()
    conversation_id = uuid4()
    history = (
        ConversationMessage(MessageRole.USER, "Prior"),
    )
    request = TuesdayRequest(
        content="/chat   Hello with spacing  ",
        conversation_id=conversation_id,
    )
    context = ConversationContext(
        conversation_id=conversation_id,
        messages=history,
    )
    original_request = request
    original_context = context

    response = asyncio.run(
        create_model_backed_orchestrator(provider).handle(request, context)
    )

    assert request == original_request
    assert request.content == "/chat   Hello with spacing  "
    assert context == original_context
    assert context.messages == history
    assert provider.requests[0].messages[-1].content == "Hello with spacing  "
    assert response.request_id == request.request_id


def test_provider_error_propagates_once_through_full_application_path() -> None:
    error = LanguageModelProviderError("provider failed")
    provider = RaisingProvider(error)

    with pytest.raises(LanguageModelProviderError) as caught:
        run_interaction(
            create_model_backed_orchestrator(provider),
            "/chat Hello",
        )

    assert caught.value is error
    assert len(provider.requests) == 1


@pytest.mark.parametrize(
    "content",
    [
        "/unknown Hello",
        "Hello",
        "Hello /chat",
    ],
)
def test_invalid_or_missing_route_never_calls_provider(content: str) -> None:
    provider = RecordingProvider()

    with pytest.raises(NoRouteFoundError):
        run_interaction(create_model_backed_orchestrator(provider), content)

    assert provider.requests == []


@pytest.mark.parametrize("content", ["/chat", "/conversation   "])
def test_known_route_without_content_never_calls_provider(content: str) -> None:
    provider = RecordingProvider()

    with pytest.raises(
        RequestPreparationError,
        match="must include content after the routing directive",
    ):
        run_interaction(create_model_backed_orchestrator(provider), content)

    assert provider.requests == []


def test_request_context_mismatch_never_calls_provider() -> None:
    provider = RecordingProvider()
    request = TuesdayRequest(content="/chat Hello")
    context = ConversationContext()

    with pytest.raises(
        ValueError,
        match="Request and context must belong to the same conversation",
    ):
        asyncio.run(
            create_model_backed_orchestrator(provider).handle(request, context)
        )

    assert provider.requests == []


def test_factory_rejects_non_provider_through_agent_contract() -> None:
    with pytest.raises(TypeError, match="BaseLanguageModelProvider"):
        create_model_backed_orchestrator(object())  # type: ignore[arg-type]


def test_factory_rejects_invalid_renderer_through_agent_contract() -> None:
    with pytest.raises(TypeError, match="ConversationalPromptRenderer or None"):
        create_model_backed_orchestrator(
            RecordingProvider(),
            renderer=object(),  # type: ignore[arg-type]
        )


def test_factory_calls_create_independent_runtime_graphs() -> None:
    first_provider = RecordingProvider("First")
    second_provider = RecordingProvider("Second")

    first = create_model_backed_orchestrator(first_provider)
    second = create_model_backed_orchestrator(second_provider)

    assert first is not second
    assert first._registry is not second._registry
    assert first._router is not second._router
    assert first._request_preparer is not second._request_preparer
    assert first._registry.get("conversation") is not second._registry.get(
        "conversation"
    )
    assert first._registry.get("conversation")._provider is first_provider
    assert second._registry.get("conversation")._provider is second_provider


def test_default_composition_remains_deterministic_and_independent() -> None:
    request, _, response = run_interaction(
        create_default_orchestrator(),
        "/chat Hello",
    )

    assert response.content == "TUESDAY received: Hello"
    assert response.request_id == request.request_id


def test_composition_module_remains_provider_neutral() -> None:
    tree = ast.parse(inspect.getsource(composition_module))
    imported_modules = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert "openai" not in imported_modules
    assert "tuesday.language_models.openai" not in imported_modules
    assert "tuesday.config" not in imported_modules

    source = inspect.getsource(composition_module)
    assert "OpenAILanguageModelProvider" not in source
    assert "load_settings" not in source
    assert "os.environ" not in source


def test_public_composition_exports_both_factories() -> None:
    assert composition_module.__all__ == [
        "create_default_orchestrator",
        "create_model_backed_orchestrator",
    ]
