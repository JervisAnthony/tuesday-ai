"""Tests for TUESDAY's provider-neutral model-backed conversational agent."""

import ast
import asyncio
import inspect
from uuid import UUID, uuid4

import pytest

import tuesday.agents.model_backed as model_backed_module
from tuesday.agents import (
    BaseAgent,
    ConversationalAgent,
    ModelBackedConversationalAgent,
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
from tuesday.prompting import (
    DEFAULT_CONVERSATIONAL_SYSTEM_PROMPT,
    ConversationalPromptRenderer,
)


class RecordingProvider(BaseLanguageModelProvider):
    """Deterministic provider stub that records generated requests."""

    def __init__(
        self,
        responses: tuple[LanguageModelResponse, ...] | None = None,
    ) -> None:
        self.requests: list[LanguageModelRequest] = []
        self._responses = responses or (
            LanguageModelResponse(
                content="Model response",
                provider="stub",
                model="stub-model",
            ),
        )

    @property
    def name(self) -> str:
        return "stub"

    async def generate(
        self,
        request: LanguageModelRequest,
    ) -> LanguageModelResponse:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self._responses) - 1)
        return self._responses[index]


class RaisingProvider(BaseLanguageModelProvider):
    """Provider stub that always raises one configured error."""

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


class InvalidResponseProvider(BaseLanguageModelProvider):
    """Provider stub that violates the provider return contract."""

    def __init__(self) -> None:
        self.requests: list[LanguageModelRequest] = []

    @property
    def name(self) -> str:
        return "invalid"

    async def generate(self, request: LanguageModelRequest) -> LanguageModelResponse:
        self.requests.append(request)
        return object()  # type: ignore[return-value]


def run_handle(
    agent: ModelBackedConversationalAgent,
    request: TuesdayRequest,
    context: ConversationContext,
) -> TuesdayResponse:
    return asyncio.run(agent.handle(request, context))


def matching_interaction(
    content: str = "Hello",
    messages: tuple[ConversationMessage, ...] = (),
) -> tuple[TuesdayRequest, ConversationContext]:
    conversation_id = uuid4()
    return (
        TuesdayRequest(content=content, conversation_id=conversation_id),
        ConversationContext(conversation_id=conversation_id, messages=messages),
    )


def test_agent_inherits_base_agent() -> None:
    assert isinstance(ModelBackedConversationalAgent(RecordingProvider()), BaseAgent)


def test_agent_preserves_stable_conversation_registry_identity() -> None:
    agent = ModelBackedConversationalAgent(RecordingProvider())

    assert agent.name == "conversation"
    assert agent.name == ConversationalAgent().name
    assert "model" in agent.description.lower()
    assert "conversational" in agent.description.lower()


def test_handle_is_asynchronous() -> None:
    assert inspect.iscoroutinefunction(ModelBackedConversationalAgent.handle)


def test_constructor_requires_language_model_provider() -> None:
    with pytest.raises(TypeError, match="BaseLanguageModelProvider"):
        ModelBackedConversationalAgent(object())  # type: ignore[arg-type]


def test_constructor_rejects_invalid_renderer() -> None:
    with pytest.raises(TypeError, match="ConversationalPromptRenderer or None"):
        ModelBackedConversationalAgent(
            RecordingProvider(),
            renderer=object(),  # type: ignore[arg-type]
        )


def test_constructor_uses_default_renderer_when_none_is_supplied() -> None:
    provider = RecordingProvider()
    agent = ModelBackedConversationalAgent(provider)
    request, context = matching_interaction()

    run_handle(agent, request, context)

    assert provider.requests[0].messages[0] == LanguageModelMessage(
        role=MessageRole.SYSTEM,
        content=DEFAULT_CONVERSATIONAL_SYSTEM_PROMPT,
    )


def test_custom_renderer_is_used_exactly() -> None:
    provider = RecordingProvider()
    renderer = ConversationalPromptRenderer(system_prompt="Custom TUESDAY prompt")
    agent = ModelBackedConversationalAgent(provider, renderer=renderer)
    request, context = matching_interaction("Hello")

    run_handle(agent, request, context)

    assert provider.requests[0].messages[0].content == "Custom TUESDAY prompt"


def test_matching_request_and_context_generate_correlated_response() -> None:
    provider = RecordingProvider()
    agent = ModelBackedConversationalAgent(provider)
    request, context = matching_interaction("Hello")

    response = run_handle(agent, request, context)

    assert isinstance(response, TuesdayResponse)
    assert response.content == "Model response"
    assert response.conversation_id == request.conversation_id
    assert response.request_id == request.request_id
    assert isinstance(response.response_id, UUID)


def test_empty_context_renders_system_then_current_user_message() -> None:
    provider = RecordingProvider()
    agent = ModelBackedConversationalAgent(provider)
    request, context = matching_interaction("Hello TUESDAY")

    run_handle(agent, request, context)

    assert provider.requests == [
        LanguageModelRequest(
            messages=(
                LanguageModelMessage(
                    role=MessageRole.SYSTEM,
                    content=DEFAULT_CONVERSATIONAL_SYSTEM_PROMPT,
                ),
                LanguageModelMessage(
                    role=MessageRole.USER,
                    content="Hello TUESDAY",
                ),
            )
        )
    ]


def test_prior_history_is_forwarded_in_exact_order_and_roles() -> None:
    prior_messages = (
        ConversationMessage(MessageRole.USER, "First"),
        ConversationMessage(MessageRole.ASSISTANT, "Second"),
        ConversationMessage(MessageRole.SYSTEM, "Third"),
    )
    provider = RecordingProvider()
    agent = ModelBackedConversationalAgent(provider)
    request, context = matching_interaction("Fourth", prior_messages)

    run_handle(agent, request, context)

    assert [
        (message.role, message.content)
        for message in provider.requests[0].messages
    ] == [
        (MessageRole.SYSTEM, DEFAULT_CONVERSATIONAL_SYSTEM_PROMPT),
        (MessageRole.USER, "First"),
        (MessageRole.ASSISTANT, "Second"),
        (MessageRole.SYSTEM, "Third"),
        (MessageRole.USER, "Fourth"),
    ]


def test_meaningful_whitespace_and_multiline_content_are_preserved() -> None:
    prior = ConversationMessage(
        MessageRole.ASSISTANT,
        "  Prior response\nline two  ",
    )
    provider = RecordingProvider()
    agent = ModelBackedConversationalAgent(provider)
    request, context = matching_interaction(
        "  Current request\nline two  ",
        (prior,),
    )

    run_handle(agent, request, context)

    messages = provider.requests[0].messages
    assert messages[1].content == "  Prior response\nline two  "
    assert messages[2].content == "  Current request\nline two  "


def test_provider_is_awaited_exactly_once_per_handle_call() -> None:
    provider = RecordingProvider()
    agent = ModelBackedConversationalAgent(provider)
    request, context = matching_interaction()

    run_handle(agent, request, context)

    assert len(provider.requests) == 1


def test_model_response_content_is_preserved_exactly() -> None:
    provider = RecordingProvider(
        responses=(
            LanguageModelResponse(
                content="  Generated answer\nsecond line  ",
                provider="stub",
                model="stub-model",
            ),
        )
    )
    agent = ModelBackedConversationalAgent(provider)
    request, context = matching_interaction()

    response = run_handle(agent, request, context)

    assert response.content == "  Generated answer\nsecond line  "


def test_provider_and_model_metadata_do_not_leak_into_tuesday_response() -> None:
    provider = RecordingProvider(
        responses=(
            LanguageModelResponse(
                content="Answer",
                provider="provider-secret-name",
                model="model-secret-name",
            ),
        )
    )
    agent = ModelBackedConversationalAgent(provider)
    request, context = matching_interaction()

    response = run_handle(agent, request, context)

    assert response.content == "Answer"
    assert not hasattr(response, "provider")
    assert not hasattr(response, "model")
    assert "provider-secret-name" not in response.content
    assert "model-secret-name" not in response.content


def test_provider_error_propagates_unchanged_without_retry() -> None:
    error = LanguageModelProviderError("provider failed")
    provider = RaisingProvider(error)
    agent = ModelBackedConversationalAgent(provider)
    request, context = matching_interaction()

    with pytest.raises(LanguageModelProviderError) as caught:
        run_handle(agent, request, context)

    assert caught.value is error
    assert len(provider.requests) == 1


def test_mismatched_context_fails_before_provider_execution() -> None:
    provider = RecordingProvider()
    agent = ModelBackedConversationalAgent(provider)
    request = TuesdayRequest(content="Hello")
    context = ConversationContext()

    with pytest.raises(
        ValueError,
        match="Request and context must belong to the same conversation",
    ):
        run_handle(agent, request, context)

    assert provider.requests == []


def test_malformed_context_fails_before_provider_execution() -> None:
    provider = RecordingProvider()
    agent = ModelBackedConversationalAgent(provider)
    request, _ = matching_interaction()
    context = ConversationContext(
        conversation_id=request.conversation_id,
        messages=(object(),),  # type: ignore[arg-type]
    )

    with pytest.raises(
        TypeError,
        match="context messages must be ConversationMessage instances",
    ):
        run_handle(agent, request, context)

    assert provider.requests == []


def test_invalid_provider_response_is_rejected_explicitly() -> None:
    provider = InvalidResponseProvider()
    agent = ModelBackedConversationalAgent(provider)
    request, context = matching_interaction()

    with pytest.raises(
        TypeError,
        match="provider must return a LanguageModelResponse",
    ):
        run_handle(agent, request, context)

    assert len(provider.requests) == 1


def test_successful_execution_does_not_mutate_request_or_context() -> None:
    prior = ConversationMessage(MessageRole.USER, "Earlier")
    request, context = matching_interaction("Current", (prior,))
    original_request = request
    original_context = context
    provider = RecordingProvider()
    agent = ModelBackedConversationalAgent(provider)

    run_handle(agent, request, context)

    assert request == original_request
    assert context == original_context
    assert context.messages[0] is prior


def test_sequential_requests_remain_independent() -> None:
    provider = RecordingProvider(
        responses=(
            LanguageModelResponse("First answer", "stub", "stub-model"),
            LanguageModelResponse("Second answer", "stub", "stub-model"),
        )
    )
    agent = ModelBackedConversationalAgent(provider)
    first_request, first_context = matching_interaction("First question")
    second_request, second_context = matching_interaction("Second question")

    first = run_handle(agent, first_request, first_context)
    second = run_handle(agent, second_request, second_context)

    assert first.content == "First answer"
    assert second.content == "Second answer"
    assert provider.requests[0] is not provider.requests[1]
    assert provider.requests[0].messages[-1].content == "First question"
    assert provider.requests[1].messages[-1].content == "Second question"


def test_multiple_executions_create_independent_response_ids() -> None:
    provider = RecordingProvider()
    agent = ModelBackedConversationalAgent(provider)
    request, context = matching_interaction()

    first = run_handle(agent, request, context)
    second = run_handle(agent, request, context)

    assert first.response_id != second.response_id


def test_agent_retains_only_provider_and_renderer_dependencies() -> None:
    provider = RecordingProvider()
    renderer = ConversationalPromptRenderer()
    agent = ModelBackedConversationalAgent(provider, renderer=renderer)
    request, context = matching_interaction()

    run_handle(agent, request, context)

    assert not hasattr(agent, "__dict__")
    assert agent._provider is provider
    assert agent._renderer is renderer


def test_handle_contract_has_no_registry_routing_or_configuration_parameters() -> None:
    parameters = tuple(
        inspect.signature(ModelBackedConversationalAgent.handle).parameters
    )

    assert parameters == ("self", "request", "context")


def test_model_backed_module_has_provider_neutral_dependencies_only() -> None:
    tree = ast.parse(inspect.getsource(model_backed_module))
    imported_modules = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert imported_modules == {
        "tuesday.agents.base",
        "tuesday.domain",
        "tuesday.language_models",
        "tuesday.prompting",
    }


def test_model_backed_module_has_no_openai_or_composition_dependency() -> None:
    source = inspect.getsource(model_backed_module)

    for forbidden in (
        "openai",
        "OpenAILanguageModelProvider",
        "LanguageModelSettings",
        "AgentRegistry",
        "BaseRouter",
        "TuesdayOrchestrator",
        "DirectiveRequestPreparer",
        "create_default_orchestrator",
    ):
        assert forbidden not in source


def test_agent_does_not_implement_retry_or_fallback_behavior() -> None:
    source = inspect.getsource(ModelBackedConversationalAgent.handle)

    assert "while " not in source
    assert "for " not in source
    assert "retry" not in source.lower()
    assert "fallback" not in source.lower()


def test_public_agents_package_exports_model_backed_agent() -> None:
    import tuesday.agents as agents

    assert agents.ModelBackedConversationalAgent is ModelBackedConversationalAgent
