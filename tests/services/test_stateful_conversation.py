"""Tests for TUESDAY's stateful conversation service."""

import asyncio
import inspect
from uuid import UUID, uuid4

import pytest

import tuesday.services.conversation as service_module
from tuesday.composition import create_default_orchestrator
from tuesday.conversations import (
    BaseConversationRepository,
    ConversationRepositoryError,
    InMemoryConversationRepository,
)
from tuesday.domain import (
    ConversationContext,
    ConversationMessage,
    MessageRole,
    TuesdayRequest,
    TuesdayResponse,
)
from tuesday.orchestration import TuesdayOrchestrator
from tuesday.services import StatefulConversationService


class RecordingRepository(BaseConversationRepository):
    """Repository double with deterministic call recording."""

    def __init__(
        self,
        context: ConversationContext,
        *,
        events: list[str] | None = None,
        get_error: Exception | None = None,
        append_error: Exception | None = None,
    ) -> None:
        self.context = context
        self.events = events if events is not None else []
        self.get_error = get_error
        self.append_error = append_error
        self.get_calls: list[UUID] = []
        self.append_calls: list[
            tuple[UUID, tuple[ConversationMessage, ...]]
        ] = []

    async def get_context(self, conversation_id: UUID) -> ConversationContext:
        self.events.append("get")
        self.get_calls.append(conversation_id)
        if self.get_error is not None:
            raise self.get_error
        return self.context

    async def append_messages(
        self,
        conversation_id: UUID,
        messages: tuple[ConversationMessage, ...],
    ) -> ConversationContext:
        self.events.append("append")
        self.append_calls.append((conversation_id, messages))
        if self.append_error is not None:
            raise self.append_error
        return ConversationContext(
            conversation_id=conversation_id,
            messages=(*self.context.messages, *messages),
        )


class RecordingOrchestrator(TuesdayOrchestrator):
    """Orchestrator double with deterministic call recording."""

    def __init__(
        self,
        response: TuesdayResponse,
        *,
        events: list[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.events = events if events is not None else []
        self.error = error
        self.calls: list[
            tuple[TuesdayRequest, ConversationContext]
        ] = []

    async def handle(
        self,
        request: TuesdayRequest,
        context: ConversationContext,
    ) -> TuesdayResponse:
        self.events.append("handle")
        self.calls.append((request, context))
        if self.error is not None:
            raise self.error
        return self.response


def run_handle(
    service: StatefulConversationService,
    request: TuesdayRequest,
) -> TuesdayResponse:
    return asyncio.run(service.handle(request))


def make_response(
    request: TuesdayRequest,
    content: str = "Assistant reply",
) -> TuesdayResponse:
    return TuesdayResponse(
        content=content,
        conversation_id=request.conversation_id,
        request_id=request.request_id,
    )


def test_service_contract_is_async_and_slotted() -> None:
    assert inspect.iscoroutinefunction(StatefulConversationService.handle)
    assert StatefulConversationService.__slots__ == (
        "_orchestrator",
        "_repository",
    )
    parameters = inspect.signature(
        StatefulConversationService.handle
    ).parameters
    assert tuple(parameters) == ("self", "request")


def test_constructor_preserves_dependency_identity() -> None:
    orchestrator = create_default_orchestrator()
    repository = InMemoryConversationRepository()

    service = StatefulConversationService(orchestrator, repository)

    assert service._orchestrator is orchestrator
    assert service._repository is repository
    assert not hasattr(service, "__dict__")


@pytest.mark.parametrize("invalid_orchestrator", [None, object(), "invalid"])
def test_constructor_rejects_non_orchestrator(
    invalid_orchestrator: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="orchestrator must be a TuesdayOrchestrator",
    ):
        StatefulConversationService(
            invalid_orchestrator,  # type: ignore[arg-type]
            InMemoryConversationRepository(),
        )


@pytest.mark.parametrize("invalid_repository", [None, object(), "invalid"])
def test_constructor_rejects_non_repository(
    invalid_repository: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="repository must be a BaseConversationRepository",
    ):
        StatefulConversationService(
            create_default_orchestrator(),
            invalid_repository,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("invalid_request", [None, object(), "invalid", 123])
def test_handle_rejects_non_request_before_repository_access(
    invalid_request: object,
) -> None:
    conversation_id = uuid4()
    valid_request = TuesdayRequest(
        content="/chat Hello",
        conversation_id=conversation_id,
    )
    repository = RecordingRepository(
        ConversationContext(conversation_id=conversation_id)
    )
    orchestrator = RecordingOrchestrator(make_response(valid_request))
    service = StatefulConversationService(orchestrator, repository)

    with pytest.raises(TypeError, match="request must be a TuesdayRequest"):
        asyncio.run(
            service.handle(invalid_request)  # type: ignore[arg-type]
        )

    assert repository.get_calls == []
    assert orchestrator.calls == []
    assert repository.append_calls == []


def test_successful_lifecycle_is_load_handle_append() -> None:
    events: list[str] = []
    conversation_id = uuid4()
    request = TuesdayRequest(
        content="/chat Hello",
        conversation_id=conversation_id,
    )
    context = ConversationContext(conversation_id=conversation_id)
    response = make_response(request)
    repository = RecordingRepository(context, events=events)
    orchestrator = RecordingOrchestrator(response, events=events)

    returned = run_handle(
        StatefulConversationService(orchestrator, repository),
        request,
    )

    assert returned is response
    assert events == ["get", "handle", "append"]
    assert repository.get_calls == [conversation_id]
    assert orchestrator.calls == [(request, context)]


def test_repository_result_must_be_conversation_context() -> None:
    conversation_id = uuid4()
    request = TuesdayRequest(
        content="/chat Hello",
        conversation_id=conversation_id,
    )
    repository = RecordingRepository(
        ConversationContext(conversation_id=conversation_id)
    )
    repository.context = object()  # type: ignore[assignment]
    orchestrator = RecordingOrchestrator(make_response(request))

    with pytest.raises(TypeError, match="must return a ConversationContext"):
        run_handle(
            StatefulConversationService(orchestrator, repository),
            request,
        )

    assert orchestrator.calls == []
    assert repository.append_calls == []


def test_repository_context_must_match_request_conversation() -> None:
    request = TuesdayRequest(
        content="/chat Hello",
        conversation_id=uuid4(),
    )
    repository = RecordingRepository(
        ConversationContext(conversation_id=uuid4())
    )
    orchestrator = RecordingOrchestrator(make_response(request))

    with pytest.raises(
        ValueError,
        match="Stored conversation context must match",
    ):
        run_handle(
            StatefulConversationService(orchestrator, repository),
            request,
        )

    assert orchestrator.calls == []
    assert repository.append_calls == []


def test_success_persists_exact_original_user_and_assistant_content() -> None:
    conversation_id = uuid4()
    request = TuesdayRequest(
        content="/chat   Hello with spacing  ",
        conversation_id=conversation_id,
    )
    response = make_response(request, "  Assistant\nreply  ")
    repository = RecordingRepository(
        ConversationContext(conversation_id=conversation_id)
    )
    orchestrator = RecordingOrchestrator(response)

    returned = run_handle(
        StatefulConversationService(orchestrator, repository),
        request,
    )

    assert returned is response
    appended_id, messages = repository.append_calls[0]
    assert appended_id == conversation_id
    assert tuple(message.role for message in messages) == (
        MessageRole.USER,
        MessageRole.ASSISTANT,
    )
    assert tuple(message.content for message in messages) == (
        "/chat   Hello with spacing  ",
        "  Assistant\nreply  ",
    )
    assert all(isinstance(message, ConversationMessage) for message in messages)
    assert messages[0].message_id != messages[1].message_id
    assert messages[0].created_at.tzinfo is not None
    assert messages[1].created_at.tzinfo is not None


def test_source_request_and_context_are_not_mutated() -> None:
    conversation_id = uuid4()
    prior = ConversationMessage(MessageRole.USER, "Prior")
    context = ConversationContext(
        conversation_id=conversation_id,
        messages=(prior,),
    )
    request = TuesdayRequest(
        content="/chat   Hello  ",
        conversation_id=conversation_id,
    )
    repository = RecordingRepository(context)
    orchestrator = RecordingOrchestrator(make_response(request))

    run_handle(StatefulConversationService(orchestrator, repository), request)

    assert request.content == "/chat   Hello  "
    assert context.messages == (prior,)


def test_repository_load_failure_prevents_execution_and_append() -> None:
    conversation_id = uuid4()
    request = TuesdayRequest(
        content="/chat Hello",
        conversation_id=conversation_id,
    )
    error = ConversationRepositoryError("load failed")
    repository = RecordingRepository(
        ConversationContext(conversation_id=conversation_id),
        get_error=error,
    )
    orchestrator = RecordingOrchestrator(make_response(request))

    with pytest.raises(ConversationRepositoryError) as caught:
        run_handle(
            StatefulConversationService(orchestrator, repository),
            request,
        )

    assert caught.value is error
    assert orchestrator.calls == []
    assert repository.append_calls == []


def test_orchestrator_failure_prevents_turn_persistence() -> None:
    conversation_id = uuid4()
    request = TuesdayRequest(
        content="/chat Hello",
        conversation_id=conversation_id,
    )
    error = RuntimeError("execution failed")
    repository = RecordingRepository(
        ConversationContext(conversation_id=conversation_id)
    )
    orchestrator = RecordingOrchestrator(
        make_response(request),
        error=error,
    )

    with pytest.raises(RuntimeError) as caught:
        run_handle(
            StatefulConversationService(orchestrator, repository),
            request,
        )

    assert caught.value is error
    assert len(orchestrator.calls) == 1
    assert repository.append_calls == []


def test_append_failure_propagates_without_retrying_execution() -> None:
    conversation_id = uuid4()
    request = TuesdayRequest(
        content="/chat Hello",
        conversation_id=conversation_id,
    )
    error = ConversationRepositoryError("append failed")
    repository = RecordingRepository(
        ConversationContext(conversation_id=conversation_id),
        append_error=error,
    )
    orchestrator = RecordingOrchestrator(make_response(request))

    with pytest.raises(ConversationRepositoryError) as caught:
        run_handle(
            StatefulConversationService(orchestrator, repository),
            request,
        )

    assert caught.value is error
    assert len(repository.get_calls) == 1
    assert len(orchestrator.calls) == 1
    assert len(repository.append_calls) == 1


def test_default_runtime_automatically_reuses_persisted_history() -> None:
    repository = InMemoryConversationRepository()
    service = StatefulConversationService(
        create_default_orchestrator(),
        repository,
    )
    conversation_id = uuid4()

    first = TuesdayRequest(
        content="/chat First",
        conversation_id=conversation_id,
    )
    second = TuesdayRequest(
        content="/conversation Second",
        conversation_id=conversation_id,
    )

    first_response = run_handle(service, first)
    second_response = run_handle(service, second)
    context = asyncio.run(repository.get_context(conversation_id))

    assert first_response.content == "TUESDAY received: First"
    assert second_response.content == (
        "TUESDAY received: Second (2 prior messages in context)"
    )
    assert [message.role for message in context.messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert [message.content for message in context.messages] == [
        "/chat First",
        "TUESDAY received: First",
        "/conversation Second",
        second_response.content,
    ]


def test_separate_conversations_remain_isolated() -> None:
    repository = InMemoryConversationRepository()
    service = StatefulConversationService(
        create_default_orchestrator(),
        repository,
    )
    first_id = uuid4()
    second_id = uuid4()

    first_response = run_handle(
        service,
        TuesdayRequest(content="/chat First", conversation_id=first_id),
    )
    second_response = run_handle(
        service,
        TuesdayRequest(content="/chat Second", conversation_id=second_id),
    )

    assert first_response.content == "TUESDAY received: First"
    assert second_response.content == "TUESDAY received: Second"
    assert len(asyncio.run(repository.get_context(first_id)).messages) == 2
    assert len(asyncio.run(repository.get_context(second_id)).messages) == 2


def test_service_module_has_no_provider_or_composition_dependency() -> None:
    source = inspect.getsource(service_module)

    assert "openai" not in source.lower()
    assert "tuesday.language_models" not in source
    assert "tuesday.composition" not in source


def test_public_services_package_exports_stateful_service() -> None:
    import tuesday.services as services

    assert services.StatefulConversationService is StatefulConversationService
