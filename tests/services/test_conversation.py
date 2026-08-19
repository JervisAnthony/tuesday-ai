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
    """Repository test double that records operations and arguments."""

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
    """Orchestrator test double that records execution inputs."""

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


def test_service_has_exact_slots() -> None:
    assert StatefulConversationService.__slots__ == (
        "_orchestrator",
        "_repository",
    )


def test_service_has_no_dynamic_instance_dictionary() -> None:
    service = StatefulConversationService(
        create_default_orchestrator(),
        InMemoryConversationRepository(),
    )

    assert not hasattr(service, "__dict__")


def test_constructor_preserves_dependency_identity() -> None:
    orchestrator = create_default_orchestrator()
    repository = InMemoryConversationRepository()

    service = StatefulConversationService(orchestrator, repository)

    assert service._orchestrator is orchestrator
    assert service._repository is repository


@pytest.mark.parametrize("orchestrator", [None, object(), "orchestrator"])
def test_constructor_rejects_non_orchestrator(orchestrator: object) -> None:
    with pytest.raises(
        TypeError,
        match="orchestrator must be a TuesdayOrchestrator",
    ):
        StatefulConversationService(
            orchestrator,  # type: ignore[arg-type]
            InMemoryConversationRepository(),
        )


@pytest.mark.parametrize("repository", [None, object(), "repository"])
def test_constructor_rejects_non_repository(repository: object) -> None:
    with pytest.raises(
        TypeError,
        match="repository must be a BaseConversationRepository",
    ):
        StatefulConversationService(
            create_default_orchestrator(),
            repository,  # type: ignore[arg-type]
        )


def test_handle_is_asynchronous() -> None:
    assert inspect.iscoroutinefunction(StatefulConversationService.handle)


def test_handle_signature_accepts_only_request() -> None:
    parameters = inspect.signature(
        StatefulConversationService.handle
    ).parameters
    assert tuple(parameters) == ("self", "request")


@pytest.mark.parametrize("request", [None, object(), "request", 123])
def test_handle_rejects_non_request_before_repository_access(
    request: object,
) -> None:
    conversation_id = uuid4()
    context = ConversationContext(conversation_id=conversation_id)
    repository = RecordingRepository(context)
    valid_request = TuesdayRequest(
        content="/chat Hello",
        conversation_id=conversation_id,
    )
    orchestrator = RecordingOrchestrator(make_response(valid_request))
    service = StatefulConversationService(orchestrator, repository)

    with pytest.raises(TypeError, match="request must be a TuesdayRequest"):
        asyncio.run(service.handle(request))  # type: ignore[arg-type]

    assert repository.get_calls == []
    assert orchestrator.calls == []
    assert repository.append_calls == []


def test_service_executes_load_handle_append_in_order() -> None:
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


def test_service_loads_request_conversation_context() -> None:
    conversation_id = uuid4()
    request = TuesdayRequest(
        content="/chat Hello",
        conversation_id=conversation_id,
    )
    context = ConversationContext(conversation_id=conversation_id)
    repository = RecordingRepository(context)
    orchestrator = RecordingOrchestrator(make_response(request))

    run_handle(StatefulConversationService(orchestrator, repository), request)

    assert repository.get_calls == [conversation_id]
    assert orchestrator.calls[0][0] is request
    assert orchestrator.calls[0][1] is context


def test_repository_must_return_conversation_context() -> None:
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

    with pytest.raises(
        TypeError,
        match="must return a ConversationContext",
    ):
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


def test_success_persists_exact_user_and_assistant_turn() -> None:
    conversation_id = uuid4()
    request = TuesdayRequest(
        content="/chat   Hello with spacing  ",
        conversation_id=conversation_id,
    )
    context = ConversationContext(conversation_id=conversation_id)
    response = make_response(request, "  Assistant\nreply  ")
    repository = RecordingRepository(context)
    orchestrator = RecordingOrchestrator(response)

    returned = run_handle(
        StatefulConversationService(orchestrator, repository),
        request,
    )

    assert returned is response
    assert len(repository.append_calls) == 1
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


def test_persisted_messages_are_fresh_immutable_domain_objects() -> None:
    conversation_id = uuid4()
    request = TuesdayRequest(
        content="/chat Hello",
        conversation_id=conversation_id,
    )
    repository = RecordingRepository(
        ConversationContext(conversation_id=conversation_id)
    )
    orchestrator = RecordingOrchestrator(make_response(request))

    run_handle(StatefulConversationService(orchestrator, repository), request)

    user_message, assistant_message = repository.append_calls[0][1]
    assert isinstance(user_message, ConversationMessage)
    assert isinstance(assistant_message, ConversationMessage)
    assert user_message is not assistant_message
    assert user_message.message_id != assistant_message.message_id
    assert user_message.created_at.tzinfo is not None
    assert assistant_message.created_at.tzinfo is not None
    with pytest.raises(AttributeError):
        user_message.content = "changed"  # type: ignore[misc]


def test_source_request_and_loaded_context_are_not_mutated() -> None:
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


def test_repository_load_error_prevents_execution_and_append() -> None:
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


def test_orchestrator_error_prevents_turn_persistence() -> None:
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


def test_append_error_propagates_without_retrying_execution() -> None:
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


def test_default_runtime_first_turn_is_persisted() -> None:
    repository = InMemoryConversationRepository()
    service = StatefulConversationService(
        create_default_orchestrator(),
        repository,
    )
    conversation_id = uuid4()
    request = TuesdayRequest(
        content="/chat Hello",
        conversation_id=conversation_id,
    )

    response = run_handle(service, request)
    context = asyncio.run(repository.get_context(conversation_id))

    assert response.content == "TUESDAY received: Hello"
    assert tuple(message.role for message in context.messages) == (
        MessageRole.USER,
        MessageRole.ASSISTANT,
    )
    assert tuple(message.content for message in context.messages) == (
        "/chat Hello",
        "TUESDAY received: Hello",
    )


def test_default_runtime_second_turn_receives_stored_history() -> None:
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

    run_handle(service, first)
    response = run_handle(service, second)
    context = asyncio.run(repository.get_context(conversation_id))

    assert response.content == (
        "TUESDAY received: Second (2 prior messages in context)"
    )
    assert len(context.messages) == 4
    assert context.messages[2].role is MessageRole.USER
    assert context.messages[2].content == "/conversation Second"
    assert context.messages[3].role is MessageRole.ASSISTANT
    assert context.messages[3].content == response.content


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


def test_services_can_share_repository_state_explicitly() -> None:
    repository = InMemoryConversationRepository()
    first_service = StatefulConversationService(
        create_default_orchestrator(),
        repository,
    )
    second_service = StatefulConversationService(
        create_default_orchestrator(),
        repository,
    )
    conversation_id = uuid4()

    run_handle(
        first_service,
        TuesdayRequest(
            content="/chat First",
            conversation_id=conversation_id,
        ),
    )
    response = run_handle(
        second_service,
        TuesdayRequest(
            content="/chat Second",
            conversation_id=conversation_id,
        ),
    )

    assert response.content == (
        "TUESDAY received: Second (2 prior messages in context)"
    )


def test_service_module_has_no_provider_or_composition_dependency() -> None:
    source = inspect.getsource(service_module)

    assert "openai" not in source.lower()
    assert "tuesday.language_models" not in source
    assert "tuesday.composition" not in source


def test_public_services_package_exports_stateful_service() -> None:
    import tuesday.services as services

    assert services.StatefulConversationService is StatefulConversationService
