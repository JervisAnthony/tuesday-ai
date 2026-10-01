"""Tests for the in-memory conversation-history repository."""

import ast
import asyncio
import inspect
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

import tuesday.conversations.memory as memory_module
from tuesday.conversations import (
    BaseConversationRepository,
    InMemoryConversationRepository,
)
from tuesday.domain import ConversationContext, ConversationMessage, MessageRole


def run_get(
    repository: InMemoryConversationRepository,
    conversation_id: UUID,
) -> ConversationContext:
    return asyncio.run(repository.get_context(conversation_id))


def run_append(
    repository: InMemoryConversationRepository,
    conversation_id: UUID,
    messages: tuple[ConversationMessage, ...],
) -> ConversationContext:
    return asyncio.run(repository.append_messages(conversation_id, messages))


def test_repository_inherits_abstract_contract() -> None:
    assert isinstance(InMemoryConversationRepository(), BaseConversationRepository)


def test_repository_operations_are_asynchronous() -> None:
    assert inspect.iscoroutinefunction(InMemoryConversationRepository.get_context)
    assert inspect.iscoroutinefunction(
        InMemoryConversationRepository.append_messages
    )


def test_new_repository_has_no_stored_conversations() -> None:
    repository = InMemoryConversationRepository()

    assert repository._messages_by_conversation == {}


def test_unknown_conversation_returns_empty_correlated_context() -> None:
    repository = InMemoryConversationRepository()
    conversation_id = uuid4()

    context = run_get(repository, conversation_id)

    assert context == ConversationContext(conversation_id=conversation_id)
    assert context.conversation_id is conversation_id
    assert context.messages == ()


def test_unknown_conversation_read_does_not_create_storage_entry() -> None:
    repository = InMemoryConversationRepository()

    run_get(repository, uuid4())

    assert repository._messages_by_conversation == {}


def test_repeated_reads_return_independent_immutable_snapshots() -> None:
    repository = InMemoryConversationRepository()
    conversation_id = uuid4()

    first = run_get(repository, conversation_id)
    second = run_get(repository, conversation_id)

    assert first == second
    assert first is not second
    with pytest.raises(AttributeError):
        first.messages = ()  # type: ignore[misc]


@pytest.mark.parametrize("conversation_id", ["conversation", 123, None, object()])
def test_get_rejects_non_uuid_conversation_identifier(
    conversation_id: object,
) -> None:
    repository = InMemoryConversationRepository()

    with pytest.raises(TypeError, match="conversation_id must be a UUID"):
        asyncio.run(repository.get_context(conversation_id))  # type: ignore[arg-type]

    assert repository._messages_by_conversation == {}


@pytest.mark.parametrize("conversation_id", ["conversation", 123, None, object()])
def test_append_rejects_non_uuid_conversation_identifier(
    conversation_id: object,
) -> None:
    repository = InMemoryConversationRepository()
    message = ConversationMessage(MessageRole.USER, "Hello")

    with pytest.raises(TypeError, match="conversation_id must be a UUID"):
        asyncio.run(
            repository.append_messages(  # type: ignore[arg-type]
                conversation_id,
                (message,),
            )
        )

    assert repository._messages_by_conversation == {}


def test_append_requires_tuple_messages() -> None:
    repository = InMemoryConversationRepository()
    conversation_id = uuid4()
    message = ConversationMessage(MessageRole.USER, "Hello")

    with pytest.raises(TypeError, match="messages must be a tuple"):
        asyncio.run(
            repository.append_messages(
                conversation_id,
                [message],  # type: ignore[arg-type]
            )
        )

    assert repository._messages_by_conversation == {}


def test_append_rejects_empty_message_tuple() -> None:
    repository = InMemoryConversationRepository()
    conversation_id = uuid4()

    with pytest.raises(ValueError, match="at least one ConversationMessage"):
        run_append(repository, conversation_id, ())

    assert repository._messages_by_conversation == {}


def test_append_rejects_non_message_member_without_partial_write() -> None:
    repository = InMemoryConversationRepository()
    conversation_id = uuid4()
    valid = ConversationMessage(MessageRole.USER, "Valid")

    with pytest.raises(TypeError, match="only ConversationMessage instances"):
        asyncio.run(
            repository.append_messages(
                conversation_id,
                (valid, object()),  # type: ignore[arg-type]
            )
        )

    assert repository._messages_by_conversation == {}


def test_single_message_append_creates_conversation_history() -> None:
    repository = InMemoryConversationRepository()
    conversation_id = uuid4()
    message = ConversationMessage(MessageRole.USER, "Hello")

    context = run_append(repository, conversation_id, (message,))

    assert context.conversation_id == conversation_id
    assert context.messages == (message,)
    assert context.messages[0] is message
    assert run_get(repository, conversation_id) == context


def test_batch_append_preserves_exact_order_and_identity() -> None:
    repository = InMemoryConversationRepository()
    conversation_id = uuid4()
    messages = (
        ConversationMessage(MessageRole.SYSTEM, "System"),
        ConversationMessage(MessageRole.USER, "Question"),
        ConversationMessage(MessageRole.ASSISTANT, "Answer"),
    )

    context = run_append(repository, conversation_id, messages)

    assert context.messages == messages
    assert all(
        stored is supplied
        for stored, supplied in zip(context.messages, messages, strict=True)
    )


def test_append_preserves_message_identifiers_and_timestamps() -> None:
    repository = InMemoryConversationRepository()
    conversation_id = uuid4()
    message_id = uuid4()
    created_at = datetime(2026, 8, 18, 17, 0, tzinfo=UTC)
    message = ConversationMessage(
        MessageRole.USER,
        "Exact message",
        message_id=message_id,
        created_at=created_at,
    )

    stored = run_append(repository, conversation_id, (message,)).messages[0]

    assert stored is message
    assert stored.message_id is message_id
    assert stored.created_at is created_at


def test_append_preserves_meaningful_whitespace_and_multiline_content() -> None:
    repository = InMemoryConversationRepository()
    conversation_id = uuid4()
    message = ConversationMessage(
        MessageRole.USER,
        "  First line\nsecond line  ",
    )

    stored = run_append(repository, conversation_id, (message,)).messages[0]

    assert stored.content == "  First line\nsecond line  "


def test_second_append_extends_existing_history_in_order() -> None:
    repository = InMemoryConversationRepository()
    conversation_id = uuid4()
    first = ConversationMessage(MessageRole.USER, "First")
    second = ConversationMessage(MessageRole.ASSISTANT, "Second")
    third = ConversationMessage(MessageRole.USER, "Third")

    run_append(repository, conversation_id, (first, second))
    context = run_append(repository, conversation_id, (third,))

    assert context.messages == (first, second, third)


def test_previous_snapshot_is_not_mutated_by_later_append() -> None:
    repository = InMemoryConversationRepository()
    conversation_id = uuid4()
    first = ConversationMessage(MessageRole.USER, "First")
    second = ConversationMessage(MessageRole.ASSISTANT, "Second")

    previous = run_append(repository, conversation_id, (first,))
    current = run_append(repository, conversation_id, (second,))

    assert previous.messages == (first,)
    assert current.messages == (first, second)
    assert previous.messages is not current.messages


def test_supplied_message_tuple_is_not_mutated_or_replaced() -> None:
    repository = InMemoryConversationRepository()
    conversation_id = uuid4()
    first = ConversationMessage(MessageRole.USER, "First")
    second = ConversationMessage(MessageRole.ASSISTANT, "Second")
    supplied = (first, second)

    run_append(repository, conversation_id, supplied)

    assert supplied == (first, second)
    assert supplied[0] is first
    assert supplied[1] is second


def test_conversations_are_isolated_by_identifier() -> None:
    repository = InMemoryConversationRepository()
    first_id = uuid4()
    second_id = uuid4()
    first_message = ConversationMessage(MessageRole.USER, "First conversation")
    second_message = ConversationMessage(MessageRole.USER, "Second conversation")

    run_append(repository, first_id, (first_message,))
    run_append(repository, second_id, (second_message,))

    assert run_get(repository, first_id).messages == (first_message,)
    assert run_get(repository, second_id).messages == (second_message,)


def test_repository_instances_do_not_share_state() -> None:
    first_repository = InMemoryConversationRepository()
    second_repository = InMemoryConversationRepository()
    conversation_id = uuid4()
    message = ConversationMessage(MessageRole.USER, "Private to first repository")

    run_append(first_repository, conversation_id, (message,))

    assert run_get(first_repository, conversation_id).messages == (message,)
    assert run_get(second_repository, conversation_id).messages == ()


def test_failed_append_does_not_change_existing_history() -> None:
    repository = InMemoryConversationRepository()
    conversation_id = uuid4()
    existing = ConversationMessage(MessageRole.USER, "Existing")
    run_append(repository, conversation_id, (existing,))

    with pytest.raises(TypeError):
        asyncio.run(
            repository.append_messages(
                conversation_id,
                (object(),),  # type: ignore[arg-type]
            )
        )

    assert run_get(repository, conversation_id).messages == (existing,)


def test_storage_uses_immutable_message_tuples() -> None:
    repository = InMemoryConversationRepository()
    conversation_id = uuid4()
    message = ConversationMessage(MessageRole.USER, "Hello")

    run_append(repository, conversation_id, (message,))

    stored = repository._messages_by_conversation[conversation_id]
    assert isinstance(stored, tuple)
    assert stored == (message,)


def test_repository_has_only_declared_in_memory_state_slot() -> None:
    repository = InMemoryConversationRepository()

    assert InMemoryConversationRepository.__slots__ == ("_messages_by_conversation",)
    assert not hasattr(repository, "__dict__")


def test_memory_module_has_domain_and_repository_dependencies_only() -> None:
    tree = ast.parse(inspect.getsource(memory_module))
    imported_modules = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert imported_modules == {
        "uuid",
        "tuesday.conversations.base",
        "tuesday.domain",
    }


def test_memory_repository_has_no_external_persistence_or_model_dependency() -> None:
    source = inspect.getsource(memory_module).lower()

    for forbidden in (
        "openai",
        "sqlite",
        "sqlalchemy",
        "postgres",
        "redis",
        "mongodb",
        "motor",
        "pymongo",
        "requests",
        "httpx",
        "pathlib",
        "tempfile",
        "language_models",
        "orchestrator",
        "agentregistry",
    ):
        assert forbidden not in source


def test_public_package_exports_in_memory_repository() -> None:
    import tuesday.conversations as conversations

    assert conversations.InMemoryConversationRepository is (
        InMemoryConversationRepository
    )
