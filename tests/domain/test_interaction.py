"""Tests for the core interaction domain contracts."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from tuesday.domain import (
    ConversationContext,
    ConversationMessage,
    MessageRole,
    TuesdayRequest,
    TuesdayResponse,
)

INVALID_CONTENT = ["", " ", "\t", "\n", "   \n\t "]


def test_message_role_values() -> None:
    assert MessageRole.USER == "user"
    assert MessageRole.ASSISTANT == "assistant"
    assert MessageRole.SYSTEM == "system"


def test_conversation_message_creation_and_defaults() -> None:
    message = ConversationMessage(role=MessageRole.USER, content="  Hello  ")

    assert message.role is MessageRole.USER
    assert message.content == "  Hello  "
    assert isinstance(message.message_id, UUID)
    assert message.created_at.tzinfo is not None
    assert message.created_at.utcoffset() is not None


def test_conversation_message_preserves_explicit_values() -> None:
    message_id = uuid4()
    created_at = datetime(2026, 1, 1, tzinfo=UTC)

    message = ConversationMessage(
        role=MessageRole.SYSTEM,
        content="Instructions",
        message_id=message_id,
        created_at=created_at,
    )

    assert message.message_id == message_id
    assert message.created_at == created_at


@pytest.mark.parametrize("content", INVALID_CONTENT)
def test_conversation_message_rejects_empty_content(content: str) -> None:
    with pytest.raises(ValueError, match="Message content must not be empty"):
        ConversationMessage(role=MessageRole.USER, content=content)


def test_conversation_message_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="created_at must be timezone-aware"):
        ConversationMessage(
            role=MessageRole.USER,
            content="Hello",
            created_at=datetime(2026, 1, 1),
        )


def test_conversation_message_is_immutable() -> None:
    message = ConversationMessage(role=MessageRole.USER, content="Hello")

    with pytest.raises(FrozenInstanceError):
        message.content = "Changed"  # type: ignore[misc]


def test_tuesday_request_creation_and_defaults() -> None:
    request = TuesdayRequest(content="  Help me plan  ")

    assert request.content == "  Help me plan  "
    assert isinstance(request.conversation_id, UUID)
    assert isinstance(request.request_id, UUID)
    assert request.conversation_id != request.request_id


def test_tuesday_request_preserves_explicit_identifiers() -> None:
    conversation_id = uuid4()
    request_id = uuid4()

    request = TuesdayRequest(
        content="Hello",
        conversation_id=conversation_id,
        request_id=request_id,
    )

    assert request.conversation_id == conversation_id
    assert request.request_id == request_id


@pytest.mark.parametrize("content", INVALID_CONTENT)
def test_tuesday_request_rejects_empty_content(content: str) -> None:
    with pytest.raises(ValueError, match="Request content must not be empty"):
        TuesdayRequest(content=content)


def test_tuesday_request_is_immutable() -> None:
    request = TuesdayRequest(content="Hello")

    with pytest.raises(FrozenInstanceError):
        request.content = "Changed"  # type: ignore[misc]


def test_tuesday_response_creation_and_correlation() -> None:
    conversation_id = uuid4()
    request_id = uuid4()

    response = TuesdayResponse(
        content="  Certainly  ",
        conversation_id=conversation_id,
        request_id=request_id,
    )

    assert response.content == "  Certainly  "
    assert response.conversation_id == conversation_id
    assert response.request_id == request_id
    assert isinstance(response.response_id, UUID)
    assert response.response_id not in {conversation_id, request_id}


@pytest.mark.parametrize("content", INVALID_CONTENT)
def test_tuesday_response_rejects_empty_content(content: str) -> None:
    with pytest.raises(ValueError, match="Response content must not be empty"):
        TuesdayResponse(
            content=content,
            conversation_id=uuid4(),
            request_id=uuid4(),
        )


def test_tuesday_response_is_immutable() -> None:
    response = TuesdayResponse(
        content="Hello",
        conversation_id=uuid4(),
        request_id=uuid4(),
    )

    with pytest.raises(FrozenInstanceError):
        response.content = "Changed"  # type: ignore[misc]


def test_empty_conversation_context_creation() -> None:
    conversation_id = uuid4()
    context = ConversationContext(conversation_id=conversation_id)

    assert context.conversation_id == conversation_id
    assert context.messages == ()
    assert isinstance(context.messages, tuple)


def test_conversation_context_stores_messages_immutably() -> None:
    message = ConversationMessage(role=MessageRole.USER, content="Hello")
    context = ConversationContext(messages=(message,))

    assert context.messages == (message,)
    with pytest.raises(FrozenInstanceError):
        context.messages = ()  # type: ignore[misc]


def test_add_message_returns_new_correlated_context() -> None:
    context = ConversationContext()
    message = ConversationMessage(role=MessageRole.ASSISTANT, content="Hello")

    updated_context = context.add_message(message)

    assert updated_context is not context
    assert context.messages == ()
    assert updated_context.messages == (message,)
    assert updated_context.conversation_id == context.conversation_id


def test_uuid_defaults_are_independent_between_objects() -> None:
    first_message = ConversationMessage(role=MessageRole.USER, content="First")
    second_message = ConversationMessage(role=MessageRole.USER, content="Second")
    first_request = TuesdayRequest(content="First")
    second_request = TuesdayRequest(content="Second")
    first_context = ConversationContext()
    second_context = ConversationContext()

    assert first_message.message_id != second_message.message_id
    assert first_request.request_id != second_request.request_id
    assert first_request.conversation_id != second_request.conversation_id
    assert first_context.conversation_id != second_context.conversation_id
