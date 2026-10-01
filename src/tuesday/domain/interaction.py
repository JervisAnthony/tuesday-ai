"""Framework-independent contracts for TUESDAY interactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

__all__ = [
    "ConversationContext",
    "ConversationMessage",
    "MessageRole",
    "TuesdayRequest",
    "TuesdayResponse",
]


class MessageRole(StrEnum):
    """The participant that produced a conversation message."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _require_content(content: str, model_name: str) -> None:
    if not content.strip():
        raise ValueError(f"{model_name} content must not be empty.")


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    """A single immutable message in conversation history."""

    role: MessageRole
    content: str
    message_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        _require_content(self.content, "Message")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("Message created_at must be timezone-aware.")


@dataclass(frozen=True, slots=True)
class TuesdayRequest:
    """An incoming user request with correlation identifiers."""

    content: str
    conversation_id: UUID = field(default_factory=uuid4)
    request_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        _require_content(self.content, "Request")


@dataclass(frozen=True, slots=True)
class TuesdayResponse:
    """A response correlated with its originating request and conversation."""

    content: str
    conversation_id: UUID
    request_id: UUID
    response_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        _require_content(self.content, "Response")


@dataclass(frozen=True, slots=True)
class ConversationContext:
    """An immutable snapshot of conversation history."""

    conversation_id: UUID = field(default_factory=uuid4)
    messages: tuple[ConversationMessage, ...] = ()

    def add_message(self, message: ConversationMessage) -> ConversationContext:
        """Return a new context containing the appended message."""
        return ConversationContext(
            conversation_id=self.conversation_id,
            messages=(*self.messages, message),
        )
