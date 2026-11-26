"""Read-only history queries and immutable paginated results."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from tuesday.domain import ConversationMessage

__all__ = [
    "BaseConversationHistoryReader",
    "ConversationSummary",
    "ConversationSummaryPage",
    "ConversationMessagePage",
]


def _validate_pagination(limit: int, offset: int) -> None:
    if type(limit) is not int or type(offset) is not int:
        raise TypeError("limit and offset must be integers.")
    if not 1 <= limit <= 200:
        raise ValueError("limit must be between 1 and 200.")
    if not 0 <= offset <= 2**63 - 1:
        raise ValueError("offset must be a nonnegative signed 64-bit integer.")


def _validate_conversation_id(conversation_id: UUID) -> None:
    if not isinstance(conversation_id, UUID):
        raise TypeError("conversation_id must be a UUID.")


def _validate_timestamp(value: datetime) -> None:
    if not isinstance(value, datetime):
        raise TypeError("Message timestamps must be datetime values.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Message timestamps must be timezone-aware.")


def _validate_next_offset(next_offset: int | None) -> None:
    if next_offset is not None:
        if type(next_offset) is not int:
            raise TypeError("next_offset must be an integer or None.")
        if next_offset < 0:
            raise ValueError("next_offset must be nonnegative.")


@dataclass(frozen=True, slots=True)
class ConversationSummary:
    """Count and timestamps of the first and last appended messages."""

    conversation_id: UUID
    message_count: int
    first_message_at: datetime
    last_message_at: datetime

    def __post_init__(self) -> None:
        _validate_conversation_id(self.conversation_id)
        if type(self.message_count) is not int:
            raise TypeError("message_count must be an integer.")
        if self.message_count < 0:
            raise ValueError("message_count must be nonnegative.")
        _validate_timestamp(self.first_message_at)
        _validate_timestamp(self.last_message_at)


@dataclass(frozen=True, slots=True)
class ConversationSummaryPage:
    """One immutable page in conversation first-append order."""

    items: tuple[ConversationSummary, ...]
    limit: int
    offset: int
    next_offset: int | None

    def __post_init__(self) -> None:
        _validate_pagination(self.limit, self.offset)
        _validate_next_offset(self.next_offset)
        if not isinstance(self.items, tuple) or not all(
            isinstance(item, ConversationSummary) for item in self.items
        ):
            raise TypeError("items must be a tuple of ConversationSummary values.")


@dataclass(frozen=True, slots=True)
class ConversationMessagePage:
    """One immutable message page, without a transcript-wide total count."""

    conversation_id: UUID
    messages: tuple[ConversationMessage, ...]
    limit: int
    offset: int
    next_offset: int | None

    def __post_init__(self) -> None:
        _validate_conversation_id(self.conversation_id)
        _validate_pagination(self.limit, self.offset)
        _validate_next_offset(self.next_offset)
        if not isinstance(self.messages, tuple) or not all(
            isinstance(message, ConversationMessage) for message in self.messages
        ):
            raise TypeError("messages must be a tuple of ConversationMessage values.")


class BaseConversationHistoryReader(ABC):
    """Query current committed history without requiring write operations."""

    __slots__ = ()

    @abstractmethod
    async def list_conversations(
        self, *, limit: int = 50, offset: int = 0
    ) -> ConversationSummaryPage:
        """List conversations in first-successful-append order."""

    @abstractmethod
    async def get_messages(
        self, conversation_id: UUID, *, limit: int = 50, offset: int = 0
    ) -> ConversationMessagePage:
        """Read messages in append order, selecting at most one page."""
