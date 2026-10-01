"""Conversation-history repository contracts."""

from abc import ABC, abstractmethod
from uuid import UUID

from tuesday.domain import ConversationContext, ConversationMessage

__all__ = [
    "BaseConversationRepository",
    "ConversationRepositoryError",
]


class ConversationRepositoryError(RuntimeError):
    """Raised when a conversation repository cannot complete an operation."""


class BaseConversationRepository(ABC):
    """Abstract asynchronous storage boundary for conversation history."""

    __slots__ = ()

    @abstractmethod
    async def get_context(self, conversation_id: UUID) -> ConversationContext:
        """Return an immutable snapshot for one conversation."""

    @abstractmethod
    async def append_messages(
        self,
        conversation_id: UUID,
        messages: tuple[ConversationMessage, ...],
    ) -> ConversationContext:
        """Append messages and return the resulting immutable snapshot."""
