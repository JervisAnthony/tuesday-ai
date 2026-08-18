"""In-memory conversation-history repository."""

from uuid import UUID

from tuesday.conversations.base import BaseConversationRepository
from tuesday.domain import ConversationContext, ConversationMessage

__all__ = ["InMemoryConversationRepository"]


class InMemoryConversationRepository(BaseConversationRepository):
    """Store immutable conversation-message tuples in process memory."""

    __slots__ = ("_messages_by_conversation",)

    def __init__(self) -> None:
        self._messages_by_conversation: dict[
            UUID,
            tuple[ConversationMessage, ...],
        ] = {}

    async def get_context(self, conversation_id: UUID) -> ConversationContext:
        """Return the current immutable snapshot for one conversation."""
        self._validate_conversation_id(conversation_id)
        return ConversationContext(
            conversation_id=conversation_id,
            messages=self._messages_by_conversation.get(conversation_id, ()),
        )

    async def append_messages(
        self,
        conversation_id: UUID,
        messages: tuple[ConversationMessage, ...],
    ) -> ConversationContext:
        """Append one immutable message batch and return the new snapshot."""
        self._validate_conversation_id(conversation_id)
        self._validate_messages(messages)

        existing_messages = self._messages_by_conversation.get(conversation_id, ())
        updated_messages = (*existing_messages, *messages)
        self._messages_by_conversation[conversation_id] = updated_messages

        return ConversationContext(
            conversation_id=conversation_id,
            messages=updated_messages,
        )

    @staticmethod
    def _validate_conversation_id(conversation_id: UUID) -> None:
        if not isinstance(conversation_id, UUID):
            raise TypeError("conversation_id must be a UUID.")

    @staticmethod
    def _validate_messages(messages: tuple[ConversationMessage, ...]) -> None:
        if not isinstance(messages, tuple):
            raise TypeError("messages must be a tuple of ConversationMessage instances.")
        if not messages:
            raise ValueError("messages must contain at least one ConversationMessage.")
        if not all(isinstance(message, ConversationMessage) for message in messages):
            raise TypeError("messages must contain only ConversationMessage instances.")
