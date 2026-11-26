"""In-memory conversation-history repository."""

from itertools import islice
from uuid import UUID

from tuesday.conversations.base import BaseConversationRepository
from tuesday.conversations.history import (
    BaseConversationHistoryReader,
    ConversationMessagePage,
    ConversationSummary,
    ConversationSummaryPage,
    _validate_pagination,
)
from tuesday.domain import ConversationContext, ConversationMessage

__all__ = ["InMemoryConversationRepository"]


class InMemoryConversationRepository(
    BaseConversationRepository, BaseConversationHistoryReader
):
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

    async def list_conversations(
        self, *, limit: int = 50, offset: int = 0
    ) -> ConversationSummaryPage:
        _validate_pagination(limit, offset)
        entries = tuple(
            islice(
                islice(
                    self._messages_by_conversation.items(),
                    min(offset, len(self._messages_by_conversation)),
                    None,
                ),
                limit + 1,
            )
        )
        items = tuple(
            ConversationSummary(
                identifier,
                len(messages),
                messages[0].created_at,
                messages[-1].created_at,
            )
            for identifier, messages in entries[:limit]
        )
        return ConversationSummaryPage(
            items, limit, offset, offset + len(items) if len(entries) > limit else None
        )

    async def get_messages(
        self, conversation_id: UUID, *, limit: int = 50, offset: int = 0
    ) -> ConversationMessagePage:
        self._validate_conversation_id(conversation_id)
        _validate_pagination(limit, offset)
        found = self._messages_by_conversation.get(conversation_id, ())[
            offset : offset + limit + 1
        ]
        messages = found[:limit]
        return ConversationMessagePage(
            conversation_id,
            messages,
            limit,
            offset,
            offset + len(messages) if len(found) > limit else None,
        )

    @staticmethod
    def _validate_conversation_id(conversation_id: UUID) -> None:
        if not isinstance(conversation_id, UUID):
            raise TypeError("conversation_id must be a UUID.")

    @staticmethod
    def _validate_messages(messages: tuple[ConversationMessage, ...]) -> None:
        if not isinstance(messages, tuple):
            raise TypeError(
                "messages must be a tuple of ConversationMessage instances."
            )
        if not messages:
            raise ValueError("messages must contain at least one ConversationMessage.")
        if not all(isinstance(message, ConversationMessage) for message in messages):
            raise TypeError("messages must contain only ConversationMessage instances.")
