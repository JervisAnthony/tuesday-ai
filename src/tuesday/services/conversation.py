"""Stateful conversation interaction service."""

from tuesday.conversations import BaseConversationRepository
from tuesday.domain import (
    ConversationContext,
    ConversationMessage,
    MessageRole,
    TuesdayRequest,
    TuesdayResponse,
)
from tuesday.orchestration import TuesdayOrchestrator

__all__ = ["StatefulConversationService"]


class StatefulConversationService:
    """Load, execute, and persist one successful conversation interaction."""

    __slots__ = ("_orchestrator", "_repository")

    def __init__(
        self,
        orchestrator: TuesdayOrchestrator,
        repository: BaseConversationRepository,
    ) -> None:
        if not isinstance(orchestrator, TuesdayOrchestrator):
            raise TypeError("orchestrator must be a TuesdayOrchestrator.")
        if not isinstance(repository, BaseConversationRepository):
            raise TypeError(
                "repository must be a BaseConversationRepository."
            )

        self._orchestrator = orchestrator
        self._repository = repository

    async def handle(self, request: TuesdayRequest) -> TuesdayResponse:
        """Execute one request against stored history and persist the turn."""
        if not isinstance(request, TuesdayRequest):
            raise TypeError("request must be a TuesdayRequest.")

        context = await self._repository.get_context(request.conversation_id)
        self._validate_context(request, context)

        response = await self._orchestrator.handle(request, context)

        messages = (
            ConversationMessage(
                role=MessageRole.USER,
                content=request.content,
            ),
            ConversationMessage(
                role=MessageRole.ASSISTANT,
                content=response.content,
            ),
        )
        await self._repository.append_messages(
            request.conversation_id,
            messages,
        )
        return response

    @staticmethod
    def _validate_context(
        request: TuesdayRequest,
        context: ConversationContext,
    ) -> None:
        if not isinstance(context, ConversationContext):
            raise TypeError(
                "Conversation repository must return a ConversationContext."
            )
        if context.conversation_id != request.conversation_id:
            raise ValueError(
                "Stored conversation context must match the request conversation."
            )
