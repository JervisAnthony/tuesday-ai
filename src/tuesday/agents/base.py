"""Framework-independent base contract for TUESDAY agents."""

from abc import ABC, abstractmethod

from tuesday.domain import ConversationContext, TuesdayRequest, TuesdayResponse

__all__ = ["BaseAgent"]


class BaseAgent(ABC):
    """An executable component that handles a TUESDAY interaction."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the agent's stable, machine-friendly name."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a short human-readable description of the agent's purpose."""

    @abstractmethod
    async def handle(
        self,
        request: TuesdayRequest,
        context: ConversationContext,
    ) -> TuesdayResponse:
        """Handle a request and return a response preserving its identifiers."""

    @staticmethod
    def _validate_context(
        request: TuesdayRequest,
        context: ConversationContext,
    ) -> None:
        """Ensure the request and context refer to the same conversation."""
        if request.conversation_id != context.conversation_id:
            raise ValueError(
                "Request and context must belong to the same conversation."
            )
