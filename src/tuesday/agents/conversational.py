"""Deterministic baseline conversational agent."""

from tuesday.agents.base import BaseAgent
from tuesday.domain import ConversationContext, TuesdayRequest, TuesdayResponse

__all__ = ["ConversationalAgent"]


class ConversationalAgent(BaseAgent):
    """Provide deterministic conversational responses for TUESDAY."""

    @property
    def name(self) -> str:
        """Return the agent's stable registry name."""
        return "conversation"

    @property
    def description(self) -> str:
        """Return the agent's deterministic baseline purpose."""
        return "Provides deterministic conversational responses for TUESDAY."

    async def handle(
        self,
        request: TuesdayRequest,
        context: ConversationContext,
    ) -> TuesdayResponse:
        """Acknowledge the request and report the prior-message count."""
        self._validate_context(request, context)

        content = f"TUESDAY received: {request.content}"
        message_count = len(context.messages)
        if message_count:
            suffix = "" if message_count == 1 else "s"
            content += (
                f" ({message_count} prior message{suffix} in context)"
            )

        return TuesdayResponse(
            content=content,
            conversation_id=request.conversation_id,
            request_id=request.request_id,
        )
