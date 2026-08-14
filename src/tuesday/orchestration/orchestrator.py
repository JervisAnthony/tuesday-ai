"""Core coordination of routing and agent execution."""

from tuesday.agents import AgentRegistry
from tuesday.domain import ConversationContext, TuesdayRequest, TuesdayResponse
from tuesday.routing import BaseRouter

__all__ = ["InvalidAgentResponseError", "TuesdayOrchestrator"]


class InvalidAgentResponseError(ValueError):
    """Raised when an agent response is not correlated to its request."""


class TuesdayOrchestrator:
    """Coordinate one routed agent execution for a TUESDAY interaction."""

    def __init__(self, router: BaseRouter, registry: AgentRegistry) -> None:
        self._router = router
        self._registry = registry

    async def handle(
        self,
        request: TuesdayRequest,
        context: ConversationContext,
    ) -> TuesdayResponse:
        """Route, execute, and validate one interaction."""
        self._validate_context(request, context)
        decision = await self._router.route(request, context, self._registry)
        agent = self._registry.get(decision.agent_name)
        response = await agent.handle(request, context)
        self._validate_response(request, response)
        return response

    @staticmethod
    def _validate_context(
        request: TuesdayRequest,
        context: ConversationContext,
    ) -> None:
        """Ensure the request and context share a conversation."""
        if request.conversation_id != context.conversation_id:
            raise ValueError(
                "Request and context must belong to the same conversation."
            )

    @staticmethod
    def _validate_response(
        request: TuesdayRequest,
        response: TuesdayResponse,
    ) -> None:
        """Ensure the response preserves request correlation identifiers."""
        if response.conversation_id != request.conversation_id:
            raise InvalidAgentResponseError(
                "Agent response conversation_id does not match the request."
            )
        if response.request_id != request.request_id:
            raise InvalidAgentResponseError(
                "Agent response request_id does not match the request."
            )
