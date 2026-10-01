"""Framework-independent contracts for TUESDAY routing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from tuesday.agents import AgentRegistry
from tuesday.domain import ConversationContext, TuesdayRequest

__all__ = ["BaseRouter", "RoutingDecision"]


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """An immutable result identifying the agent selected for a request."""

    agent_name: str
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.agent_name, str) or not self.agent_name.strip():
            raise ValueError("Routing decision agent_name must be non-empty text.")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("Routing decision reason must be non-empty text.")


class BaseRouter(ABC):
    """Abstract contract for choosing an agent for a TUESDAY request."""

    @abstractmethod
    async def route(
        self,
        request: TuesdayRequest,
        context: ConversationContext,
        registry: AgentRegistry,
    ) -> RoutingDecision:
        """Return a routing decision without executing the selected agent."""

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
