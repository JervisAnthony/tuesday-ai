"""Default production composition for TUESDAY."""

from tuesday.agents import AgentRegistry, ConversationalAgent
from tuesday.orchestration import TuesdayOrchestrator
from tuesday.routing import DeterministicRouter

__all__ = ["create_default_orchestrator"]


def create_default_orchestrator() -> TuesdayOrchestrator:
    """Create a fresh deterministic TUESDAY application composition."""
    registry = AgentRegistry()
    conversational_agent = ConversationalAgent()
    registry.register(conversational_agent)

    router = DeterministicRouter(
        {
            "chat": conversational_agent.name,
            "conversation": conversational_agent.name,
        }
    )
    return TuesdayOrchestrator(router=router, registry=registry)
