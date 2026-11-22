"""Application composition helpers for TUESDAY."""

from tuesday.agents import (
    AgentRegistry,
    BaseAgent,
    ConversationalAgent,
    GuardedOpenAIConversationalAgent,
    ModelBackedConversationalAgent,
)
from tuesday.language_models.base import BaseLanguageModelProvider
from tuesday.language_models.tools import LanguageModelToolDefinition
from tuesday.orchestration import TuesdayOrchestrator
from tuesday.preparation import DirectiveRequestPreparer
from tuesday.prompting import ConversationalPromptRenderer
from tuesday.routing import DeterministicRouter
from tuesday.services.openai_tool_turn import OpenAIToolTurnService

__all__ = [
    "create_default_orchestrator",
    "create_guarded_openai_orchestrator",
    "create_model_backed_orchestrator",
]


def _compose_conversational_orchestrator(
    conversational_agent: BaseAgent,
) -> TuesdayOrchestrator:
    """Compose one conversational agent into TUESDAY's routed runtime."""
    registry = AgentRegistry()
    registry.register(conversational_agent)

    router = DeterministicRouter(
        {
            "chat": conversational_agent.name,
            "conversation": conversational_agent.name,
        }
    )
    request_preparer = DirectiveRequestPreparer()
    return TuesdayOrchestrator(
        router=router,
        registry=registry,
        request_preparer=request_preparer,
    )


def create_default_orchestrator() -> TuesdayOrchestrator:
    """Create a fresh deterministic TUESDAY application composition."""
    return _compose_conversational_orchestrator(ConversationalAgent())


def create_guarded_openai_orchestrator(
    tool_turn_service: OpenAIToolTurnService,
    tools: tuple[LanguageModelToolDefinition, ...],
    *,
    renderer: ConversationalPromptRenderer | None = None,
) -> TuesdayOrchestrator:
    """Compose an opt-in guarded agent with caller-configured tool capabilities."""
    conversational_agent = GuardedOpenAIConversationalAgent(
        tool_turn_service,
        tools,
        renderer=renderer,
    )
    return _compose_conversational_orchestrator(conversational_agent)


def create_model_backed_orchestrator(
    provider: BaseLanguageModelProvider,
    *,
    renderer: ConversationalPromptRenderer | None = None,
) -> TuesdayOrchestrator:
    """Create a fresh model-backed TUESDAY application composition."""
    conversational_agent = ModelBackedConversationalAgent(
        provider,
        renderer=renderer,
    )
    return _compose_conversational_orchestrator(conversational_agent)
