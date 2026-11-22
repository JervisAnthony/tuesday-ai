"""Public contracts for TUESDAY agents."""

from tuesday.agents.base import BaseAgent
from tuesday.agents.conversational import ConversationalAgent
from tuesday.agents.model_backed import (
    ModelBackedConversationalAgent,
    UnsupportedModelToolCallError,
)
from tuesday.agents.registry import (
    AgentNotFoundError,
    AgentRegistrationError,
    AgentRegistry,
)

# Initialize AgentRegistry before the guarded agent imports the services package.
# isort: split
from tuesday.agents.guarded_openai import (
    GuardedOpenAIConversationalAgent,
    IncompleteGuardedToolTurnError,
)

__all__ = [
    "AgentNotFoundError",
    "AgentRegistrationError",
    "AgentRegistry",
    "BaseAgent",
    "ConversationalAgent",
    "GuardedOpenAIConversationalAgent",
    "IncompleteGuardedToolTurnError",
    "ModelBackedConversationalAgent",
    "UnsupportedModelToolCallError",
]
