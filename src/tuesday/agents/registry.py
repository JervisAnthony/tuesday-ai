"""Explicit registration and lookup for TUESDAY agents."""

from tuesday.agents.base import BaseAgent

__all__ = [
    "AgentNotFoundError",
    "AgentRegistrationError",
    "AgentRegistry",
]


class AgentRegistrationError(ValueError):
    """Raised when an agent cannot be registered."""


class AgentNotFoundError(LookupError):
    """Raised when a requested agent is not registered."""


class AgentRegistry:
    """A deterministic collection of explicitly registered agent instances."""

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        """Register an agent under its exact stable name."""
        name = agent.name
        if not isinstance(name, str) or not name.strip():
            raise AgentRegistrationError(
                "Agent name must be a non-empty string."
            )
        if name in self._agents:
            raise AgentRegistrationError(
                f"An agent named {name!r} is already registered."
            )

        self._agents[name] = agent

    def get(self, name: str) -> BaseAgent:
        """Return the registered agent with the requested exact name."""
        try:
            return self._agents[name]
        except KeyError as error:
            raise AgentNotFoundError(
                f"No agent named {name!r} is registered."
            ) from error

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered names in deterministic registration order."""
        return tuple(self._agents)

    def __contains__(self, name: object) -> bool:
        """Return whether an exact agent name is registered."""
        return name in self._agents

    def __len__(self) -> int:
        """Return the number of registered agents."""
        return len(self._agents)
