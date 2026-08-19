"""Provider-neutral tool authorization policy contracts."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from tuesday.tools.base import ToolInvocation

__all__ = [
    "BaseToolAuthorizationPolicy",
    "ToolAuthorizationDecision",
    "ToolAuthorizationOutcome",
]


class ToolAuthorizationOutcome(StrEnum):
    """Possible policy outcomes for one explicit tool invocation."""

    ALLOW = "allow"
    REQUIRE_CONFIRMATION = "require_confirmation"
    DENY = "deny"


def _require_meaningful_text(value: object, label: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    if not value.strip():
        raise ValueError(f"{label} must not be empty.")
    if value != value.strip():
        raise ValueError(f"{label} must not have surrounding whitespace.")


@dataclass(frozen=True, slots=True)
class ToolAuthorizationDecision:
    """An immutable policy decision correlated to one tool invocation."""

    tool_name: str
    invocation_id: UUID
    outcome: ToolAuthorizationOutcome
    reason: str

    def __post_init__(self) -> None:
        _require_meaningful_text(
            self.tool_name,
            "Tool authorization decision name",
        )
        if not isinstance(self.invocation_id, UUID):
            raise TypeError(
                "Tool authorization decision invocation_id must be a UUID."
            )
        if not isinstance(self.outcome, ToolAuthorizationOutcome):
            raise TypeError(
                "Tool authorization decision outcome must be a "
                "ToolAuthorizationOutcome."
            )
        _require_meaningful_text(
            self.reason,
            "Tool authorization decision reason",
        )


class BaseToolAuthorizationPolicy(ABC):
    """Abstract policy for deciding whether one invocation may proceed."""

    __slots__ = ()

    @abstractmethod
    async def authorize(
        self,
        invocation: ToolInvocation,
    ) -> ToolAuthorizationDecision:
        """Return an authorization decision without executing the tool."""
