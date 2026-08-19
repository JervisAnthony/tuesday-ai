"""Fail-closed authorization-aware tool execution."""

from tuesday.tools.authorization import (
    BaseToolAuthorizationPolicy,
    ToolAuthorizationDecision,
    ToolAuthorizationOutcome,
)
from tuesday.tools.base import ToolInvocation, ToolResult
from tuesday.tools.executor import DeterministicToolExecutor

__all__ = [
    "GuardedToolExecutor",
    "InvalidToolAuthorizationDecisionError",
    "ToolAuthorizationBlockedError",
    "ToolAuthorizationDeniedError",
    "ToolConfirmationRequiredError",
]


class InvalidToolAuthorizationDecisionError(ValueError):
    """Raised when a policy decision is malformed or incorrectly correlated."""


class ToolAuthorizationBlockedError(RuntimeError):
    """Base error for a valid decision that blocks immediate execution."""

    __slots__ = ("decision",)

    def __init__(self, decision: ToolAuthorizationDecision) -> None:
        if not isinstance(decision, ToolAuthorizationDecision):
            raise TypeError("decision must be a ToolAuthorizationDecision.")
        self.decision = decision
        super().__init__(decision.reason)


class ToolConfirmationRequiredError(ToolAuthorizationBlockedError):
    """Raised when execution requires confirmation that is not yet modeled."""


class ToolAuthorizationDeniedError(ToolAuthorizationBlockedError):
    """Raised when a valid authorization decision denies execution."""


class GuardedToolExecutor:
    """Authorize one invocation before deterministic tool execution."""

    __slots__ = ("_policy", "_executor")

    def __init__(
        self,
        policy: BaseToolAuthorizationPolicy,
        executor: DeterministicToolExecutor,
    ) -> None:
        if not isinstance(policy, BaseToolAuthorizationPolicy):
            raise TypeError(
                "policy must be a BaseToolAuthorizationPolicy instance."
            )
        if not isinstance(executor, DeterministicToolExecutor):
            raise TypeError(
                "executor must be a DeterministicToolExecutor instance."
            )
        self._policy = policy
        self._executor = executor

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        """Authorize and, only when allowed, execute one invocation."""
        if not isinstance(invocation, ToolInvocation):
            raise TypeError("invocation must be a ToolInvocation.")

        decision = await self._policy.authorize(invocation)
        if not isinstance(decision, ToolAuthorizationDecision):
            raise InvalidToolAuthorizationDecisionError(
                "Authorization policy must return a ToolAuthorizationDecision."
            )
        if decision.tool_name != invocation.tool_name:
            raise InvalidToolAuthorizationDecisionError(
                "Authorization decision tool_name does not match the invocation."
            )
        if decision.invocation_id != invocation.invocation_id:
            raise InvalidToolAuthorizationDecisionError(
                "Authorization decision invocation_id does not match the invocation."
            )

        if decision.outcome is ToolAuthorizationOutcome.ALLOW:
            return await self._executor.execute(invocation)
        if decision.outcome is ToolAuthorizationOutcome.REQUIRE_CONFIRMATION:
            raise ToolConfirmationRequiredError(decision)
        raise ToolAuthorizationDeniedError(decision)
