"""Deterministic exact-name tool authorization policy."""

from collections.abc import Mapping
from types import MappingProxyType

from tuesday.tools.authorization import (
    BaseToolAuthorizationPolicy,
    ToolAuthorizationDecision,
    ToolAuthorizationOutcome,
)
from tuesday.tools.base import ToolInvocation

__all__ = ["StaticToolAuthorizationPolicy"]


class StaticToolAuthorizationPolicy(BaseToolAuthorizationPolicy):
    """Authorize exact tool names from an immutable rule snapshot."""

    __slots__ = ("_rules",)

    def __init__(
        self,
        rules: Mapping[str, ToolAuthorizationOutcome],
    ) -> None:
        if not isinstance(rules, Mapping):
            raise TypeError("rules must be a mapping.")

        validated_rules: dict[str, ToolAuthorizationOutcome] = {}
        for name, outcome in rules.items():
            if not isinstance(name, str):
                raise TypeError("Static authorization rule name must be a string.")
            if not name.strip():
                raise ValueError(
                    "Static authorization rule name must not be empty."
                )
            if name != name.strip():
                raise ValueError(
                    "Static authorization rule name must not have surrounding "
                    "whitespace."
                )
            if not isinstance(outcome, ToolAuthorizationOutcome):
                raise TypeError(
                    "Static authorization rule outcome must be a "
                    "ToolAuthorizationOutcome."
                )
            validated_rules[name] = outcome

        self._rules = MappingProxyType(validated_rules)

    async def authorize(
        self,
        invocation: ToolInvocation,
    ) -> ToolAuthorizationDecision:
        """Return a correlated decision for one exact tool name."""
        if not isinstance(invocation, ToolInvocation):
            raise TypeError("invocation must be a ToolInvocation.")

        outcome = self._rules.get(invocation.tool_name)
        if outcome is ToolAuthorizationOutcome.ALLOW:
            reason = (
                "Tool is explicitly allowed by the static authorization policy."
            )
        elif outcome is ToolAuthorizationOutcome.REQUIRE_CONFIRMATION:
            reason = (
                "Tool requires confirmation under the static authorization policy."
            )
        elif outcome is ToolAuthorizationOutcome.DENY:
            reason = (
                "Tool is explicitly denied by the static authorization policy."
            )
        else:
            outcome = ToolAuthorizationOutcome.DENY
            reason = (
                "Tool is not configured and is denied by the static authorization "
                "policy."
            )

        return ToolAuthorizationDecision(
            tool_name=invocation.tool_name,
            invocation_id=invocation.invocation_id,
            outcome=outcome,
            reason=reason,
        )
