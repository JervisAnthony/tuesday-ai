"""Tests for the deterministic static tool authorization policy."""

import ast
import asyncio
import inspect
from collections import UserDict
from pathlib import Path
from types import MappingProxyType

import pytest

import tuesday.tools as tools
from tuesday.tools import (
    BaseTool,
    BaseToolAuthorizationPolicy,
    DeterministicToolExecutor,
    GuardedToolExecutor,
    StaticToolAuthorizationPolicy,
    ToolAuthorizationDecision,
    ToolAuthorizationDeniedError,
    ToolAuthorizationOutcome,
    ToolInvocation,
    ToolRegistry,
    ToolResult,
)

ALLOW_REASON = "Tool is explicitly allowed by the static authorization policy."
CONFIRM_REASON = (
    "Tool requires confirmation under the static authorization policy."
)
DENY_REASON = "Tool is explicitly denied by the static authorization policy."
UNKNOWN_REASON = (
    "Tool is not configured and is denied by the static authorization policy."
)


class RecordingTool(BaseTool):
    """Small tool used to demonstrate guarded-policy compatibility."""

    def __init__(self, name: str) -> None:
        self._name = name
        self.execute_calls = 0
        self.invocations: list[ToolInvocation] = []
        self.result: ToolResult | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Record static-policy integration execution."

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        self.execute_calls += 1
        self.invocations.append(invocation)
        self.result = ToolResult(self.name, invocation.invocation_id, "done")
        return self.result


def authorize(
    policy: StaticToolAuthorizationPolicy,
    invocation: ToolInvocation,
) -> ToolAuthorizationDecision:
    return asyncio.run(policy.authorize(invocation))


@pytest.mark.parametrize(
    "rules",
    [
        {"calendar.read": ToolAuthorizationOutcome.ALLOW},
        MappingProxyType(
            {"calendar.read": ToolAuthorizationOutcome.ALLOW}
        ),
        UserDict({"calendar.read": ToolAuthorizationOutcome.ALLOW}),
    ],
)
def test_constructor_accepts_mapping_implementations(rules: object) -> None:
    policy = StaticToolAuthorizationPolicy(rules)  # type: ignore[arg-type]

    decision = authorize(policy, ToolInvocation(tool_name="calendar.read"))

    assert decision.outcome is ToolAuthorizationOutcome.ALLOW


@pytest.mark.parametrize("invalid_rules", [None, [], (), "rules", object()])
def test_constructor_rejects_non_mapping_rules(invalid_rules: object) -> None:
    with pytest.raises(TypeError, match="rules must be a mapping"):
        StaticToolAuthorizationPolicy(invalid_rules)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_name", [None, 1, object()])
def test_constructor_rejects_non_string_rule_names(invalid_name: object) -> None:
    with pytest.raises(TypeError, match="rule name must be a string"):
        StaticToolAuthorizationPolicy(
            {invalid_name: ToolAuthorizationOutcome.ALLOW}  # type: ignore[dict-item]
        )


@pytest.mark.parametrize("invalid_name", ["", " ", "\t", "\n"])
def test_constructor_rejects_empty_rule_names(invalid_name: str) -> None:
    with pytest.raises(ValueError, match="rule name must not be empty"):
        StaticToolAuthorizationPolicy(
            {invalid_name: ToolAuthorizationOutcome.ALLOW}
        )


@pytest.mark.parametrize(
    "invalid_name",
    [" calendar.read", "calendar.read ", " calendar.read "],
)
def test_constructor_rejects_rule_names_with_surrounding_whitespace(
    invalid_name: str,
) -> None:
    with pytest.raises(ValueError, match="must not have surrounding whitespace"):
        StaticToolAuthorizationPolicy(
            {invalid_name: ToolAuthorizationOutcome.ALLOW}
        )


@pytest.mark.parametrize(
    "invalid_outcome",
    [None, "allow", "require_confirmation", "deny", 1, object()],
)
def test_constructor_requires_actual_outcome_enum(
    invalid_outcome: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="rule outcome must be a ToolAuthorizationOutcome",
    ):
        StaticToolAuthorizationPolicy(
            {"calendar.read": invalid_outcome}  # type: ignore[dict-item]
        )


def test_empty_policy_denies_everything_with_correlated_decision() -> None:
    invocation = ToolInvocation(tool_name="calendar.read")

    decision = authorize(StaticToolAuthorizationPolicy({}), invocation)

    assert isinstance(decision, ToolAuthorizationDecision)
    assert decision.tool_name == invocation.tool_name
    assert decision.invocation_id is invocation.invocation_id
    assert decision.outcome is ToolAuthorizationOutcome.DENY
    assert decision.reason == UNKNOWN_REASON


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        (ToolAuthorizationOutcome.ALLOW, ALLOW_REASON),
        (ToolAuthorizationOutcome.REQUIRE_CONFIRMATION, CONFIRM_REASON),
        (ToolAuthorizationOutcome.DENY, DENY_REASON),
    ],
)
def test_configured_outcomes_return_deterministic_correlated_decisions(
    outcome: ToolAuthorizationOutcome,
    reason: str,
) -> None:
    invocation = ToolInvocation(tool_name="calendar.read")
    policy = StaticToolAuthorizationPolicy({invocation.tool_name: outcome})

    decision = authorize(policy, invocation)

    assert isinstance(decision, ToolAuthorizationDecision)
    assert decision.tool_name == invocation.tool_name
    assert decision.invocation_id is invocation.invocation_id
    assert decision.outcome is outcome
    assert decision.reason == reason


@pytest.mark.parametrize(
    "invalid_invocation",
    [None, object(), "calendar.read", 123, {}],
)
def test_authorize_rejects_invalid_invocation_before_field_access(
    invalid_invocation: object,
) -> None:
    policy = StaticToolAuthorizationPolicy({})

    with pytest.raises(TypeError, match="invocation must be a ToolInvocation"):
        asyncio.run(
            policy.authorize(invalid_invocation)  # type: ignore[arg-type]
        )


def test_lookup_is_exact_and_case_sensitive() -> None:
    policy = StaticToolAuthorizationPolicy(
        {"calendar.read": ToolAuthorizationOutcome.ALLOW}
    )

    exact = authorize(policy, ToolInvocation(tool_name="calendar.read"))
    different_case = authorize(
        policy,
        ToolInvocation(tool_name="Calendar.Read"),
    )

    assert exact.outcome is ToolAuthorizationOutcome.ALLOW
    assert exact.reason == ALLOW_REASON
    assert different_case.outcome is ToolAuthorizationOutcome.DENY
    assert different_case.reason == UNKNOWN_REASON


def test_caller_mutation_cannot_change_snapshotted_rules() -> None:
    rules = {"calendar.read": ToolAuthorizationOutcome.ALLOW}
    policy = StaticToolAuthorizationPolicy(rules)
    rules["calendar.read"] = ToolAuthorizationOutcome.DENY
    rules["calendar.delete"] = ToolAuthorizationOutcome.ALLOW

    original = authorize(policy, ToolInvocation(tool_name="calendar.read"))
    added_later = authorize(
        policy,
        ToolInvocation(tool_name="calendar.delete"),
    )

    assert original.outcome is ToolAuthorizationOutcome.ALLOW
    assert added_later.outcome is ToolAuthorizationOutcome.DENY


def test_constructor_does_not_mutate_caller_mapping() -> None:
    rules = {
        "Calendar.Read": ToolAuthorizationOutcome.ALLOW,
        "calendar.delete": ToolAuthorizationOutcome.REQUIRE_CONFIRMATION,
    }
    before = rules.copy()

    StaticToolAuthorizationPolicy(rules)

    assert rules == before


def test_internal_rule_snapshot_is_immutable() -> None:
    policy = StaticToolAuthorizationPolicy(
        {"calendar.read": ToolAuthorizationOutcome.ALLOW}
    )

    with pytest.raises(TypeError):
        policy._rules["calendar.read"] = ToolAuthorizationOutcome.DENY  # type: ignore[index]


def test_repeated_authorization_is_stateless_and_returns_fresh_decisions() -> None:
    policy = StaticToolAuthorizationPolicy(
        {"calendar.read": ToolAuthorizationOutcome.ALLOW}
    )
    invocation = ToolInvocation(tool_name="calendar.read")
    rules_before = policy._rules

    first = authorize(policy, invocation)
    second = authorize(policy, invocation)

    assert first is not second
    assert first == second
    assert first.invocation_id is invocation.invocation_id
    assert second.invocation_id is invocation.invocation_id
    assert policy._rules is rules_before
    assert not hasattr(policy, "_history")


def test_different_invocations_receive_their_own_correlation() -> None:
    policy = StaticToolAuthorizationPolicy(
        {"calendar.read": ToolAuthorizationOutcome.ALLOW}
    )
    first_invocation = ToolInvocation(tool_name="calendar.read")
    second_invocation = ToolInvocation(tool_name="calendar.read")

    first = authorize(policy, first_invocation)
    second = authorize(policy, second_invocation)

    assert first.invocation_id is first_invocation.invocation_id
    assert second.invocation_id is second_invocation.invocation_id
    assert first.invocation_id != second.invocation_id


def test_unknown_tool_returns_deny_decision_not_exception() -> None:
    invocation = ToolInvocation(tool_name="unregistered.tool")

    decision = authorize(StaticToolAuthorizationPolicy({}), invocation)

    assert isinstance(decision, ToolAuthorizationDecision)
    assert decision.outcome is ToolAuthorizationOutcome.DENY
    assert decision.reason == UNKNOWN_REASON


def test_policy_does_not_inspect_invocation_arguments() -> None:
    policy = StaticToolAuthorizationPolicy(
        {"calendar.read": ToolAuthorizationOutcome.ALLOW}
    )
    first = ToolInvocation(
        tool_name="calendar.read",
        arguments={"event_id": "one"},
    )
    second = ToolInvocation(
        tool_name="calendar.read",
        arguments={"event_id": "two", "private": True},
    )

    assert authorize(policy, first).outcome is ToolAuthorizationOutcome.ALLOW
    assert authorize(policy, second).outcome is ToolAuthorizationOutcome.ALLOW


def test_class_shape_matches_authorization_policy_contract() -> None:
    policy = StaticToolAuthorizationPolicy({})

    assert issubclass(
        StaticToolAuthorizationPolicy,
        BaseToolAuthorizationPolicy,
    )
    assert inspect.iscoroutinefunction(StaticToolAuthorizationPolicy.authorize)
    assert tuple(
        inspect.signature(StaticToolAuthorizationPolicy.authorize).parameters
    ) == ("self", "invocation")
    assert StaticToolAuthorizationPolicy.__slots__ == ("_rules",)
    assert not hasattr(policy, "__dict__")


def test_static_policy_is_publicly_exported() -> None:
    assert tools.StaticToolAuthorizationPolicy is StaticToolAuthorizationPolicy


def test_allow_rule_integrates_with_guarded_executor() -> None:
    invocation = ToolInvocation(tool_name="calendar.read")
    tool = RecordingTool(invocation.tool_name)
    registry = ToolRegistry()
    registry.register(tool)
    guarded = GuardedToolExecutor(
        StaticToolAuthorizationPolicy(
            {invocation.tool_name: ToolAuthorizationOutcome.ALLOW}
        ),
        DeterministicToolExecutor(registry),
    )

    returned = asyncio.run(guarded.execute(invocation))

    assert returned is tool.result
    assert tool.execute_calls == 1
    assert tool.invocations[0] is invocation


def test_unconfigured_rule_is_blocked_by_guarded_executor() -> None:
    invocation = ToolInvocation(tool_name="calendar.read")
    tool = RecordingTool(invocation.tool_name)
    registry = ToolRegistry()
    registry.register(tool)
    guarded = GuardedToolExecutor(
        StaticToolAuthorizationPolicy({}),
        DeterministicToolExecutor(registry),
    )

    with pytest.raises(ToolAuthorizationDeniedError) as caught:
        asyncio.run(guarded.execute(invocation))

    assert caught.value.decision.outcome is ToolAuthorizationOutcome.DENY
    assert caught.value.decision.reason == UNKNOWN_REASON
    assert tool.execute_calls == 0


def test_static_policy_module_has_only_allowed_dependencies() -> None:
    source_path = (
        Path(__file__).parents[2]
        / "src"
        / "tuesday"
        / "tools"
        / "static_policy.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )

    assert imports == {
        "collections.abc",
        "types",
        "tuesday.tools.authorization",
        "tuesday.tools.base",
    }
