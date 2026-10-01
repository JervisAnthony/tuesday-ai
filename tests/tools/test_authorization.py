"""Tests for provider-neutral tool authorization policy contracts."""

import ast
import asyncio
import inspect
from dataclasses import FrozenInstanceError
from enum import StrEnum
from pathlib import Path
from uuid import UUID, uuid4

import pytest

import tuesday.tools as tools
from tuesday.tools import (
    BaseTool,
    BaseToolAuthorizationPolicy,
    ToolAuthorizationDecision,
    ToolAuthorizationOutcome,
    ToolInvocation,
    ToolResult,
)


class StaticAuthorizationPolicy(BaseToolAuthorizationPolicy):
    """Test-only policy returning one preconfigured decision."""

    def __init__(self, decision: ToolAuthorizationDecision) -> None:
        self.decision = decision
        self.invocations: list[ToolInvocation] = []

    async def authorize(
        self,
        invocation: ToolInvocation,
    ) -> ToolAuthorizationDecision:
        self.invocations.append(invocation)
        return self.decision


class ExecutionCountingTool(BaseTool):
    """Test tool used to prove authorization does not execute tools."""

    def __init__(self) -> None:
        self.execute_calls = 0

    @property
    def name(self) -> str:
        return "calendar.delete"

    @property
    def description(self) -> str:
        return "Delete a calendar event."

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        self.execute_calls += 1
        return ToolResult(self.name, invocation.invocation_id, None)


def make_decision(
    *,
    tool_name: str = "calendar.delete",
    invocation_id: UUID | None = None,
    outcome: ToolAuthorizationOutcome = ToolAuthorizationOutcome.ALLOW,
    reason: str = "Invocation is permitted by the current policy.",
) -> ToolAuthorizationDecision:
    return ToolAuthorizationDecision(
        tool_name=tool_name,
        invocation_id=invocation_id or uuid4(),
        outcome=outcome,
        reason=reason,
    )


def test_outcome_is_str_enum_with_exact_members_and_values() -> None:
    assert issubclass(ToolAuthorizationOutcome, StrEnum)
    assert set(ToolAuthorizationOutcome.__members__) == {
        "ALLOW",
        "REQUIRE_CONFIRMATION",
        "DENY",
    }
    assert len(ToolAuthorizationOutcome) == 3
    assert ToolAuthorizationOutcome.ALLOW.value == "allow"
    assert (
        ToolAuthorizationOutcome.REQUIRE_CONFIRMATION.value
        == "require_confirmation"
    )
    assert ToolAuthorizationOutcome.DENY.value == "deny"
    assert isinstance(ToolAuthorizationOutcome.ALLOW, str)
    assert str(ToolAuthorizationOutcome.ALLOW) == "allow"


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        (
            ToolAuthorizationOutcome.ALLOW,
            "Invocation is permitted by the current policy.",
        ),
        (
            ToolAuthorizationOutcome.REQUIRE_CONFIRMATION,
            "Invocation requires explicit user confirmation.",
        ),
        (
            ToolAuthorizationOutcome.DENY,
            "Invocation is denied by the current policy.",
        ),
    ],
)
def test_valid_decision_preserves_all_fields(
    outcome: ToolAuthorizationOutcome,
    reason: str,
) -> None:
    invocation_id = uuid4()

    decision = make_decision(
        tool_name="Calendar.Delete",
        invocation_id=invocation_id,
        outcome=outcome,
        reason=reason,
    )

    assert decision.tool_name == "Calendar.Delete"
    assert decision.invocation_id is invocation_id
    assert decision.outcome is outcome
    assert decision.reason == reason


def test_decision_is_frozen_and_slotted() -> None:
    decision = make_decision()

    assert not hasattr(decision, "__dict__")
    with pytest.raises(FrozenInstanceError):
        decision.reason = "Changed."  # type: ignore[misc]


@pytest.mark.parametrize("invalid_tool_name", [None, 1, object()])
def test_decision_tool_name_requires_string(
    invalid_tool_name: object,
) -> None:
    with pytest.raises(TypeError, match="decision name must be a string"):
        make_decision(tool_name=invalid_tool_name)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_tool_name", ["", " ", "\t", "\n"])
def test_decision_tool_name_requires_meaningful_text(
    invalid_tool_name: str,
) -> None:
    with pytest.raises(ValueError, match="decision name must not be empty"):
        make_decision(tool_name=invalid_tool_name)


@pytest.mark.parametrize(
    "invalid_tool_name",
    [" calendar.read", "calendar.read ", " calendar.read "],
)
def test_decision_tool_name_rejects_surrounding_whitespace(
    invalid_tool_name: str,
) -> None:
    with pytest.raises(ValueError, match="must not have surrounding whitespace"):
        make_decision(tool_name=invalid_tool_name)


@pytest.mark.parametrize("invalid_invocation_id", [None, "uuid", 123, object()])
def test_decision_invocation_id_requires_uuid(
    invalid_invocation_id: object,
) -> None:
    with pytest.raises(TypeError, match="invocation_id must be a UUID"):
        ToolAuthorizationDecision(
            tool_name="calendar.read",
            invocation_id=invalid_invocation_id,  # type: ignore[arg-type]
            outcome=ToolAuthorizationOutcome.ALLOW,
            reason="Invocation is permitted.",
        )


@pytest.mark.parametrize(
    "invalid_outcome",
    [None, "allow", "deny", "require_confirmation", 1, object()],
)
def test_decision_outcome_requires_actual_enum(
    invalid_outcome: object,
) -> None:
    with pytest.raises(TypeError, match="outcome must be a ToolAuthorizationOutcome"):
        ToolAuthorizationDecision(
            tool_name="calendar.read",
            invocation_id=uuid4(),
            outcome=invalid_outcome,  # type: ignore[arg-type]
            reason="Invocation is permitted.",
        )


@pytest.mark.parametrize("invalid_reason", [None, 1, object()])
def test_decision_reason_requires_string(invalid_reason: object) -> None:
    with pytest.raises(TypeError, match="decision reason must be a string"):
        make_decision(reason=invalid_reason)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_reason", ["", " ", "\t", "\n"])
def test_decision_reason_requires_meaningful_text(invalid_reason: str) -> None:
    with pytest.raises(ValueError, match="decision reason must not be empty"):
        make_decision(reason=invalid_reason)


@pytest.mark.parametrize(
    "invalid_reason",
    [" reason", "reason ", " reason "],
)
def test_decision_reason_rejects_surrounding_whitespace(
    invalid_reason: str,
) -> None:
    with pytest.raises(ValueError, match="must not have surrounding whitespace"):
        make_decision(reason=invalid_reason)


def test_base_policy_is_abstract_slotted_async_contract() -> None:
    with pytest.raises(TypeError):
        BaseToolAuthorizationPolicy()

    assert BaseToolAuthorizationPolicy.__abstractmethods__ == {"authorize"}
    assert BaseToolAuthorizationPolicy.__slots__ == ()
    assert inspect.iscoroutinefunction(BaseToolAuthorizationPolicy.authorize)
    assert tuple(
        inspect.signature(BaseToolAuthorizationPolicy.authorize).parameters
    ) == ("self", "invocation")


def test_fake_policy_receives_exact_invocation_and_returns_exact_decision() -> None:
    invocation = ToolInvocation(tool_name="calendar.delete")
    decision = make_decision(
        tool_name=invocation.tool_name,
        invocation_id=invocation.invocation_id,
    )
    policy = StaticAuthorizationPolicy(decision)

    returned = asyncio.run(policy.authorize(invocation))

    assert policy.invocations == [invocation]
    assert policy.invocations[0] is invocation
    assert returned is decision


def test_policy_authorization_does_not_execute_tool() -> None:
    invocation = ToolInvocation(tool_name="calendar.delete")
    decision = make_decision(
        tool_name=invocation.tool_name,
        invocation_id=invocation.invocation_id,
    )
    policy = StaticAuthorizationPolicy(decision)
    tool = ExecutionCountingTool()

    asyncio.run(policy.authorize(invocation))

    assert tool.execute_calls == 0


def test_decision_exposes_both_invocation_correlation_fields() -> None:
    invocation = ToolInvocation(tool_name="calendar.delete")

    matching = make_decision(
        tool_name=invocation.tool_name,
        invocation_id=invocation.invocation_id,
    )
    different_name = make_decision(
        tool_name="calendar.read",
        invocation_id=invocation.invocation_id,
    )
    different_id = make_decision(tool_name=invocation.tool_name)

    assert matching.tool_name == invocation.tool_name
    assert matching.invocation_id is invocation.invocation_id
    assert different_name.tool_name != invocation.tool_name
    assert different_id.invocation_id != invocation.invocation_id


def test_authorization_contracts_are_publicly_exported() -> None:
    assert tools.BaseToolAuthorizationPolicy is BaseToolAuthorizationPolicy
    assert tools.ToolAuthorizationDecision is ToolAuthorizationDecision
    assert tools.ToolAuthorizationOutcome is ToolAuthorizationOutcome


def test_authorization_module_has_only_allowed_dependencies() -> None:
    source_path = (
        Path(__file__).parents[2]
        / "src"
        / "tuesday"
        / "tools"
        / "authorization.py"
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
        "abc",
        "dataclasses",
        "enum",
        "tuesday.tools.base",
        "uuid",
    }
