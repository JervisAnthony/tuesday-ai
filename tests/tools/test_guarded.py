"""Tests for fail-closed authorization-aware tool execution."""

import ast
import asyncio
import inspect
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest

import tuesday.tools as tools
from tuesday.tools import (
    BaseTool,
    BaseToolAuthorizationPolicy,
    DeterministicToolExecutor,
    GuardedToolExecutor,
    InvalidToolAuthorizationDecisionError,
    InvalidToolResultError,
    ToolAuthorizationBlockedError,
    ToolAuthorizationDecision,
    ToolAuthorizationDeniedError,
    ToolAuthorizationOutcome,
    ToolConfirmationRequiredError,
    ToolExecutionError,
    ToolInvocation,
    ToolNotFoundError,
    ToolRegistry,
    ToolResult,
)


class RecordingAuthorizationPolicy(BaseToolAuthorizationPolicy):
    """Configurable test policy that records authorization attempts."""

    def __init__(
        self,
        decision: object = None,
        *,
        error: BaseException | None = None,
    ) -> None:
        self.decision = decision
        self.error = error
        self.authorize_calls = 0
        self.invocations: list[ToolInvocation] = []

    async def authorize(
        self,
        invocation: ToolInvocation,
    ) -> ToolAuthorizationDecision:
        self.authorize_calls += 1
        self.invocations.append(invocation)
        if self.error is not None:
            raise self.error
        return cast(ToolAuthorizationDecision, self.decision)


class RecordingTool(BaseTool):
    """Configurable tool that records deterministic execution calls."""

    def __init__(
        self,
        name: str,
        *,
        result: object = None,
        error: BaseException | None = None,
    ) -> None:
        self._name = name
        self.result = result
        self.error = error
        self.execute_calls = 0
        self.invocations: list[ToolInvocation] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Record guarded tool execution."

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        self.execute_calls += 1
        self.invocations.append(invocation)
        if self.error is not None:
            raise self.error
        return cast(ToolResult, self.result)


class RecordingDeterministicExecutor(DeterministicToolExecutor):
    """Deterministic executor subclass that records delegation calls."""

    __slots__ = ("execute_calls", "invocations")

    def __init__(self, registry: ToolRegistry) -> None:
        super().__init__(registry)
        self.execute_calls = 0
        self.invocations: list[ToolInvocation] = []

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        self.execute_calls += 1
        self.invocations.append(invocation)
        return await super().execute(invocation)


class FailingDeterministicExecutor(DeterministicToolExecutor):
    """Executor subclass that raises one exact configured exception."""

    __slots__ = ("error", "execute_calls")

    def __init__(self, error: BaseException) -> None:
        super().__init__(ToolRegistry())
        self.error = error
        self.execute_calls = 0

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        self.execute_calls += 1
        raise self.error


def decision_for(
    invocation: ToolInvocation,
    outcome: ToolAuthorizationOutcome = ToolAuthorizationOutcome.ALLOW,
    *,
    tool_name: str | None = None,
    invocation_id: object = None,
) -> ToolAuthorizationDecision:
    return ToolAuthorizationDecision(
        tool_name=tool_name or invocation.tool_name,
        invocation_id=(
            invocation.invocation_id
            if invocation_id is None
            else invocation_id  # type: ignore[arg-type]
        ),
        outcome=outcome,
        reason=f"Policy outcome is {outcome.value}.",
    )


def configured_guarded(
    outcome: ToolAuthorizationOutcome = ToolAuthorizationOutcome.ALLOW,
) -> tuple[
    GuardedToolExecutor,
    RecordingAuthorizationPolicy,
    RecordingDeterministicExecutor,
    ToolRegistry,
    RecordingTool,
    ToolInvocation,
    ToolAuthorizationDecision,
    ToolResult,
]:
    invocation = ToolInvocation(
        tool_name="calendar.delete",
        arguments={"event_id": "event-1"},
    )
    decision = decision_for(invocation, outcome)
    result = ToolResult(
        tool_name=invocation.tool_name,
        invocation_id=invocation.invocation_id,
        output={"deleted": True},
    )
    tool = RecordingTool(invocation.tool_name, result=result)
    registry = ToolRegistry()
    registry.register(tool)
    executor = RecordingDeterministicExecutor(registry)
    policy = RecordingAuthorizationPolicy(decision)
    guarded = GuardedToolExecutor(policy, executor)
    return guarded, policy, executor, registry, tool, invocation, decision, result


def run_execute(
    guarded: GuardedToolExecutor,
    invocation: ToolInvocation,
) -> ToolResult:
    return asyncio.run(guarded.execute(invocation))


def test_constructor_preserves_exact_dependencies_and_slotted_state() -> None:
    components = configured_guarded()
    guarded, policy, executor = components[:3]

    assert guarded._policy is policy
    assert guarded._executor is executor
    assert GuardedToolExecutor.__slots__ == ("_policy", "_executor")
    assert not hasattr(guarded, "__dict__")


@pytest.mark.parametrize("invalid_policy", [None, object(), {}, "policy"])
def test_constructor_rejects_invalid_policy(invalid_policy: object) -> None:
    with pytest.raises(
        TypeError,
        match="policy must be a BaseToolAuthorizationPolicy instance",
    ):
        GuardedToolExecutor(
            invalid_policy,  # type: ignore[arg-type]
            DeterministicToolExecutor(ToolRegistry()),
        )


@pytest.mark.parametrize("invalid_executor", [None, object(), {}, "executor"])
def test_constructor_rejects_invalid_executor(invalid_executor: object) -> None:
    policy = RecordingAuthorizationPolicy()

    with pytest.raises(
        TypeError,
        match="executor must be a DeterministicToolExecutor instance",
    ):
        GuardedToolExecutor(
            policy,
            invalid_executor,  # type: ignore[arg-type]
        )


def test_constructor_accepts_dependency_subclasses() -> None:
    policy = RecordingAuthorizationPolicy()
    executor = RecordingDeterministicExecutor(ToolRegistry())

    guarded = GuardedToolExecutor(policy, executor)

    assert guarded._policy is policy
    assert guarded._executor is executor


def test_execute_is_async_with_exact_parameters() -> None:
    assert inspect.iscoroutinefunction(GuardedToolExecutor.execute)
    assert tuple(inspect.signature(GuardedToolExecutor.execute).parameters) == (
        "self",
        "invocation",
    )


@pytest.mark.parametrize(
    "invalid_invocation",
    [None, object(), "invalid", 123, {}],
)
def test_invalid_invocation_fails_before_policy_and_execution(
    invalid_invocation: object,
) -> None:
    guarded, policy, executor, _, tool, _, _, _ = configured_guarded()

    with pytest.raises(TypeError, match="invocation must be a ToolInvocation"):
        asyncio.run(
            guarded.execute(invalid_invocation)  # type: ignore[arg-type]
        )

    assert policy.authorize_calls == 0
    assert executor.execute_calls == 0
    assert tool.execute_calls == 0


def test_allow_preserves_identity_counts_and_registry_state() -> None:
    guarded, policy, executor, registry, tool, invocation, decision, result = (
        configured_guarded()
    )
    names_before = registry.names

    returned = run_execute(guarded, invocation)

    assert policy.authorize_calls == 1
    assert policy.invocations == [invocation]
    assert policy.invocations[0] is invocation
    assert executor.execute_calls == 1
    assert executor.invocations == [invocation]
    assert executor.invocations[0] is invocation
    assert tool.execute_calls == 1
    assert tool.invocations[0] is invocation
    assert returned is result
    assert policy.decision is decision
    assert registry.names == names_before
    assert registry.get(tool.name) is tool


@pytest.mark.parametrize(
    ("outcome", "error_type"),
    [
        (
            ToolAuthorizationOutcome.REQUIRE_CONFIRMATION,
            ToolConfirmationRequiredError,
        ),
        (ToolAuthorizationOutcome.DENY, ToolAuthorizationDeniedError),
    ],
)
def test_blocked_outcome_preserves_decision_without_execution(
    outcome: ToolAuthorizationOutcome,
    error_type: type[ToolAuthorizationBlockedError],
) -> None:
    guarded, policy, executor, _, tool, invocation, decision, _ = (
        configured_guarded(outcome)
    )

    with pytest.raises(error_type) as caught:
        run_execute(guarded, invocation)

    assert caught.value.decision is decision
    assert str(caught.value) == decision.reason
    assert policy.authorize_calls == 1
    assert policy.invocations[0] is invocation
    assert executor.execute_calls == 0
    assert tool.execute_calls == 0


def test_blocked_error_hierarchy_is_distinct_from_tool_execution_failure() -> None:
    assert issubclass(
        ToolConfirmationRequiredError,
        ToolAuthorizationBlockedError,
    )
    assert issubclass(
        ToolAuthorizationDeniedError,
        ToolAuthorizationBlockedError,
    )
    assert issubclass(ToolAuthorizationBlockedError, RuntimeError)
    assert not issubclass(ToolAuthorizationBlockedError, ToolExecutionError)


@pytest.mark.parametrize("invalid_decision", [None, object(), "decision"])
def test_blocked_error_requires_decision(invalid_decision: object) -> None:
    with pytest.raises(
        TypeError,
        match="decision must be a ToolAuthorizationDecision",
    ):
        ToolAuthorizationBlockedError(
            invalid_decision  # type: ignore[arg-type]
        )


def test_blocked_error_preserves_exact_valid_decision() -> None:
    invocation = ToolInvocation(tool_name="calendar.delete")
    decision = decision_for(invocation, ToolAuthorizationOutcome.DENY)

    error = ToolAuthorizationBlockedError(decision)

    assert error.decision is decision
    assert str(error) == decision.reason


@pytest.mark.parametrize(
    "invalid_decision",
    [None, object(), "allow", ToolAuthorizationOutcome.ALLOW, {}, ()],
)
def test_invalid_policy_result_fails_before_execution(
    invalid_decision: object,
) -> None:
    guarded, policy, executor, _, tool, invocation, _, _ = configured_guarded()
    policy.decision = invalid_decision

    with pytest.raises(
        InvalidToolAuthorizationDecisionError,
        match="Authorization policy must return a ToolAuthorizationDecision",
    ):
        run_execute(guarded, invocation)

    assert policy.authorize_calls == 1
    assert executor.execute_calls == 0
    assert tool.execute_calls == 0


@pytest.mark.parametrize(
    "outcome",
    list(ToolAuthorizationOutcome),
)
def test_tool_name_mismatch_is_invalid_before_outcome_interpretation(
    outcome: ToolAuthorizationOutcome,
) -> None:
    guarded, policy, executor, _, tool, invocation, _, _ = configured_guarded()
    decision = decision_for(
        invocation,
        outcome,
        tool_name="calendar.read",
    )
    policy.decision = decision

    with pytest.raises(
        InvalidToolAuthorizationDecisionError,
        match="tool_name does not match the invocation",
    ):
        run_execute(guarded, invocation)

    assert policy.authorize_calls == 1
    assert executor.execute_calls == 0
    assert tool.execute_calls == 0
    assert decision.tool_name == "calendar.read"


def test_invocation_id_mismatch_is_invalid_without_repair_or_execution() -> None:
    guarded, policy, executor, _, tool, invocation, _, _ = configured_guarded()
    mismatched_id = uuid4()
    decision = decision_for(invocation, invocation_id=mismatched_id)
    policy.decision = decision

    with pytest.raises(
        InvalidToolAuthorizationDecisionError,
        match="invocation_id does not match the invocation",
    ):
        run_execute(guarded, invocation)

    assert policy.authorize_calls == 1
    assert executor.execute_calls == 0
    assert tool.execute_calls == 0
    assert decision.invocation_id is mismatched_id


def test_policy_exception_propagates_by_identity_without_execution() -> None:
    guarded, policy, executor, _, tool, invocation, _, _ = configured_guarded()
    error = RuntimeError("policy failure")
    policy.error = error

    with pytest.raises(RuntimeError) as caught:
        run_execute(guarded, invocation)

    assert caught.value is error
    assert policy.authorize_calls == 1
    assert executor.execute_calls == 0
    assert tool.execute_calls == 0


@pytest.mark.parametrize(
    "error",
    [
        ToolNotFoundError("missing"),
        ToolExecutionError("tool failure"),
        InvalidToolResultError("invalid result"),
        RuntimeError("unexpected"),
    ],
)
def test_executor_exception_propagates_by_identity_without_retry(
    error: BaseException,
) -> None:
    invocation = ToolInvocation(tool_name="calendar.delete")
    decision = decision_for(invocation)
    policy = RecordingAuthorizationPolicy(decision)
    executor = FailingDeterministicExecutor(error)
    guarded = GuardedToolExecutor(policy, executor)

    with pytest.raises(type(error)) as caught:
        run_execute(guarded, invocation)

    assert caught.value is error
    assert policy.authorize_calls == 1
    assert executor.execute_calls == 1


def test_real_executor_invalid_result_propagates_without_reauthorization() -> None:
    invocation = ToolInvocation(tool_name="calendar.delete")
    policy = RecordingAuthorizationPolicy(decision_for(invocation))
    tool = RecordingTool(invocation.tool_name, result=None)
    registry = ToolRegistry()
    registry.register(tool)
    guarded = GuardedToolExecutor(
        policy,
        DeterministicToolExecutor(registry),
    )

    with pytest.raises(InvalidToolResultError):
        run_execute(guarded, invocation)

    assert policy.authorize_calls == 1
    assert tool.execute_calls == 1


def test_real_executor_missing_tool_propagates_without_reauthorization() -> None:
    invocation = ToolInvocation(tool_name="calendar.delete")
    policy = RecordingAuthorizationPolicy(decision_for(invocation))
    guarded = GuardedToolExecutor(
        policy,
        DeterministicToolExecutor(ToolRegistry()),
    )

    with pytest.raises(ToolNotFoundError):
        run_execute(guarded, invocation)

    assert policy.authorize_calls == 1


def test_real_tool_failure_propagates_by_identity_without_retry() -> None:
    invocation = ToolInvocation(tool_name="calendar.delete")
    error = ToolExecutionError("tool failure")
    policy = RecordingAuthorizationPolicy(decision_for(invocation))
    tool = RecordingTool(invocation.tool_name, error=error)
    registry = ToolRegistry()
    registry.register(tool)
    guarded = GuardedToolExecutor(
        policy,
        DeterministicToolExecutor(registry),
    )

    with pytest.raises(ToolExecutionError) as caught:
        run_execute(guarded, invocation)

    assert caught.value is error
    assert policy.authorize_calls == 1
    assert tool.execute_calls == 1


def test_multiple_registered_tools_execute_only_selected_tool() -> None:
    invocation = ToolInvocation(tool_name="calendar.delete")
    result = ToolResult(invocation.tool_name, invocation.invocation_id, None)
    selected = RecordingTool(invocation.tool_name, result=result)
    other = RecordingTool("calendar.read")
    registry = ToolRegistry()
    registry.register(selected)
    registry.register(other)
    policy = RecordingAuthorizationPolicy(decision_for(invocation))
    guarded = GuardedToolExecutor(
        policy,
        DeterministicToolExecutor(registry),
    )

    returned = run_execute(guarded, invocation)

    assert returned is result
    assert policy.authorize_calls == 1
    assert selected.execute_calls == 1
    assert other.execute_calls == 0


def test_guarded_symbols_are_publicly_exported() -> None:
    assert tools.GuardedToolExecutor is GuardedToolExecutor
    assert (
        tools.InvalidToolAuthorizationDecisionError
        is InvalidToolAuthorizationDecisionError
    )
    assert tools.ToolAuthorizationBlockedError is ToolAuthorizationBlockedError
    assert tools.ToolAuthorizationDeniedError is ToolAuthorizationDeniedError
    assert tools.ToolConfirmationRequiredError is ToolConfirmationRequiredError


def test_guarded_module_has_only_allowed_dependencies() -> None:
    source_path = (
        Path(__file__).parents[2] / "src" / "tuesday" / "tools" / "guarded.py"
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
        "tuesday.tools.authorization",
        "tuesday.tools.base",
        "tuesday.tools.executor",
    }
