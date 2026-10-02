"""Contract and guarded integration tests for model-requested execution."""

import ast
import asyncio
import inspect
from dataclasses import FrozenInstanceError, fields
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest

import tuesday.bridges as bridges
import tuesday.bridges.model_execution as model_execution
from tuesday.bridges import (
    GuardedModelToolExecutor,
    ModelToolExecution,
    ModelToolInvocationBridge,
)
from tuesday.language_models.tools import LanguageModelToolCall
from tuesday.tools import (
    BaseTool,
    BasicCalculatorTool,
    DeterministicToolExecutor,
    GuardedToolExecutor,
    InvalidToolAuthorizationDecisionError,
    InvalidToolResultError,
    StaticToolAuthorizationPolicy,
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

INVOCATION_ID = UUID("11111111-1111-1111-1111-111111111111")


class RecordingTool(BaseTool):
    """Record execution without changing any production tool."""

    def __init__(self) -> None:
        self.invocations: list[ToolInvocation] = []
        self.result: ToolResult | None = None
        self.error: Exception | None = None

    @property
    def name(self) -> str:
        return "calculator.basic"

    @property
    def description(self) -> str:
        return "Record one invocation."

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        self._validate_invocation(invocation)
        self.invocations.append(invocation)
        if self.error is not None:
            raise self.error
        self.result = ToolResult(self.name, invocation.invocation_id, 42)
        return self.result


def make_objects() -> tuple[LanguageModelToolCall, ToolInvocation, ToolResult]:
    call = LanguageModelToolCall(
        "opaque:model-call",
        "calculator.basic",
        {
            "payload": {"items": [1, True, None]},
        },
    )
    invocation = ToolInvocation(call.name, call.arguments, INVOCATION_ID)
    result = ToolResult(invocation.tool_name, invocation.invocation_id, 42)
    return call, invocation, result


def make_stack(
    rules: dict[str, ToolAuthorizationOutcome],
    tool: BaseTool | None = None,
) -> tuple[GuardedModelToolExecutor, Mock, StaticToolAuthorizationPolicy, ToolRegistry]:
    registry = ToolRegistry()
    if tool is not None:
        registry.register(tool)
    policy = StaticToolAuthorizationPolicy(rules)
    guarded = GuardedToolExecutor(policy, DeterministicToolExecutor(registry))
    factory = Mock(return_value=INVOCATION_ID)
    bridge = ModelToolInvocationBridge(invocation_id_factory=factory)
    return GuardedModelToolExecutor(bridge, guarded), factory, policy, registry


def test_execution_record_is_frozen_slotted_and_preserves_objects() -> None:
    call, invocation, result = make_objects()
    execution = ModelToolExecution(call, invocation, result)
    assert execution.tool_call is call
    assert execution.invocation is invocation
    assert execution.result is result
    assert invocation.arguments == call.arguments
    assert invocation.arguments is not call.arguments
    assert tuple(field.name for field in fields(execution)) == (
        "tool_call",
        "invocation",
        "result",
    )
    assert not hasattr(execution, "__dict__")
    for name in ("tool_call", "invocation", "result"):
        with pytest.raises(FrozenInstanceError):
            setattr(execution, name, None)


@pytest.mark.parametrize("invalid", [None, {}, object(), "object"])
@pytest.mark.parametrize("field", ["tool_call", "invocation", "result"])
def test_record_requires_actual_types(field: str, invalid: object) -> None:
    values = dict(
        zip(("tool_call", "invocation", "result"), make_objects(), strict=True)
    )
    values[field] = invalid
    expected = {
        "tool_call": "tool_call must be a LanguageModelToolCall.",
        "invocation": "invocation must be a ToolInvocation.",
        "result": "result must be a ToolResult.",
    }
    with pytest.raises(TypeError) as caught:
        ModelToolExecution(**values)
    assert str(caught.value) == expected[field]


@pytest.mark.parametrize(
    "mismatch",
    [
        "invocation name",
        "invocation arguments",
        "result name",
        "result invocation_id",
    ],
)
def test_record_rejects_independent_correlation_mismatches(mismatch: str) -> None:
    call, invocation, result = make_objects()
    if mismatch == "invocation name":
        invocation = ToolInvocation("Calculator.Basic", call.arguments, INVOCATION_ID)
    elif mismatch == "invocation arguments":
        invocation = ToolInvocation(call.name, {"different": 1}, INVOCATION_ID)
    elif mismatch == "result name":
        result = ToolResult("Calculator.Basic", INVOCATION_ID, 42)
    else:
        result = ToolResult(call.name, UUID(int=2), 42)
    target = "tool call" if mismatch.startswith("invocation") else "invocation"
    verb = "do" if mismatch == "invocation arguments" else "does"
    with pytest.raises(ValueError) as caught:
        ModelToolExecution(call, invocation, result)
    assert str(caught.value) == (
        f"Model tool execution {mismatch} {verb} not match the {target}."
    )


def test_real_calculator_allowed_end_to_end() -> None:
    executor, factory, _, _ = make_stack(
        {"calculator.basic": ToolAuthorizationOutcome.ALLOW},
        BasicCalculatorTool(),
    )
    call = LanguageModelToolCall(
        "call_calc_1",
        "calculator.basic",
        {
            "operation": "multiply",
            "left": 6,
            "right": 7,
        },
    )
    execution = asyncio.run(executor.execute(call))
    assert isinstance(execution, ModelToolExecution)
    assert execution.tool_call is call
    assert execution.tool_call.call_id == "call_calc_1"
    assert execution.invocation.tool_name == "calculator.basic"
    assert execution.invocation.arguments == call.arguments
    assert execution.invocation.invocation_id is INVOCATION_ID
    assert execution.result.tool_name == execution.invocation.tool_name
    assert execution.result.invocation_id == execution.invocation.invocation_id
    assert execution.result.output == 42
    assert str(execution.invocation.invocation_id) != call.call_id
    factory.assert_called_once_with()


@pytest.mark.parametrize(
    "outcome, error_type",
    [
        (ToolAuthorizationOutcome.ALLOW, None),
        (ToolAuthorizationOutcome.REQUIRE_CONFIRMATION, ToolConfirmationRequiredError),
        (ToolAuthorizationOutcome.DENY, ToolAuthorizationDeniedError),
    ],
)
def test_authorization_counts_order_and_identity(
    outcome: ToolAuthorizationOutcome,
    error_type: type[Exception] | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = RecordingTool()
    executor, factory, policy, registry = make_stack({tool.name: outcome}, tool)
    call, _, _ = make_objects()
    events = []
    bridge_method = ModelToolInvocationBridge.to_invocation
    guarded_method = GuardedToolExecutor.execute
    policy_method = StaticToolAuthorizationPolicy.authorize
    registry_method = ToolRegistry.get
    converted = []
    decisions = []
    blocked = []

    def convert(self, source):
        events.append("bridge")
        invocation = bridge_method(self, source)
        converted.append(invocation)
        assert source is call
        return invocation

    async def authorize(self, invocation):
        events.append("authorize")
        assert self is policy
        assert invocation is converted[0]
        decision = await policy_method(self, invocation)
        decisions.append(decision)
        return decision

    def lookup(self, name):
        events.append("lookup")
        assert self is registry
        return registry_method(self, name)

    async def guarded_execute(self, invocation):
        events.append("guarded")
        assert invocation is converted[0]
        try:
            return await guarded_method(self, invocation)
        except (ToolConfirmationRequiredError, ToolAuthorizationDeniedError) as error:
            blocked.append(error)
            raise

    monkeypatch.setattr(ModelToolInvocationBridge, "to_invocation", convert)
    monkeypatch.setattr(GuardedToolExecutor, "execute", guarded_execute)
    monkeypatch.setattr(StaticToolAuthorizationPolicy, "authorize", authorize)
    monkeypatch.setattr(ToolRegistry, "get", lookup)
    if error_type is None:
        execution = asyncio.run(executor.execute(call))
        assert execution.tool_call is call
        assert execution.invocation is converted[0]
        assert execution.result is tool.result
        assert tool.invocations == [converted[0]]
        assert tool.invocations[0] is converted[0]
        assert events == ["bridge", "guarded", "authorize", "lookup"]
    else:
        with pytest.raises(error_type) as caught:
            asyncio.run(executor.execute(call))
        assert caught.value is blocked[0]
        assert caught.value.decision is decisions[0]
        assert caught.value.decision.tool_name == tool.name
        assert caught.value.decision.invocation_id is INVOCATION_ID
        assert tool.invocations == []
        assert events == ["bridge", "guarded", "authorize"]
    assert len(converted) == len(decisions) == 1
    factory.assert_called_once_with()


@pytest.mark.parametrize("allowed", [False, True])
def test_unknown_tool_authorization_precedes_lookup(
    allowed: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rules = {"unknown.tool": ToolAuthorizationOutcome.ALLOW} if allowed else {}
    executor, factory, _, _ = make_stack(rules)
    call = LanguageModelToolCall("call_unknown", "unknown.tool", {})
    lookup_error = ToolNotFoundError("unknown tool")
    lookup = Mock(side_effect=lookup_error)
    monkeypatch.setattr(ToolRegistry, "get", lookup)
    expected = ToolNotFoundError if allowed else ToolAuthorizationDeniedError
    with pytest.raises(expected) as caught:
        asyncio.run(executor.execute(call))
    if allowed:
        assert caught.value is lookup_error
        lookup.assert_called_once_with("unknown.tool")
    else:
        lookup.assert_not_called()
        assert caught.value.decision.tool_name == "unknown.tool"
        assert caught.value.decision.invocation_id is INVOCATION_ID
    factory.assert_called_once_with()


@pytest.mark.parametrize(
    "error", [ToolExecutionError("failed"), RuntimeError("failed")]
)
def test_tool_failure_propagates_without_retry(error: Exception) -> None:
    tool = RecordingTool()
    tool.error = error
    executor, factory, _, _ = make_stack(
        {tool.name: ToolAuthorizationOutcome.ALLOW}, tool
    )
    with pytest.raises(type(error)) as caught:
        asyncio.run(executor.execute(make_objects()[0]))
    assert caught.value is error
    assert len(tool.invocations) == 1
    assert tool.result is None
    factory.assert_called_once_with()


def test_invalid_candidate_never_begins_guarded_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, factory, _, _ = make_stack({})
    guarded = AsyncMock()
    monkeypatch.setattr(GuardedToolExecutor, "execute", guarded)
    call = LanguageModelToolCall("call_1", "calculator.basic", {"": 1})
    with pytest.raises(ValueError, match="Tool argument name must not be empty"):
        asyncio.run(executor.execute(call))
    guarded.assert_not_awaited()
    factory.assert_called_once_with()


@pytest.mark.parametrize("invalid", [None, {}, object(), "tool call"])
def test_invalid_source_does_not_touch_dependencies(
    invalid: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, factory, _, _ = make_stack({})
    bridge = Mock()
    guarded = AsyncMock()
    monkeypatch.setattr(ModelToolInvocationBridge, "to_invocation", bridge)
    monkeypatch.setattr(GuardedToolExecutor, "execute", guarded)
    with pytest.raises(TypeError) as caught:
        asyncio.run(executor.execute(invalid))
    assert str(caught.value) == "tool_call must be a LanguageModelToolCall."
    bridge.assert_not_called()
    factory.assert_not_called()
    guarded.assert_not_awaited()


@pytest.mark.parametrize("invalid", [None, {}, object(), "dependency"])
@pytest.mark.parametrize("dependency", ["bridge", "executor"])
def test_constructor_requires_actual_dependencies(
    dependency: str, invalid: object
) -> None:
    executor, _, _, _ = make_stack({})
    values = {"bridge": executor._bridge, "executor": executor._executor}
    values[dependency] = invalid
    expected = (
        "bridge must be a ModelToolInvocationBridge instance."
        if dependency == "bridge"
        else "executor must be a GuardedToolExecutor instance."
    )
    with pytest.raises(TypeError) as caught:
        GuardedModelToolExecutor(**values)
    assert str(caught.value) == expected


@pytest.mark.parametrize(
    "error",
    [
        InvalidToolAuthorizationDecisionError("invalid decision"),
        InvalidToolResultError("invalid result"),
    ],
)
def test_guarded_errors_propagate_without_success_record(
    error: Exception,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, factory, _, _ = make_stack({})
    guarded = AsyncMock(side_effect=error)
    record = Mock()
    monkeypatch.setattr(GuardedToolExecutor, "execute", guarded)
    monkeypatch.setattr(model_execution, "ModelToolExecution", record)
    with pytest.raises(type(error)) as caught:
        asyncio.run(executor.execute(make_objects()[0]))
    assert caught.value is error
    guarded.assert_awaited_once()
    factory.assert_called_once_with()
    record.assert_not_called()


def test_real_guarded_rejects_invalid_decision_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = RecordingTool()
    executor, _, _, _ = make_stack({tool.name: ToolAuthorizationOutcome.ALLOW}, tool)
    decision = ToolAuthorizationDecision(
        "other.tool", INVOCATION_ID, ToolAuthorizationOutcome.ALLOW, "allowed"
    )
    authorize = AsyncMock(return_value=decision)
    monkeypatch.setattr(StaticToolAuthorizationPolicy, "authorize", authorize)
    with pytest.raises(InvalidToolAuthorizationDecisionError):
        asyncio.run(executor.execute(make_objects()[0]))
    authorize.assert_awaited_once()
    assert tool.invocations == []


def test_real_executor_rejects_invalid_result(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = RecordingTool()
    executor, _, _, _ = make_stack({tool.name: ToolAuthorizationOutcome.ALLOW}, tool)
    execute = AsyncMock(return_value=ToolResult(tool.name, UUID(int=2), 42))
    monkeypatch.setattr(RecordingTool, "execute", execute)
    with pytest.raises(InvalidToolResultError):
        asyncio.run(executor.execute(make_objects()[0]))
    execute.assert_awaited_once()


def test_public_exports_and_async_single_call_surface() -> None:
    assert bridges.GuardedModelToolExecutor is GuardedModelToolExecutor
    assert bridges.ModelToolExecution is ModelToolExecution
    assert bridges.ModelToolInvocationBridge is ModelToolInvocationBridge
    assert bridges.__all__ == [
        "GuardedModelToolExecutor",
        "ModelToolExecution",
        "ModelToolInvocationBridge",
    ]
    executor, _, _, _ = make_stack({})
    assert not hasattr(executor, "__dict__")
    assert GuardedModelToolExecutor.__slots__ == ("_bridge", "_executor")
    assert inspect.iscoroutinefunction(GuardedModelToolExecutor.execute)
    assert tuple(inspect.signature(GuardedModelToolExecutor.execute).parameters) == (
        "self",
        "tool_call",
    )


def test_execution_module_has_only_guarded_boundary_dependencies() -> None:
    tree = ast.parse(inspect.getsource(model_execution))
    imports = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imports == {
        "dataclasses",
        "tuesday.bridges.model_tools",
        "tuesday.language_models.tools",
        "tuesday.tools.base",
        "tuesday.tools.guarded",
    }
