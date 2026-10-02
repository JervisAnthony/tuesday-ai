"""Explicit model-requested execution through the guarded tool boundary."""

from dataclasses import dataclass

from tuesday.bridges.model_tools import ModelToolInvocationBridge
from tuesday.language_models.tools import LanguageModelToolCall
from tuesday.tools.base import ToolInvocation, ToolResult
from tuesday.tools.guarded import GuardedToolExecutor


@dataclass(frozen=True, slots=True)
class ModelToolExecution:
    """Correlated domain objects from one successful guarded execution."""

    tool_call: LanguageModelToolCall
    invocation: ToolInvocation
    result: ToolResult

    def __post_init__(self) -> None:
        if not isinstance(self.tool_call, LanguageModelToolCall):
            raise TypeError("tool_call must be a LanguageModelToolCall.")
        if not isinstance(self.invocation, ToolInvocation):
            raise TypeError("invocation must be a ToolInvocation.")
        if not isinstance(self.result, ToolResult):
            raise TypeError("result must be a ToolResult.")
        if self.invocation.tool_name != self.tool_call.name:
            raise ValueError(
                "Model tool execution invocation name does not match the tool call."
            )
        if self.invocation.arguments != self.tool_call.arguments:
            raise ValueError(
                "Model tool execution invocation arguments do not match the tool call."
            )
        if self.result.tool_name != self.invocation.tool_name:
            raise ValueError(
                "Model tool execution result name does not match the invocation."
            )
        if self.result.invocation_id != self.invocation.invocation_id:
            raise ValueError(
                "Model tool execution result invocation_id "
                "does not match the invocation."
            )


class GuardedModelToolExecutor:
    """Bridge one model call and delegate exclusively to guarded execution."""

    __slots__ = ("_bridge", "_executor")

    def __init__(
        self,
        bridge: ModelToolInvocationBridge,
        executor: GuardedToolExecutor,
    ) -> None:
        if not isinstance(bridge, ModelToolInvocationBridge):
            raise TypeError("bridge must be a ModelToolInvocationBridge instance.")
        if not isinstance(executor, GuardedToolExecutor):
            raise TypeError("executor must be a GuardedToolExecutor instance.")
        self._bridge = bridge
        self._executor = executor

    async def execute(self, tool_call: LanguageModelToolCall) -> ModelToolExecution:
        """Return a correlated record only after successful guarded execution."""
        if not isinstance(tool_call, LanguageModelToolCall):
            raise TypeError("tool_call must be a LanguageModelToolCall.")
        invocation = self._bridge.to_invocation(tool_call)
        result = await self._executor.execute(invocation)
        return ModelToolExecution(
            tool_call=tool_call,
            invocation=invocation,
            result=result,
        )
