"""Deterministic execution of explicitly selected TUESDAY tools."""

from tuesday.tools.base import ToolInvocation, ToolResult
from tuesday.tools.registry import ToolRegistry

__all__ = ["DeterministicToolExecutor", "InvalidToolResultError"]


class InvalidToolResultError(ValueError):
    """Raised when a tool violates the executor result contract."""


class DeterministicToolExecutor:
    """Resolve and execute at most one explicitly requested tool."""

    __slots__ = ("_registry",)

    def __init__(self, registry: ToolRegistry) -> None:
        if not isinstance(registry, ToolRegistry):
            raise TypeError("registry must be a ToolRegistry instance.")
        self._registry = registry

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        """Execute one invocation and validate its result correlation."""
        if not isinstance(invocation, ToolInvocation):
            raise TypeError("invocation must be a ToolInvocation.")

        tool = self._registry.get(invocation.tool_name)
        result = await tool.execute(invocation)

        if not isinstance(result, ToolResult):
            raise InvalidToolResultError(
                "Tool execution must return a ToolResult."
            )
        if result.tool_name != invocation.tool_name:
            raise InvalidToolResultError(
                "Tool result name does not match the invocation."
            )
        if result.invocation_id != invocation.invocation_id:
            raise InvalidToolResultError(
                "Tool result invocation_id does not match the invocation."
            )
        return result
