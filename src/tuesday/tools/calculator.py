"""A deterministic two-operand basic arithmetic tool."""

from math import isfinite

from tuesday.tools.base import (
    BaseTool,
    ToolExecutionError,
    ToolInvocation,
    ToolResult,
)

__all__ = ["BasicCalculatorTool"]


class BasicCalculatorTool(BaseTool):
    """Perform one explicitly requested basic arithmetic operation."""

    __slots__ = ()

    @property
    def name(self) -> str:
        return "calculator.basic"

    @property
    def description(self) -> str:
        return "Perform one basic arithmetic operation on two finite numbers."

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        """Validate and execute one two-operand arithmetic invocation."""
        self._validate_invocation(invocation)

        if set(invocation.arguments) != {"operation", "left", "right"}:
            raise ToolExecutionError(
                "Calculator invocation arguments must be exactly: "
                "operation, left, right."
            )

        operation = invocation.arguments["operation"]
        left = invocation.arguments["left"]
        right = invocation.arguments["right"]

        if not isinstance(operation, str):
            raise ToolExecutionError("Calculator operation must be a string.")
        if operation not in {"add", "subtract", "multiply", "divide"}:
            raise ToolExecutionError(
                "Calculator operation must be one of: "
                "add, subtract, multiply, divide."
            )
        if type(left) not in (int, float):
            raise ToolExecutionError(
                "Calculator left operand must be an int or float."
            )
        if type(right) not in (int, float):
            raise ToolExecutionError(
                "Calculator right operand must be an int or float."
            )
        if operation == "divide" and right == 0:
            raise ToolExecutionError("Calculator cannot divide by zero.")

        try:
            if operation == "add":
                result = left + right
            elif operation == "subtract":
                result = left - right
            elif operation == "multiply":
                result = left * right
            else:
                result = left / right
        except OverflowError as error:
            raise ToolExecutionError(
                "Calculator result is outside the supported finite numeric range."
            ) from error

        if isinstance(result, float) and not isfinite(result):
            raise ToolExecutionError(
                "Calculator result is outside the supported finite numeric range."
            )

        return ToolResult(
            tool_name=self.name,
            invocation_id=invocation.invocation_id,
            output=result,
        )
