"""Translate successful execution records into model-correlated results."""

from tuesday.bridges.model_execution import ModelToolExecution
from tuesday.language_models.tools import LanguageModelToolResult


class ModelToolResultBridge:
    """Snapshot one successful result without execution or continuation."""

    __slots__ = ()

    def to_model_result(self, execution: ModelToolExecution) -> LanguageModelToolResult:
        """Preserve the original model call ID and successful tool output."""
        if not isinstance(execution, ModelToolExecution):
            raise TypeError("execution must be a ModelToolExecution.")
        return LanguageModelToolResult(
            call_id=execution.tool_call.call_id,
            output=execution.result.output,
        )
