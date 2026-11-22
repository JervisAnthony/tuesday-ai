"""One explicit OpenAI generation, guarded tool execution, and continuation."""

from dataclasses import dataclass

from tuesday.bridges import (
    GuardedModelToolExecutor,
    ModelToolExecution,
    ModelToolResultBridge,
)
from tuesday.language_models import (
    LanguageModelRequest,
    LanguageModelToolCall,
    LanguageModelToolResult,
    OpenAIGenerationResult,
    OpenAILanguageModelProvider,
)


class MultipleModelToolCallsUnsupportedError(RuntimeError):
    """Raised when a bounded tool turn receives multiple initial calls."""


class UnadvertisedModelToolCallError(RuntimeError):
    """Raised when a model requests a tool absent from its request scope."""


def _validate_execution(
    execution: ModelToolExecution,
    tool_call: LanguageModelToolCall,
) -> None:
    if not isinstance(execution, ModelToolExecution):
        raise TypeError("execution must be a ModelToolExecution.")
    if execution.tool_call is not tool_call:
        raise ValueError("execution must preserve the original model tool call.")


def _validate_model_result(
    model_result: LanguageModelToolResult,
    execution: ModelToolExecution,
) -> None:
    if not isinstance(model_result, LanguageModelToolResult):
        raise TypeError("model_result must be a LanguageModelToolResult.")
    if model_result.call_id != execution.tool_call.call_id:
        raise ValueError("model_result call_id must match the model tool call.")
    if model_result.output != execution.result.output:
        raise ValueError("model_result output must match the execution result.")


@dataclass(frozen=True, slots=True)
class OpenAIToolTurnResult:
    """Inspectable stages of one successful text-only or single-tool turn."""

    initial: OpenAIGenerationResult
    execution: ModelToolExecution | None
    model_result: LanguageModelToolResult | None
    continued: OpenAIGenerationResult | None

    def __post_init__(self) -> None:
        if not isinstance(self.initial, OpenAIGenerationResult):
            raise TypeError("initial must be an OpenAIGenerationResult.")
        if self.execution is not None and not isinstance(
            self.execution, ModelToolExecution
        ):
            raise TypeError("execution must be a ModelToolExecution or None.")
        if self.model_result is not None and not isinstance(
            self.model_result, LanguageModelToolResult
        ):
            raise TypeError("model_result must be a LanguageModelToolResult or None.")
        if self.continued is not None and not isinstance(
            self.continued, OpenAIGenerationResult
        ):
            raise TypeError("continued must be an OpenAIGenerationResult or None.")

        tool_calls = self.initial.response.tool_calls
        if len(tool_calls) > 1:
            raise ValueError("OpenAI tool turn supports at most one model tool call.")
        if not tool_calls:
            if any(
                stage is not None
                for stage in (self.execution, self.model_result, self.continued)
            ):
                raise ValueError("Text-only tool turn must not have tool stages.")
            return

        _validate_execution(self.execution, tool_calls[0])
        _validate_model_result(self.model_result, self.execution)
        if self.continued is None:
            raise ValueError("Single-tool turn requires a continued generation.")


class OpenAIToolTurnService:
    """Compose injected boundaries once without retries or automatic iteration."""

    __slots__ = ("_provider", "_executor", "_result_bridge")

    def __init__(
        self,
        provider: OpenAILanguageModelProvider,
        executor: GuardedModelToolExecutor,
        result_bridge: ModelToolResultBridge,
    ) -> None:
        if not isinstance(provider, OpenAILanguageModelProvider):
            raise TypeError("provider must be an OpenAILanguageModelProvider.")
        if not isinstance(executor, GuardedModelToolExecutor):
            raise TypeError("executor must be a GuardedModelToolExecutor.")
        if not isinstance(result_bridge, ModelToolResultBridge):
            raise TypeError("result_bridge must be a ModelToolResultBridge.")
        self._provider = provider
        self._executor = executor
        self._result_bridge = result_bridge

    async def run_once(self, request: LanguageModelRequest) -> OpenAIToolTurnResult:
        """Generate once and, for one tool call, execute and continue once."""
        if not isinstance(request, LanguageModelRequest):
            raise TypeError("request must be a LanguageModelRequest.")
        initial = await self._provider.generate_with_continuation(request)
        if not isinstance(initial, OpenAIGenerationResult):
            raise TypeError("initial must be an OpenAIGenerationResult.")

        tool_calls = initial.response.tool_calls
        if len(tool_calls) > 1:
            raise MultipleModelToolCallsUnsupportedError(
                "OpenAI tool turn supports at most one model tool call."
            )
        if not tool_calls:
            return OpenAIToolTurnResult(initial, None, None, None)

        tool_call = tool_calls[0]
        if initial.continuation.request is not request:
            raise ValueError(
                "OpenAI continuation state request must match the original tool turn "
                "request."
            )
        if not any(definition.name == tool_call.name for definition in request.tools):
            raise UnadvertisedModelToolCallError(
                "Model requested a tool that was not advertised for this request."
            )
        execution = await self._executor.execute(tool_call)
        _validate_execution(execution, tool_call)
        model_result = self._result_bridge.to_model_result(execution)
        _validate_model_result(model_result, execution)
        continued = await self._provider.continue_with_tool_results(
            initial.continuation, (model_result,)
        )
        if not isinstance(continued, OpenAIGenerationResult):
            raise TypeError("continued must be an OpenAIGenerationResult.")
        return OpenAIToolTurnResult(initial, execution, model_result, continued)
