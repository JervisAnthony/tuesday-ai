"""OpenAI implementation of TUESDAY's language-model provider contract."""

import json
from collections.abc import Mapping

from openai import AsyncOpenAI, OpenAIError

from tuesday.config import ConfigurationError, LanguageModelSettings
from tuesday.language_models.base import (
    BaseLanguageModelProvider,
    LanguageModelProviderError,
    LanguageModelRequest,
    LanguageModelResponse,
)
from tuesday.language_models.openai_continuation import (
    OpenAIContinuationState,
    OpenAIGenerationResult,
)
from tuesday.language_models.tools import (
    LanguageModelToolCall,
    LanguageModelToolResult,
    LanguageModelToolValue,
)

__all__ = ["OpenAILanguageModelProvider"]


def _thaw_tool_value(value: LanguageModelToolValue) -> object:
    """Copy frozen model-tool metadata into plain JSON-compatible containers."""
    if isinstance(value, Mapping):
        return {key: _thaw_tool_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_tool_value(item) for item in value]
    return value


def _parse_tool_call(
    call_id: str,
    name: str,
    arguments: object,
) -> LanguageModelToolCall:
    """Decode one provider function call without interpreting execution intent."""
    if not isinstance(arguments, str):
        raise LanguageModelProviderError(
            "OpenAI provider returned invalid tool call arguments."
        )
    try:
        decoded = json.loads(arguments)
    except json.JSONDecodeError as error:
        raise LanguageModelProviderError(
            "OpenAI provider returned invalid tool call arguments."
        ) from error
    if not isinstance(decoded, Mapping):
        raise LanguageModelProviderError(
            "OpenAI provider returned invalid tool call arguments."
        )
    try:
        return LanguageModelToolCall(call_id=call_id, name=name, arguments=decoded)
    except (TypeError, ValueError) as error:
        raise LanguageModelProviderError(
            "OpenAI provider returned an invalid tool call."
        ) from error


class OpenAILanguageModelProvider(BaseLanguageModelProvider):
    """Generate provider-neutral text and tool-call responses through OpenAI."""

    def __init__(
        self,
        settings: LanguageModelSettings,
        *,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if settings.provider != "openai":
            raise ValueError(
                "OpenAI provider requires model settings with provider 'openai'."
            )
        if settings.api_key is None:
            raise ConfigurationError(
                "OpenAI provider requires TUESDAY_MODEL_API_KEY."
            )
        if settings.temperature is not None and settings.temperature > 2:
            raise ConfigurationError(
                "OpenAI temperature must be between 0 and 2."
            )

        self._settings = settings
        self._client = (
            AsyncOpenAI(
                api_key=settings.api_key,
                timeout=settings.timeout_seconds,
                max_retries=0,
            )
            if client is None
            else client
        )

    @property
    def name(self) -> str:
        """Return the adapter's stable provider identity."""
        return "openai"

    async def generate(
        self,
        request: LanguageModelRequest,
    ) -> LanguageModelResponse:
        """Generate one complete response through the OpenAI Responses API."""
        arguments = self._request_arguments(
            request, self._settings.model, self._settings.temperature
        )
        raw_response = await self._create_response(arguments)
        return self._translate_response(raw_response)

    @staticmethod
    def _request_arguments(
        request: LanguageModelRequest,
        model: str,
        temperature: float | None,
    ) -> dict[str, object]:
        input_messages = [
            {
                "role": message.role.value,
                "content": message.content,
            }
            for message in request.messages
        ]
        request_arguments: dict[str, object] = {
            "model": model,
            "input": input_messages,
            "store": False,
            "stream": False,
        }
        if temperature is not None:
            request_arguments["temperature"] = temperature
        if request.tools:
            request_arguments["tools"] = [
                {
                    "type": "function",
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": _thaw_tool_value(definition.parameters),
                }
                for definition in request.tools
            ]

        return request_arguments

    async def _create_response(self, request_arguments: dict[str, object]):
        try:
            return await self._client.responses.create(**request_arguments)
        except OpenAIError as error:
            raise LanguageModelProviderError(
                "OpenAI provider request failed."
            ) from error

    def _translate_response(self, response) -> LanguageModelResponse:
        tool_calls = tuple(
            _parse_tool_call(item.call_id, item.name, item.arguments)
            for item in response.output
            if item.type == "function_call"
        )
        output_text = response.output_text
        if output_text is not None and not isinstance(output_text, str):
            raise LanguageModelProviderError(
                "OpenAI provider returned no text content."
            )
        content = output_text if output_text and output_text.strip() else None
        if content is None and not tool_calls:
            raise LanguageModelProviderError(
                "OpenAI provider returned no text content."
            )

        return LanguageModelResponse(
            content=content,
            provider=self.name,
            model=response.model,
            tool_calls=tool_calls,
        )

    @staticmethod
    def _snapshot_output(output) -> tuple[str, ...]:
        snapshots = []
        for item in output:
            try:
                dump = getattr(item, "model_dump", None)
                if not callable(dump):
                    raise TypeError("Output item has no public serializer.")
                mapping = dump(mode="json", exclude_none=True)
                if not isinstance(mapping, Mapping):
                    raise TypeError("Output serialization must be a mapping.")
                snapshots.append(
                    json.dumps(
                        dict(mapping),
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                )
            except Exception as error:
                raise LanguageModelProviderError(
                    "OpenAI provider returned output that cannot be replayed."
                ) from error
        return tuple(snapshots)

    def _generation_result(
        self,
        raw_response,
        request: LanguageModelRequest,
        model: str,
        temperature: float | None,
        prior_items: tuple[str, ...] = (),
    ) -> OpenAIGenerationResult:
        response = self._translate_response(raw_response)
        continuation = None
        if response.tool_calls:
            continuation = OpenAIContinuationState(
                request=request,
                model=model,
                temperature=temperature,
                replay_items_json=prior_items
                + self._snapshot_output(raw_response.output),
                pending_call_ids=tuple(call.call_id for call in response.tool_calls),
            )
        return OpenAIGenerationResult(response, continuation)

    async def generate_with_continuation(
        self,
        request: LanguageModelRequest,
    ) -> OpenAIGenerationResult:
        """Generate once and opt into serialized state for pending tool calls."""
        if not isinstance(request, LanguageModelRequest):
            raise TypeError("request must be a LanguageModelRequest.")
        model, temperature = self._settings.model, self._settings.temperature
        raw_response = await self._create_response(
            self._request_arguments(request, model, temperature)
        )
        return self._generation_result(raw_response, request, model, temperature)

    async def continue_with_tool_results(
        self,
        state: OpenAIContinuationState,
        results: tuple[LanguageModelToolResult, ...],
    ) -> OpenAIGenerationResult:
        """Replay context and submit every pending result in one explicit request."""
        if not isinstance(state, OpenAIContinuationState):
            raise TypeError("state must be an OpenAIContinuationState.")
        if not isinstance(results, tuple):
            raise TypeError("OpenAI continuation tool results must be a tuple.")
        if not results:
            raise ValueError("OpenAI continuation requires at least one tool result.")
        if not all(isinstance(result, LanguageModelToolResult) for result in results):
            raise TypeError(
                "OpenAI continuation tool results must be "
                "LanguageModelToolResult instances."
            )
        if tuple(result.call_id for result in results) != state.pending_call_ids:
            raise ValueError(
                "OpenAI continuation tool result call IDs "
                "must exactly match pending calls."
            )
        outputs = [
            {
                "type": "function_call_output",
                "call_id": result.call_id,
                "output": json.dumps(
                    _thaw_tool_value(result.output),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ),
            }
            for result in results
        ]
        arguments = self._request_arguments(
            state.request, state.model, state.temperature
        )
        arguments["input"].extend(json.loads(item) for item in state.replay_items_json)
        arguments["input"].extend(outputs)
        raw_response = await self._create_response(arguments)
        submitted = tuple(
            json.dumps(item, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            for item in outputs
        )
        return self._generation_result(
            raw_response,
            state.request,
            state.model,
            state.temperature,
            state.replay_items_json + submitted,
        )
