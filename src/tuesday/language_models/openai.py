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
from tuesday.language_models.tools import LanguageModelToolCall, LanguageModelToolValue

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
        input_messages = [
            {
                "role": message.role.value,
                "content": message.content,
            }
            for message in request.messages
        ]
        request_arguments: dict[str, object] = {
            "model": self._settings.model,
            "input": input_messages,
            "store": False,
            "stream": False,
        }
        if self._settings.temperature is not None:
            request_arguments["temperature"] = self._settings.temperature
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

        try:
            response = await self._client.responses.create(**request_arguments)
        except OpenAIError as error:
            raise LanguageModelProviderError(
                "OpenAI provider request failed."
            ) from error

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
