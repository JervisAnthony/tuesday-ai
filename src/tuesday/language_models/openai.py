"""OpenAI implementation of TUESDAY's language-model provider contract."""

from openai import AsyncOpenAI, OpenAIError

from tuesday.config import ConfigurationError, LanguageModelSettings
from tuesday.language_models.base import (
    BaseLanguageModelProvider,
    LanguageModelProviderError,
    LanguageModelRequest,
    LanguageModelResponse,
)

__all__ = ["OpenAILanguageModelProvider"]


class OpenAILanguageModelProvider(BaseLanguageModelProvider):
    """Generate provider-neutral text responses through OpenAI."""

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
        if request.tools:
            raise LanguageModelProviderError(
                "OpenAI provider does not yet support language model tool "
                "definitions."
            )

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

        try:
            response = await self._client.responses.create(**request_arguments)
        except OpenAIError as error:
            raise LanguageModelProviderError(
                "OpenAI provider request failed."
            ) from error

        output_text = response.output_text
        if not isinstance(output_text, str) or not output_text.strip():
            raise LanguageModelProviderError(
                "OpenAI provider returned no text content."
            )

        return LanguageModelResponse(
            content=output_text,
            provider=self.name,
            model=response.model,
        )
