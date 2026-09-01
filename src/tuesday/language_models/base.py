"""Framework-independent contracts for language-model generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from tuesday.domain import MessageRole
from tuesday.language_models.tools import LanguageModelToolDefinition

__all__ = [
    "BaseLanguageModelProvider",
    "LanguageModelMessage",
    "LanguageModelProviderError",
    "LanguageModelRequest",
    "LanguageModelResponse",
]


@dataclass(frozen=True, slots=True)
class LanguageModelMessage:
    """An immutable message supplied to a language-model provider."""

    role: MessageRole
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, MessageRole):
            raise TypeError("Language model message role must be a MessageRole.")
        if not isinstance(self.content, str):
            raise TypeError("Language model message content must be a string.")
        if not self.content.strip():
            raise ValueError("Language model message content must not be empty.")


@dataclass(frozen=True, slots=True)
class LanguageModelRequest:
    """Immutable provider-neutral input for one text generation."""

    messages: tuple[LanguageModelMessage, ...]
    tools: tuple[LanguageModelToolDefinition, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.messages, tuple):
            raise TypeError("Language model request messages must be a tuple.")
        if not self.messages:
            raise ValueError("Language model request must contain a message.")
        if not all(
            isinstance(message, LanguageModelMessage) for message in self.messages
        ):
            raise TypeError(
                "Language model request messages must be LanguageModelMessage "
                "instances."
            )
        if not isinstance(self.tools, tuple):
            raise TypeError("Language model request tools must be a tuple.")
        if not all(
            isinstance(tool, LanguageModelToolDefinition) for tool in self.tools
        ):
            raise TypeError(
                "Language model request tools must be "
                "LanguageModelToolDefinition instances."
            )
        tool_names = [tool.name for tool in self.tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("Language model request tool names must be unique.")


@dataclass(frozen=True, slots=True)
class LanguageModelResponse:
    """Immutable text returned by a language-model provider."""

    content: str
    provider: str
    model: str

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("Language model response content must be a string.")
        if not self.content.strip():
            raise ValueError("Language model response content must not be empty.")

        if not isinstance(self.provider, str):
            raise TypeError("Language model response provider must be a string.")
        if not self.provider.strip():
            raise ValueError("Language model response provider must not be empty.")
        if self.provider != self.provider.strip():
            raise ValueError(
                "Language model response provider must not have surrounding "
                "whitespace."
            )

        if not isinstance(self.model, str):
            raise TypeError("Language model response model must be a string.")
        if not self.model.strip():
            raise ValueError("Language model response model must not be empty.")
        if self.model != self.model.strip():
            raise ValueError(
                "Language model response model must not have surrounding "
                "whitespace."
            )


class LanguageModelProviderError(RuntimeError):
    """Raised when a language-model provider fails during execution."""


class BaseLanguageModelProvider(ABC):
    """Abstract asynchronous language-model generation boundary."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider adapter's stable machine-friendly name."""

    @abstractmethod
    async def generate(
        self,
        request: LanguageModelRequest,
    ) -> LanguageModelResponse:
        """Generate one complete response for the supplied request."""
