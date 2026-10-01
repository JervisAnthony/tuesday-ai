"""Render conversational TUESDAY context into provider-neutral model input."""

from dataclasses import dataclass

from tuesday.domain import (
    ConversationContext,
    ConversationMessage,
    MessageRole,
    TuesdayRequest,
)
from tuesday.language_models import LanguageModelMessage, LanguageModelRequest

__all__ = [
    "DEFAULT_CONVERSATIONAL_SYSTEM_PROMPT",
    "ConversationalPromptRenderer",
]

DEFAULT_CONVERSATIONAL_SYSTEM_PROMPT = (
    "You are TUESDAY — Task-Unifying Engine for Smart Decisions, Actions & You. "
    "Respond helpfully and clearly using the conversation context provided."
)


@dataclass(frozen=True, slots=True)
class ConversationalPromptRenderer:
    """Build immutable model input from a request and its prior conversation context."""

    system_prompt: str = DEFAULT_CONVERSATIONAL_SYSTEM_PROMPT

    def __post_init__(self) -> None:
        if not isinstance(self.system_prompt, str):
            raise TypeError("Conversational system prompt must be a string.")
        if not self.system_prompt.strip():
            raise ValueError("Conversational system prompt must not be empty.")

    def render(
        self,
        request: TuesdayRequest,
        context: ConversationContext,
    ) -> LanguageModelRequest:
        """Render prior context followed by the current user request."""
        if not isinstance(request, TuesdayRequest):
            raise TypeError("request must be a TuesdayRequest.")
        if not isinstance(context, ConversationContext):
            raise TypeError("context must be a ConversationContext.")
        if request.conversation_id != context.conversation_id:
            raise ValueError(
                "Request and context must belong to the same conversation."
            )
        if not all(
            isinstance(message, ConversationMessage) for message in context.messages
        ):
            raise TypeError(
                "Conversation context messages must be ConversationMessage instances."
            )

        messages = (
            LanguageModelMessage(
                role=MessageRole.SYSTEM,
                content=self.system_prompt,
            ),
            *(
                LanguageModelMessage(role=message.role, content=message.content)
                for message in context.messages
            ),
            LanguageModelMessage(
                role=MessageRole.USER,
                content=request.content,
            ),
        )
        return LanguageModelRequest(messages=messages)
