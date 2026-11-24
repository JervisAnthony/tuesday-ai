"""Render conversational TUESDAY context into provider-neutral model input."""

from dataclasses import dataclass, field

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
    max_history_messages: int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.system_prompt, str):
            raise TypeError("Conversational system prompt must be a string.")
        if not self.system_prompt.strip():
            raise ValueError("Conversational system prompt must not be empty.")
        if self.max_history_messages is not None:
            if type(self.max_history_messages) is not int:
                raise TypeError("max_history_messages must be an int or None.")
            if self.max_history_messages < 0:
                raise ValueError("max_history_messages must be nonnegative.")

    def render(
        self,
        request: TuesdayRequest,
        context: ConversationContext,
    ) -> LanguageModelRequest:
        """Validate context, then render selected history and the current request."""
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

        if self.max_history_messages is None:
            history = context.messages
        elif self.max_history_messages == 0:
            history = ()
        else:
            history = context.messages[-self.max_history_messages:]

        messages = (
            LanguageModelMessage(
                role=MessageRole.SYSTEM,
                content=self.system_prompt,
            ),
            *(
                LanguageModelMessage(role=message.role, content=message.content)
                for message in history
            ),
            LanguageModelMessage(
                role=MessageRole.USER,
                content=request.content,
            ),
        )
        return LanguageModelRequest(messages=messages)
