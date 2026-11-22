"""Opt-in conversational adaptation of one bounded guarded OpenAI tool turn."""

from tuesday.agents.base import BaseAgent
from tuesday.domain import ConversationContext, TuesdayRequest, TuesdayResponse
from tuesday.language_models import LanguageModelRequest, LanguageModelToolDefinition
from tuesday.prompting import ConversationalPromptRenderer
from tuesday.services.openai_tool_turn import (
    OpenAIToolTurnResult,
    OpenAIToolTurnService,
)

__all__ = ["GuardedOpenAIConversationalAgent", "IncompleteGuardedToolTurnError"]


class IncompleteGuardedToolTurnError(RuntimeError):
    """Raised when a bounded tool turn has unresolved model tool calls."""


class GuardedOpenAIConversationalAgent(BaseAgent):
    """Render an interaction and delegate one explicitly configured tool turn."""

    __slots__ = ("_tool_turn_service", "_tools", "_renderer")

    def __init__(
        self,
        tool_turn_service: OpenAIToolTurnService,
        tools: tuple[LanguageModelToolDefinition, ...],
        *,
        renderer: ConversationalPromptRenderer | None = None,
    ) -> None:
        if not isinstance(tool_turn_service, OpenAIToolTurnService):
            raise TypeError("tool_turn_service must be an OpenAIToolTurnService.")
        if not isinstance(tools, tuple):
            raise TypeError("tools must be a tuple.")
        if not tools:
            raise ValueError("tools must contain at least one model tool definition.")
        if not all(isinstance(tool, LanguageModelToolDefinition) for tool in tools):
            raise TypeError(
                "tools must contain only LanguageModelToolDefinition instances."
            )
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError("Model tool definition names must be unique.")
        if renderer is not None and not isinstance(
            renderer, ConversationalPromptRenderer
        ):
            raise TypeError("renderer must be a ConversationalPromptRenderer or None.")

        self._tool_turn_service = tool_turn_service
        self._tools = tools
        self._renderer = (
            renderer if renderer is not None else ConversationalPromptRenderer()
        )

    @property
    def name(self) -> str:
        """Return the stable conversation registry name."""
        return "conversation"

    @property
    def description(self) -> str:
        """Describe the explicitly configured, bounded conversational adapter."""
        return (
            "Provides explicitly configured OpenAI-backed conversational responses "
            "with bounded guarded tool use."
        )

    async def handle(
        self,
        request: TuesdayRequest,
        context: ConversationContext,
    ) -> TuesdayResponse:
        """Return only completed model text, preserving interaction correlation."""
        if not isinstance(request, TuesdayRequest):
            raise TypeError("request must be a TuesdayRequest.")
        if not isinstance(context, ConversationContext):
            raise TypeError("context must be a ConversationContext.")
        self._validate_context(request, context)

        rendered_request = self._renderer.render(request, context)
        if not isinstance(rendered_request, LanguageModelRequest):
            raise TypeError("renderer must return a LanguageModelRequest.")
        if rendered_request.tools:
            raise ValueError(
                "Conversational renderer must not preconfigure model tools."
            )
        model_request = LanguageModelRequest(
            messages=rendered_request.messages, tools=self._tools
        )
        result = await self._tool_turn_service.run_once(model_request)
        if not isinstance(result, OpenAIToolTurnResult):
            raise TypeError("tool_turn_service must return an OpenAIToolTurnResult.")

        final_response = (
            result.initial.response
            if result.continued is None
            else result.continued.response
        )
        if final_response.tool_calls:
            raise IncompleteGuardedToolTurnError(
                "Guarded OpenAI conversational turn has unresolved model tool calls."
            )
        return TuesdayResponse(
            content=final_response.content,
            conversation_id=request.conversation_id,
            request_id=request.request_id,
        )
