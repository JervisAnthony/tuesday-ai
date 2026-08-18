"""Provider-neutral model-backed conversational agent."""

from tuesday.agents.base import BaseAgent
from tuesday.domain import ConversationContext, TuesdayRequest, TuesdayResponse
from tuesday.language_models import BaseLanguageModelProvider, LanguageModelResponse
from tuesday.prompting import ConversationalPromptRenderer

__all__ = ["ModelBackedConversationalAgent"]


class ModelBackedConversationalAgent(BaseAgent):
    """Generate conversational responses through a language-model provider."""

    __slots__ = ("_provider", "_renderer")

    def __init__(
        self,
        provider: BaseLanguageModelProvider,
        *,
        renderer: ConversationalPromptRenderer | None = None,
    ) -> None:
        if not isinstance(provider, BaseLanguageModelProvider):
            raise TypeError("provider must be a BaseLanguageModelProvider.")
        if renderer is not None and not isinstance(
            renderer, ConversationalPromptRenderer
        ):
            raise TypeError(
                "renderer must be a ConversationalPromptRenderer or None."
            )

        self._provider = provider
        self._renderer = (
            renderer if renderer is not None else ConversationalPromptRenderer()
        )

    @property
    def name(self) -> str:
        """Return the stable conversational-agent registry name."""
        return "conversation"

    @property
    def description(self) -> str:
        """Return the agent's model-backed conversational purpose."""
        return "Generates conversational responses through a language model provider."

    async def handle(
        self,
        request: TuesdayRequest,
        context: ConversationContext,
    ) -> TuesdayResponse:
        """Render context, generate model output, and preserve TUESDAY correlation."""
        self._validate_context(request, context)

        model_request = self._renderer.render(request, context)
        model_response = await self._provider.generate(model_request)
        if not isinstance(model_response, LanguageModelResponse):
            raise TypeError(
                "Language model provider must return a LanguageModelResponse."
            )

        return TuesdayResponse(
            content=model_response.content,
            conversation_id=request.conversation_id,
            request_id=request.request_id,
        )
