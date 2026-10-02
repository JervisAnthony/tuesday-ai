"""Translate model tool intent into an execution-domain candidate."""

from collections.abc import Callable
from uuid import UUID, uuid4

from tuesday.language_models.tools import LanguageModelToolCall
from tuesday.tools.base import ToolInvocation


class ModelToolInvocationBridge:
    """Create one invocation candidate without authorization or execution."""

    __slots__ = ("_invocation_id_factory",)

    def __init__(
        self,
        *,
        invocation_id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        if not callable(invocation_id_factory):
            raise TypeError("invocation_id_factory must be callable.")
        self._invocation_id_factory = invocation_id_factory

    def to_invocation(self, tool_call: LanguageModelToolCall) -> ToolInvocation:
        """Snapshot one model call with a separate execution UUID."""
        if not isinstance(tool_call, LanguageModelToolCall):
            raise TypeError("tool_call must be a LanguageModelToolCall.")
        invocation_id = self._invocation_id_factory()
        if not isinstance(invocation_id, UUID):
            raise TypeError("invocation_id_factory must return a UUID.")
        return ToolInvocation(
            tool_name=tool_call.name,
            arguments=tool_call.arguments,
            invocation_id=invocation_id,
        )
