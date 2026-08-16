"""Framework-independent contracts for preparing TUESDAY requests."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from tuesday.domain import TuesdayRequest

__all__ = ["BaseRequestPreparer", "PreparedRequest"]


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    """An immutable agent-facing view of an original request."""

    source_request: TuesdayRequest
    content: str
    route_directive: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_request, TuesdayRequest):
            raise TypeError("source_request must be a TuesdayRequest.")
        if not isinstance(self.content, str):
            raise TypeError("Prepared request content must be a string.")
        if not self.content.strip():
            raise ValueError("Prepared request content must not be empty.")

        if self.route_directive is None:
            return
        if not isinstance(self.route_directive, str):
            raise TypeError("Route directive must be a string or None.")
        if not self.route_directive.strip():
            raise ValueError("Route directive must not be empty.")
        if self.route_directive.startswith("/"):
            raise ValueError("Route directive must not begin with '/'.")
        if any(character.isspace() for character in self.route_directive):
            raise ValueError("Route directive must not contain whitespace.")

    def to_request(self) -> TuesdayRequest:
        """Materialize the prepared view with the source correlation IDs."""
        return TuesdayRequest(
            content=self.content,
            conversation_id=self.source_request.conversation_id,
            request_id=self.source_request.request_id,
        )


class BaseRequestPreparer(ABC):
    """Abstract contract for deterministic local request preparation."""

    @abstractmethod
    def prepare(self, request: TuesdayRequest) -> PreparedRequest:
        """Return a prepared view while preserving the original request."""
