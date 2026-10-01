"""Directive-aware preparation of agent-facing TUESDAY requests."""

from tuesday.domain import TuesdayRequest
from tuesday.preparation.base import BaseRequestPreparer, PreparedRequest

__all__ = ["DirectiveRequestPreparer", "RequestPreparationError"]


class RequestPreparationError(ValueError):
    """Raised when a request cannot produce meaningful prepared content."""


class DirectiveRequestPreparer(BaseRequestPreparer):
    """Remove one leading routing directive from agent-facing content."""

    def prepare(self, request: TuesdayRequest) -> PreparedRequest:
        """Parse a leading directive without modifying the source request."""
        content = request.content
        if not content.startswith("/"):
            raise RequestPreparationError(
                "Request must begin with a routing directive."
            )

        separator_index = next(
            (
                index
                for index, character in enumerate(content[1:], start=1)
                if character.isspace()
            ),
            None,
        )
        if separator_index is None:
            if content == "/":
                raise RequestPreparationError(
                    "Routing directive must not be empty."
                )
            raise RequestPreparationError(
                "Request must include content after the routing directive."
            )

        route_directive = content[1:separator_index]
        if not route_directive:
            raise RequestPreparationError("Routing directive must not be empty.")

        content_index = separator_index
        while content_index < len(content) and content[content_index].isspace():
            content_index += 1
        prepared_content = content[content_index:]
        if not prepared_content.strip():
            raise RequestPreparationError(
                "Request must include content after the routing directive."
            )

        return PreparedRequest(
            source_request=request,
            content=prepared_content,
            route_directive=route_directive,
        )
