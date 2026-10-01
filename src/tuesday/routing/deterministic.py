"""Explicit directive-based routing for TUESDAY agents."""

from collections.abc import Mapping

from tuesday.agents import AgentRegistry
from tuesday.domain import ConversationContext, TuesdayRequest
from tuesday.routing.base import BaseRouter, RoutingDecision

__all__ = [
    "DeterministicRouter",
    "NoRouteFoundError",
    "RoutedAgentUnavailableError",
    "RoutingConfigurationError",
]


class RoutingConfigurationError(ValueError):
    """Raised when deterministic route configuration is invalid."""


class NoRouteFoundError(LookupError):
    """Raised when a request has no configured explicit route."""


class RoutedAgentUnavailableError(LookupError):
    """Raised when a route targets an agent absent from the registry."""


class DeterministicRouter(BaseRouter):
    """Select registered agents using configured leading directives."""

    def __init__(self, routes: Mapping[str, str]) -> None:
        copied_routes = dict(routes)
        for route_key, agent_name in copied_routes.items():
            self._validate_route(route_key, agent_name)
        self._routes = copied_routes

    @property
    def routes(self) -> tuple[tuple[str, str], ...]:
        """Return an immutable snapshot of configured routes."""
        return tuple(self._routes.items())

    async def route(
        self,
        request: TuesdayRequest,
        context: ConversationContext,
        registry: AgentRegistry,
    ) -> RoutingDecision:
        """Select an available agent from the request's explicit directive."""
        self._validate_context(request, context)
        route_key = self._extract_route_key(request.content)

        try:
            agent_name = self._routes[route_key]
        except KeyError as error:
            raise NoRouteFoundError(
                f"No route is configured for directive '/{route_key}'."
            ) from error

        if agent_name not in registry:
            raise RoutedAgentUnavailableError(
                f"Route '/{route_key}' targets agent {agent_name!r}, "
                "but that agent is not registered."
            )

        return RoutingDecision(
            agent_name=agent_name,
            reason=(
                f"Explicit route '/{route_key}' selected agent {agent_name!r}."
            ),
        )

    @staticmethod
    def _validate_route(route_key: object, agent_name: object) -> None:
        if not isinstance(route_key, str) or not route_key.strip():
            raise RoutingConfigurationError(
                "Routing key must be a non-empty string."
            )
        if route_key.startswith("/"):
            raise RoutingConfigurationError(
                f"Routing key {route_key!r} must not include a leading '/'."
            )
        if any(character.isspace() for character in route_key):
            raise RoutingConfigurationError(
                f"Routing key {route_key!r} must not contain whitespace."
            )
        if not isinstance(agent_name, str) or not agent_name.strip():
            raise RoutingConfigurationError(
                f"Agent name for route {route_key!r} must be non-empty text."
            )

    @staticmethod
    def _extract_route_key(content: str) -> str:
        if not content.startswith("/"):
            raise NoRouteFoundError("No explicit route directive was found.")

        directive = content.split(maxsplit=1)[0]
        route_key = directive[1:]
        if not route_key:
            raise NoRouteFoundError("No explicit route directive was found.")
        return route_key
