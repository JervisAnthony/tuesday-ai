"""Tests for explicit directive-based deterministic routing."""

import asyncio
from collections.abc import Iterator, Mapping
from uuid import uuid4

import pytest

from tuesday.agents import AgentRegistry, BaseAgent
from tuesday.domain import ConversationContext, TuesdayRequest, TuesdayResponse
from tuesday.routing import (
    BaseRouter,
    DeterministicRouter,
    NoRouteFoundError,
    RoutedAgentUnavailableError,
    RoutingConfigurationError,
    RoutingDecision,
)


class TrackingAgent(BaseAgent):
    """Configurable test agent that records execution attempts."""

    def __init__(self, name: str) -> None:
        self._name = name
        self.was_executed = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "A deterministic-router test agent."

    async def handle(
        self,
        request: TuesdayRequest,
        context: ConversationContext,
    ) -> TuesdayResponse:
        self.was_executed = True
        return TuesdayResponse(
            content="executed",
            conversation_id=request.conversation_id,
            request_id=request.request_id,
        )


def run_route(
    router: DeterministicRouter,
    content: str,
    registry: AgentRegistry,
) -> tuple[TuesdayRequest, ConversationContext, RoutingDecision]:
    conversation_id = uuid4()
    request = TuesdayRequest(content=content, conversation_id=conversation_id)
    context = ConversationContext(conversation_id=conversation_id)
    decision = asyncio.run(router.route(request, context, registry))
    return request, context, decision


def registry_with(*agents: TrackingAgent) -> AgentRegistry:
    registry = AgentRegistry()
    for agent in agents:
        registry.register(agent)
    return registry


def test_deterministic_router_inherits_base_router() -> None:
    assert isinstance(DeterministicRouter({"plan": "planner"}), BaseRouter)


def test_valid_route_configuration_is_accepted() -> None:
    router = DeterministicRouter({"plan": "planner"})

    assert router.routes == (("plan", "planner"),)


def test_router_copies_caller_owned_mapping() -> None:
    routes = {"plan": "planner"}
    router = DeterministicRouter(routes)

    routes["research"] = "researcher"

    assert router.routes == (("plan", "planner"),)


@pytest.mark.parametrize("route_key", ["", " ", "\t", "\n"])
def test_empty_or_whitespace_routing_key_is_rejected(route_key: str) -> None:
    with pytest.raises(
        RoutingConfigurationError,
        match="Routing key must be a non-empty string",
    ):
        DeterministicRouter({route_key: "planner"})


@pytest.mark.parametrize("route_key", ["plan now", "plan\tnow", "plan\nnow"])
def test_routing_key_containing_whitespace_is_rejected(route_key: str) -> None:
    with pytest.raises(
        RoutingConfigurationError,
        match="must not contain whitespace",
    ):
        DeterministicRouter({route_key: "planner"})


def test_routing_key_with_leading_slash_is_rejected() -> None:
    with pytest.raises(
        RoutingConfigurationError,
        match="must not include a leading '/'",
    ):
        DeterministicRouter({"/plan": "planner"})


@pytest.mark.parametrize("agent_name", ["", " ", "\t", "\n"])
def test_empty_or_whitespace_agent_name_is_rejected(agent_name: str) -> None:
    with pytest.raises(
        RoutingConfigurationError,
        match="Agent name for route 'plan' must be non-empty text",
    ):
        DeterministicRouter({"plan": agent_name})


def test_valid_configuration_preserves_exact_values() -> None:
    router = DeterministicRouter({"Plan-Now": "  Planner  "})

    assert router.routes == (("Plan-Now", "  Planner  "),)


def test_routes_property_is_an_immutable_snapshot() -> None:
    router = DeterministicRouter({"plan": "planner"})

    routes = router.routes
    routes += (("research", "researcher"),)

    assert router.routes == (("plan", "planner"),)


def test_matching_directive_returns_expected_decision() -> None:
    router = DeterministicRouter({"plan": "planner"})
    registry = registry_with(TrackingAgent("planner"))

    _, _, decision = run_route(router, "/plan organise my day", registry)

    assert isinstance(decision, RoutingDecision)
    assert decision.agent_name == "planner"
    assert decision.reason == "Explicit route '/plan' selected agent 'planner'."


def test_request_context_mismatch_is_rejected_without_mutation() -> None:
    request = TuesdayRequest(content="/plan organise my day")
    context = ConversationContext()
    original_request = request
    original_context = context
    router = DeterministicRouter({"plan": "planner"})

    with pytest.raises(
        ValueError,
        match="Request and context must belong to the same conversation",
    ):
        asyncio.run(router.route(request, context, AgentRegistry()))

    assert request == original_request
    assert context == original_context


@pytest.mark.parametrize(
    "content",
    ["organise my day", "organise /plan my day", "/"],
)
def test_missing_or_misplaced_directive_is_unroutable(content: str) -> None:
    router = DeterministicRouter({"plan": "planner"})

    with pytest.raises(
        NoRouteFoundError,
        match="No explicit route directive was found",
    ):
        run_route(router, content, registry_with(TrackingAgent("planner")))


def test_unknown_directive_reports_route_identifier() -> None:
    router = DeterministicRouter({"plan": "planner"})

    with pytest.raises(
        NoRouteFoundError,
        match="No route is configured for directive '/weather'",
    ):
        run_route(router, "/weather tomorrow", AgentRegistry())


def test_route_matching_is_case_sensitive() -> None:
    router = DeterministicRouter({"plan": "planner"})

    with pytest.raises(NoRouteFoundError, match="'/Plan'"):
        run_route(router, "/Plan organise my day", AgentRegistry())


def test_configured_unregistered_agent_reports_route_and_agent() -> None:
    router = DeterministicRouter({"plan": "planner"})

    with pytest.raises(
        RoutedAgentUnavailableError,
        match="Route '/plan' targets agent 'planner'.*not registered",
    ):
        run_route(router, "/plan organise my day", AgentRegistry())


def test_empty_registry_uses_normal_unavailable_agent_failure() -> None:
    router = DeterministicRouter({"plan": "planner"})

    with pytest.raises(RoutedAgentUnavailableError):
        run_route(router, "/plan", AgentRegistry())


def test_registered_agent_is_not_executed_during_routing() -> None:
    agent = TrackingAgent("planner")
    registry = registry_with(agent)

    run_route(DeterministicRouter({"plan": "planner"}), "/plan task", registry)

    assert agent.was_executed is False


def test_registry_remains_unchanged_after_routing() -> None:
    planner = TrackingAgent("planner")
    researcher = TrackingAgent("researcher")
    registry = registry_with(planner, researcher)
    original_names = registry.names

    run_route(DeterministicRouter({"plan": "planner"}), "/plan task", registry)

    assert registry.names == original_names
    assert registry.get("planner") is planner
    assert registry.get("researcher") is researcher


def test_request_remains_unchanged_after_routing() -> None:
    router = DeterministicRouter({"plan": "planner"})
    registry = registry_with(TrackingAgent("planner"))

    request, _, _ = run_route(router, "/plan organise my day", registry)

    assert request.content == "/plan organise my day"


def test_router_configuration_remains_unchanged_after_routing() -> None:
    router = DeterministicRouter({"plan": "planner"})
    original_routes = router.routes

    run_route(router, "/plan task", registry_with(TrackingAgent("planner")))

    assert router.routes == original_routes


@pytest.mark.parametrize(
    ("content", "expected_agent"),
    [
        ("/plan organise my day", "planner"),
        ("/research compare databases", "researcher"),
    ],
)
def test_multiple_routes_select_corresponding_agents(
    content: str,
    expected_agent: str,
) -> None:
    router = DeterministicRouter(
        {"plan": "planner", "research": "researcher"}
    )
    registry = registry_with(
        TrackingAgent("planner"),
        TrackingAgent("researcher"),
    )

    _, _, decision = run_route(router, content, registry)

    assert decision.agent_name == expected_agent


def test_directive_without_trailing_message_is_routable() -> None:
    router = DeterministicRouter({"plan": "planner"})
    registry = registry_with(TrackingAgent("planner"))

    _, _, decision = run_route(router, "/plan", registry)

    assert decision.agent_name == "planner"


def test_only_first_token_is_used_as_directive() -> None:
    router = DeterministicRouter(
        {"plan": "planner", "research": "researcher"}
    )
    registry = registry_with(
        TrackingAgent("planner"),
        TrackingAgent("researcher"),
    )

    _, _, decision = run_route(router, "/plan /research this", registry)

    assert decision.agent_name == "planner"


def test_constructor_accepts_general_mapping() -> None:
    class RouteMapping(Mapping[str, str]):
        def __getitem__(self, key: str) -> str:
            if key != "plan":
                raise KeyError(key)
            return "planner"

        def __iter__(self) -> Iterator[str]:
            return iter(("plan",))

        def __len__(self) -> int:
            return 1

    assert DeterministicRouter(RouteMapping()).routes == (("plan", "planner"),)
