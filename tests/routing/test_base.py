"""Tests for the framework-independent routing contracts."""

import asyncio
import inspect
from dataclasses import FrozenInstanceError
from uuid import uuid4

import pytest

from tuesday.agents import AgentRegistry, BaseAgent
from tuesday.domain import ConversationContext, TuesdayRequest, TuesdayResponse
from tuesday.routing import BaseRouter, RoutingDecision


class StubRouter(BaseRouter):
    """Constant test-only implementation of the routing contract."""

    def __init__(self) -> None:
        self.received_registry: AgentRegistry | None = None

    async def route(
        self,
        request: TuesdayRequest,
        context: ConversationContext,
        registry: AgentRegistry,
    ) -> RoutingDecision:
        self._validate_context(request, context)
        self.received_registry = registry
        return RoutingDecision(
            agent_name="stub",
            reason="Deterministic test route.",
        )


class IncompleteRouter(BaseRouter):
    """Test subclass intentionally missing the required route method."""


class TrackingAgent(BaseAgent):
    """Test agent that records whether routing executes it."""

    def __init__(self) -> None:
        self.was_executed = False

    @property
    def name(self) -> str:
        return "stub"

    @property
    def description(self) -> str:
        return "An agent used to detect execution."

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


def test_routing_decision_creation() -> None:
    decision = RoutingDecision(agent_name="planner", reason="Best fit.")

    assert decision.agent_name == "planner"
    assert decision.reason == "Best fit."


def test_routing_decision_preserves_agent_name_exactly() -> None:
    decision = RoutingDecision(agent_name="  Planner  ", reason="Best fit.")

    assert decision.agent_name == "  Planner  "


def test_routing_decision_preserves_reason_exactly() -> None:
    decision = RoutingDecision(agent_name="planner", reason="  Best fit.  ")

    assert decision.reason == "  Best fit.  "


@pytest.mark.parametrize("agent_name", ["", " ", "\t", "\n"])
def test_meaningless_agent_name_is_rejected(agent_name: str) -> None:
    with pytest.raises(ValueError, match="agent_name must be non-empty text"):
        RoutingDecision(agent_name=agent_name, reason="Best fit.")


@pytest.mark.parametrize("reason", ["", " ", "\t", "\n"])
def test_meaningless_reason_is_rejected(reason: str) -> None:
    with pytest.raises(ValueError, match="reason must be non-empty text"):
        RoutingDecision(agent_name="planner", reason=reason)


def test_routing_decision_is_immutable() -> None:
    decision = RoutingDecision(agent_name="planner", reason="Best fit.")

    with pytest.raises(FrozenInstanceError):
        decision.agent_name = "other"  # type: ignore[misc]


def test_base_router_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseRouter()


def test_incomplete_router_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        IncompleteRouter()


def test_complete_router_can_be_instantiated() -> None:
    assert isinstance(StubRouter(), BaseRouter)


def test_route_defines_an_async_contract() -> None:
    assert inspect.iscoroutinefunction(BaseRouter.route)
    assert inspect.iscoroutinefunction(StubRouter.route)


def test_route_accepts_domain_contracts_and_registry() -> None:
    conversation_id = uuid4()
    request = TuesdayRequest(content="Plan my day", conversation_id=conversation_id)
    context = ConversationContext(conversation_id=conversation_id)
    registry = AgentRegistry()

    decision = asyncio.run(StubRouter().route(request, context, registry))

    assert isinstance(decision, RoutingDecision)
    assert decision.agent_name == "stub"
    assert decision.reason == "Deterministic test route."


def test_matching_request_and_context_are_accepted() -> None:
    conversation_id = uuid4()
    request = TuesdayRequest(content="Hello", conversation_id=conversation_id)
    context = ConversationContext(conversation_id=conversation_id)

    decision = asyncio.run(StubRouter().route(request, context, AgentRegistry()))

    assert isinstance(decision, RoutingDecision)


def test_mismatched_request_and_context_are_rejected_without_mutation() -> None:
    request = TuesdayRequest(content="Hello")
    context = ConversationContext()
    original_request = request
    original_context = context

    with pytest.raises(
        ValueError,
        match="Request and context must belong to the same conversation",
    ):
        asyncio.run(StubRouter().route(request, context, AgentRegistry()))

    assert request == original_request
    assert context == original_context


def test_registry_is_passed_by_instance_without_mutation() -> None:
    conversation_id = uuid4()
    request = TuesdayRequest(content="Hello", conversation_id=conversation_id)
    context = ConversationContext(conversation_id=conversation_id)
    registry = AgentRegistry()
    router = StubRouter()

    asyncio.run(router.route(request, context, registry))

    assert router.received_registry is registry
    assert len(registry) == 0
    assert registry.names == ()


def test_routing_does_not_execute_the_selected_agent() -> None:
    conversation_id = uuid4()
    request = TuesdayRequest(content="Hello", conversation_id=conversation_id)
    context = ConversationContext(conversation_id=conversation_id)
    registry = AgentRegistry()
    agent = TrackingAgent()
    registry.register(agent)

    decision = asyncio.run(StubRouter().route(request, context, registry))

    assert decision.agent_name == agent.name
    assert agent.was_executed is False


def test_decision_does_not_require_named_agent_to_be_registered() -> None:
    registry = AgentRegistry()

    decision = RoutingDecision(agent_name="not-registered", reason="test")

    assert decision.agent_name not in registry
