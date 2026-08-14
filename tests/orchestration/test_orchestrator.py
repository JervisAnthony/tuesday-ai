"""Tests for the core TUESDAY orchestrator."""

import asyncio
import inspect
from uuid import UUID, uuid4

import pytest

from tuesday.agents import AgentNotFoundError, AgentRegistry, BaseAgent
from tuesday.domain import ConversationContext, TuesdayRequest, TuesdayResponse
from tuesday.orchestration import (
    InvalidAgentResponseError,
    TuesdayOrchestrator,
)
from tuesday.routing import BaseRouter, RoutingDecision


class TrackingRouter(BaseRouter):
    """Test router that records calls and returns a fixed decision."""

    def __init__(
        self,
        agent_name: str = "selected",
        error: Exception | None = None,
    ) -> None:
        self.agent_name = agent_name
        self.error = error
        self.route_calls = 0
        self.received_request: TuesdayRequest | None = None
        self.received_context: ConversationContext | None = None
        self.received_registry: AgentRegistry | None = None

    async def route(
        self,
        request: TuesdayRequest,
        context: ConversationContext,
        registry: AgentRegistry,
    ) -> RoutingDecision:
        self.route_calls += 1
        self.received_request = request
        self.received_context = context
        self.received_registry = registry
        if self.error is not None:
            raise self.error
        return RoutingDecision(
            agent_name=self.agent_name,
            reason="Fixed orchestration test route.",
        )


class TrackingAgent(BaseAgent):
    """Test agent that records calls and can return invalid correlations."""

    def __init__(
        self,
        name: str,
        *,
        conversation_id: UUID | None = None,
        request_id: UUID | None = None,
        error: Exception | None = None,
    ) -> None:
        self._name = name
        self._conversation_id = conversation_id
        self._request_id = request_id
        self.error = error
        self.handle_calls = 0
        self.received_request: TuesdayRequest | None = None
        self.received_context: ConversationContext | None = None
        self.returned_response: TuesdayResponse | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "An orchestration test agent."

    async def handle(
        self,
        request: TuesdayRequest,
        context: ConversationContext,
    ) -> TuesdayResponse:
        self.handle_calls += 1
        self.received_request = request
        self.received_context = context
        if self.error is not None:
            raise self.error
        self.returned_response = TuesdayResponse(
            content=f"response from {self.name}",
            conversation_id=(
                self._conversation_id
                if self._conversation_id is not None
                else request.conversation_id
            ),
            request_id=(
                self._request_id
                if self._request_id is not None
                else request.request_id
            ),
        )
        return self.returned_response


def matching_interaction(content: str = "Hello") -> tuple[
    TuesdayRequest,
    ConversationContext,
]:
    conversation_id = uuid4()
    return (
        TuesdayRequest(content=content, conversation_id=conversation_id),
        ConversationContext(conversation_id=conversation_id),
    )


def registry_with(*agents: TrackingAgent) -> AgentRegistry:
    registry = AgentRegistry()
    for agent in agents:
        registry.register(agent)
    return registry


def run_handle(
    orchestrator: TuesdayOrchestrator,
    request: TuesdayRequest,
    context: ConversationContext,
) -> TuesdayResponse:
    return asyncio.run(orchestrator.handle(request, context))


def test_orchestrator_accepts_router_and_registry_dependencies() -> None:
    router = TrackingRouter()
    registry = AgentRegistry()

    orchestrator = TuesdayOrchestrator(router, registry)

    assert orchestrator._router is router
    assert orchestrator._registry is registry


def test_handle_is_asynchronous() -> None:
    assert inspect.iscoroutinefunction(TuesdayOrchestrator.handle)


def test_matching_request_and_context_are_accepted() -> None:
    request, context = matching_interaction()
    agent = TrackingAgent("selected")
    orchestrator = TuesdayOrchestrator(
        TrackingRouter(),
        registry_with(agent),
    )

    response = run_handle(orchestrator, request, context)

    assert response.conversation_id == request.conversation_id


def test_context_mismatch_fails_before_routing_without_mutation() -> None:
    request = TuesdayRequest(content="Hello")
    context = ConversationContext()
    original_request = request
    original_context = context
    router = TrackingRouter()
    orchestrator = TuesdayOrchestrator(router, AgentRegistry())

    with pytest.raises(
        ValueError,
        match="Request and context must belong to the same conversation",
    ):
        run_handle(orchestrator, request, context)

    assert router.route_calls == 0
    assert request == original_request
    assert context == original_context


def test_router_receives_exact_interaction_and_registry_once() -> None:
    request, context = matching_interaction()
    router = TrackingRouter()
    registry = registry_with(TrackingAgent("selected"))

    run_handle(TuesdayOrchestrator(router, registry), request, context)

    assert router.route_calls == 1
    assert router.received_request is request
    assert router.received_context is context
    assert router.received_registry is registry


def test_decision_is_resolved_to_exact_registered_agent() -> None:
    request, context = matching_interaction()
    selected = TrackingAgent("selected")
    registry = registry_with(selected)

    run_handle(
        TuesdayOrchestrator(TrackingRouter("selected"), registry),
        request,
        context,
    )

    assert registry.get("selected") is selected
    assert selected.handle_calls == 1


def test_only_selected_agent_executes_exactly_once() -> None:
    request, context = matching_interaction()
    selected = TrackingAgent("selected")
    other = TrackingAgent("other")
    registry = registry_with(selected, other)

    run_handle(TuesdayOrchestrator(TrackingRouter(), registry), request, context)

    assert selected.handle_calls == 1
    assert other.handle_calls == 0


def test_selected_agent_receives_exact_request_and_context() -> None:
    request, context = matching_interaction()
    agent = TrackingAgent("selected")

    run_handle(
        TuesdayOrchestrator(TrackingRouter(), registry_with(agent)),
        request,
        context,
    )

    assert agent.received_request is request
    assert agent.received_context is context


def test_successful_response_is_returned_unchanged_with_all_ids() -> None:
    request, context = matching_interaction()
    agent = TrackingAgent("selected")

    response = run_handle(
        TuesdayOrchestrator(TrackingRouter(), registry_with(agent)),
        request,
        context,
    )

    assert response is agent.returned_response
    assert response.conversation_id == request.conversation_id
    assert response.request_id == request.request_id
    assert response.response_id == agent.returned_response.response_id


def test_unregistered_decision_propagates_agent_not_found_without_execution() -> None:
    request, context = matching_interaction()
    registered = TrackingAgent("registered")
    registry = registry_with(registered)
    orchestrator = TuesdayOrchestrator(TrackingRouter("missing"), registry)

    with pytest.raises(AgentNotFoundError, match="missing"):
        run_handle(orchestrator, request, context)

    assert registered.handle_calls == 0


def test_wrong_response_conversation_id_is_rejected_without_repair() -> None:
    request, context = matching_interaction()
    wrong_conversation_id = uuid4()
    agent = TrackingAgent(
        "selected",
        conversation_id=wrong_conversation_id,
    )

    with pytest.raises(
        InvalidAgentResponseError,
        match="conversation_id does not match the request",
    ):
        run_handle(
            TuesdayOrchestrator(TrackingRouter(), registry_with(agent)),
            request,
            context,
        )

    assert agent.returned_response is not None
    assert agent.returned_response.conversation_id == wrong_conversation_id


def test_wrong_response_request_id_is_rejected_without_repair() -> None:
    request, context = matching_interaction()
    wrong_request_id = uuid4()
    agent = TrackingAgent("selected", request_id=wrong_request_id)

    with pytest.raises(
        InvalidAgentResponseError,
        match="request_id does not match the request",
    ):
        run_handle(
            TuesdayOrchestrator(TrackingRouter(), registry_with(agent)),
            request,
            context,
        )

    assert agent.returned_response is not None
    assert agent.returned_response.request_id == wrong_request_id


def test_registry_is_not_mutated() -> None:
    request, context = matching_interaction()
    agent = TrackingAgent("selected")
    registry = registry_with(agent)
    original_names = registry.names

    run_handle(TuesdayOrchestrator(TrackingRouter(), registry), request, context)

    assert registry.names == original_names
    assert registry.get("selected") is agent


def test_request_and_context_are_not_mutated() -> None:
    request, context = matching_interaction("/plan organise my day")
    original_request = request
    original_context = context

    run_handle(
        TuesdayOrchestrator(
            TrackingRouter(),
            registry_with(TrackingAgent("selected")),
        ),
        request,
        context,
    )

    assert request == original_request
    assert request.content == "/plan organise my day"
    assert context == original_context


def test_handle_returns_response_without_exposing_routing_decision() -> None:
    request, context = matching_interaction()
    response = run_handle(
        TuesdayOrchestrator(
            TrackingRouter(),
            registry_with(TrackingAgent("selected")),
        ),
        request,
        context,
    )

    assert isinstance(response, TuesdayResponse)
    assert not isinstance(response, tuple)


def test_orchestrator_retains_only_composition_dependencies() -> None:
    router = TrackingRouter()
    registry = registry_with(TrackingAgent("selected"))
    orchestrator = TuesdayOrchestrator(router, registry)
    request, context = matching_interaction()

    run_handle(orchestrator, request, context)

    assert vars(orchestrator) == {"_router": router, "_registry": registry}


def test_sequential_interactions_do_not_leak_request_state() -> None:
    router = TrackingRouter()
    agent = TrackingAgent("selected")
    orchestrator = TuesdayOrchestrator(router, registry_with(agent))
    first_request, first_context = matching_interaction("first")
    second_request, second_context = matching_interaction("second")

    first_response = run_handle(orchestrator, first_request, first_context)
    second_response = run_handle(orchestrator, second_request, second_context)

    assert first_response.request_id == first_request.request_id
    assert second_response.request_id == second_request.request_id
    assert second_response.conversation_id == second_context.conversation_id
    assert router.route_calls == 2
    assert agent.handle_calls == 2


def test_agent_exception_propagates_without_retry() -> None:
    request, context = matching_interaction()
    error = RuntimeError("agent failed")
    agent = TrackingAgent("selected", error=error)

    with pytest.raises(RuntimeError, match="agent failed") as raised:
        run_handle(
            TuesdayOrchestrator(TrackingRouter(), registry_with(agent)),
            request,
            context,
        )

    assert raised.value is error
    assert agent.handle_calls == 1


def test_router_exception_propagates_without_agent_execution() -> None:
    request, context = matching_interaction()
    error = RuntimeError("router failed")
    router = TrackingRouter(error=error)
    agent = TrackingAgent("selected")

    with pytest.raises(RuntimeError, match="router failed") as raised:
        run_handle(
            TuesdayOrchestrator(router, registry_with(agent)),
            request,
            context,
        )

    assert raised.value is error
    assert router.route_calls == 1
    assert agent.handle_calls == 0
