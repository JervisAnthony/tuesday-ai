"""Tests for the framework-independent base agent contract."""

import asyncio
import inspect
from uuid import uuid4

import pytest

from tuesday.agents import BaseAgent
from tuesday.domain import ConversationContext, TuesdayRequest, TuesdayResponse


class StubAgent(BaseAgent):
    """Minimal test-only implementation of the agent contract."""

    @property
    def name(self) -> str:
        return "stub"

    @property
    def description(self) -> str:
        return "A deterministic test agent."

    async def handle(
        self,
        request: TuesdayRequest,
        context: ConversationContext,
    ) -> TuesdayResponse:
        self._validate_context(request, context)
        return TuesdayResponse(
            content="stub response",
            conversation_id=request.conversation_id,
            request_id=request.request_id,
        )


class IncompleteAgent(BaseAgent):
    """Test subclass intentionally missing the required members."""


def test_base_agent_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseAgent()


def test_incomplete_agent_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        IncompleteAgent()


def test_complete_agent_can_be_instantiated() -> None:
    assert isinstance(StubAgent(), BaseAgent)


def test_agent_exposes_identity_metadata() -> None:
    agent = StubAgent()

    assert agent.name == "stub"
    assert agent.description == "A deterministic test agent."


def test_handle_defines_an_async_contract() -> None:
    assert inspect.iscoroutinefunction(BaseAgent.handle)
    assert inspect.iscoroutinefunction(StubAgent.handle)


def test_handle_accepts_domain_contracts_and_returns_correlated_response() -> None:
    conversation_id = uuid4()
    request = TuesdayRequest(
        content="Hello",
        conversation_id=conversation_id,
    )
    context = ConversationContext(conversation_id=conversation_id)

    response = asyncio.run(StubAgent().handle(request, context))

    assert isinstance(response, TuesdayResponse)
    assert response.content == "stub response"
    assert response.conversation_id == request.conversation_id
    assert response.request_id == request.request_id


def test_matching_request_and_context_are_accepted() -> None:
    conversation_id = uuid4()
    request = TuesdayRequest(content="Hello", conversation_id=conversation_id)
    context = ConversationContext(conversation_id=conversation_id)

    response = asyncio.run(StubAgent().handle(request, context))

    assert response.conversation_id == conversation_id


def test_mismatched_request_and_context_are_rejected_without_mutation() -> None:
    request = TuesdayRequest(content="Hello")
    context = ConversationContext()
    original_request = request
    original_context = context

    with pytest.raises(
        ValueError,
        match="Request and context must belong to the same conversation",
    ):
        asyncio.run(StubAgent().handle(request, context))

    assert request == original_request
    assert context == original_context


def test_multiple_executions_produce_independent_response_ids() -> None:
    conversation_id = uuid4()
    request = TuesdayRequest(content="Hello", conversation_id=conversation_id)
    context = ConversationContext(conversation_id=conversation_id)
    agent = StubAgent()

    first_response = asyncio.run(agent.handle(request, context))
    second_response = asyncio.run(agent.handle(request, context))

    assert first_response.response_id != second_response.response_id
