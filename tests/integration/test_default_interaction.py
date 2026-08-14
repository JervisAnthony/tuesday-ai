"""Integration tests for TUESDAY's default application composition."""

import asyncio
from uuid import UUID, uuid4

import pytest

from tuesday.agents import AgentRegistry, ConversationalAgent
from tuesday.composition import create_default_orchestrator
from tuesday.domain import (
    ConversationContext,
    ConversationMessage,
    MessageRole,
    TuesdayRequest,
    TuesdayResponse,
)
from tuesday.orchestration import TuesdayOrchestrator
from tuesday.routing import DeterministicRouter, NoRouteFoundError


def run_interaction(
    orchestrator: TuesdayOrchestrator,
    content: str,
    messages: tuple[ConversationMessage, ...] = (),
) -> tuple[TuesdayRequest, ConversationContext, TuesdayResponse]:
    conversation_id = uuid4()
    request = TuesdayRequest(content=content, conversation_id=conversation_id)
    context = ConversationContext(
        conversation_id=conversation_id,
        messages=messages,
    )
    response = asyncio.run(orchestrator.handle(request, context))
    return request, context, response


def test_factory_returns_orchestrator_with_real_default_components() -> None:
    orchestrator = create_default_orchestrator()

    assert isinstance(orchestrator, TuesdayOrchestrator)
    assert isinstance(orchestrator._router, DeterministicRouter)
    assert isinstance(orchestrator._registry, AgentRegistry)
    assert orchestrator._registry.names == ("conversation",)
    assert isinstance(
        orchestrator._registry.get("conversation"),
        ConversationalAgent,
    )


def test_default_composition_has_exact_route_policy() -> None:
    orchestrator = create_default_orchestrator()

    assert orchestrator._router.routes == (
        ("chat", "conversation"),
        ("conversation", "conversation"),
    )


@pytest.mark.parametrize(
    ("content", "expected_content"),
    [
        ("/chat Hello", "TUESDAY received: /chat Hello"),
        (
            "/conversation Hello",
            "TUESDAY received: /conversation Hello",
        ),
    ],
)
def test_default_directives_complete_end_to_end_interaction(
    content: str,
    expected_content: str,
) -> None:
    request, _, response = run_interaction(
        create_default_orchestrator(),
        content,
    )

    assert response.content == expected_content
    assert response.conversation_id == request.conversation_id
    assert response.request_id == request.request_id
    assert isinstance(response.response_id, UUID)
    assert response.response_id != request.request_id


def test_one_prior_message_is_reflected_with_singular_wording() -> None:
    messages = (
        ConversationMessage(role=MessageRole.USER, content="Earlier message"),
    )

    _, _, response = run_interaction(
        create_default_orchestrator(),
        "/chat Hello again",
        messages,
    )

    assert response.content == (
        "TUESDAY received: /chat Hello again (1 prior message in context)"
    )


def test_multiple_prior_messages_use_plural_without_leaking_content() -> None:
    first_prior_content = "private first message"
    second_prior_content = "private second message"
    messages = (
        ConversationMessage(
            role=MessageRole.USER,
            content=first_prior_content,
        ),
        ConversationMessage(
            role=MessageRole.ASSISTANT,
            content=second_prior_content,
        ),
    )

    _, _, response = run_interaction(
        create_default_orchestrator(),
        "/chat Hello again",
        messages,
    )

    assert response.content == (
        "TUESDAY received: /chat Hello again (2 prior messages in context)"
    )
    assert first_prior_content not in response.content
    assert second_prior_content not in response.content


def test_successful_interaction_does_not_mutate_request_or_context() -> None:
    messages = (
        ConversationMessage(role=MessageRole.USER, content="Earlier message"),
    )
    conversation_id = uuid4()
    request = TuesdayRequest(
        content="/chat Hello",
        conversation_id=conversation_id,
    )
    context = ConversationContext(
        conversation_id=conversation_id,
        messages=messages,
    )
    original_request = request
    original_context = context

    asyncio.run(create_default_orchestrator().handle(request, context))

    assert request == original_request
    assert request.content == "/chat Hello"
    assert context == original_context
    assert context.messages == messages


@pytest.mark.parametrize(
    "content",
    [
        "/unknown Hello",
        "Hello",
        "Hello /chat",
    ],
)
def test_unconfigured_or_missing_directive_remains_an_explicit_failure(
    content: str,
) -> None:
    with pytest.raises(NoRouteFoundError):
        run_interaction(create_default_orchestrator(), content)


def test_route_selection_remains_case_sensitive() -> None:
    with pytest.raises(NoRouteFoundError, match="'/Chat'"):
        run_interaction(create_default_orchestrator(), "/Chat Hello")


def test_request_context_mismatch_fails() -> None:
    request = TuesdayRequest(content="/chat Hello")
    context = ConversationContext()
    orchestrator = create_default_orchestrator()

    with pytest.raises(
        ValueError,
        match="Request and context must belong to the same conversation",
    ):
        asyncio.run(orchestrator.handle(request, context))

    valid_context = ConversationContext(
        conversation_id=request.conversation_id,
    )
    response = asyncio.run(orchestrator.handle(request, valid_context))
    assert response.request_id == request.request_id


def test_factory_calls_create_independent_runtime_graphs() -> None:
    first = create_default_orchestrator()
    second = create_default_orchestrator()

    assert first is not second
    assert first._registry is not second._registry
    assert first._router is not second._router
    assert first._registry.get("conversation") is not second._registry.get(
        "conversation"
    )
    assert first._registry.names == ("conversation",)
    assert second._registry.names == ("conversation",)


def test_sequential_interactions_remain_independent_and_correlated() -> None:
    orchestrator = create_default_orchestrator()

    first_request, _, first_response = run_interaction(
        orchestrator,
        "/chat First",
    )
    second_request, _, second_response = run_interaction(
        orchestrator,
        "/conversation Second",
    )

    assert first_response.content == "TUESDAY received: /chat First"
    assert second_response.content == (
        "TUESDAY received: /conversation Second"
    )
    assert first_response.conversation_id == first_request.conversation_id
    assert first_response.request_id == first_request.request_id
    assert second_response.conversation_id == second_request.conversation_id
    assert second_response.request_id == second_request.request_id
    assert first_response.response_id != second_response.response_id
