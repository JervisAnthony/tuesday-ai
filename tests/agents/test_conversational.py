"""Tests for TUESDAY's deterministic conversational agent."""

import asyncio
import inspect
from uuid import UUID, uuid4

import pytest

from tuesday.agents import BaseAgent, ConversationalAgent
from tuesday.agents import conversational as conversational_module
from tuesday.domain import (
    ConversationContext,
    ConversationMessage,
    MessageRole,
    TuesdayRequest,
    TuesdayResponse,
)


def run_handle(
    agent: ConversationalAgent,
    request: TuesdayRequest,
    context: ConversationContext,
) -> TuesdayResponse:
    return asyncio.run(agent.handle(request, context))


def matching_interaction(
    content: str = "Hello",
    messages: tuple[ConversationMessage, ...] = (),
) -> tuple[TuesdayRequest, ConversationContext]:
    conversation_id = uuid4()
    return (
        TuesdayRequest(content=content, conversation_id=conversation_id),
        ConversationContext(
            conversation_id=conversation_id,
            messages=messages,
        ),
    )


def test_conversational_agent_inherits_base_agent() -> None:
    assert isinstance(ConversationalAgent(), BaseAgent)


def test_agent_has_stable_truthful_identity() -> None:
    agent = ConversationalAgent()

    assert agent.name == "conversation"
    assert agent.description
    assert "deterministic" in agent.description.lower()
    assert "conversational" in agent.description.lower()


def test_handle_is_asynchronous() -> None:
    assert inspect.iscoroutinefunction(ConversationalAgent.handle)


def test_matching_request_and_context_are_accepted() -> None:
    request, context = matching_interaction()

    response = run_handle(ConversationalAgent(), request, context)

    assert isinstance(response, TuesdayResponse)


def test_mismatch_uses_established_error_without_mutation() -> None:
    request = TuesdayRequest(content="Hello")
    context = ConversationContext()
    original_request = request
    original_context = context

    with pytest.raises(
        ValueError,
        match="Request and context must belong to the same conversation",
    ):
        run_handle(ConversationalAgent(), request, context)

    assert request == original_request
    assert context == original_context


def test_empty_context_returns_correlated_deterministic_response() -> None:
    request, context = matching_interaction("Hello")

    response = run_handle(ConversationalAgent(), request, context)

    assert response.content == "TUESDAY received: Hello"
    assert response.conversation_id == request.conversation_id
    assert response.request_id == request.request_id
    assert isinstance(response.response_id, UUID)


@pytest.mark.parametrize(
    ("request_content", "expected_content"),
    [
        ("Hello", "TUESDAY received: Hello"),
        ("  Hello  ", "TUESDAY received:   Hello  "),
        ("/chat hello", "TUESDAY received: /chat hello"),
        ("/conversation hello", "TUESDAY received: /conversation hello"),
    ],
)
def test_request_text_is_preserved_exactly(
    request_content: str,
    expected_content: str,
) -> None:
    request, context = matching_interaction(request_content)

    response = run_handle(ConversationalAgent(), request, context)

    assert response.content == expected_content


def test_one_prior_message_uses_singular_wording() -> None:
    messages = (
        ConversationMessage(role=MessageRole.USER, content="Earlier message"),
    )
    request, context = matching_interaction("Hello", messages)

    response = run_handle(ConversationalAgent(), request, context)

    assert response.content == (
        "TUESDAY received: Hello (1 prior message in context)"
    )


def test_multiple_prior_messages_use_plural_wording() -> None:
    messages = (
        ConversationMessage(role=MessageRole.USER, content="First"),
        ConversationMessage(role=MessageRole.ASSISTANT, content="Second"),
    )
    request, context = matching_interaction("Hello", messages)

    response = run_handle(ConversationalAgent(), request, context)

    assert response.content == (
        "TUESDAY received: Hello (2 prior messages in context)"
    )


def test_prior_message_content_is_not_copied_into_response() -> None:
    prior_content = "private prior content"
    messages = (
        ConversationMessage(role=MessageRole.USER, content=prior_content),
    )
    request, context = matching_interaction("Current request", messages)

    response = run_handle(ConversationalAgent(), request, context)

    assert prior_content not in response.content
    assert "Current request" in response.content


@pytest.mark.parametrize("role", list(MessageRole))
def test_message_role_does_not_change_response_behavior(role: MessageRole) -> None:
    messages = (ConversationMessage(role=role, content="Earlier message"),)
    request, context = matching_interaction("Hello", messages)

    response = run_handle(ConversationalAgent(), request, context)

    assert response.content == (
        "TUESDAY received: Hello (1 prior message in context)"
    )


def test_successful_execution_does_not_mutate_request_or_context() -> None:
    messages = (
        ConversationMessage(role=MessageRole.USER, content="Earlier message"),
    )
    request, context = matching_interaction("Hello", messages)
    original_request = request
    original_context = context

    run_handle(ConversationalAgent(), request, context)

    assert request == original_request
    assert context == original_context
    assert context.messages == messages


def test_multiple_executions_create_independent_response_ids() -> None:
    request, context = matching_interaction()
    agent = ConversationalAgent()

    first = run_handle(agent, request, context)
    second = run_handle(agent, request, context)

    assert first.response_id != second.response_id


def test_sequential_executions_do_not_leak_state() -> None:
    agent = ConversationalAgent()
    first_request, first_context = matching_interaction("First")
    prior_message = ConversationMessage(
        role=MessageRole.USER,
        content="Prior",
    )
    second_request, second_context = matching_interaction(
        "Second",
        (prior_message,),
    )

    first = run_handle(agent, first_request, first_context)
    second = run_handle(agent, second_request, second_context)

    assert first.content == "TUESDAY received: First"
    assert second.content == (
        "TUESDAY received: Second (1 prior message in context)"
    )


def test_agent_stores_no_per_request_state() -> None:
    agent = ConversationalAgent()
    request, context = matching_interaction()

    run_handle(agent, request, context)

    assert vars(agent) == {}


def test_response_content_is_deterministic_for_identical_input_shape() -> None:
    messages = (
        ConversationMessage(role=MessageRole.USER, content="First history"),
        ConversationMessage(role=MessageRole.SYSTEM, content="Second history"),
    )
    first_request, first_context = matching_interaction("Hello", messages)
    second_request, second_context = matching_interaction("Hello", messages)
    agent = ConversationalAgent()

    first = run_handle(agent, first_request, first_context)
    second = run_handle(agent, second_request, second_context)

    assert first.content == second.content


def test_handle_contract_has_no_registry_or_routing_dependency() -> None:
    parameters = tuple(inspect.signature(ConversationalAgent.handle).parameters)

    assert parameters == ("self", "request", "context")


def test_agent_module_has_no_registry_or_routing_operations() -> None:
    module_names = vars(conversational_module)

    assert "AgentRegistry" not in module_names
    assert "BaseRouter" not in module_names
    assert "RoutingDecision" not in module_names


def test_agent_handle_has_no_external_io_operations() -> None:
    external_io_names = {
        "open",
        "socket",
        "subprocess",
        "urlopen",
        "requests",
    }

    assert external_io_names.isdisjoint(ConversationalAgent.handle.__code__.co_names)
