"""Tests for the provider-neutral conversational prompt renderer."""

import ast
import inspect
from dataclasses import FrozenInstanceError, fields
from datetime import UTC, datetime
from uuid import uuid4

import pytest

import tuesday.prompting.conversational as conversational_prompting
from tuesday.domain import (
    ConversationContext,
    ConversationMessage,
    MessageRole,
    TuesdayRequest,
)
from tuesday.language_models import LanguageModelMessage, LanguageModelRequest
from tuesday.prompting import (
    DEFAULT_CONVERSATIONAL_SYSTEM_PROMPT,
    ConversationalPromptRenderer,
)


def make_request(content: str = "Hello") -> TuesdayRequest:
    conversation_id = uuid4()
    return TuesdayRequest(content=content, conversation_id=conversation_id)


def make_context(
    conversation_id,
    *messages: ConversationMessage,
) -> ConversationContext:
    return ConversationContext(conversation_id=conversation_id, messages=messages)


def test_renderer_is_frozen_and_slotted() -> None:
    renderer = ConversationalPromptRenderer()

    assert not hasattr(renderer, "__dict__")
    with pytest.raises(FrozenInstanceError):
        renderer.system_prompt = "Changed"  # type: ignore[misc]


def test_default_system_prompt_is_stable_and_meaningful() -> None:
    renderer = ConversationalPromptRenderer()

    assert renderer.system_prompt == DEFAULT_CONVERSATIONAL_SYSTEM_PROMPT
    assert "TUESDAY" in renderer.system_prompt
    assert "Task-Unifying Engine for Smart Decisions, Actions & You" in (
        renderer.system_prompt
    )


def test_custom_system_prompt_is_preserved_exactly() -> None:
    prompt = "  You are a specialised TUESDAY test assistant.  "
    renderer = ConversationalPromptRenderer(system_prompt=prompt)

    assert renderer.system_prompt == prompt


@pytest.mark.parametrize("prompt", ["", "   ", "\t\n"])
def test_empty_or_whitespace_system_prompt_is_rejected(prompt: str) -> None:
    with pytest.raises(ValueError, match="system prompt must not be empty"):
        ConversationalPromptRenderer(system_prompt=prompt)


@pytest.mark.parametrize("prompt", [None, 42, object()])
def test_non_string_system_prompt_is_rejected(prompt: object) -> None:
    with pytest.raises(TypeError, match="system prompt must be a string"):
        ConversationalPromptRenderer(system_prompt=prompt)  # type: ignore[arg-type]


def test_render_requires_tuesday_request() -> None:
    renderer = ConversationalPromptRenderer()
    context = ConversationContext()

    with pytest.raises(TypeError, match="request must be a TuesdayRequest"):
        renderer.render(object(), context)  # type: ignore[arg-type]


def test_render_requires_conversation_context() -> None:
    renderer = ConversationalPromptRenderer()
    request = make_request()

    with pytest.raises(TypeError, match="context must be a ConversationContext"):
        renderer.render(request, object())  # type: ignore[arg-type]


def test_render_rejects_mismatched_conversation_ids() -> None:
    renderer = ConversationalPromptRenderer()
    request = make_request()
    context = ConversationContext()

    with pytest.raises(
        ValueError,
        match="Request and context must belong to the same conversation",
    ):
        renderer.render(request, context)


def test_empty_context_renders_system_prompt_then_current_user_request() -> None:
    renderer = ConversationalPromptRenderer()
    request = make_request("Hello TUESDAY")
    context = make_context(request.conversation_id)

    rendered = renderer.render(request, context)

    assert isinstance(rendered, LanguageModelRequest)
    assert rendered.messages == (
        LanguageModelMessage(
            role=MessageRole.SYSTEM,
            content=DEFAULT_CONVERSATIONAL_SYSTEM_PROMPT,
        ),
        LanguageModelMessage(
            role=MessageRole.USER,
            content="Hello TUESDAY",
        ),
    )


def test_history_is_rendered_between_system_prompt_and_current_request() -> None:
    renderer = ConversationalPromptRenderer()
    request = make_request("What next?")
    prior_user = ConversationMessage(
        role=MessageRole.USER,
        content="Plan my afternoon",
    )
    prior_assistant = ConversationMessage(
        role=MessageRole.ASSISTANT,
        content="You have two open tasks.",
    )
    context = make_context(
        request.conversation_id,
        prior_user,
        prior_assistant,
    )

    rendered = renderer.render(request, context)

    assert [(message.role, message.content) for message in rendered.messages] == [
        (MessageRole.SYSTEM, DEFAULT_CONVERSATIONAL_SYSTEM_PROMPT),
        (MessageRole.USER, "Plan my afternoon"),
        (MessageRole.ASSISTANT, "You have two open tasks."),
        (MessageRole.USER, "What next?"),
    ]


def test_all_existing_domain_roles_are_preserved_from_context() -> None:
    renderer = ConversationalPromptRenderer()
    request = make_request("Continue")
    context = make_context(
        request.conversation_id,
        ConversationMessage(MessageRole.SYSTEM, "Prior system context"),
        ConversationMessage(MessageRole.USER, "Prior user context"),
        ConversationMessage(MessageRole.ASSISTANT, "Prior assistant context"),
    )

    rendered = renderer.render(request, context)

    assert [message.role for message in rendered.messages] == [
        MessageRole.SYSTEM,
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.USER,
    ]


def test_history_order_is_preserved_exactly() -> None:
    renderer = ConversationalPromptRenderer()
    request = make_request("Fourth")
    history = tuple(
        ConversationMessage(role, content)
        for role, content in (
            (MessageRole.USER, "First"),
            (MessageRole.ASSISTANT, "Second"),
            (MessageRole.USER, "Third"),
        )
    )
    context = make_context(request.conversation_id, *history)

    rendered = renderer.render(request, context)

    assert [message.content for message in rendered.messages[1:]] == [
        "First",
        "Second",
        "Third",
        "Fourth",
    ]


def test_meaningful_whitespace_and_multiline_content_are_preserved() -> None:
    renderer = ConversationalPromptRenderer(system_prompt="  System prompt  ")
    request = make_request("  Current request\nline two  ")
    prior = ConversationMessage(
        role=MessageRole.ASSISTANT,
        content="  Prior response\nline two  ",
    )
    context = make_context(request.conversation_id, prior)

    rendered = renderer.render(request, context)

    assert rendered.messages[0].content == "  System prompt  "
    assert rendered.messages[1].content == "  Prior response\nline two  "
    assert rendered.messages[2].content == "  Current request\nline two  "


def test_render_does_not_mutate_request_or_context() -> None:
    renderer = ConversationalPromptRenderer()
    request = make_request("Hello")
    prior = ConversationMessage(MessageRole.USER, "Earlier")
    messages = (prior,)
    context = ConversationContext(
        conversation_id=request.conversation_id,
        messages=messages,
    )
    original_request_id = request.request_id
    original_message_id = prior.message_id
    original_created_at = prior.created_at

    renderer.render(request, context)

    assert request.content == "Hello"
    assert request.request_id == original_request_id
    assert context.messages is messages
    assert context.messages[0] is prior
    assert prior.message_id == original_message_id
    assert prior.created_at == original_created_at


def test_rendered_model_messages_are_new_immutable_value_objects() -> None:
    renderer = ConversationalPromptRenderer()
    request = make_request()
    prior = ConversationMessage(MessageRole.USER, "Earlier")
    context = make_context(request.conversation_id, prior)

    rendered = renderer.render(request, context)

    assert isinstance(rendered.messages[1], LanguageModelMessage)
    assert rendered.messages[1] is not prior
    assert rendered.messages[1].role is prior.role
    assert rendered.messages[1].content == prior.content


def test_rendered_request_contains_only_language_model_input() -> None:
    renderer = ConversationalPromptRenderer()
    request = make_request()
    context = make_context(request.conversation_id)

    rendered = renderer.render(request, context)

    assert tuple(field.name for field in fields(rendered)) == ("messages",)
    for excluded in (
        "conversation_id",
        "request_id",
        "response_id",
        "provider",
        "model",
        "api_key",
    ):
        assert not hasattr(rendered, excluded)


def test_correlation_identifiers_and_timestamps_do_not_leak_into_content() -> None:
    renderer = ConversationalPromptRenderer()
    request = make_request("Hello")
    created_at = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
    prior = ConversationMessage(
        MessageRole.USER,
        "Earlier",
        created_at=created_at,
    )
    context = make_context(request.conversation_id, prior)

    rendered = renderer.render(request, context)
    rendered_content = "\n".join(message.content for message in rendered.messages)

    assert str(request.conversation_id) not in rendered_content
    assert str(request.request_id) not in rendered_content
    assert str(prior.message_id) not in rendered_content
    assert created_at.isoformat() not in rendered_content


def test_current_request_is_always_rendered_as_user_message() -> None:
    renderer = ConversationalPromptRenderer()
    request = make_request("Current")
    context = make_context(
        request.conversation_id,
        ConversationMessage(MessageRole.ASSISTANT, "Prior"),
    )

    rendered = renderer.render(request, context)

    assert rendered.messages[-1].role is MessageRole.USER
    assert rendered.messages[-1].content == "Current"


def test_malformed_context_member_is_rejected_explicitly() -> None:
    renderer = ConversationalPromptRenderer()
    request = make_request()
    context = ConversationContext(
        conversation_id=request.conversation_id,
        messages=(object(),),  # type: ignore[arg-type]
    )

    with pytest.raises(
        TypeError,
        match="context messages must be ConversationMessage instances",
    ):
        renderer.render(request, context)


def test_sequential_renders_are_independent() -> None:
    renderer = ConversationalPromptRenderer()
    first_request = make_request("First")
    second_request = make_request("Second")

    first = renderer.render(
        first_request,
        make_context(first_request.conversation_id),
    )
    second = renderer.render(
        second_request,
        make_context(second_request.conversation_id),
    )

    assert first is not second
    assert first.messages[-1].content == "First"
    assert second.messages[-1].content == "Second"
    assert first.messages is not second.messages


def test_renderer_performs_no_provider_execution() -> None:
    source = inspect.getsource(ConversationalPromptRenderer.render)

    assert ".generate(" not in source
    assert "responses.create" not in source
    assert "AsyncOpenAI" not in source


def test_prompting_module_has_only_provider_neutral_dependencies() -> None:
    tree = ast.parse(inspect.getsource(conversational_prompting))
    imported_modules = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert imported_modules == {
        "dataclasses",
        "tuesday.domain",
        "tuesday.language_models",
    }


def test_prompting_module_does_not_expose_provider_specific_types() -> None:
    import tuesday.prompting as prompting

    assert prompting.ConversationalPromptRenderer is ConversationalPromptRenderer
    assert (
        prompting.DEFAULT_CONVERSATIONAL_SYSTEM_PROMPT
        == DEFAULT_CONVERSATIONAL_SYSTEM_PROMPT
    )
    assert not hasattr(prompting, "OpenAILanguageModelProvider")
    assert not hasattr(prompting, "AsyncOpenAI")
