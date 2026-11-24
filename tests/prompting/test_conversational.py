"""Tests for the provider-neutral conversational prompt renderer."""

import ast
import inspect
from dataclasses import FrozenInstanceError, fields, replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

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
@pytest.mark.parametrize("limit", [None, 0, 1])
def test_empty_or_whitespace_system_prompt_is_rejected(prompt: str, limit) -> None:
    with pytest.raises(ValueError, match="system prompt must not be empty"):
        ConversationalPromptRenderer(system_prompt=prompt, max_history_messages=limit)


@pytest.mark.parametrize("prompt", [None, 42, object()])
@pytest.mark.parametrize("limit", [None, 0, 1])
def test_non_string_system_prompt_is_rejected(prompt: object, limit) -> None:
    with pytest.raises(TypeError, match="system prompt must be a string"):
        ConversationalPromptRenderer(
            system_prompt=prompt, max_history_messages=limit  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("limit", [None, 0, 1])
def test_render_requires_tuesday_request(limit) -> None:
    renderer = ConversationalPromptRenderer(max_history_messages=limit)
    context = ConversationContext()

    with pytest.raises(TypeError, match="request must be a TuesdayRequest"):
        renderer.render(object(), context)  # type: ignore[arg-type]


@pytest.mark.parametrize("limit", [None, 0, 1])
def test_render_requires_conversation_context(limit) -> None:
    renderer = ConversationalPromptRenderer(max_history_messages=limit)
    request = make_request()

    with pytest.raises(TypeError, match="context must be a ConversationContext"):
        renderer.render(request, object())  # type: ignore[arg-type]


@pytest.mark.parametrize("limit", [None, 0, 1])
def test_render_rejects_mismatched_conversation_ids(limit) -> None:
    renderer = ConversationalPromptRenderer(max_history_messages=limit)
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

    assert tuple(field.name for field in fields(rendered)) == ("messages", "tools")
    assert rendered.tools == ()
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


@pytest.fixture
def deterministic_history():
    return tuple(
        ConversationMessage(
            role=role,
            content=content,
            message_id=UUID(int=index + 1),
            # Deliberately reverse timestamps: tuple order governs selection.
            created_at=datetime(2026, 11, 24, tzinfo=UTC) - timedelta(minutes=index),
        )
        for index, (role, content) in enumerate(
            (
                (MessageRole.SYSTEM, "  Old system\ncontext  "),
                (MessageRole.ASSISTANT, "First answer"),
                (MessageRole.USER, "  Second question\nline two  "),
                (MessageRole.ASSISTANT, "Second answer"),
                (MessageRole.SYSTEM, "  Recent system\ncontext  "),
                (MessageRole.ASSISTANT, "  Latest answer\nline two  "),
            )
        )
    )


@pytest.mark.parametrize("limit", [None, 0, 1, 2, 100])
def test_history_limit_accepts_nonnegative_int_or_none_without_coercion(limit):
    renderer = ConversationalPromptRenderer(max_history_messages=limit)

    assert renderer.max_history_messages is limit


@pytest.mark.parametrize("invalid", [True, False, 1.5, "3", [], {}, object()])
def test_history_limit_rejects_non_integer_types(invalid):
    with pytest.raises(TypeError, match="max_history_messages must be an int or None"):
        ConversationalPromptRenderer(max_history_messages=invalid)


@pytest.mark.parametrize("invalid", [-1, -100])
def test_history_limit_rejects_negative_integers(invalid):
    with pytest.raises(ValueError, match="max_history_messages must be nonnegative"):
        ConversationalPromptRenderer(max_history_messages=invalid)


def test_history_limit_is_keyword_only_and_system_prompt_remains_positional():
    parameters = inspect.signature(ConversationalPromptRenderer).parameters
    assert tuple(parameters) == ("system_prompt", "max_history_messages")
    assert parameters["system_prompt"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["max_history_messages"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["max_history_messages"].default is None
    assert ConversationalPromptRenderer().max_history_messages is None
    positional = ConversationalPromptRenderer("Custom prompt")
    assert positional.system_prompt == "Custom prompt"
    renderer = ConversationalPromptRenderer("Custom prompt", max_history_messages=2)
    assert renderer.system_prompt == "Custom prompt"
    assert renderer.max_history_messages == 2
    with pytest.raises(TypeError):
        ConversationalPromptRenderer("Custom prompt", 2)


@pytest.mark.parametrize(
    "limit,indices",
    [
        (None, (0, 1, 2, 3, 4, 5)),
        (0, ()),
        (1, (5,)),
        (2, (4, 5)),
        (4, (2, 3, 4, 5)),
        (20, (0, 1, 2, 3, 4, 5)),
    ],
)
@pytest.mark.parametrize("empty", [False, True])
def test_history_window_preserves_roles_text_order_and_prompt_boundaries(
    deterministic_history, limit, indices, empty
):
    renderer = ConversationalPromptRenderer(
        "  Configured system\nprompt  ", max_history_messages=limit
    )
    request = make_request("  Current user request\nline two  ")
    context = make_context(
        request.conversation_id, *(() if empty else deterministic_history)
    )

    rendered = renderer.render(request, context)

    expected = () if empty else tuple(deterministic_history[index] for index in indices)
    assert [(message.role, message.content) for message in rendered.messages] == [
        (MessageRole.SYSTEM, renderer.system_prompt),
        *((message.role, message.content) for message in expected),
        (MessageRole.USER, request.content),
    ]
    assert len(rendered.messages) == len(expected) + 2
    assert rendered.tools == ()


def test_bounded_rendering_preserves_source_values_and_immutable_new_messages(
    deterministic_history,
):
    renderer = ConversationalPromptRenderer(max_history_messages=2)
    request = make_request("  Continue\nplease  ")
    context = ConversationContext(request.conversation_id, deterministic_history)
    original_request, original_context = replace(request), replace(context)
    original_messages = tuple(replace(message) for message in context.messages)

    first = renderer.render(request, context)
    second = renderer.render(request, context)

    assert not hasattr(renderer, "__dict__")
    with pytest.raises(FrozenInstanceError):
        renderer.max_history_messages = 0
    assert request == original_request
    assert context == original_context
    assert context.messages is deterministic_history
    assert context.messages == original_messages
    for original, retained in zip(deterministic_history, context.messages, strict=True):
        assert retained is original
    assert first == second and first is not second
    for selected, rendered, repeated in zip(
        deterministic_history[-2:],
        first.messages[1:-1],
        second.messages[1:-1],
        strict=True,
    ):
        assert type(rendered) is LanguageModelMessage
        assert rendered is not selected and rendered is not repeated
        assert (rendered.role, rendered.content) == (selected.role, selected.content)
        with pytest.raises(FrozenInstanceError):
            rendered.content = "Changed"


@pytest.mark.parametrize("limit", [0, 1])
def test_omitted_invalid_history_member_still_fails_validation(limit):
    renderer = ConversationalPromptRenderer(max_history_messages=limit)
    request = make_request()
    context = make_context(
        request.conversation_id,
        object(),
        ConversationMessage(MessageRole.ASSISTANT, "Latest valid reply"),
    )
    with pytest.raises(
        TypeError, match="context messages must be ConversationMessage instances"
    ):
        renderer.render(request, context)


def test_interleaved_bounded_renders_use_only_the_supplied_context(
    deterministic_history,
):
    renderer = ConversationalPromptRenderer(max_history_messages=2)
    request_a, request_b = make_request("A current"), make_request("B current")
    context_a = make_context(request_a.conversation_id, *deterministic_history)
    history_b = tuple(
        replace(message, content="B " + message.content)
        for message in deterministic_history
    )
    context_b = make_context(request_b.conversation_id, *history_b)
    first_a = renderer.render(request_a, context_a)
    rendered_b = renderer.render(request_b, context_b)
    second_a = renderer.render(request_a, context_a)

    assert first_a == second_a
    assert [message.content for message in first_a.messages[1:-1]] == [
        message.content for message in deterministic_history[-2:]
    ]
    assert [message.content for message in rendered_b.messages[1:-1]] == [
        message.content for message in history_b[-2:]
    ]
    assert tuple(field.name for field in fields(renderer)) == (
        "system_prompt",
        "max_history_messages",
    )
