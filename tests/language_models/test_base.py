"""Tests for provider-neutral language-model contracts."""

import ast
import asyncio
import inspect
from dataclasses import FrozenInstanceError, fields

import pytest

import tuesday.language_models.base as language_model_base
from tuesday.domain import MessageRole
from tuesday.language_models import (
    BaseLanguageModelProvider,
    LanguageModelMessage,
    LanguageModelProviderError,
    LanguageModelRequest,
    LanguageModelResponse,
    LanguageModelToolCall,
    LanguageModelToolDefinition,
)


class StubLanguageModelProvider(BaseLanguageModelProvider):
    """Deterministic test-only provider implementation."""

    def __init__(self, response: LanguageModelResponse) -> None:
        self.response = response
        self.received_requests: list[LanguageModelRequest] = []

    @property
    def name(self) -> str:
        return "stub"

    async def generate(
        self,
        request: LanguageModelRequest,
    ) -> LanguageModelResponse:
        self.received_requests.append(request)
        return self.response


class ProviderWithoutName(BaseLanguageModelProvider):
    """Test provider intentionally missing the name property."""

    async def generate(
        self,
        request: LanguageModelRequest,
    ) -> LanguageModelResponse:
        raise AssertionError("An incomplete provider cannot execute.")


class ProviderWithoutGenerate(BaseLanguageModelProvider):
    """Test provider intentionally missing the generate method."""

    @property
    def name(self) -> str:
        return "incomplete"


@pytest.fixture
def user_message() -> LanguageModelMessage:
    return LanguageModelMessage(role=MessageRole.USER, content="Hello")


@pytest.fixture
def response() -> LanguageModelResponse:
    return LanguageModelResponse(
        content="Deterministic response",
        provider="stub",
        model="stub-model",
    )


def test_language_model_message_is_frozen_and_slotted() -> None:
    message = LanguageModelMessage(role=MessageRole.USER, content="Hello")

    assert not hasattr(message, "__dict__")
    with pytest.raises(FrozenInstanceError):
        message.content = "Changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "role",
    [MessageRole.USER, MessageRole.ASSISTANT, MessageRole.SYSTEM],
)
def test_message_accepts_and_preserves_domain_roles(role: MessageRole) -> None:
    message = LanguageModelMessage(role=role, content="Hello")

    assert message.role is role
    assert LanguageModelMessage.__annotations__["role"] is MessageRole


@pytest.mark.parametrize("role", ["user", None, object()])
def test_message_rejects_non_domain_roles(role: object) -> None:
    with pytest.raises(TypeError, match="role must be a MessageRole"):
        LanguageModelMessage(role=role, content="Hello")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "content",
    ["Hello", "  Hello  ", "Line one\nLine two"],
)
def test_message_accepts_and_preserves_meaningful_content(content: str) -> None:
    message = LanguageModelMessage(role=MessageRole.USER, content=content)

    assert message.content == content


@pytest.mark.parametrize("content", ["", "   ", "\t\n"])
def test_message_rejects_empty_or_whitespace_content(content: str) -> None:
    with pytest.raises(ValueError, match="content must not be empty"):
        LanguageModelMessage(role=MessageRole.USER, content=content)


@pytest.mark.parametrize("content", [None, 42, object()])
def test_message_rejects_non_string_content(content: object) -> None:
    with pytest.raises(TypeError, match="content must be a string"):
        LanguageModelMessage(
            role=MessageRole.USER,
            content=content,  # type: ignore[arg-type]
        )


def test_language_model_request_is_frozen_and_slotted(
    user_message: LanguageModelMessage,
) -> None:
    request = LanguageModelRequest(messages=(user_message,))

    assert not hasattr(request, "__dict__")
    with pytest.raises(FrozenInstanceError):
        request.messages = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        request.tools = ()  # type: ignore[misc]


def test_request_preserves_exact_tuple_and_message_identities(
    user_message: LanguageModelMessage,
) -> None:
    system_message = LanguageModelMessage(
        role=MessageRole.SYSTEM,
        content="You are TUESDAY.",
    )
    messages = (system_message, user_message)

    request = LanguageModelRequest(messages=messages)

    assert request.messages is messages
    assert request.messages[0] is system_message
    assert request.messages[1] is user_message
    assert system_message.content == "You are TUESDAY."
    assert user_message.content == "Hello"


def test_request_accepts_one_message_tuple(
    user_message: LanguageModelMessage,
) -> None:
    request = LanguageModelRequest(messages=(user_message,))

    assert request.messages == (user_message,)


def test_request_defaults_to_empty_tools_tuple(
    user_message: LanguageModelMessage,
) -> None:
    request = LanguageModelRequest(messages=(user_message,))

    assert request.tools == ()
    assert isinstance(request.tools, tuple)


def test_request_accepts_explicit_empty_tools_tuple(
    user_message: LanguageModelMessage,
) -> None:
    request = LanguageModelRequest(messages=(user_message,), tools=())

    assert request.tools == ()


def test_request_preserves_tool_tuple_identities_and_order(
    user_message: LanguageModelMessage,
) -> None:
    first = LanguageModelToolDefinition(
        name="calendar.lookup",
        description="Look up calendar events.",
        parameters={},
    )
    second = LanguageModelToolDefinition(
        name="calculator.basic",
        description="Perform basic arithmetic.",
        parameters={},
    )
    tools = (first, second)

    request = LanguageModelRequest(messages=(user_message,), tools=tools)

    assert request.tools is tools
    assert request.tools == (first, second)
    assert request.tools[0] is first
    assert request.tools[1] is second


@pytest.mark.parametrize("invalid_tools", [[], set(), {}, None, object()])
def test_request_tools_must_be_tuple(
    user_message: LanguageModelMessage,
    invalid_tools: object,
) -> None:
    with pytest.raises(TypeError, match="tools must be a tuple"):
        LanguageModelRequest(
            messages=(user_message,),
            tools=invalid_tools,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_tool",
    [
        object(),
        None,
        "calculator.basic",
        {},
        LanguageModelToolCall(
            call_id="call_1",
            name="calculator.basic",
            arguments={},
        ),
    ],
)
def test_request_tools_require_definition_instances(
    user_message: LanguageModelMessage,
    invalid_tool: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be LanguageModelToolDefinition instances",
    ):
        LanguageModelRequest(
            messages=(user_message,),
            tools=(invalid_tool,),  # type: ignore[arg-type]
        )


def test_request_rejects_exact_duplicate_tool_names(
    user_message: LanguageModelMessage,
) -> None:
    first = LanguageModelToolDefinition(
        name="calculator.basic",
        description="First description.",
        parameters={"version": 1},
    )
    second = LanguageModelToolDefinition(
        name="calculator.basic",
        description="Different description.",
        parameters={"version": 2},
    )

    with pytest.raises(ValueError, match="tool names must be unique"):
        LanguageModelRequest(
            messages=(user_message,),
            tools=(first, second),
        )


def test_request_allows_case_distinct_tool_names(
    user_message: LanguageModelMessage,
) -> None:
    lower = LanguageModelToolDefinition(
        name="calculator.basic",
        description="Lowercase capability.",
        parameters={},
    )
    upper = LanguageModelToolDefinition(
        name="Calculator.Basic",
        description="Uppercase capability.",
        parameters={},
    )

    request = LanguageModelRequest(
        messages=(user_message,),
        tools=(lower, upper),
    )

    assert request.tools == (lower, upper)
    assert request.tools[0] is lower
    assert request.tools[1] is upper


def test_request_rejects_empty_tuple() -> None:
    with pytest.raises(ValueError, match="must contain a message"):
        LanguageModelRequest(messages=())


@pytest.mark.parametrize("messages", [[], set(), object()])
def test_request_rejects_non_tuple_messages(messages: object) -> None:
    with pytest.raises(TypeError, match="messages must be a tuple"):
        LanguageModelRequest(messages=messages)  # type: ignore[arg-type]


def test_request_rejects_tuple_containing_non_message() -> None:
    with pytest.raises(TypeError, match="must be LanguageModelMessage instances"):
        LanguageModelRequest(messages=(object(),))  # type: ignore[arg-type]


def test_request_has_only_model_input_fields(
    user_message: LanguageModelMessage,
) -> None:
    request = LanguageModelRequest(messages=(user_message,))

    assert tuple(field.name for field in fields(request)) == ("messages", "tools")
    for excluded_field in (
        "conversation_id",
        "request_id",
        "response_id",
        "provider",
        "model",
        "api_key",
        "timeout",
        "temperature",
        "invocation_id",
        "authorization",
        "confirmed",
    ):
        assert not hasattr(request, excluded_field)


def test_language_model_response_is_frozen_and_slotted() -> None:
    response = LanguageModelResponse(
        content="Hello",
        provider="stub",
        model="stub-model",
    )

    assert not hasattr(response, "__dict__")
    with pytest.raises(FrozenInstanceError):
        response.content = "Changed"  # type: ignore[misc]


def test_response_preserves_meaningful_values_exactly() -> None:
    response = LanguageModelResponse(
        content="  Hello  ",
        provider="CustomProvider",
        model="organisation/custom-model",
    )

    assert response.content == "  Hello  "
    assert response.provider == "CustomProvider"
    assert response.model == "organisation/custom-model"
    assert tuple(field.name for field in fields(response)) == (
        "content",
        "provider",
        "model",
    )


@pytest.mark.parametrize("content", ["", "   ", "\t\n"])
def test_response_rejects_empty_or_whitespace_content(content: str) -> None:
    with pytest.raises(ValueError, match="content must not be empty"):
        LanguageModelResponse(
            content=content,
            provider="stub",
            model="stub-model",
        )


def test_response_rejects_non_string_content() -> None:
    with pytest.raises(TypeError, match="content must be a string"):
        LanguageModelResponse(
            content=42,  # type: ignore[arg-type]
            provider="stub",
            model="stub-model",
        )


@pytest.mark.parametrize("provider", ["", "   ", "\t\n"])
def test_response_rejects_empty_or_whitespace_provider(provider: str) -> None:
    with pytest.raises(ValueError, match="provider must not be empty"):
        LanguageModelResponse(
            content="Hello",
            provider=provider,
            model="stub-model",
        )


@pytest.mark.parametrize("provider", [" stub", "stub "])
def test_response_rejects_provider_surrounding_whitespace(provider: str) -> None:
    with pytest.raises(ValueError, match="provider must not have surrounding"):
        LanguageModelResponse(
            content="Hello",
            provider=provider,
            model="stub-model",
        )


def test_response_rejects_non_string_provider() -> None:
    with pytest.raises(TypeError, match="provider must be a string"):
        LanguageModelResponse(
            content="Hello",
            provider=42,  # type: ignore[arg-type]
            model="stub-model",
        )


@pytest.mark.parametrize("model", ["", "   ", "\t\n"])
def test_response_rejects_empty_or_whitespace_model(model: str) -> None:
    with pytest.raises(ValueError, match="model must not be empty"):
        LanguageModelResponse(content="Hello", provider="stub", model=model)


@pytest.mark.parametrize("model", [" stub-model", "stub-model "])
def test_response_rejects_model_surrounding_whitespace(model: str) -> None:
    with pytest.raises(ValueError, match="model must not have surrounding"):
        LanguageModelResponse(content="Hello", provider="stub", model=model)


def test_response_rejects_non_string_model() -> None:
    with pytest.raises(TypeError, match="model must be a string"):
        LanguageModelResponse(
            content="Hello",
            provider="stub",
            model=42,  # type: ignore[arg-type]
        )


def test_base_provider_is_abstract_and_incomplete_subclasses_fail() -> None:
    assert inspect.isabstract(BaseLanguageModelProvider)
    with pytest.raises(TypeError):
        BaseLanguageModelProvider()
    with pytest.raises(TypeError):
        ProviderWithoutName()
    with pytest.raises(TypeError):
        ProviderWithoutGenerate()


def test_complete_provider_exposes_name_and_async_generate(
    response: LanguageModelResponse,
) -> None:
    provider = StubLanguageModelProvider(response)

    assert isinstance(provider, BaseLanguageModelProvider)
    assert provider.name == "stub"
    assert inspect.iscoroutinefunction(BaseLanguageModelProvider.generate)
    assert inspect.iscoroutinefunction(StubLanguageModelProvider.generate)


def test_generate_receives_exact_request_and_returns_exact_response(
    user_message: LanguageModelMessage,
    response: LanguageModelResponse,
) -> None:
    request = LanguageModelRequest(messages=(user_message,))
    provider = StubLanguageModelProvider(response)

    returned = asyncio.run(provider.generate(request))

    assert provider.received_requests == [request]
    assert provider.received_requests[0] is request
    assert returned is response
    assert isinstance(returned, LanguageModelResponse)


def test_sequential_generate_calls_remain_independent(
    response: LanguageModelResponse,
) -> None:
    first = LanguageModelRequest(
        messages=(LanguageModelMessage(MessageRole.USER, "First"),)
    )
    second = LanguageModelRequest(
        messages=(LanguageModelMessage(MessageRole.USER, "Second"),)
    )
    provider = StubLanguageModelProvider(response)

    first_response = asyncio.run(provider.generate(first))
    second_response = asyncio.run(provider.generate(second))

    assert provider.received_requests == [first, second]
    assert first_response is response
    assert second_response is response


def test_provider_contract_has_no_shared_mutable_state(
    response: LanguageModelResponse,
) -> None:
    first = StubLanguageModelProvider(response)
    second = StubLanguageModelProvider(response)

    first.received_requests.append(
        LanguageModelRequest(
            messages=(LanguageModelMessage(MessageRole.USER, "Hello"),)
        )
    )

    assert len(first.received_requests) == 1
    assert second.received_requests == []


def test_provider_error_is_stable_runtime_error_contract() -> None:
    assert issubclass(LanguageModelProviderError, RuntimeError)
    assert isinstance(LanguageModelProviderError("provider failed"), RuntimeError)


def test_contract_module_has_only_allowed_dependencies() -> None:
    tree = ast.parse(inspect.getsource(language_model_base))
    imported_modules = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert imported_modules == {
        "abc",
        "dataclasses",
        "tuesday.domain",
        "tuesday.language_models.tools",
    }
