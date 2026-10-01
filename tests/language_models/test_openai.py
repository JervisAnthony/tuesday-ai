"""Tests for the OpenAI language-model provider adapter."""

import ast
import asyncio
import inspect
import json
import tomllib
from pathlib import Path
from types import MappingProxyType

import pytest
from openai import OpenAIError

import tuesday.language_models.openai as openai_adapter
from tuesday.config import ConfigurationError, LanguageModelSettings
from tuesday.domain import MessageRole
from tuesday.language_models import (
    BaseLanguageModelProvider,
    LanguageModelMessage,
    LanguageModelProviderError,
    LanguageModelRequest,
    LanguageModelResponse,
    LanguageModelToolCall,
    LanguageModelToolDefinition,
    OpenAILanguageModelProvider,
)


class FakeOutputItem:
    """Small output item fake; call_id deliberately differs from the item ID."""

    def __init__(
        self,
        type: str = "function_call",
        *,
        call_id: object = "call_1",
        name: object = "calculator.basic",
        arguments: object = '{"operation":"multiply","left":6,"right":7}',
    ) -> None:
        self.type = type
        self.id = "fc_separate_item_id"
        self.call_id = call_id
        self.name = name
        self.arguments = arguments


class FakeOpenAIResponse:
    """Minimal fake of the public OpenAI response values the adapter reads."""

    def __init__(
        self,
        output_text: object,
        model: str = "returned-model",
        *,
        output: tuple[FakeOutputItem, ...] = (),
    ) -> None:
        self.output_text = output_text
        self.model = model
        self.output = output


class FakeResponses:
    """Record Responses API calls and return or raise configured behavior."""

    def __init__(
        self,
        response: FakeOpenAIResponse | None = None,
        error: OpenAIError | None = None,
    ) -> None:
        self.response = response or FakeOpenAIResponse("Generated text")
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> FakeOpenAIResponse:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeAsyncOpenAI:
    """Minimal fake client exposing only the Responses API resource."""

    def __init__(self, responses: FakeResponses | None = None) -> None:
        self.responses = responses or FakeResponses()


def make_settings(**overrides: object) -> LanguageModelSettings:
    values: dict[str, object] = {
        "provider": "openai",
        "model": "example-model",
        "api_key": "test-openai-key",
        "timeout_seconds": 30.0,
        "temperature": None,
    }
    values.update(overrides)
    return LanguageModelSettings(**values)  # type: ignore[arg-type]


def make_request(
    *messages: LanguageModelMessage,
) -> LanguageModelRequest:
    if not messages:
        messages = (LanguageModelMessage(MessageRole.USER, "Hello"),)
    return LanguageModelRequest(messages=messages)


def run_generate(
    provider: OpenAILanguageModelProvider,
    request: LanguageModelRequest | None = None,
) -> LanguageModelResponse:
    return asyncio.run(provider.generate(request or make_request()))


def run_output(output_text: object, *items: FakeOutputItem) -> LanguageModelResponse:
    responses = FakeResponses(FakeOpenAIResponse(output_text, output=items))
    provider = OpenAILanguageModelProvider(
        make_settings(),
        client=FakeAsyncOpenAI(responses),  # type: ignore[arg-type]
    )
    return run_generate(provider)


def test_provider_implements_async_base_contract_and_exact_name() -> None:
    provider = OpenAILanguageModelProvider(
        make_settings(),
        client=FakeAsyncOpenAI(),  # type: ignore[arg-type]
    )

    assert isinstance(provider, BaseLanguageModelProvider)
    assert provider.name == "openai"
    assert inspect.iscoroutinefunction(OpenAILanguageModelProvider.generate)


def test_valid_settings_and_injected_client_are_retained_exactly() -> None:
    settings = make_settings(
        model="organisation/custom-model",
        timeout_seconds=45.5,
        temperature=0.2,
    )
    client = FakeAsyncOpenAI()

    provider = OpenAILanguageModelProvider(
        settings,
        client=client,  # type: ignore[arg-type]
    )

    assert provider._settings is settings
    assert provider._client is client
    assert provider._settings.model == "organisation/custom-model"
    assert provider._settings.timeout_seconds == 45.5
    assert provider._settings.temperature == 0.2


@pytest.mark.parametrize("provider_name", ["anthropic", "OpenAI", "OPENAI"])
def test_non_openai_provider_is_rejected_without_normalization(
    provider_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="requires model settings with provider 'openai'",
    ):
        OpenAILanguageModelProvider(make_settings(provider=provider_name))


def test_missing_api_key_is_rejected_without_environment_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")

    with pytest.raises(
        ConfigurationError,
        match="requires TUESDAY_MODEL_API_KEY",
    ) as raised:
        OpenAILanguageModelProvider(make_settings(api_key=None))

    assert "environment-secret" not in str(raised.value)


@pytest.mark.parametrize("temperature", [0.0, 2.0])
def test_openai_temperature_boundaries_are_accepted(temperature: float) -> None:
    provider = OpenAILanguageModelProvider(
        make_settings(temperature=temperature),
        client=FakeAsyncOpenAI(),  # type: ignore[arg-type]
    )

    assert provider._settings.temperature == temperature


def test_temperature_above_openai_limit_is_rejected() -> None:
    with pytest.raises(
        ConfigurationError,
        match="OpenAI temperature must be between 0 and 2",
    ):
        OpenAILanguageModelProvider(make_settings(temperature=2.1))


def test_provider_representation_does_not_expose_api_key() -> None:
    secret = "test-openai-key"
    provider = OpenAILanguageModelProvider(
        make_settings(api_key=secret),
        client=FakeAsyncOpenAI(),  # type: ignore[arg-type]
    )

    assert secret not in repr(provider)
    assert secret not in repr(provider._settings)


def test_default_client_is_constructed_once_with_explicit_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeAsyncOpenAI()
    construction_calls: list[dict[str, object]] = []

    def fake_async_openai(**kwargs: object) -> FakeAsyncOpenAI:
        construction_calls.append(kwargs)
        return client

    monkeypatch.setattr(openai_adapter, "AsyncOpenAI", fake_async_openai)
    settings = make_settings(timeout_seconds=45.0)

    provider = OpenAILanguageModelProvider(settings)

    assert provider._client is client
    assert construction_calls == [
        {
            "api_key": "test-openai-key",
            "timeout": 45.0,
            "max_retries": 0,
        }
    ]
    assert client.responses.calls == []


@pytest.mark.parametrize(
    ("role", "expected_role"),
    [
        (MessageRole.SYSTEM, "system"),
        (MessageRole.USER, "user"),
        (MessageRole.ASSISTANT, "assistant"),
    ],
)
def test_each_domain_role_translates_through_its_exact_value(
    role: MessageRole,
    expected_role: str,
) -> None:
    message = LanguageModelMessage(role, "Hello")
    request = make_request(message)
    responses = FakeResponses()
    provider = OpenAILanguageModelProvider(
        make_settings(),
        client=FakeAsyncOpenAI(responses),  # type: ignore[arg-type]
    )

    run_generate(provider, request)

    assert responses.calls[0]["input"] == [
        {"role": expected_role, "content": "Hello"}
    ]


def test_multi_message_order_and_exact_content_are_preserved() -> None:
    messages = (
        LanguageModelMessage(MessageRole.SYSTEM, "You are TUESDAY."),
        LanguageModelMessage(MessageRole.USER, "  Hello  "),
        LanguageModelMessage(MessageRole.ASSISTANT, "Line one\nLine two"),
        LanguageModelMessage(MessageRole.USER, "Continue"),
    )
    request = LanguageModelRequest(messages=messages)
    responses = FakeResponses()
    provider = OpenAILanguageModelProvider(
        make_settings(),
        client=FakeAsyncOpenAI(responses),  # type: ignore[arg-type]
    )

    run_generate(provider, request)

    assert responses.calls[0]["input"] == [
        {"role": "system", "content": "You are TUESDAY."},
        {"role": "user", "content": "  Hello  "},
        {"role": "assistant", "content": "Line one\nLine two"},
        {"role": "user", "content": "Continue"},
    ]
    assert request.messages is messages
    assert all(
        request.messages[index] is message
        for index, message in enumerate(messages)
    )
    assert messages[1].content == "  Hello  "
    assert messages[2].content == "Line one\nLine two"


def test_responses_api_call_is_stateless_non_streaming_and_minimal() -> None:
    responses = FakeResponses()
    provider = OpenAILanguageModelProvider(
        make_settings(),
        client=FakeAsyncOpenAI(responses),  # type: ignore[arg-type]
    )

    run_generate(provider)

    assert responses.calls == [
        {
            "model": "example-model",
            "input": [{"role": "user", "content": "Hello"}],
            "store": False,
            "stream": False,
        }
    ]
    call = responses.calls[0]
    for excluded_argument in (
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "max_tool_calls",
        "previous_response_id",
        "conversation",
        "conversation_id",
        "request_id",
        "response_format",
    ):
        assert excluded_argument not in call


@pytest.mark.parametrize("output_text", [None, "  I'll calculate that.  "])
def test_tool_definition_serialization_and_function_call_translation(
    output_text: str | None,
) -> None:
    schema = {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["add", "multiply"]},
            "left": {"type": "number"},
            "right": {"type": "number"},
        },
        "required": ["operation", "left", "right"],
        "additionalProperties": False,
        "metadata": {"nullable": None, "default": 1.5, "version": 2},
    }
    definition = LanguageModelToolDefinition(
        name="calculator.basic",
        description="Perform basic\n arithmetic.",
        parameters=schema,
    )
    request = LanguageModelRequest(
        messages=(LanguageModelMessage(MessageRole.USER, "Calculate"),),
        tools=(definition,),
    )
    calls = request.tools
    parameters = definition.parameters
    properties = parameters["properties"]
    required = parameters["required"]
    responses = FakeResponses(
        FakeOpenAIResponse(output_text, output=(FakeOutputItem(),))
    )
    provider = OpenAILanguageModelProvider(
        make_settings(),
        client=FakeAsyncOpenAI(responses),  # type: ignore[arg-type]
    )

    response = run_generate(provider, request)

    assert len(responses.calls) == 1
    payload = responses.calls[0]
    assert payload == {
        "model": "example-model",
        "input": [{"role": "user", "content": "Calculate"}],
        "store": False,
        "stream": False,
        "tools": [
            {
                "type": "function",
                "name": "calculator.basic",
                "description": "Perform basic\n arithmetic.",
                "parameters": schema,
            }
        ],
    }
    serialized = payload["tools"][0]["parameters"]
    assert type(serialized) is dict
    assert type(serialized["properties"]) is dict
    assert type(serialized["properties"]["operation"]) is dict
    assert type(serialized["properties"]["operation"]["enum"]) is list
    assert type(serialized["required"]) is list
    for key, value in schema["metadata"].items():
        assert type(serialized["metadata"][key]) is type(value)
    assert json.loads(json.dumps(serialized)) == schema
    assert response.content == output_text
    assert response.provider == "openai"
    assert response.model == "returned-model"
    assert isinstance(response.tool_calls, tuple)
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert isinstance(call, LanguageModelToolCall)
    assert call.call_id == "call_1"
    assert call.name == "calculator.basic"
    assert call.arguments == {"operation": "multiply", "left": 6, "right": 7}

    serialized["properties"]["operation"]["enum"].append("changed")
    serialized["required"].append("changed")
    serialized["metadata"]["version"] = 99
    assert request.tools is calls
    assert request.tools[0] is definition
    assert definition.parameters is parameters
    assert parameters["properties"] is properties
    assert parameters["required"] is required
    assert properties["operation"]["enum"] == ("add", "multiply")
    assert required == ("operation", "left", "right")
    assert parameters["metadata"]["version"] == 2
    with pytest.raises(TypeError):
        properties["operation"]["type"] = "changed"


def test_multiple_tool_definitions_preserve_exact_order_and_are_independent() -> None:
    first = LanguageModelToolDefinition("Z.lookup", "Look up data.", {})
    second = LanguageModelToolDefinition("a.calculate", "Calculate.", {})
    request = LanguageModelRequest(make_request().messages, (first, second))
    responses = FakeResponses()
    provider = OpenAILanguageModelProvider(
        make_settings(temperature=0.2),
        client=FakeAsyncOpenAI(responses),  # type: ignore[arg-type]
    )

    run_generate(provider, request)
    run_generate(provider, request)
    run_generate(provider)

    for payload in responses.calls[:2]:
        assert payload["tools"] == [
            {
                "type": "function",
                "name": "Z.lookup",
                "description": "Look up data.",
                "parameters": {},
            },
            {
                "type": "function",
                "name": "a.calculate",
                "description": "Calculate.",
                "parameters": {},
            },
        ]
        assert payload["temperature"] == 0.2
        assert "tool_choice" not in payload
        assert "parallel_tool_calls" not in payload
        assert "max_tool_calls" not in payload
    assert responses.calls[0]["tools"] is not responses.calls[1]["tools"]
    assert responses.calls[0]["tools"][0] is not responses.calls[0]["tools"][1]
    assert (
        responses.calls[0]["tools"][0]["parameters"]
        is not responses.calls[1]["tools"][0]["parameters"]
    )
    assert "tools" not in responses.calls[2]


@pytest.mark.parametrize("output_text", [None, "", "   ", "\t\n"])
def test_valid_function_call_without_meaningful_text(output_text: object) -> None:
    response = run_output(output_text, FakeOutputItem())

    assert response.content is None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].call_id == "call_1"


def test_multiple_calls_preserve_order_and_ignore_unrelated_items() -> None:
    response = run_output(
        "  Mixed response  ",
        FakeOutputItem("reasoning"),
        FakeOutputItem(call_id="z_call", name="Z.unknown", arguments='{"value":1}'),
        FakeOutputItem("message"),
        FakeOutputItem(call_id="a_call", name="a.unknown", arguments='{"value":2}'),
        FakeOutputItem("provider_metadata"),
        FakeOutputItem("Function_call"),
    )

    assert response.content == "  Mixed response  "
    assert [
        (call.call_id, call.name, dict(call.arguments)) for call in response.tool_calls
    ] == [
        ("z_call", "Z.unknown", {"value": 1}),
        ("a_call", "a.unknown", {"value": 2}),
    ]


def test_nested_arguments_delegate_freezing_to_model_tool_call() -> None:
    response = run_output(
        None,
        FakeOutputItem(
            name="provider.unknown",
            arguments=('{"options":{"items":[1,2.5,"text",true,null,{"ok":false}]}}'),
        ),
    )
    call = response.tool_calls[0]
    assert call.name == "provider.unknown"
    assert isinstance(call.arguments, MappingProxyType)
    assert isinstance(call.arguments["options"], MappingProxyType)
    items = call.arguments["options"]["items"]
    assert items == (1, 2.5, "text", True, None, {"ok": False})
    assert isinstance(items, tuple)
    assert isinstance(items[-1], MappingProxyType)
    with pytest.raises(TypeError):
        items[-1]["ok"] = True


def test_malformed_json_has_safe_error_and_chained_cause() -> None:
    arguments = '{"private_left":'

    with pytest.raises(LanguageModelProviderError) as caught:
        run_output(None, FakeOutputItem(arguments=arguments))

    assert str(caught.value) == "OpenAI provider returned invalid tool call arguments."
    assert arguments not in str(caught.value)
    assert isinstance(caught.value.__cause__, json.JSONDecodeError)


@pytest.mark.parametrize("arguments", [None, 42, {}, [], True, b"{}"])
def test_non_string_arguments_fail_safely(arguments: object) -> None:
    with pytest.raises(LanguageModelProviderError) as caught:
        run_output(None, FakeOutputItem(arguments=arguments))

    assert str(caught.value) == "OpenAI provider returned invalid tool call arguments."


@pytest.mark.parametrize(
    "arguments", ["[]", "[1,2]", "42", '"hello"', "true", "false", "null"]
)
def test_non_object_json_arguments_fail_safely(arguments: str) -> None:
    with pytest.raises(LanguageModelProviderError) as caught:
        run_output(None, FakeOutputItem(arguments=arguments))

    assert str(caught.value) == "OpenAI provider returned invalid tool call arguments."


@pytest.mark.parametrize("field", ["call_id", "name"])
@pytest.mark.parametrize("value", ["", " ", " private_value ", None, 42])
def test_invalid_call_metadata_has_safe_error_and_cause(
    field: str,
    value: object,
) -> None:
    with pytest.raises(LanguageModelProviderError) as caught:
        run_output(None, FakeOutputItem(**{field: value}))

    assert str(caught.value) == "OpenAI provider returned an invalid tool call."
    assert isinstance(caught.value.__cause__, (TypeError, ValueError))


@pytest.mark.parametrize(
    "arguments", ['{"value":NaN}', '{"value":Infinity}', '{"value":1e999}']
)
def test_invalid_recursive_arguments_fail_through_domain_validation(
    arguments: str,
) -> None:
    with pytest.raises(LanguageModelProviderError) as caught:
        run_output(None, FakeOutputItem(arguments=arguments))

    assert str(caught.value) == "OpenAI provider returned an invalid tool call."
    assert isinstance(caught.value.__cause__, ValueError)


@pytest.mark.parametrize("with_call", [False, True])
@pytest.mark.parametrize("output_text", [42, True, {}, []])
def test_invalid_text_type_fails_even_with_tool_calls(
    with_call: bool,
    output_text: object,
) -> None:
    items = (FakeOutputItem(),) if with_call else ()
    with pytest.raises(LanguageModelProviderError) as caught:
        run_output(output_text, *items)

    assert str(caught.value) == "OpenAI provider returned no text content."


def test_duplicate_provider_call_ids_are_rejected_without_deduplication() -> None:
    with pytest.raises(ValueError, match="tool call IDs must be unique"):
        run_output(None, FakeOutputItem(), FakeOutputItem(name="different.tool"))


def test_case_distinct_call_ids_and_repeated_tool_names_are_preserved() -> None:
    response = run_output(None, FakeOutputItem(), FakeOutputItem(call_id="Call_1"))

    assert [call.call_id for call in response.tool_calls] == ["call_1", "Call_1"]
    assert [call.name for call in response.tool_calls] == [
        "calculator.basic",
        "calculator.basic",
    ]


def test_configured_temperature_and_model_are_passed_exactly() -> None:
    responses = FakeResponses()
    provider = OpenAILanguageModelProvider(
        make_settings(model="configured-model", temperature=0.2),
        client=FakeAsyncOpenAI(responses),  # type: ignore[arg-type]
    )

    run_generate(provider)

    assert responses.calls[0]["model"] == "configured-model"
    assert responses.calls[0]["temperature"] == 0.2


def test_unset_temperature_is_omitted_without_inventing_default() -> None:
    responses = FakeResponses()
    provider = OpenAILanguageModelProvider(
        make_settings(temperature=None),
        client=FakeAsyncOpenAI(responses),  # type: ignore[arg-type]
    )

    run_generate(provider)

    assert "temperature" not in responses.calls[0]


def test_openai_output_translates_to_provider_neutral_response() -> None:
    raw_response = FakeOpenAIResponse(
        output_text="  Generated text  ",
        model="actual-returned-model",
    )
    provider = OpenAILanguageModelProvider(
        make_settings(),
        client=FakeAsyncOpenAI(FakeResponses(raw_response)),  # type: ignore[arg-type]
    )

    response = run_generate(provider)

    assert isinstance(response, LanguageModelResponse)
    assert response is not raw_response
    assert response.content == "  Generated text  "
    assert response.provider == "openai"
    assert response.model == "actual-returned-model"
    assert response.tool_calls == ()


@pytest.mark.parametrize("output_text", [None, "", "   ", "\t\n"])
def test_empty_openai_output_raises_safe_provider_error(
    output_text: object,
) -> None:
    secret = "test-openai-key"
    user_content = "private user content"
    responses = FakeResponses(FakeOpenAIResponse(output_text))
    provider = OpenAILanguageModelProvider(
        make_settings(api_key=secret),
        client=FakeAsyncOpenAI(responses),  # type: ignore[arg-type]
    )
    request = make_request(LanguageModelMessage(MessageRole.USER, user_content))

    with pytest.raises(
        LanguageModelProviderError,
        match="OpenAI provider returned no text content",
    ) as raised:
        run_generate(provider, request)

    assert secret not in str(raised.value)
    assert user_content not in str(raised.value)
    assert responses.calls and len(responses.calls) == 1


def test_openai_sdk_error_is_translated_once_with_original_cause() -> None:
    secret = "test-openai-key"
    user_content = "private user content"
    sdk_error = OpenAIError(f"raw failure {secret} {user_content}")
    responses = FakeResponses(error=sdk_error)
    provider = OpenAILanguageModelProvider(
        make_settings(api_key=secret),
        client=FakeAsyncOpenAI(responses),  # type: ignore[arg-type]
    )
    request = make_request(LanguageModelMessage(MessageRole.USER, user_content))

    with pytest.raises(
        LanguageModelProviderError,
        match="^OpenAI provider request failed\\.$",
    ) as raised:
        run_generate(provider, request)

    assert raised.value.__cause__ is sdk_error
    assert secret not in str(raised.value)
    assert user_content not in str(raised.value)
    assert len(responses.calls) == 1


def test_sequential_requests_do_not_leak_provider_state() -> None:
    responses = FakeResponses(FakeOpenAIResponse("First response", "first-model"))
    provider = OpenAILanguageModelProvider(
        make_settings(),
        client=FakeAsyncOpenAI(responses),  # type: ignore[arg-type]
    )
    first_request = make_request(LanguageModelMessage(MessageRole.USER, "First"))
    second_request = make_request(LanguageModelMessage(MessageRole.USER, "Second"))

    first_response = run_generate(provider, first_request)
    responses.response = FakeOpenAIResponse("Second response", "second-model")
    second_response = run_generate(provider, second_request)

    assert first_response.content == "First response"
    assert first_response.model == "first-model"
    assert second_response.content == "Second response"
    assert second_response.model == "second-model"
    assert responses.calls[0]["input"] == [
        {"role": "user", "content": "First"}
    ]
    assert responses.calls[1]["input"] == [
        {"role": "user", "content": "Second"}
    ]
    assert vars(provider) == {
        "_settings": provider._settings,
        "_client": provider._client,
    }


def test_separate_provider_instances_are_independent() -> None:
    first_client = FakeAsyncOpenAI()
    second_client = FakeAsyncOpenAI()

    first = OpenAILanguageModelProvider(
        make_settings(model="first-model"),
        client=first_client,  # type: ignore[arg-type]
    )
    second = OpenAILanguageModelProvider(
        make_settings(model="second-model"),
        client=second_client,  # type: ignore[arg-type]
    )

    assert first._client is first_client
    assert second._client is second_client
    assert first._client is not second._client
    assert first._settings.model == "first-model"
    assert second._settings.model == "second-model"


def test_adapter_has_only_allowed_architectural_dependencies() -> None:
    tree = ast.parse(inspect.getsource(openai_adapter))
    imported_modules = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert imported_modules == {
        "json",
        "collections.abc",
        "openai",
        "tuesday.config",
        "tuesday.language_models.base",
        "tuesday.language_models.tools",
    }


def test_openai_sdk_import_is_isolated_to_adapter() -> None:
    source_root = Path(openai_adapter.__file__).parents[2]
    openai_importers: list[Path] = []
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(
            (
                isinstance(node, ast.Import)
                and any(alias.name == "openai" for alias in node.names)
            )
            or (
                isinstance(node, ast.ImportFrom)
                and node.module == "openai"
            )
            for node in ast.walk(tree)
        ):
            openai_importers.append(path.relative_to(source_root))

    assert openai_importers == [Path("tuesday/language_models/openai.py")]


def test_openai_is_the_only_declared_runtime_dependency() -> None:
    project_root = Path(__file__).parents[2]
    project = tomllib.loads(
        (project_root / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]

    assert project["dependencies"] == ["openai>=2,<3"]


def test_public_package_exports_adapter_without_sdk_types() -> None:
    import tuesday.language_models as language_models

    assert language_models.OpenAILanguageModelProvider is OpenAILanguageModelProvider
    assert "OpenAILanguageModelProvider" in language_models.__all__
    assert "AsyncOpenAI" not in language_models.__all__
    assert "OpenAIError" not in language_models.__all__
    assert not hasattr(language_models, "LanguageModelProviderRegistry")
    assert not hasattr(language_models, "create_language_model_provider")
