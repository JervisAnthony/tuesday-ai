"""Deterministic CLI contracts and real service/storage boundaries."""

import argparse
import asyncio
import importlib
import json
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import pytest
from openai import OpenAIError

import tuesday
import tuesday.cli as cli
import tuesday.language_models.openai as openai_adapter
from tuesday.agents import UnsupportedModelToolCallError
from tuesday.conversations import (
    ConversationRepositoryError,
    InMemoryConversationRepository,
    SQLiteConversationRepository,
)
from tuesday.domain import MessageRole
from tuesday.services.conversation import StatefulConversationService

CONVERSATION_ID = "00000000-0000-0000-0000-000000000044"


@pytest.mark.parametrize("process_argv", [False, True])
def test_leading_history_dispatch_preserves_arguments(monkeypatch, process_argv):
    arguments = ["history", "show", CONVERSATION_ID, "--database", "explicit.db"]
    history = Mock(return_value=130)
    monkeypatch.setattr(cli, "history_main", history)
    monkeypatch.setattr(cli, "_parser", Mock(side_effect=AssertionError("No parser")))
    if process_argv:
        monkeypatch.setattr(sys, "argv", ["tuesday", *arguments])
        assert cli.main() == 130
    else:
        assert cli.main(tuple(arguments)) == 130
    history.assert_called_once_with(arguments[1:])


@pytest.mark.parametrize(
    "arguments,expected",
    [
        (["/chat Tell me about history"], "TUESDAY received: Tell me about history\n"),
        (["--json", "/chat history"], None),
        (["--runtime", "deterministic", "history"], None),
    ],
)
def test_only_a_leading_token_selects_history(monkeypatch, capsys, arguments, expected):
    history = Mock(side_effect=AssertionError("Not a leading history command"))
    monkeypatch.setattr(cli, "history_main", history)
    status = cli.main(arguments)
    output = capsys.readouterr()
    if expected is not None:
        assert status == 0 and output.out == expected and output.err == ""
    elif "--json" in arguments:
        assert status == 0
        assert json.loads(output.out)["content"] == "TUESDAY received: history"
    else:
        assert status == 1 and output.out == "" and "explicit /chat" in output.err
    history.assert_not_called()


def test_top_level_help_mentions_history_dispatch(capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])
    assert caught.value.code == 0
    output = capsys.readouterr()
    assert "tuesday history {list,show} --help" in output.out
    assert output.err == ""


def stored(path, conversation_id=CONVERSATION_ID):
    context = asyncio.run(
        SQLiteConversationRepository(path).get_context(UUID(conversation_id))
    )
    return [(item.role, item.content) for item in context.messages]


@pytest.mark.parametrize("directive", ["/chat", "/conversation"])
def test_real_default_interaction_outputs_only_completed_text(capsys, directive):
    assert cli.main([directive + " Hello TUESDAY"]) == 0
    output = capsys.readouterr()
    assert output.out == "TUESDAY received: Hello TUESDAY\n"
    assert output.err == ""


def test_argv_none_uses_process_arguments(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["tuesday", "/chat From argv"])
    assert cli.main() == 0
    assert capsys.readouterr().out == "TUESDAY received: From argv\n"


@pytest.mark.parametrize("as_json", [False, True])
def test_multiline_whitespace_and_unicode_are_preserved(capsys, as_json):
    args = ["--json"] if as_json else []
    assert cli.main([*args, "/chat   café नमस्ते 🌍\nsecond line  "]) == 0
    output = capsys.readouterr()
    expected = "TUESDAY received: café नमस्ते 🌍\nsecond line  "
    assert (
        json.loads(output.out)["content"] if as_json else output.out[:-1]
    ) == expected
    assert output.err == ""


@pytest.mark.parametrize("explicit_id", [False, True])
def test_json_uses_actual_response_and_one_real_stateful_invocation(
    monkeypatch, capsys, explicit_id
):
    original = StatefulConversationService.handle
    interactions = []

    async def record(self, request):
        response = await original(self, request)
        interactions.append((self, request, response))
        return response

    monkeypatch.setattr(StatefulConversationService, "handle", record)
    factory = Mock(wraps=cli.create_default_orchestrator)
    service_factory = Mock(wraps=StatefulConversationService)
    monkeypatch.setattr(cli, "create_default_orchestrator", factory)
    monkeypatch.setattr(cli, "StatefulConversationService", service_factory)
    args = ["--json"]
    if explicit_id:
        args.extend(["--conversation-id", CONVERSATION_ID])
    source = "/conversation   Original\ncontent  "
    assert cli.main([*args, source]) == 0
    assert len(interactions) == 1
    service, request, response = interactions[0]
    assert isinstance(service._repository, InMemoryConversationRepository)
    assert request.content == source
    assert response.conversation_id == request.conversation_id
    assert response.request_id == request.request_id
    if explicit_id:
        assert request.conversation_id == UUID(CONVERSATION_ID)
    factory.assert_called_once_with()
    service_factory.assert_called_once()
    output = capsys.readouterr()
    assert json.loads(output.out) == {
        "content": response.content,
        "conversation_id": str(response.conversation_id),
        "request_id": str(response.request_id),
        "response_id": str(response.response_id),
    }
    assert output.err == ""
    assert (
        len({response.conversation_id, response.request_id, response.response_id}) == 3
    )


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        [""],
        ["  \t\n"],
        ["--unknown"],
        ["--conversation-id", "bad", "/chat Hello"],
        ["--conversation-id"],
        ["--database"],
        ["/chat Hello", "extra"],
        ["--json"],
        ["--dat", "somewhere.db", "/chat Hello"],
        ["--js", "/chat Hello"],
        ["--runtime", "unknown", "/chat Hello"],
        ["--runtime"],
        ["--runtime", "openai", "--max-history-messages", "-1", "/chat Hello"],
        ["--runtime", "openai", "--max-history-messages", "1.5", "/chat Hello"],
        ["--runtime", "openai", "--max-history-messages", "bad", "/chat Hello"],
        ["--runtime", "openai", "--max-history-messages"],
        ["--max-history-messages", "0", "/chat Hello"],
        ["--runtime", "deterministic", "--max-history-messages", "2", "/chat Hello"],
        ["--interactive", "/chat Hello"],
        ["--interactive", "--json"],
        ["--interactive", "--runtime", "openai", "--json"],
        ["--interactive", "--max-history-messages", "0"],
    ],
)
def test_argument_failures_exit_two_before_runtime_or_database_creation(
    monkeypatch, capsys, tmp_path, arguments
):
    spies = []
    for name in (
        "InMemoryConversationRepository",
        "SQLiteConversationRepository",
        "create_default_orchestrator",
        "StatefulConversationService",
        "load_settings",
        "OpenAILanguageModelProvider",
        "create_model_backed_orchestrator",
    ):
        spy = Mock(
            side_effect=AssertionError("Argument errors must not construct runtime")
        )
        monkeypatch.setattr(cli, name, spy)
        spies.append(spy)
    path = tmp_path / "must-not-exist.sqlite3"
    with pytest.raises(SystemExit) as caught:
        cli.main(["--database", str(path), *arguments])
    assert caught.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "usage:" in output.err
    assert not path.exists()
    for spy in spies:
        spy.assert_not_called()


@pytest.mark.parametrize("option", ["-h", "--help", "--version"])
@pytest.mark.parametrize("runtime", ["deterministic", "openai"])
@pytest.mark.parametrize("interactive", [False, True])
def test_help_and_version_exit_zero_without_constructing_runtime(
    monkeypatch, capsys, tmp_path, option, runtime, interactive
):
    for name in (
        "InMemoryConversationRepository",
        "SQLiteConversationRepository",
        "create_default_orchestrator",
        "StatefulConversationService",
        "load_settings",
        "OpenAILanguageModelProvider",
    ):
        monkeypatch.setattr(cli, name, Mock(side_effect=AssertionError("No runtime")))
    path = tmp_path / "not-created.db"
    with pytest.raises(SystemExit) as caught:
        cli.main(
            [
                "--runtime",
                runtime,
                "--database",
                str(path),
                *(["--interactive"] if interactive else []),
                option,
            ]
        )
    assert caught.value.code == 0
    output = capsys.readouterr()
    assert output.err == ""
    if option == "--version":
        assert output.out == f"TUESDAY {tuesday.__version__}\n"
    else:
        for text in (
            "REQUEST",
            "/chat",
            "/conversation",
            "--conversation-id",
            "--database",
            "durable",
            "--json",
        ):
            assert text in output.out
    assert not path.exists()


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize(
    "source,error",
    [
        ("Hello", "explicit /chat or /conversation"),
        ("/unknown Hello", "explicit /chat or /conversation"),
        ("/chat", "content after the routing directive"),
        ("/conversation", "content after the routing directive"),
    ],
)
def test_expected_application_errors_use_stderr_and_exit_one(
    capsys, source, error, as_json
):
    assert cli.main((["--json"] if as_json else []) + [source]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert error in output.err
    assert "Traceback" not in output.err


@pytest.mark.parametrize("path", ["", " \t", ":memory:", "file:history.db"])
def test_invalid_database_configuration_is_reported_without_execution(path, capsys):
    assert cli.main(["--database", path, "/chat Hello"]) == 1
    assert capsys.readouterr() == ("", "tuesday: invalid database path.\n")


@pytest.mark.parametrize("failure", ["missing-parent", "corrupt", "directory"])
def test_real_database_initialization_failures_do_not_expose_paths(
    tmp_path, capsys, failure
):
    path = tmp_path / "private-conversations.db"
    if failure == "missing-parent":
        path = tmp_path / "missing" / "private-conversations.db"
    elif failure == "corrupt":
        path.write_bytes(b"not sqlite")
    else:
        path.mkdir()
    assert cli.main(["--database", str(path), "/chat Hello"]) == 1
    assert capsys.readouterr() == ("", "tuesday: conversation storage failed.\n")


@pytest.mark.parametrize("operation", ["get_context", "append_messages"])
def test_repository_operation_failure_is_not_a_success(monkeypatch, capsys, operation):
    spy = AsyncMock(side_effect=ConversationRepositoryError("private stored content"))
    monkeypatch.setattr(InMemoryConversationRepository, operation, spy)
    assert cli.main(["--json", "/chat Hello"]) == 1
    spy.assert_awaited_once()
    assert capsys.readouterr() == ("", "tuesday: conversation storage failed.\n")


@pytest.mark.parametrize("stage", ["construction", "interaction"])
def test_keyboard_interrupt_returns_130_without_retry(monkeypatch, capsys, stage):
    if stage == "construction":
        spy = Mock(side_effect=KeyboardInterrupt)
        monkeypatch.setattr(cli, "create_default_orchestrator", spy)
    else:
        spy = AsyncMock(side_effect=KeyboardInterrupt)
        monkeypatch.setattr(StatefulConversationService, "handle", spy)
    assert cli.main(["/chat Hello"]) == 130
    spy.assert_called_once()
    assert capsys.readouterr() == ("", "tuesday: interrupted.\n")


@pytest.mark.parametrize("error_type", [TypeError, ValueError, RuntimeError])
def test_unexpected_internal_failures_propagate(monkeypatch, capsys, error_type):
    error = error_type("Programming defect")
    spy = AsyncMock(side_effect=error)
    monkeypatch.setattr(StatefulConversationService, "handle", spy)
    with pytest.raises(error_type) as caught:
        cli.main(["/chat Hello"])
    assert caught.value is error
    spy.assert_awaited_once()
    assert capsys.readouterr() == ("", "")


def test_sqlite_reuses_history_and_preserves_original_directives(tmp_path, capsys):
    path = tmp_path / "history.sqlite3"
    options = ["--database", str(path), "--conversation-id", CONVERSATION_ID]
    assert cli.main([*options, "/chat Hello"]) == 0
    assert capsys.readouterr().out == "TUESDAY received: Hello\n"
    assert cli.main([*options, "/conversation Continue"]) == 0
    answer = "TUESDAY received: Continue (2 prior messages in context)"
    assert capsys.readouterr().out == answer + "\n"
    assert stored(path) == [
        (MessageRole.USER, "/chat Hello"),
        (MessageRole.ASSISTANT, "TUESDAY received: Hello"),
        (MessageRole.USER, "/conversation Continue"),
        (MessageRole.ASSISTANT, answer),
    ]


def test_database_generated_id_is_recoverable_through_json(tmp_path, capsys):
    path = tmp_path / "generated.sqlite3"
    assert cli.main(["--database", str(path), "--json", "/chat First"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert UUID(result["conversation_id"])
    assert stored(path, result["conversation_id"]) == [
        (MessageRole.USER, "/chat First"),
        (MessageRole.ASSISTANT, result["content"]),
    ]


@pytest.mark.parametrize("source", ["/unknown Hello", "/chat", "/conversation"])
def test_failed_sqlite_interaction_preserves_previous_history(tmp_path, capsys, source):
    path = tmp_path / "history.sqlite3"
    options = ["--database", str(path), "--conversation-id", CONVERSATION_ID]
    assert cli.main([*options, "/chat Earlier"]) == 0
    capsys.readouterr()
    before = stored(path)
    assert cli.main([*options, source]) == 1
    output = capsys.readouterr()
    assert output.out == "" and output.err
    assert stored(path) == before


def test_storage_and_conversation_ids_are_isolated(tmp_path, capsys):
    first, second = tmp_path / "first.db", tmp_path / "second.db"
    other_id = str(uuid4())
    for path, identifier in (
        (first, CONVERSATION_ID),
        (first, other_id),
        (second, CONVERSATION_ID),
    ):
        assert (
            cli.main(
                [
                    "--database",
                    str(path),
                    "--conversation-id",
                    identifier,
                    "/chat Independent",
                ]
            )
            == 0
        )
        assert capsys.readouterr().out == "TUESDAY received: Independent\n"
        assert len(stored(path, identifier)) == 2
    for _ in range(2):
        assert (
            cli.main(["--conversation-id", CONVERSATION_ID, "/chat Fresh memory"]) == 0
        )
        assert capsys.readouterr().out == "TUESDAY received: Fresh memory\n"


def test_module_guard_propagates_main_exit_status(monkeypatch):
    main = Mock(return_value=130)
    monkeypatch.setattr(cli, "main", main)
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("tuesday", run_name="__main__")
    assert caught.value.code == 130
    main.assert_called_once_with()


def test_ordinary_main_module_import_does_not_parse_or_execute(monkeypatch):
    main = Mock(side_effect=AssertionError("Import must not execute main"))
    monkeypatch.setattr(cli, "main", main)
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        Mock(side_effect=AssertionError("Import must not parse arguments")),
    )
    module = importlib.import_module("tuesday.__main__")
    importlib.reload(module)
    main.assert_not_called()


@pytest.mark.parametrize("explicit", [False, True])
def test_model_configuration_never_selects_model_or_tool_execution(
    monkeypatch, capsys, explicit
):
    import tuesday.agents as agents
    import tuesday.config as config
    import tuesday.language_models as models
    from tuesday.services.openai_tool_turn import OpenAIToolTurnService

    monkeypatch.setenv("TUESDAY_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("TUESDAY_MODEL_NAME", "unused-model")
    monkeypatch.setenv("TUESDAY_MODEL_API_KEY", "fake-not-a-real-key")
    forbidden = Mock(side_effect=AssertionError("No model or guarded tool activation"))
    monkeypatch.setattr(config, "load_settings", forbidden)
    monkeypatch.setattr(cli, "load_settings", forbidden)
    for component in (
        models.OpenAILanguageModelProvider,
        agents.ModelBackedConversationalAgent,
        agents.GuardedOpenAIConversationalAgent,
        OpenAIToolTurnService,
    ):
        monkeypatch.setattr(component, "__init__", forbidden)
    options = ["--runtime", "deterministic"] if explicit else []
    assert cli.main([*options, "/chat Deterministic"]) == 0
    assert capsys.readouterr() == ("TUESDAY received: Deterministic\n", "")
    forbidden.assert_not_called()


@pytest.fixture
def sdk(monkeypatch):
    """Fake SDK I/O while retaining the production application stack."""
    for name in ("PROVIDER", "NAME", "API_KEY", "TIMEOUT_SECONDS", "TEMPERATURE"):
        monkeypatch.delenv("TUESDAY_MODEL_" + name, raising=False)
    monkeypatch.setenv("TUESDAY_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("TUESDAY_MODEL_NAME", "fake-model")
    monkeypatch.setenv("TUESDAY_MODEL_API_KEY", "fake-cli-key")
    state = SimpleNamespace(
        response=SimpleNamespace(
            output=(), output_text="  Model reply\nline two  ", model="fake-model"
        ),
        loops=[],
    )

    async def generate(**kwargs):
        state.loops.append(asyncio.get_running_loop())
        return state.response

    async def close():
        state.loops.append(asyncio.get_running_loop())

    state.create = AsyncMock(side_effect=generate)
    state.close = AsyncMock(side_effect=close)
    state.client = SimpleNamespace(
        responses=SimpleNamespace(create=state.create), close=state.close
    )

    def construct(**kwargs):
        state.loops.append(asyncio.get_running_loop())
        return state.client

    state.factory = Mock(side_effect=construct)
    monkeypatch.setattr(openai_adapter, "AsyncOpenAI", state.factory)
    return state


@pytest.mark.parametrize("directive", ["/chat", "/conversation"])
@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("explicit_id", [False, True])
def test_openai_runtime_uses_real_stack_actual_response_and_one_owned_loop(
    sdk, monkeypatch, capsys, directive, as_json, explicit_id
):
    original = StatefulConversationService.handle
    interactions = []

    async def record(self, request):
        response = await original(self, request)
        interactions.append((request, response))
        return response

    monkeypatch.setattr(StatefulConversationService, "handle", record)
    factory = Mock(wraps=cli.create_model_backed_orchestrator)
    monkeypatch.setattr(cli, "create_model_backed_orchestrator", factory)
    options = ["--runtime", "openai"]
    if explicit_id:
        options += ["--conversation-id", CONVERSATION_ID]
    source = directive + "   Original café\nrequest  "
    assert cli.main([*options, *(["--json"] if as_json else []), source]) == 0
    output = capsys.readouterr()
    request, response = interactions[0]
    assert len(interactions) == 1
    assert request.content == source
    assert response.conversation_id == request.conversation_id
    if explicit_id:
        assert response.conversation_id == UUID(CONVERSATION_ID)
    assert response.request_id == request.request_id
    if as_json:
        assert json.loads(output.out) == {
            "content": response.content,
            "conversation_id": str(response.conversation_id),
            "request_id": str(response.request_id),
            "response_id": str(response.response_id),
        }
    else:
        assert output.out == "  Model reply\nline two  \n"
    assert output.err == ""
    sdk.factory.assert_called_once_with(
        api_key="fake-cli-key", timeout=30.0, max_retries=0
    )
    sdk.create.assert_awaited_once()
    sdk.close.assert_awaited_once()
    assert len(sdk.loops) == 3 and all(loop is sdk.loops[0] for loop in sdk.loops)
    payload = sdk.create.call_args.kwargs
    assert payload["input"][-1] == {
        "role": "user",
        "content": "Original café\nrequest  ",
    }
    assert payload["store"] is payload["stream"] is False
    assert "tools" not in payload
    renderer = factory.call_args.kwargs["renderer"]
    assert renderer.max_history_messages is None


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "provider",
        "model",
        "unsupported",
        "case",
        "key",
        "blank-key",
        "temperature",
    ],
)
def test_openai_configuration_errors_are_fixed_and_do_not_expose_secrets(
    sdk, monkeypatch, capsys, failure
):
    if failure == "missing":
        for name in ("PROVIDER", "NAME", "API_KEY"):
            monkeypatch.delenv("TUESDAY_MODEL_" + name)
    elif failure in ("provider", "model", "key"):
        name = {"provider": "PROVIDER", "model": "NAME", "key": "API_KEY"}[failure]
        monkeypatch.delenv("TUESDAY_MODEL_" + name)
    elif failure in ("unsupported", "case"):
        monkeypatch.setenv(
            "TUESDAY_MODEL_PROVIDER", "other" if failure == "unsupported" else "OpenAI"
        )
    elif failure == "blank-key":
        monkeypatch.setenv("TUESDAY_MODEL_API_KEY", "  ")
    else:
        monkeypatch.setenv("TUESDAY_MODEL_TEMPERATURE", "3")
    assert cli.main(["--runtime", "openai", "--json", "/chat Private request"]) == 1
    assert capsys.readouterr() == ("", "tuesday: invalid OpenAI model configuration.\n")
    sdk.factory.assert_not_called()


def test_openai_failure_is_reported_without_fallback_retry_or_secret_output(
    sdk, capsys
):
    sdk.create.side_effect = OpenAIError("fake-cli-key Private request raw SDK detail")
    assert cli.main(["--runtime", "openai", "--json", "/chat Private request"]) == 1
    assert capsys.readouterr() == ("", "tuesday: model operation failed.\n")
    sdk.create.assert_awaited_once()
    sdk.close.assert_awaited_once()


@pytest.mark.parametrize("operation", ["get_context", "append_messages"])
def test_openai_client_is_closed_on_downstream_repository_failure(
    sdk, monkeypatch, capsys, operation
):
    monkeypatch.setattr(
        InMemoryConversationRepository,
        operation,
        AsyncMock(side_effect=ConversationRepositoryError("private")),
    )
    assert cli.main(["--runtime", "openai", "/chat Hello"]) == 1
    assert capsys.readouterr() == ("", "tuesday: conversation storage failed.\n")
    assert sdk.create.await_count == int(operation == "append_messages")
    sdk.close.assert_awaited_once()


@pytest.mark.parametrize(
    "cleanup_error", [OpenAIError, RuntimeError, KeyboardInterrupt]
)
def test_cleanup_never_hides_primary_composition_failure(
    sdk, monkeypatch, capsys, cleanup_error
):
    primary = ValueError("Unexpected composition defect")
    monkeypatch.setattr(
        cli, "create_model_backed_orchestrator", Mock(side_effect=primary)
    )
    sdk.close.side_effect = cleanup_error("Secondary cleanup failure")
    with pytest.raises(ValueError) as caught:
        cli.main(["--runtime", "openai", "/chat Hello"])
    assert caught.value is primary
    sdk.close.assert_awaited_once()
    sdk.create.assert_not_awaited()
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("primary_type", [OpenAIError, RuntimeError, KeyboardInterrupt])
def test_cleanup_failure_preserves_request_failure_or_interruption(
    sdk, capsys, primary_type
):
    primary = primary_type("private primary failure")
    sdk.create.side_effect = primary
    sdk.close.side_effect = OpenAIError("private cleanup failure")
    if primary_type is RuntimeError:
        with pytest.raises(RuntimeError) as caught:
            cli.main(["--runtime", "openai", "/chat Hello"])
        assert caught.value is primary
        assert capsys.readouterr() == ("", "")
    else:
        expected = 130 if primary_type is KeyboardInterrupt else 1
        assert cli.main(["--runtime", "openai", "/chat Hello"]) == expected
        error = "interrupted" if expected == 130 else "model operation failed"
        assert capsys.readouterr() == ("", f"tuesday: {error}.\n")
    sdk.create.assert_awaited_once()
    sdk.close.assert_awaited_once()


@pytest.mark.parametrize("error_type", [OpenAIError, RuntimeError])
def test_cleanup_failure_after_success_is_not_silently_ignored(sdk, capsys, error_type):
    error = error_type("private cleanup detail")
    sdk.close.side_effect = error
    if error_type is OpenAIError:
        assert cli.main(["--runtime", "openai", "/chat Hello"]) == 1
        assert capsys.readouterr() == ("", "tuesday: model operation failed.\n")
    else:
        with pytest.raises(RuntimeError) as caught:
            cli.main(["--runtime", "openai", "/chat Hello"])
        assert caught.value is error
        assert capsys.readouterr() == ("", "")
    sdk.create.assert_awaited_once()
    sdk.close.assert_awaited_once()


@pytest.mark.parametrize("source", ["/chat", "/unknown Hello"])
def test_model_runtime_reuses_routing_failures_and_closes_without_generation(
    sdk, capsys, source
):
    assert cli.main(["--runtime", "openai", source]) == 1
    assert capsys.readouterr().out == ""
    sdk.create.assert_not_awaited()
    sdk.close.assert_awaited_once()


def test_model_tool_calls_retain_text_only_agent_failure_without_execution(sdk, capsys):
    sdk.response = SimpleNamespace(
        output=(
            SimpleNamespace(
                type="function_call",
                call_id="call_unexpected",
                name="calculator.basic",
                arguments="{}",
            ),
        ),
        output_text=None,
        model="fake-model",
    )
    with pytest.raises(UnsupportedModelToolCallError):
        cli.main(["--runtime", "openai", "/chat Do not execute tools"])
    sdk.create.assert_awaited_once()
    sdk.close.assert_awaited_once()
    assert capsys.readouterr() == ("", "")
