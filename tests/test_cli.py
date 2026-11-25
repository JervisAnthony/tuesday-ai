"""Deterministic CLI contracts and real service/storage boundaries."""

import argparse
import asyncio
import importlib
import json
import runpy
import sys
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import pytest

import tuesday
import tuesday.cli as cli
from tuesday.conversations import (
    ConversationRepositoryError,
    InMemoryConversationRepository,
    SQLiteConversationRepository,
)
from tuesday.domain import MessageRole
from tuesday.services.conversation import StatefulConversationService

CONVERSATION_ID = "00000000-0000-0000-0000-000000000044"


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
def test_help_and_version_exit_zero_without_constructing_runtime(
    monkeypatch, capsys, tmp_path, option
):
    for name in (
        "InMemoryConversationRepository",
        "SQLiteConversationRepository",
        "create_default_orchestrator",
        "StatefulConversationService",
    ):
        monkeypatch.setattr(cli, name, Mock(side_effect=AssertionError("No runtime")))
    path = tmp_path / "not-created.db"
    with pytest.raises(SystemExit) as caught:
        cli.main(["--database", str(path), option])
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


def test_model_configuration_never_selects_model_or_tool_execution(monkeypatch, capsys):
    import tuesday.agents as agents
    import tuesday.config as config
    import tuesday.language_models as models
    from tuesday.services.openai_tool_turn import OpenAIToolTurnService

    monkeypatch.setenv("TUESDAY_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("TUESDAY_MODEL_NAME", "unused-model")
    monkeypatch.setenv("TUESDAY_MODEL_API_KEY", "fake-not-a-real-key")
    forbidden = Mock(side_effect=AssertionError("No model or guarded tool activation"))
    monkeypatch.setattr(config, "load_settings", forbidden)
    for component in (
        models.OpenAILanguageModelProvider,
        agents.ModelBackedConversationalAgent,
        agents.GuardedOpenAIConversationalAgent,
        OpenAIToolTurnService,
    ):
        monkeypatch.setattr(component, "__init__", forbidden)
    assert cli.main(["/chat Deterministic"]) == 0
    assert capsys.readouterr() == ("TUESDAY received: Deterministic\n", "")
    forbidden.assert_not_called()
