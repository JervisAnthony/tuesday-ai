"""Real module and installed-command processes with isolated temporary storage."""

import asyncio
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import sysconfig
from contextlib import closing
from importlib.metadata import distribution
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest
from openai import OpenAIError

import tuesday
import tuesday.language_models.openai as openai_adapter
from tuesday.agents import (
    GuardedOpenAIConversationalAgent,
    UnsupportedModelToolCallError,
)
from tuesday.cli import main
from tuesday.conversations import SQLiteConversationRepository
from tuesday.domain import MessageRole
from tuesday.prompting import DEFAULT_CONVERSATIONAL_SYSTEM_PROMPT
from tuesday.services.openai_tool_turn import OpenAIToolTurnService
from tuesday.tools import BasicCalculatorTool

CONVERSATION_ID = "00000000-0000-0000-0000-000000000044"
OTHER_ID = "00000000-0000-0000-0000-000000000045"


def invoke(tmp_path, *arguments, env=None, input_text=None):
    return subprocess.run(
        [sys.executable, "-m", "tuesday", *arguments],
        cwd=tmp_path,
        env=env,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=30,
    )


def history(path, conversation_id):
    return asyncio.run(
        SQLiteConversationRepository(path).get_context(UUID(conversation_id))
    )


@pytest.mark.parametrize("option", ["--help", "--version"])
@pytest.mark.parametrize("runtime", ["deterministic", "openai"])
def test_module_help_and_version_do_not_create_storage(tmp_path, option, runtime):
    result = invoke(tmp_path, "--runtime", runtime, option)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    if option == "--version":
        assert result.stdout == f"TUESDAY {tuesday.__version__}\n"
    else:
        assert "usage: tuesday" in result.stdout
        assert "/chat" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_module_text_and_json_success(tmp_path):
    plain = invoke(tmp_path, "/chat Hello TUESDAY")
    assert plain.returncode == 0, plain.stderr
    assert plain.stdout == "TUESDAY received: Hello TUESDAY\n"
    assert plain.stderr == ""
    structured = invoke(tmp_path, "--json", "/conversation Hello TUESDAY")
    assert structured.returncode == 0, structured.stderr
    assert structured.stderr == ""
    payload = json.loads(structured.stdout)
    assert set(payload) == {"content", "conversation_id", "request_id", "response_id"}
    assert payload["content"] == "TUESDAY received: Hello TUESDAY"
    for key in ("conversation_id", "request_id", "response_id"):
        assert str(UUID(payload[key])) == payload[key]


@pytest.mark.parametrize(
    "arguments,exit_code",
    [
        ([], 2),
        (["--conversation-id", "invalid", "/chat Hello"], 2),
        (["--unknown"], 2),
        (["/unknown Hello"], 1),
        (["/chat"], 1),
        (["/conversation"], 1),
        (["--runtime", "unknown", "/chat Hello"], 2),
        (
            [
                "--runtime",
                "deterministic",
                "--max-history-messages",
                "0",
                "/chat Hello",
            ],
            2,
        ),
    ],
)
def test_process_exit_codes_and_stderr(tmp_path, arguments, exit_code):
    result = invoke(tmp_path, *arguments)
    assert result.returncode == exit_code
    assert result.stdout == ""
    assert result.stderr and "Traceback" not in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_cross_process_durability_with_generated_conversation_id(tmp_path):
    path = tmp_path / "history.sqlite3"
    first_source = "/chat First question"
    first = invoke(tmp_path, "--json", "--database", str(path), first_source)
    assert first.returncode == 0, first.stderr
    one = json.loads(first.stdout)
    second_source = "/conversation Second question"
    second = invoke(
        tmp_path,
        "--json",
        "--database",
        str(path),
        "--conversation-id",
        one["conversation_id"],
        second_source,
    )
    assert second.returncode == 0, second.stderr
    assert first.stderr == second.stderr == ""
    two = json.loads(second.stdout)
    assert one["conversation_id"] == two["conversation_id"]
    assert one["request_id"] != two["request_id"]
    assert one["response_id"] != two["response_id"]
    assert one["content"] == "TUESDAY received: First question"
    assert (
        two["content"]
        == "TUESDAY received: Second question (2 prior messages in context)"
    )
    saved = history(path, one["conversation_id"])
    assert [(message.role, message.content) for message in saved.messages] == [
        (MessageRole.USER, first_source),
        (MessageRole.ASSISTANT, one["content"]),
        (MessageRole.USER, second_source),
        (MessageRole.ASSISTANT, two["content"]),
    ]
    assert len(saved.messages) == 4


def test_process_conversation_and_database_isolation(tmp_path):
    first_path, other_path = tmp_path / "first.db", tmp_path / "other.db"
    for path, identifier in (
        (first_path, CONVERSATION_ID),
        (first_path, OTHER_ID),
        (other_path, CONVERSATION_ID),
    ):
        result = invoke(
            tmp_path,
            "--database",
            str(path),
            "--conversation-id",
            identifier,
            "/chat Independent",
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == "TUESDAY received: Independent\n"
        assert len(history(path, identifier).messages) == 2


def test_default_memory_does_not_cross_process_boundaries(tmp_path):
    for _ in range(2):
        result = invoke(tmp_path, "--conversation-id", CONVERSATION_ID, "/chat Fresh")
        assert result.returncode == 0, result.stderr
        assert result.stdout == "TUESDAY received: Fresh\n"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("source", ["/unknown Hello", "/chat", "/conversation"])
def test_failed_process_does_not_append_to_existing_history(tmp_path, source):
    path = tmp_path / "history.db"
    options = ("--database", str(path), "--conversation-id", CONVERSATION_ID)
    first = invoke(tmp_path, *options, "/chat Previously committed")
    assert first.returncode == 0, first.stderr
    before = history(path, CONVERSATION_ID)
    failed = invoke(tmp_path, *options, source)
    assert failed.returncode == 1
    assert failed.stdout == "" and failed.stderr
    assert history(path, CONVERSATION_ID) == before


def test_model_environment_does_not_activate_live_execution(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "TUESDAY_MODEL_PROVIDER": "openai",
            "TUESDAY_MODEL_NAME": "unused-model",
            "TUESDAY_MODEL_API_KEY": "fake-not-a-real-key",
            "TUESDAY_MODEL_TIMEOUT_SECONDS": "invalid-and-must-not-be-read",
        }
    )
    result = invoke(tmp_path, "/chat Deterministic", env=env)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "TUESDAY received: Deterministic\n"
    assert result.stderr == ""


def test_imports_have_no_cli_runtime_database_or_network_side_effects(tmp_path):
    script = """
import argparse, socket, sqlite3, sys
def forbidden(*args, **kwargs):
    raise AssertionError("Import must not execute application logic")
sqlite3.connect = forbidden
socket.create_connection = forbidden
argparse.ArgumentParser.parse_args = forbidden
import tuesday.terminal_session
import tuesday.composition
import tuesday.config
import tuesday.services.conversation
tuesday.composition.create_default_orchestrator = forbidden
tuesday.config.load_settings = forbidden
tuesday.services.conversation.StatefulConversationService = forbidden
sys.argv = ["unrelated-program", "--unknown-option"]
import tuesday
import tuesday.cli
import tuesday.terminal_session
import tuesday.__main__
assert tuesday.__all__ == ["__version__"]
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == result.stderr == ""
    assert list(tmp_path.iterdir()) == []


def test_installed_entry_point_metadata_resolves_to_same_main():
    points = [
        point
        for point in distribution("tuesday-ai").entry_points
        if point.group == "console_scripts" and point.name == "tuesday"
    ]
    assert len(points) == 1
    assert points[0].value == "tuesday.cli:main"
    assert points[0].load() is main


@pytest.mark.parametrize(
    "arguments,expected,exit_code",
    [
        (["--version"], f"TUESDAY {tuesday.__version__}\n", 0),
        (["/chat Hello TUESDAY"], "TUESDAY received: Hello TUESDAY\n", 0),
        (["--help"], None, 0),
        ([], "", 2),
        (["/unknown Hello"], "", 1),
        (["--runtime", "openai", "--version"], f"TUESDAY {tuesday.__version__}\n", 0),
        (["--runtime", "openai", "--help"], None, 0),
    ],
)
def test_actual_installed_console_script(tmp_path, arguments, expected, exit_code):
    # Discover the launcher for this interpreter, independently of the caller's PATH.
    executable = shutil.which("tuesday", path=sysconfig.get_path("scripts"))
    assert executable is not None, "Refresh this environment with pip install -e .[dev]"
    result = subprocess.run(
        [executable, *arguments],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == exit_code, result.stderr
    if expected is None:
        assert "usage: tuesday" in result.stdout
    else:
        assert result.stdout == expected
    assert bool(result.stderr) is (exit_code != 0)


@pytest.fixture
def model_sdk(monkeypatch):
    """In-process fake SDK I/O over the real CLI, provider, and SQLite stack."""
    for name in ("PROVIDER", "NAME", "API_KEY", "TIMEOUT_SECONDS", "TEMPERATURE"):
        monkeypatch.delenv("TUESDAY_MODEL_" + name, raising=False)
    monkeypatch.setenv("TUESDAY_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("TUESDAY_MODEL_NAME", "fake-model")
    monkeypatch.setenv("TUESDAY_MODEL_API_KEY", "fake-not-a-real-key")
    create = AsyncMock()
    clients = []

    def construct(**kwargs):
        client = SimpleNamespace(
            responses=SimpleNamespace(create=create), close=AsyncMock()
        )
        clients.append(client)
        return client

    factory = Mock(side_effect=construct)
    monkeypatch.setattr(openai_adapter, "AsyncOpenAI", factory)
    forbidden = Mock(side_effect=AssertionError("No guarded runtime or tools"))
    monkeypatch.setattr(GuardedOpenAIConversationalAgent, "__init__", forbidden)
    monkeypatch.setattr(OpenAIToolTurnService, "__init__", forbidden)
    monkeypatch.setattr(BasicCalculatorTool, "execute", forbidden)
    return SimpleNamespace(
        create=create, clients=clients, factory=factory, tools=forbidden
    )


def model_text(content):
    return SimpleNamespace(output=(), output_text=content, model="fake-model")


@pytest.mark.parametrize("limit", [None, 0, 2])
def test_in_process_model_cli_reconstructs_sqlite_services_and_bounds_only_prompts(
    tmp_path, capsys, model_sdk, limit
):
    path = tmp_path / "model-history.sqlite3"
    options = [
        "--runtime",
        "openai",
        "--json",
        "--database",
        str(path),
        "--conversation-id",
        CONVERSATION_ID,
    ]
    if limit is not None:
        options += ["--max-history-messages", str(limit)]
    answers = [f"  答え {index}\nline two  " for index in range(4)]
    model_sdk.create.side_effect = [model_text(answer) for answer in answers]
    expected = []
    responses = []
    for index, answer in enumerate(answers):
        source = ("/chat" if index % 2 == 0 else "/conversation") + f" Question {index}"
        prior = expected if limit is None else [] if limit == 0 else expected[-2:]
        assert main([*options, source]) == 0
        captured = capsys.readouterr()
        assert captured.err == ""
        payload = json.loads(captured.out)
        responses.append(payload)
        assert set(payload) == {
            "content",
            "conversation_id",
            "request_id",
            "response_id",
        }
        assert payload["content"] == answer
        assert payload["conversation_id"] == CONVERSATION_ID
        assert model_sdk.create.call_args.kwargs["input"] == [
            {"role": "system", "content": DEFAULT_CONVERSATIONAL_SYSTEM_PROMPT},
            *({"role": role.value, "content": content} for role, content in prior),
            {"role": "user", "content": f"Question {index}"},
        ]
        assert "tools" not in model_sdk.create.call_args.kwargs
        expected.extend([(MessageRole.USER, source), (MessageRole.ASSISTANT, answer)])
        saved = history(path, CONVERSATION_ID)
        assert [(item.role, item.content) for item in saved.messages] == expected
        assert len(saved.messages) == (index + 1) * 2
    assert len(saved.messages) == 8
    assert len({payload["request_id"] for payload in responses}) == 4
    assert len({payload["response_id"] for payload in responses}) == 4
    assert model_sdk.create.await_count == model_sdk.factory.call_count == 4
    assert len(model_sdk.clients) == 4
    for client in model_sdk.clients:
        client.close.assert_awaited_once()
    model_sdk.tools.assert_not_called()


@pytest.mark.parametrize("failure", ["provider", "append", "tool-call"])
def test_in_process_failed_model_turn_preserves_committed_sqlite_history_without_retry(
    tmp_path, capsys, model_sdk, failure
):
    path = tmp_path / "model-failures.sqlite3"
    options = [
        "--runtime",
        "openai",
        "--database",
        str(path),
        "--conversation-id",
        CONVERSATION_ID,
    ]
    model_sdk.create.side_effect = [model_text("Earlier answer")]
    assert main([*options, "/chat Earlier question"]) == 0
    capsys.readouterr()
    before = history(path, CONVERSATION_ID)
    if failure == "provider":
        model_sdk.create.side_effect = OpenAIError("private SDK detail")
    elif failure == "append":
        with closing(sqlite3.connect(path)) as connection:
            connection.execute("""
                CREATE TRIGGER reject_reply BEFORE INSERT
                ON tuesday_conversation_messages WHEN NEW.content = 'Reject reply'
                BEGIN SELECT RAISE(ABORT, 'Cannot persist reply'); END
            """)
        model_sdk.create.side_effect = [model_text("Reject reply")]
    else:
        tool_call = SimpleNamespace(
            type="function_call",
            call_id="call_not_executed",
            name="calculator.basic",
            arguments="{}",
        )
        model_sdk.create.side_effect = [
            SimpleNamespace(output=(tool_call,), output_text=None, model="fake-model")
        ]
    if failure == "tool-call":
        with pytest.raises(UnsupportedModelToolCallError):
            main([*options, "/chat Do not execute a tool"])
    else:
        assert main([*options, "/chat Failed later request"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    if failure != "tool-call":
        assert captured.err in (
            "tuesday: model operation failed.\n",
            "tuesday: conversation storage failed.\n",
        )
    assert history(path, CONVERSATION_ID) == before
    assert len(before.messages) == 2
    assert model_sdk.create.await_count == 2
    assert model_sdk.factory.call_count == 2
    for client in model_sdk.clients:
        client.close.assert_awaited_once()
    model_sdk.tools.assert_not_called()


def test_process_openai_missing_configuration_fails_without_network_or_database(
    tmp_path,
):
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("TUESDAY_MODEL_")
    }
    result = invoke(tmp_path, "--runtime", "openai", "/chat Hello", env=env)
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "tuesday: invalid OpenAI model configuration.\n"
    assert list(tmp_path.iterdir()) == []


def test_cleanup_failure_does_not_undo_a_committed_model_turn(
    tmp_path, capsys, model_sdk
):
    path = tmp_path / "committed-before-cleanup.sqlite3"
    model_sdk.create.side_effect = [model_text("Committed answer")]
    original = model_sdk.factory.side_effect

    def failing_cleanup(**kwargs):
        client = original(**kwargs)
        client.close.side_effect = OpenAIError("private cleanup details")
        return client

    model_sdk.factory.side_effect = failing_cleanup
    assert (
        main(
            [
                "--runtime",
                "openai",
                "--database",
                str(path),
                "--conversation-id",
                CONVERSATION_ID,
                "/chat Committed question",
            ]
        )
        == 1
    )
    assert capsys.readouterr() == ("", "tuesday: model operation failed.\n")
    saved = history(path, CONVERSATION_ID)
    assert [(message.role, message.content) for message in saved.messages] == [
        (MessageRole.USER, "/chat Committed question"),
        (MessageRole.ASSISTANT, "Committed answer"),
    ]
    model_sdk.create.assert_awaited_once()
    model_sdk.clients[0].close.assert_awaited_once()


@pytest.mark.parametrize("launcher", ["module", "installed"])
@pytest.mark.parametrize("ending", ["", ":exit\n/chat Not processed\n"])
def test_actual_interactive_entry_points_use_scripted_stdin_and_clean_stdout(
    tmp_path, launcher, ending
):
    arguments = ["--interactive", "--conversation-id", CONVERSATION_ID]
    source = "/chat Hello\n/conversation Continue\n" + ending
    if launcher == "module":
        result = invoke(tmp_path, *arguments, input_text=source)
    else:
        executable = shutil.which("tuesday", path=sysconfig.get_path("scripts"))
        assert executable is not None
        result = subprocess.run(
            [executable, *arguments],
            input=source,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
    assert result.returncode == 0, result.stderr
    assert result.stdout == (
        "TUESDAY received: Hello\n"
        "TUESDAY received: Continue (2 prior messages in context)\n"
    )
    assert CONVERSATION_ID in result.stderr
    assert "tuesday>" not in result.stderr
    assert "Hello" not in result.stderr and "Continue" not in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_interactive_process_switches_and_recovers_conversations(tmp_path):
    source = (
        "/chat First\n:new\n/chat New conversation\n"
        f":use {CONVERSATION_ID}\n/conversation Back\n:use {OTHER_ID}\n"
        "/chat Unknown selection\n:help\n:id\n:exit\n"
    )
    result = invoke(
        tmp_path,
        "--interactive",
        "--conversation-id",
        CONVERSATION_ID,
        input_text=source,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == (
        "TUESDAY received: First\nTUESDAY received: New conversation\n"
        "TUESDAY received: Back (2 prior messages in context)\n"
        "TUESDAY received: Unknown selection\n"
    )
    assert ":use UUID" in result.stderr
    assert "tuesday>" not in result.stderr


def test_interactive_sqlite_history_survives_separate_session_processes(tmp_path):
    path = tmp_path / "interactive-history.sqlite3"
    first = invoke(
        tmp_path,
        "--interactive",
        "--database",
        str(path),
        input_text="/chat First question\n:exit\n",
    )
    assert first.returncode == 0, first.stderr
    identifier = first.stderr.strip().removeprefix("Conversation: ")
    assert str(UUID(identifier)) == identifier
    second = invoke(
        tmp_path,
        "--interactive",
        "--database",
        str(path),
        "--conversation-id",
        identifier,
        input_text="/conversation Second question\n",
    )
    assert second.returncode == 0, second.stderr
    assert first.stdout == "TUESDAY received: First question\n"
    answer = "TUESDAY received: Second question (2 prior messages in context)"
    assert second.stdout == answer + "\n"
    assert [
        (message.role, message.content)
        for message in history(path, identifier).messages
    ] == [
        (MessageRole.USER, "/chat First question"),
        (MessageRole.ASSISTANT, "TUESDAY received: First question"),
        (MessageRole.USER, "/conversation Second question"),
        (MessageRole.ASSISTANT, answer),
    ]


def test_scripted_session_errors_recover_without_persisting_failed_turns(
    tmp_path,
):
    path = tmp_path / "session-errors.sqlite3"
    source = (
        ":use private-bad-id\n:unknown private\n/unknown private\n/chat\n/chat Saved\n"
    )
    result = invoke(
        tmp_path,
        "--interactive",
        "--database",
        str(path),
        "--conversation-id",
        CONVERSATION_ID,
        input_text=source,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "TUESDAY received: Saved\n"
    assert result.stderr.count("tuesday:") == 4
    assert "private" not in result.stderr
    assert [
        (message.role, message.content)
        for message in history(path, CONVERSATION_ID).messages
    ] == [
        (MessageRole.USER, "/chat Saved"),
        (MessageRole.ASSISTANT, "TUESDAY received: Saved"),
    ]
