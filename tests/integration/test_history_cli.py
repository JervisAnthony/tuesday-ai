"""Real history command processes, existing-file inspection, and storage safety."""

import asyncio
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import sysconfig
from contextlib import closing
from datetime import datetime, timedelta
from uuid import UUID

import pytest

from tuesday.cli import main
from tuesday.conversations import SQLiteConversationRepository
from tuesday.domain import ConversationMessage, MessageRole

A, B, UNKNOWN, MESSAGE_ID = (UUID(int=value) for value in (481, 482, 499, 1))
STAMP = datetime.fromisoformat("2026-11-26T12:30:45.123456+05:30")
CONTENT = "  /chat café नमस्ते 🌍\nsecond line\t  "


def invoke(tmp_path, launcher, *arguments):
    if launcher == "module":
        prefix = [sys.executable, "-m", "tuesday"]
    else:
        executable = shutil.which("tuesday", path=sysconfig.get_path("scripts"))
        assert executable is not None
        prefix = [executable]
    environment = dict(os.environ)
    environment.update(
        PYTHONIOENCODING="utf-8",
        TUESDAY_MODEL_PROVIDER="invalid-provider",
        TUESDAY_MODEL_TIMEOUT_SECONDS="invalid-number",
        TUESDAY_MODEL_TEMPERATURE="invalid-number",
    )
    return subprocess.run(
        [*prefix, *arguments],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "existing # ü.sqlite3"
    writer = SQLiteConversationRepository(path)
    messages = (
        ConversationMessage(MessageRole.SYSTEM, "System", MESSAGE_ID, STAMP),
        ConversationMessage(MessageRole.USER, CONTENT, MESSAGE_ID, STAMP),
        ConversationMessage(
            MessageRole.ASSISTANT, "Reply\n", MESSAGE_ID, STAMP - timedelta(days=1)
        ),
    )
    asyncio.run(writer.append_messages(A, messages[:1]))
    asyncio.run(writer.append_messages(B, messages[:1]))
    asyncio.run(writer.append_messages(A, messages[1:]))
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("PRAGMA user_version=48")
        connection.execute("CREATE TABLE unrelated (value TEXT)")
        connection.execute("INSERT INTO unrelated VALUES ('preserved')")
    return path, messages


def inspect_database(path):
    with closing(sqlite3.connect(path)) as connection:
        return (
            connection.execute("SELECT * FROM sqlite_master ORDER BY name").fetchall(),
            connection.execute("PRAGMA user_version").fetchone(),
            connection.execute("SELECT * FROM unrelated").fetchall(),
            connection.execute(
                "SELECT * FROM tuesday_conversation_messages ORDER BY sequence"
            ).fetchall(),
        )


@pytest.mark.parametrize("launcher", ["module", "installed"])
def test_real_pages_preserve_database_and_exact_persisted_values(
    tmp_path, database, launcher
):
    path, messages = database
    before_bytes, before_state = path.read_bytes(), inspect_database(path)
    before_files = set(tmp_path.iterdir())
    options = ["--database", str(path), "--limit", "1", "--json"]
    listed = []
    for offset, identifier, next_offset in ((0, A, 1), (1, B, None)):
        result = invoke(
            tmp_path, launcher, "history", "list", *options, "--offset", str(offset)
        )
        assert result.returncode == 0, result.stderr
        assert result.stderr == "" and result.stdout.count("\n") == 1
        payload = json.loads(result.stdout)
        assert set(payload) == {"items", "limit", "offset", "next_offset"}
        assert payload["limit"] == 1 and payload["offset"] == offset
        assert payload["next_offset"] == next_offset
        listed.extend(payload["items"])
        assert payload["items"][0]["conversation_id"] == str(identifier)
    assert listed[0] == {
        "conversation_id": str(A),
        "message_count": 3,
        "first_message_at": STAMP.isoformat(),
        "last_message_at": (STAMP - timedelta(days=1)).isoformat(),
    }
    actual, offset = [], 0
    while True:
        result = invoke(
            tmp_path,
            launcher,
            "history",
            "show",
            str(A),
            *options,
            "--offset",
            str(offset),
        )
        assert result.returncode == 0, result.stderr
        assert result.stderr == "" and result.stdout.count("\n") == 1
        payload = json.loads(result.stdout)
        assert set(payload) == {
            "conversation_id",
            "messages",
            "limit",
            "offset",
            "next_offset",
        }
        assert payload["conversation_id"] == str(A)
        assert payload["limit"] == 1 and payload["offset"] == offset
        assert len(payload["messages"]) == 1
        actual.extend(payload["messages"])
        if payload["next_offset"] is None:
            break
        assert payload["next_offset"] == offset + 1
        offset = payload["next_offset"]
    assert actual == [
        {
            "message_id": str(message.message_id),
            "role": message.role.value,
            "content": message.content,
            "created_at": message.created_at.isoformat(),
        }
        for message in messages
    ]
    assert path.read_bytes() == before_bytes
    assert inspect_database(path) == before_state
    assert set(tmp_path.iterdir()) == before_files


@pytest.mark.parametrize("launcher", ["module", "installed"])
@pytest.mark.parametrize(
    "command,identifier,offset",
    [
        ("list", None, 99),
        ("show", A, 99),
        ("show", UNKNOWN, 0),
    ],
)
@pytest.mark.parametrize("as_json", [False, True])
def test_real_empty_pages_succeed_without_claiming_no_history(
    tmp_path, database, launcher, command, identifier, offset, as_json
):
    path, _ = database
    before = path.read_bytes()
    result = invoke(
        tmp_path,
        launcher,
        "history",
        command,
        *([str(identifier)] if identifier else []),
        "--database",
        str(path),
        "--offset",
        str(offset),
        *(["--json"] if as_json else []),
    )
    assert result.returncode == 0 and result.stderr == ""
    if as_json:
        expected = {"limit": 50, "offset": offset, "next_offset": None}
        expected.update(
            {"items": []}
            if command == "list"
            else {
                "conversation_id": str(identifier),
                "messages": [],
            }
        )
        assert json.loads(result.stdout) == expected
    else:
        prefix = "" if command == "list" else f"Conversation: {identifier}\n\n"
        noun = "conversations" if command == "list" else "messages"
        assert result.stdout == (
            f"{prefix}No {noun} on this page.\n\nLimit: 50\nOffset: {offset}\n"
            "Next offset: None\n"
        )
    assert path.read_bytes() == before


@pytest.mark.parametrize("launcher", ["module", "installed"])
@pytest.mark.parametrize("command", ["list", "show"])
def test_missing_database_and_parent_remain_missing(tmp_path, launcher, command):
    path = tmp_path / "missing" / "private.sqlite3"
    result = invoke(
        tmp_path,
        launcher,
        "history",
        command,
        *([str(A)] if command == "show" else []),
        "--database",
        str(path),
        "--json",
    )
    assert result.returncode == 1
    assert (
        result.stdout == ""
        and result.stderr == "tuesday: conversation storage failed.\n"
    )
    assert not path.exists() and not path.parent.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "failure",
    [
        "corrupt",
        "missing-table",
        "incompatible",
        "uuid",
        "role",
        "content",
        "timestamp",
    ],
)
def test_real_storage_failures_are_private_and_leave_bytes_unchanged(
    tmp_path, database, failure
):
    path, _ = database
    command = ["show", str(A)]
    if failure == "corrupt":
        path.write_bytes(b"private corrupted content")
    else:
        with closing(sqlite3.connect(path)) as connection, connection:
            if failure == "missing-table":
                connection.execute("DROP TABLE tuesday_conversation_messages")
            elif failure == "incompatible":
                connection.execute(
                    "ALTER TABLE tuesday_conversation_messages ADD extra TEXT"
                )
            else:
                connection.execute("PRAGMA ignore_check_constraints=ON")
                column, value = {
                    "uuid": ("message_id", "secret invalid UUID"),
                    "role": ("role", "secret invalid role"),
                    "content": ("content", " \t"),
                    "timestamp": ("created_at", "secret invalid timestamp"),
                }[failure]
                connection.execute(
                    f"UPDATE tuesday_conversation_messages SET {column}=? "
                    "WHERE sequence=1",
                    (value,),
                )
    before = path.read_bytes()
    result = invoke(
        tmp_path, "module", "history", *command, "--database", str(path), "--json"
    )
    assert result.returncode == 1
    assert (
        result.stdout == ""
        and result.stderr == "tuesday: conversation storage failed.\n"
    )
    assert path.read_bytes() == before


@pytest.mark.parametrize("path", ["", " \t", ":memory:", "file:private.sqlite3"])
def test_real_invalid_path_configuration_uses_exit_one(tmp_path, path):
    result = invoke(tmp_path, "module", "history", "list", "--database", path)
    assert result.returncode == 1
    assert result.stdout == "" and result.stderr == "tuesday: invalid database path.\n"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("launcher", ["module", "installed"])
@pytest.mark.parametrize(
    "argv",
    [
        ["--help"],
        ["history", "--help"],
        ["history", "list", "--help"],
        ["history", "show", "--help"],
    ],
)
def test_real_help_without_storage_or_valid_model_configuration(
    tmp_path, launcher, argv
):
    result = invoke(tmp_path, launcher, *argv)
    assert result.returncode == 0 and result.stderr == ""
    assert "usage:" in result.stdout and "history" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_real_invalid_syntax_has_exit_two_and_no_database_creation(tmp_path):
    path = tmp_path / "never-created.sqlite3"
    result = invoke(
        tmp_path,
        "module",
        "history",
        "show",
        "private-invalid-uuid",
        "--database",
        str(path),
    )
    assert result.returncode == 2 and result.stdout == ""
    assert result.stderr == "tuesday: invalid history arguments; use --help.\n"
    assert not path.exists()


def test_listed_uuid_can_resume_existing_conversation_command(tmp_path, capsys):
    path = tmp_path / "resumable.sqlite3"
    assert main(["--database", str(path), "/chat First"]) == 0
    capsys.readouterr()
    result = invoke(
        tmp_path, "module", "history", "list", "--database", str(path), "--json"
    )
    assert result.returncode == 0 and result.stderr == ""
    identifier = json.loads(result.stdout)["items"][0]["conversation_id"]
    assert (
        main(
            [
                "--database",
                str(path),
                "--conversation-id",
                identifier,
                "/conversation Continue",
            ]
        )
        == 0
    )
    assert capsys.readouterr() == (
        "TUESDAY received: Continue (2 prior messages in context)\n",
        "",
    )


def test_clean_imports_and_history_execution_never_activate_application_layers(
    tmp_path,
):
    script = """
import sys
import tuesday.cli as cli
import tuesday.history_cli
import tuesday.agents as agents
import tuesday.config as config
import tuesday.language_models as models
from tuesday.conversations import SQLiteConversationRepository
from tuesday.services.openai_tool_turn import OpenAIToolTurnService
from tuesday.tools import BasicCalculatorTool
def forbidden(*args, **kwargs):
    raise AssertionError("History must not activate application components")
config.load_settings = cli.load_settings = forbidden
cli.create_default_orchestrator = cli.create_model_backed_orchestrator = forbidden
cli.StatefulConversationService = cli.InMemoryConversationRepository = forbidden
for component in (SQLiteConversationRepository, models.OpenAILanguageModelProvider,
    agents.ModelBackedConversationalAgent, agents.GuardedOpenAIConversationalAgent,
    OpenAIToolTurnService, BasicCalculatorTool):
    component.__init__ = forbidden
assert cli.main(["history", "list", "--database", sys.argv[1], "--json"]) == 0
"""
    path = tmp_path / "empty.sqlite3"
    SQLiteConversationRepository(path)
    before = path.read_bytes()
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == "" and json.loads(result.stdout)["items"] == []
    assert path.read_bytes() == before
