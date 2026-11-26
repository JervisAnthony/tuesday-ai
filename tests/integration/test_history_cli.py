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
from tuesday.conversation_export import render_transcript
from tuesday.conversations import SQLiteConversationRepository
from tuesday.domain import ConversationContext, ConversationMessage, MessageRole

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
        ["history", "export", "--help"],
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


@pytest.mark.parametrize("command", ["list", "export"])
def test_clean_imports_and_history_execution_never_activate_application_layers(
    tmp_path, command
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
arguments = ["history", sys.argv[2]]
if sys.argv[2] == "export":
    arguments += ["00000000-0000-0000-0000-000000000049"]
else:
    arguments += ["--json"]
assert cli.main([*arguments, "--database", sys.argv[1]]) == 0
"""
    path = tmp_path / "empty.sqlite3"
    SQLiteConversationRepository(path)
    before = path.read_bytes()
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), command],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    key = "items" if command == "list" else "messages"
    assert result.stderr == "" and json.loads(result.stdout)[key] == []
    assert path.read_bytes() == before


@pytest.fixture
def export_database(database):
    path, original = database
    messages = original + tuple(
        ConversationMessage(
            MessageRole.USER,
            f"  /conversation Message {index}\t\n# Heading\n<html>\n````md\n```  ",
            MESSAGE_ID,
            STAMP - timedelta(seconds=index),
        )
        for index in range(3, 501)
    )
    asyncio.run(SQLiteConversationRepository(path).append_messages(A, messages[3:]))
    return path, ConversationContext(A, messages)


@pytest.mark.parametrize("launcher", ["module", "installed"])
@pytest.mark.parametrize("format", ["json", "markdown"])
def test_real_export_stdout_collects_multiple_pages_and_preserves_source(
    tmp_path, export_database, launcher, format
):
    path, transcript = export_database
    before_bytes, before_state = path.read_bytes(), inspect_database(path)
    before_files = set(tmp_path.iterdir())
    options = [] if format == "json" else ["--format", "markdown"]
    result = invoke(
        tmp_path,
        launcher,
        "history",
        "export",
        str(A),
        "--database",
        str(path),
        *options,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == "" and result.stdout.endswith("\n")
    if format == "json":
        payload = json.loads(result.stdout)
        assert set(payload) == {"format", "version", "conversation_id", "messages"}
        assert (
            payload["format"] == "tuesday.conversation"
            and type(payload["version"]) is int
        )
        assert payload["version"] == 1 and payload["conversation_id"] == str(A)
        assert payload["messages"] == [
            {
                "message_id": str(message.message_id),
                "role": message.role.value,
                "content": message.content,
                "created_at": message.created_at.isoformat(),
            }
            for message in transcript.messages
        ]
        assert result.stdout == json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    else:
        assert result.stdout == render_transcript(transcript, "markdown")
        assert "Messages: 501\n" in result.stdout
        assert result.stdout.count("\n## Message ") == 501
        assert "`````text\n" in result.stdout
        for message in transcript.messages:
            assert message.content in result.stdout
    assert path.read_bytes() == before_bytes and inspect_database(path) == before_state
    assert set(tmp_path.iterdir()) == before_files


@pytest.mark.parametrize("launcher", ["module", "installed"])
@pytest.mark.parametrize("format", ["json", "markdown"])
def test_real_file_export_no_clobber_and_explicit_overwrite(
    tmp_path, export_database, launcher, format
):
    path, transcript = export_database
    before_bytes, before_state = path.read_bytes(), inspect_database(path)
    destination = tmp_path / "transcript.txt"
    options = [
        "history",
        "export",
        str(A),
        "--database",
        str(path),
        "--format",
        format,
        "--output",
        str(destination),
    ]
    result = invoke(tmp_path, launcher, *options)
    assert result.returncode == 0 and result.stdout == result.stderr == ""
    expected = render_transcript(transcript, format).encode("utf-8")
    assert destination.read_bytes() == expected
    assert b"\r\n" not in expected and not expected.startswith(b"\xef\xbb\xbf")
    result = invoke(tmp_path, launcher, *options)
    assert result.returncode == 1 and result.stdout == ""
    assert result.stderr == "tuesday: transcript output failed.\n"
    assert destination.read_bytes() == expected
    destination.write_bytes(b"previous destination bytes")
    result = invoke(tmp_path, launcher, *options, "--overwrite")
    assert result.returncode == 0 and result.stdout == result.stderr == ""
    assert destination.read_bytes() == expected
    assert path.read_bytes() == before_bytes and inspect_database(path) == before_state
    assert not list(tmp_path.glob(".tuesday-transcript-*"))


@pytest.mark.parametrize("launcher", ["module", "installed"])
@pytest.mark.parametrize("format", ["json", "markdown"])
def test_real_unknown_conversation_exports_a_valid_empty_document(
    tmp_path, database, launcher, format
):
    path, _ = database
    result = invoke(
        tmp_path,
        launcher,
        "history",
        "export",
        str(UNKNOWN),
        "--database",
        str(path),
        "--format",
        format,
    )
    assert result.returncode == 0 and result.stderr == ""
    if format == "json":
        assert json.loads(result.stdout) == {
            "format": "tuesday.conversation",
            "version": 1,
            "conversation_id": str(UNKNOWN),
            "messages": [],
        }
    else:
        assert result.stdout == (
            f"# Conversation transcript\n\nConversation: {UNKNOWN}\n"
            "Format: tuesday.conversation\nVersion: 1\nMessages: 0\n"
        )


@pytest.mark.parametrize("failure", ["missing", "corrupt", "later-page"])
@pytest.mark.parametrize("existing", [False, True])
def test_real_failed_export_never_creates_or_replaces_destination(
    tmp_path, export_database, failure, existing
):
    path, _ = export_database
    if failure == "missing":
        path = tmp_path / "missing" / "private.sqlite3"
    elif failure == "corrupt":
        path.write_bytes(b"private corrupt database")
    else:
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute(
                "UPDATE tuesday_conversation_messages SET created_at=? "
                "WHERE sequence=502",
                ("private invalid last-page timestamp",),
            )
    destination = tmp_path / "transcript.txt"
    if existing:
        destination.write_bytes(b"original output bytes")
    before = path.read_bytes() if path.exists() else None
    result = invoke(
        tmp_path,
        "module",
        "history",
        "export",
        str(A),
        "--database",
        str(path),
        "--output",
        str(destination),
        "--overwrite",
    )
    assert result.returncode == 1 and result.stdout == ""
    assert result.stderr == "tuesday: conversation storage failed.\n"
    assert (
        destination.read_bytes() == b"original output bytes"
        if existing
        else not destination.exists()
    )
    if before is None:
        assert not path.exists() and not path.parent.exists()
    else:
        assert path.read_bytes() == before
    assert not list(tmp_path.glob(".tuesday-transcript-*"))


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
def test_real_export_rejects_source_and_sidecar_destinations(
    tmp_path, database, suffix
):
    path, _ = database
    destination = str(path) + suffix
    before = path.read_bytes()
    result = invoke(
        tmp_path,
        "module",
        "history",
        "export",
        str(A),
        "--database",
        str(path),
        "--output",
        destination,
        "--overwrite",
    )
    assert result.returncode == 1 and result.stdout == ""
    assert result.stderr == "tuesday: transcript output failed.\n"
    assert path.read_bytes() == before
    if suffix:
        assert not os.path.exists(destination)


def test_real_export_missing_output_parent_stays_missing(tmp_path, database):
    path, _ = database
    destination = tmp_path / "missing" / "transcript.json"
    result = invoke(
        tmp_path,
        "installed",
        "history",
        "export",
        str(A),
        "--database",
        str(path),
        "--output",
        str(destination),
    )
    assert result.returncode == 1 and result.stdout == ""
    assert result.stderr == "tuesday: transcript output failed.\n"
    assert not destination.parent.exists()


def test_clean_export_imports_do_not_parse_connect_or_create_files(tmp_path):
    script = """
import argparse, os, sqlite3, tempfile
def forbidden(*args, **kwargs):
    raise AssertionError("Import must not parse, connect, execute, or write")
argparse.ArgumentParser.parse_args = forbidden
sqlite3.connect = tempfile.NamedTemporaryFile = forbidden
os.link = os.replace = forbidden
import tuesday.conversation_export
import tuesday.history_cli
import tuesday.cli
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
