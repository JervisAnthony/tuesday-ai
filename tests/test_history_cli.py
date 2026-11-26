"""History command parsing, one-page query boundaries, formatting, and failures."""

import importlib
import json
import sys
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest

import tuesday.cli as cli
import tuesday.history_cli as history_cli
from tuesday.conversation_export import render_transcript
from tuesday.conversations import (
    ConversationMessagePage,
    ConversationRepositoryError,
    ConversationSummary,
    ConversationSummaryPage,
)
from tuesday.domain import ConversationContext, ConversationMessage, MessageRole

A, B, MESSAGE_ID = UUID(int=481), UUID(int=482), UUID(int=1)
STAMP = datetime.fromisoformat("2026-11-26T12:30:45.123456+05:30")
EARLIER = STAMP - timedelta(days=1)
CONTENT = "  /chat café नमस्ते 🌍\nsecond line\t  "
ARGUMENT_ERROR = "tuesday: invalid history arguments; use --help.\n"


@pytest.fixture
def reader(monkeypatch):
    reader = Mock()
    reader.list_conversations = AsyncMock(
        return_value=ConversationSummaryPage(
            (
                ConversationSummary(A, 4, STAMP, EARLIER),
                ConversationSummary(B, 1, STAMP, STAMP),
            ),
            2,
            7,
            9,
        )
    )
    reader.get_messages = AsyncMock(
        return_value=ConversationMessagePage(
            A,
            (
                ConversationMessage(MessageRole.USER, CONTENT, MESSAGE_ID, STAMP),
                ConversationMessage(
                    MessageRole.ASSISTANT, "Reply\n", MESSAGE_ID, EARLIER
                ),
            ),
            2,
            7,
            9,
        )
    )
    factory = Mock(return_value=reader)
    monkeypatch.setattr(history_cli, "SQLiteConversationHistoryReader", factory)
    return reader, factory


@pytest.fixture(autouse=True)
def forbid_conversation_execution(monkeypatch):
    forbidden = Mock(
        side_effect=AssertionError("History must not execute conversations")
    )
    for name in (
        "InMemoryConversationRepository",
        "SQLiteConversationRepository",
        "StatefulConversationService",
        "create_default_orchestrator",
        "create_model_backed_orchestrator",
        "OpenAILanguageModelProvider",
        "load_settings",
        "_run_interactive",
        "_openai_runtime",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setenv("TUESDAY_MODEL_PROVIDER", "invalid-provider")
    monkeypatch.setenv("TUESDAY_MODEL_TIMEOUT_SECONDS", "not-a-number")
    monkeypatch.setenv("TUESDAY_MODEL_TEMPERATURE", "invalid")
    yield
    forbidden.assert_not_called()


def arguments(command):
    return [command, *([str(A)] if command == "show" else [])]


@pytest.mark.parametrize("command", ["list", "show"])
@pytest.mark.parametrize("process_argv", [False, True])
@pytest.mark.parametrize("limit,offset", [(1, 0), (50, 0), (200, 2**63 - 1)])
def test_exact_construction_and_one_query(
    reader, monkeypatch, capsys, command, process_argv, limit, offset
):
    instance, factory = reader
    argv = [*arguments(command), "--database", "explicit.db", "--json"]
    if (limit, offset) != (50, 0):
        argv += ["--limit", str(limit), "--offset", str(offset)]
    if process_argv:
        monkeypatch.setattr(sys, "argv", ["tuesday", "history", *argv])
        assert cli.main() == 0
    else:
        assert cli.main(["history", *argv]) == 0
    factory.assert_called_once_with("explicit.db")
    if command == "list":
        instance.list_conversations.assert_awaited_once_with(limit=limit, offset=offset)
        instance.get_messages.assert_not_called()
    else:
        instance.get_messages.assert_awaited_once_with(A, limit=limit, offset=offset)
        instance.list_conversations.assert_not_called()
    output = capsys.readouterr()
    assert output.err == "" and output.out.endswith("\n")
    assert isinstance(json.loads(output.out), dict)


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["unknown"],
        ["list"],
        ["show"],
        ["show", str(A)],
        ["list", str(A), "--database", "secret.db"],
        ["show", "--database", "secret.db"],
        ["show", "secret-invalid-uuid", "--database", "secret.db"],
        ["show", str(A), str(B), "--database", "secret.db"],
        ["list", "--database"],
        ["--database", "secret.db", "list"],
        ["list", "--database", "secret.db", "--unknown"],
    ]
    + [
        [
            command,
            *([str(A)] if command == "show" else []),
            "--database",
            "secret.db",
            *bad,
        ]
        for command in ("list", "show")
        for bad in (
            ["--limit", "0"],
            ["--limit", "-1"],
            ["--limit", "201"],
            ["--limit", "1.5"],
            ["--limit", "True"],
            ["--limit", "bad"],
            ["--limit"],
            ["--offset", "-1"],
            ["--offset", str(2**63)],
            ["--offset", "1.0"],
            ["--offset", "False"],
            ["--offset"],
            ["--runtime", "openai"],
            ["--interactive"],
            ["--max-history-messages", "1"],
            ["--conversation-id", str(A)],
            ["--dat", "secret.db"],
            ["--lim", "1"],
            ["--off", "0"],
            ["--js"],
        )
    ],
)
def test_invalid_syntax_is_fixed_private_and_precedes_construction(
    reader, capsys, argv
):
    with pytest.raises(SystemExit) as caught:
        cli.main(["history", *argv])
    assert caught.value.code == 2
    assert capsys.readouterr() == ("", ARGUMENT_ERROR)
    reader[1].assert_not_called()


@pytest.mark.parametrize("argv", [[], ["list"], ["show"]])
def test_help_requires_no_database_uuid_configuration_or_runtime(reader, capsys, argv):
    with pytest.raises(SystemExit) as caught:
        cli.main(["history", *argv, "--help"])
    assert caught.value.code == 0
    output = capsys.readouterr()
    assert output.err == "" and "usage: tuesday history" in output.out
    if argv:
        for option in ("--database", "--limit", "--offset", "--json"):
            assert option in output.out
    else:
        assert "list" in output.out and "show" in output.out
    reader[1].assert_not_called()


def test_list_json_exact_fields_types_and_first_append_metadata(reader, capsys):
    assert cli.main(["history", "list", "--database", "db", "--json"]) == 0
    output = capsys.readouterr()
    assert output.err == "" and output.out.count("\n") == 1
    assert json.loads(output.out) == {
        "items": [
            {
                "conversation_id": str(A),
                "message_count": 4,
                "first_message_at": STAMP.isoformat(),
                "last_message_at": EARLIER.isoformat(),
            },
            {
                "conversation_id": str(B),
                "message_count": 1,
                "first_message_at": STAMP.isoformat(),
                "last_message_at": STAMP.isoformat(),
            },
        ],
        "limit": 2,
        "offset": 7,
        "next_offset": 9,
    }


def test_show_json_preserves_exact_content_times_order_and_duplicate_ids(
    reader, capsys
):
    assert cli.main(["history", "show", str(A), "--database", "db", "--json"]) == 0
    output = capsys.readouterr()
    assert output.err == "" and output.out.count("\n") == 1
    assert json.loads(output.out) == {
        "conversation_id": str(A),
        "messages": [
            {
                "message_id": str(MESSAGE_ID),
                "role": "user",
                "content": CONTENT,
                "created_at": STAMP.isoformat(),
            },
            {
                "message_id": str(MESSAGE_ID),
                "role": "assistant",
                "content": "Reply\n",
                "created_at": EARLIER.isoformat(),
            },
        ],
        "limit": 2,
        "offset": 7,
        "next_offset": 9,
    }


def test_exact_list_text(reader, capsys):
    assert cli.main(["history", "list", "--database", "db"]) == 0
    assert capsys.readouterr() == (
        f"Conversation: {A}\nMessages: 4\nFirst appended: {STAMP.isoformat()}\n"
        f"Last appended: {EARLIER.isoformat()}\n\n"
        f"Conversation: {B}\nMessages: 1\nFirst appended: {STAMP.isoformat()}\n"
        f"Last appended: {STAMP.isoformat()}\n\nLimit: 2\nOffset: 7\nNext offset: 9\n",
        "",
    )


def test_exact_show_text_preserves_whitespace(reader, capsys):
    assert cli.main(["history", "show", str(A), "--database", "db"]) == 0
    assert capsys.readouterr() == (
        f"Conversation: {A}\n\nRole: user\nCreated at: {STAMP.isoformat()}\n"
        f"Message: {MESSAGE_ID}\nContent:\n{CONTENT}\n\n"
        f"Role: assistant\nCreated at: {EARLIER.isoformat()}\n"
        f"Message: {MESSAGE_ID}\nContent:\nReply\n\n\n"
        "Limit: 2\nOffset: 7\nNext offset: 9\n",
        "",
    )


@pytest.mark.parametrize("command", ["list", "show"])
@pytest.mark.parametrize("offset", [0, 99])
@pytest.mark.parametrize("as_json", [False, True])
def test_empty_pages_use_neutral_wording_and_null_next_offset(
    reader, capsys, command, offset, as_json
):
    instance, _ = reader
    instance.list_conversations.return_value = ConversationSummaryPage(
        (), 50, offset, None
    )
    instance.get_messages.return_value = ConversationMessagePage(
        A, (), 50, offset, None
    )
    assert (
        cli.main(
            [
                "history",
                *arguments(command),
                "--database",
                "db",
                "--offset",
                str(offset),
                *(["--json"] if as_json else []),
            ]
        )
        == 0
    )
    output = capsys.readouterr()
    assert output.err == ""
    if as_json:
        expected = {"limit": 50, "offset": offset, "next_offset": None}
        expected.update(
            {"items": []}
            if command == "list"
            else {
                "conversation_id": str(A),
                "messages": [],
            }
        )
        assert json.loads(output.out) == expected
    else:
        prefix = "" if command == "list" else f"Conversation: {A}\n\n"
        noun = "conversations" if command == "list" else "messages"
        assert output.out == (
            f"{prefix}No {noun} on this page.\n\n"
            f"Limit: 50\nOffset: {offset}\nNext offset: None\n"
        )


@pytest.mark.parametrize("error_type", [TypeError, ValueError])
def test_path_validation_failure_is_narrow_and_private(reader, capsys, error_type):
    reader[1].side_effect = error_type("secret path and credentials")
    assert cli.main(["history", "list", "--database", "secret"]) == 1
    assert capsys.readouterr() == ("", "tuesday: invalid database path.\n")
    reader[1].assert_called_once()
    reader[0].list_conversations.assert_not_called()


@pytest.mark.parametrize("stage", ["constructor", "query", "format"])
@pytest.mark.parametrize("interrupted", [False, True])
def test_expected_failures_have_no_partial_output_or_retry(
    reader, monkeypatch, capsys, stage, interrupted
):
    error = (
        KeyboardInterrupt() if interrupted else ConversationRepositoryError("secret")
    )
    operation = reader[1] if stage == "constructor" else reader[0].get_messages
    if stage == "format":
        operation = Mock(side_effect=error)
        monkeypatch.setattr(history_cli, "_format_page", operation)
    else:
        operation.side_effect = error
    assert cli.main(["history", "show", str(A), "--database", "secret", "--json"]) == (
        130 if interrupted else 1
    )
    expected = "interrupted." if interrupted else "conversation storage failed."
    assert capsys.readouterr() == ("", f"tuesday: {expected}\n")
    operation.assert_called_once()


@pytest.mark.parametrize(
    "stage,error_type",
    [("constructor", RuntimeError)]
    + [
        (stage, error_type)
        for stage in ("query", "format")
        for error_type in (TypeError, ValueError, RuntimeError)
    ],
)
def test_unexpected_internal_failures_propagate_without_partial_stdout(
    reader, monkeypatch, capsys, stage, error_type
):
    error = error_type("programming defect")
    operation = reader[1] if stage == "constructor" else reader[0].list_conversations
    if stage == "format":
        operation = Mock(side_effect=error)
        monkeypatch.setattr(history_cli, "_format_page", operation)
    else:
        operation.side_effect = error
    with pytest.raises(error_type) as caught:
        cli.main(["history", "list", "--database", "db"])
    assert caught.value is error
    assert capsys.readouterr() == ("", "")
    operation.assert_called_once()


def test_direct_history_main_uses_process_argv(reader, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["tuesday history", "list", "--database", "db"])
    assert history_cli.main() == 0
    assert capsys.readouterr().err == ""
    reader[1].assert_called_once_with("db")


def test_import_does_not_parse_construct_or_execute(reader, monkeypatch):
    import argparse

    parse = Mock(side_effect=AssertionError("No parsing during import"))
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse)
    importlib.reload(history_cli)
    parse.assert_not_called()
    reader[1].assert_not_called()


@pytest.fixture
def export_reader(reader):
    instance, factory = reader
    messages = instance.get_messages.return_value.messages
    instance.get_messages.return_value = ConversationMessagePage(
        A, messages, 200, 0, None
    )
    return instance, factory, ConversationContext(A, messages)


@pytest.mark.parametrize("format", [None, "json", "markdown"])
def test_export_stdout_has_exact_completed_document(export_reader, capsys, format):
    instance, factory, transcript = export_reader
    options = [] if format is None else ["--format", format]
    assert (
        cli.main(["history", "export", str(A), "--database", "explicit.db", *options])
        == 0
    )
    factory.assert_called_once_with("explicit.db")
    instance.get_messages.assert_awaited_once_with(A, limit=200, offset=0)
    instance.list_conversations.assert_not_called()
    assert capsys.readouterr() == (render_transcript(transcript, format or "json"), "")


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [str(A)],
        [str(A), "--database"],
        ["invalid-private-uuid", "--database", "private.db"],
        [str(A), str(B), "--database", "private.db"],
    ]
    + [
        [str(A), "--database", "private.db", *options]
        for options in (
            ["--overwrite"],
            ["--format", "xml"],
            ["--format"],
            ["--output"],
            ["--limit", "1"],
            ["--offset", "0"],
            ["--json"],
            ["--runtime", "openai"],
            ["--interactive"],
            ["--max-history-messages", "1"],
            ["--conversation-id", str(A)],
            ["--out", "private.txt"],
            ["--over"],
            ["--for", "json"],
            ["--unknown"],
        )
    ],
)
def test_export_syntax_is_validated_before_reader_construction(reader, capsys, bad):
    with pytest.raises(SystemExit) as caught:
        cli.main(["history", "export", *bad])
    assert caught.value.code == 2
    reader[1].assert_not_called()
    assert capsys.readouterr() == ("", ARGUMENT_ERROR)


def test_export_help_requires_no_reader_or_database(reader, capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(["history", "export", "--help"])
    assert caught.value.code == 0
    reader[1].assert_not_called()
    output = capsys.readouterr()
    assert output.err == ""
    for word in ("UUID", "--database", "--format", "--output", "--overwrite"):
        assert word in output.out
    for word in ("--limit", "--offset", "--json", "--runtime"):
        assert word not in output.out


@pytest.mark.parametrize("format", ["json", "markdown"])
@pytest.mark.parametrize("overwrite", [False, True])
def test_export_file_is_quiet_utf8_and_default_conflict_preserves_bytes(
    export_reader, capsys, tmp_path, format, overwrite
):
    destination = tmp_path / "transcript.txt"
    if overwrite:
        destination.write_bytes(b"old destination")
    options = [
        "history",
        "export",
        str(A),
        "--database",
        "explicit.db",
        "--format",
        format,
        "--output",
        str(destination),
        *(["--overwrite"] if overwrite else []),
    ]
    assert cli.main(options) == 0
    expected = render_transcript(export_reader[2], format).encode("utf-8")
    assert destination.read_bytes() == expected
    assert capsys.readouterr() == ("", "")
    if not overwrite:
        assert cli.main(options) == 1
        assert capsys.readouterr() == ("", "tuesday: transcript output failed.\n")
        assert destination.read_bytes() == expected


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure_stage", ["later-read", "render"])
def test_failed_preparation_never_opens_or_changes_destination(
    export_reader, monkeypatch, capsys, tmp_path, existing, failure_stage
):
    instance, _, _ = export_reader
    destination = tmp_path / "transcript.txt"
    if existing:
        destination.write_bytes(b"existing bytes")
    if failure_stage == "later-read":
        instance.get_messages.side_effect = [
            ConversationMessagePage(
                A, instance.get_messages.return_value.messages, 200, 0, 2
            ),
            ConversationRepositoryError("private data"),
        ]
    else:
        monkeypatch.setattr(
            history_cli, "render_transcript", Mock(side_effect=ValueError("bug"))
        )
    publish = Mock(side_effect=AssertionError("No file may be opened"))
    monkeypatch.setattr(history_cli, "publish_transcript", publish)
    options = [
        "history",
        "export",
        str(A),
        "--database",
        "db",
        "--output",
        str(destination),
        "--overwrite",
    ]
    if failure_stage == "later-read":
        assert cli.main(options) == 1
        assert capsys.readouterr() == ("", "tuesday: conversation storage failed.\n")
        assert instance.get_messages.await_count == 2
    else:
        with pytest.raises(ValueError, match="bug"):
            cli.main(options)
        assert capsys.readouterr() == ("", "")
    publish.assert_not_called()
    assert (
        destination.read_bytes() == b"existing bytes"
        if existing
        else not destination.exists()
    )


@pytest.mark.parametrize("stage", ["read", "render", "publish", "stdout"])
def test_export_interruption_returns_130_without_retry(
    export_reader, monkeypatch, capsys, stage
):
    instance, _, _ = export_reader
    options = ["history", "export", str(A), "--database", "db"]
    operation = Mock(side_effect=KeyboardInterrupt())
    if stage == "read":
        operation = instance.get_messages
        operation.side_effect = KeyboardInterrupt()
    elif stage == "render":
        monkeypatch.setattr(history_cli, "render_transcript", operation)
    elif stage == "publish":
        monkeypatch.setattr(history_cli, "publish_transcript", operation)
        options += ["--output", "explicit.txt"]
    else:
        monkeypatch.setattr(history_cli.sys.stdout, "write", operation)
    assert cli.main(options) == 130
    operation.assert_called_once()
    assert capsys.readouterr() == ("", "tuesday: interrupted.\n")


@pytest.mark.parametrize("stage", ["read", "render", "publish"])
@pytest.mark.parametrize("error_type", [TypeError, ValueError, RuntimeError])
def test_export_internal_failures_propagate(
    export_reader, monkeypatch, capsys, stage, error_type
):
    instance, _, _ = export_reader
    failure = error_type("programming defect")
    operation = Mock(side_effect=failure)
    options = ["history", "export", str(A), "--database", "db"]
    if stage == "read":
        operation = instance.get_messages
        operation.side_effect = failure
    elif stage == "render":
        monkeypatch.setattr(history_cli, "render_transcript", operation)
    else:
        monkeypatch.setattr(history_cli, "publish_transcript", operation)
        options += ["--output", "explicit.txt"]
    with pytest.raises(error_type) as caught:
        cli.main(options)
    assert caught.value is failure
    operation.assert_called_once()
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("file_output", [False, True])
@pytest.mark.parametrize("error_type", [OSError, UnicodeError])
def test_export_output_failures_are_private_and_scoped_to_io(
    export_reader, monkeypatch, capsys, file_output, error_type
):
    operation = Mock(side_effect=error_type("private path or document"))
    options = ["history", "export", str(A), "--database", "db"]
    if file_output:
        monkeypatch.setattr(history_cli, "publish_transcript", operation)
        options += ["--output", "explicit.txt"]
    else:
        monkeypatch.setattr(history_cli.sys.stdout, "write", operation)
    assert cli.main(options) == 1
    operation.assert_called_once()
    assert capsys.readouterr() == ("", "tuesday: transcript output failed.\n")


@pytest.mark.parametrize("interrupted", [False, True])
def test_export_flush_failures_are_handled_after_one_write(
    export_reader, monkeypatch, capsys, interrupted
):
    write = Mock()
    error = KeyboardInterrupt() if interrupted else OSError("private stdout detail")
    flush = Mock(side_effect=error)
    with monkeypatch.context() as patch:
        patch.setattr(history_cli.sys.stdout, "write", write)
        patch.setattr(history_cli.sys.stdout, "flush", flush)
        assert cli.main(["history", "export", str(A), "--database", "db"]) == (
            130 if interrupted else 1
        )
    write.assert_called_once_with(render_transcript(export_reader[2]))
    flush.assert_called_once()
    expected = "interrupted." if interrupted else "transcript output failed."
    assert capsys.readouterr() == ("", f"tuesday: {expected}\n")
