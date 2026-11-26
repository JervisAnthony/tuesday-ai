"""Transcript collection, deterministic serialization, and safe file publication."""

import asyncio
import json
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call
from uuid import UUID

import pytest

import tuesday.conversation_export as export
from tuesday.conversations import (
    ConversationMessagePage,
    ConversationRepositoryError,
    InMemoryConversationRepository,
    SQLiteConversationHistoryReader,
    SQLiteConversationRepository,
)
from tuesday.domain import ConversationContext, ConversationMessage, MessageRole

A, B, MESSAGE_ID = UUID(int=491), UUID(int=492), UUID(int=1)
STAMP = datetime.fromisoformat("2026-11-27T12:30:45.123456+05:30")
CONTENT = (
    "  /chat café नमस्ते 🌍\n\ttrailing  \n# Heading\n<script>raw</script>\n```md\n````  "
)


@pytest.fixture
def transcript():
    return ConversationContext(
        A,
        (
            ConversationMessage(MessageRole.USER, CONTENT, MESSAGE_ID, STAMP),
            ConversationMessage(
                MessageRole.ASSISTANT, "Reply\n", MESSAGE_ID, STAMP - timedelta(days=1)
            ),
        ),
    )


@pytest.mark.parametrize("count", [0, 1, 200, 201, 501])
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_collects_all_messages_once_in_append_order(
    tmp_path, transcript, count, backend
):
    if backend == "memory":
        writer = reader = InMemoryConversationRepository()
    else:
        path = tmp_path / "history.sqlite3"
        writer = SQLiteConversationRepository(path)
        reader = SQLiteConversationHistoryReader(path)
    expected = tuple(transcript.messages[index % 2] for index in range(count))
    if expected:
        asyncio.run(writer.append_messages(A, expected))
    asyncio.run(writer.append_messages(B, transcript.messages))
    result = asyncio.run(export.collect_transcript(reader, A))
    assert result == ConversationContext(A, expected)
    assert isinstance(result.messages, tuple)
    assert asyncio.run(export.collect_transcript(reader, UUID(int=499))).messages == ()


def test_follows_returned_offsets_without_listing_or_get_context(transcript):
    reader = Mock()
    reader.get_messages = AsyncMock(
        side_effect=[
            ConversationMessagePage(A, transcript.messages[:1], 200, 0, 1),
            ConversationMessagePage(A, transcript.messages[1:], 200, 1, None),
        ]
    )
    result = asyncio.run(export.collect_transcript(reader, A))
    assert result == transcript
    assert reader.get_messages.await_args_list == [
        call(A, limit=200, offset=0),
        call(A, limit=200, offset=1),
    ]
    reader.list_conversations.assert_not_called()
    reader.get_context.assert_not_called()


@pytest.mark.parametrize("identifier", ["bad", None, True])
def test_collection_validates_uuid_before_reading(identifier):
    reader = Mock(get_messages=AsyncMock())
    with pytest.raises(TypeError):
        asyncio.run(export.collect_transcript(reader, identifier))
    reader.get_messages.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [
        "type",
        "conversation",
        "limit",
        "offset",
        "too-many",
        "same-offset",
        "wrong-next",
        "overflow",
    ],
)
def test_rejects_inconsistent_pages_before_looping_or_returning(transcript, failure):
    good = ConversationMessagePage(A, transcript.messages, 200, 0, None)
    page = {
        "type": object(),
        "conversation": replace(good, conversation_id=B),
        "limit": replace(good, limit=199),
        "offset": replace(good, offset=1),
        "too-many": replace(good, messages=(transcript.messages[0],) * 201),
        "same-offset": replace(good, messages=(), next_offset=0),
        "wrong-next": replace(good, next_offset=1),
        "overflow": replace(good, next_offset=2**63),
    }[failure]
    reader = Mock(get_messages=AsyncMock(return_value=page))
    with pytest.raises(ConversationRepositoryError, match="Inconsistent"):
        asyncio.run(export.collect_transcript(reader, A))
    reader.get_messages.assert_awaited_once_with(A, limit=200, offset=0)


def test_rejects_later_page_with_stale_offset(transcript):
    reader = Mock(
        get_messages=AsyncMock(
            side_effect=[
                ConversationMessagePage(A, transcript.messages, 200, 0, 2),
                ConversationMessagePage(A, transcript.messages, 200, 0, 2),
            ]
        )
    )
    with pytest.raises(ConversationRepositoryError):
        asyncio.run(export.collect_transcript(reader, A))
    assert reader.get_messages.await_count == 2


def test_later_read_failure_propagates_without_retry(transcript):
    failure = ConversationRepositoryError("private persisted content")
    reader = Mock(
        get_messages=AsyncMock(
            side_effect=[
                ConversationMessagePage(A, transcript.messages, 200, 0, 2),
                failure,
            ]
        )
    )
    with pytest.raises(ConversationRepositoryError) as caught:
        asyncio.run(export.collect_transcript(reader, A))
    assert caught.value is failure and reader.get_messages.await_count == 2


def test_json_has_exact_version_one_schema_and_lossless_content(transcript):
    document = export.render_transcript(transcript)
    assert document == export.render_transcript(transcript, "json")
    assert document == export.render_transcript(transcript)
    assert document.endswith("\n") and not document.endswith("\n\n")
    payload = json.loads(document)
    assert payload == {
        "format": "tuesday.conversation",
        "version": 1,
        "conversation_id": str(A),
        "messages": [
            {
                "message_id": str(message.message_id),
                "role": message.role.value,
                "content": message.content,
                "created_at": message.created_at.isoformat(),
            }
            for message in transcript.messages
        ],
    }
    assert type(payload["version"]) is int
    assert document == json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    assert "café नमस्ते 🌍" in document
    assert payload["messages"][0]["message_id"] == payload["messages"][1]["message_id"]


def test_exact_empty_json_and_markdown():
    transcript = ConversationContext(A)
    assert export.render_transcript(transcript) == (
        '{\n  "format": "tuesday.conversation",\n  "version": 1,\n'
        f'  "conversation_id": "{A}",\n  "messages": []\n}}\n'
    )
    assert export.render_transcript(transcript, "markdown") == (
        f"# Conversation transcript\n\nConversation: {A}\n"
        "Format: tuesday.conversation\nVersion: 1\nMessages: 0\n"
    )


@pytest.mark.parametrize(
    "content,fence",
    [
        ("Plain\ntext  ", "```"),
        ("`inline`", "```"),
        ("```md\n# Heading\n<html>\n```", "````"),
        ("``````\n~", "```````"),
        ("First\r\nsecond\t  \n", "```"),
    ],
)
def test_markdown_literal_fences_preserve_content_without_structure(
    transcript, content, fence
):
    message = replace(transcript.messages[0], content=content)
    context = ConversationContext(A, (message, message))
    expected = (
        f"# Conversation transcript\n\nConversation: {A}\n"
        "Format: tuesday.conversation\nVersion: 1\nMessages: 2\n\n"
        f"## Message 1\n\nRole: user\nMessage: {MESSAGE_ID}\n"
        f"Created at: {STAMP.isoformat()}\n\n{fence}text\n{content}\n{fence}\n\n"
        f"## Message 2\n\nRole: user\nMessage: {MESSAGE_ID}\n"
        f"Created at: {STAMP.isoformat()}\n\n{fence}text\n{content}\n{fence}\n"
    )
    assert export.render_transcript(context, "markdown") == expected
    assert export.render_transcript(context, "markdown") == expected


def test_unsupported_render_format_is_a_programmatic_error(transcript):
    with pytest.raises(ValueError, match="Unsupported"):
        export.render_transcript(transcript, "xml")


@pytest.fixture
def files(tmp_path):
    source, destination = tmp_path / "source.sqlite3", tmp_path / "transcript.json"
    source.write_bytes(b"source bytes")
    return source, destination


@pytest.mark.parametrize("overwrite", [False, True])
def test_publication_is_quiet_utf8_without_bom_or_generated_crlf(
    files, transcript, overwrite
):
    source, destination = files
    if overwrite:
        destination.write_bytes(b"previous destination")
    document = export.render_transcript(transcript, "markdown")
    export.publish_transcript(
        document, database=source, output=destination, overwrite=overwrite
    )
    assert destination.read_bytes() == document.encode("utf-8")
    assert b"\r\n" not in destination.read_bytes()
    assert not destination.read_bytes().startswith(b"\xef\xbb\xbf")
    assert source.read_bytes() == b"source bytes"
    assert sorted(path.name for path in source.parent.iterdir()) == [
        source.name,
        destination.name,
    ]


def test_no_clobber_is_enforced_at_publication_even_when_destination_appears(
    files, monkeypatch
):
    source, destination = files
    original = export.os.link

    def race(temporary, target):
        destination.write_bytes(b"concurrently created destination")
        return original(temporary, target)

    monkeypatch.setattr(export.os, "link", race)
    with pytest.raises(FileExistsError):
        export.publish_transcript("complete\n", database=source, output=destination)
    assert destination.read_bytes() == b"concurrently created destination"
    assert not list(source.parent.glob(".tuesday-transcript-*"))


def test_default_conflict_preserves_existing_file_and_unrelated_temp(files):
    source, destination = files
    destination.write_bytes(b"existing bytes")
    unrelated = source.parent / ".tuesday-transcript-not-owned.tmp"
    unrelated.write_bytes(b"unrelated")
    with pytest.raises(FileExistsError):
        export.publish_transcript("replacement\n", database=source, output=destination)
    assert destination.read_bytes() == b"existing bytes"
    assert list(source.parent.glob(".tuesday-transcript-*")) == [unrelated]
    assert unrelated.read_bytes() == b"unrelated"


@pytest.mark.parametrize("stage", ["write", "close", "replace", "create"])
@pytest.mark.parametrize("interrupted", [False, True])
def test_preparation_or_replace_failure_preserves_previous_destination_and_cleans(
    files, monkeypatch, stage, interrupted
):
    source, destination = files
    destination.write_bytes(b"previous bytes")
    failure = KeyboardInterrupt() if interrupted else OSError("private path")
    original = export.tempfile.NamedTemporaryFile
    operation = Mock(side_effect=failure)

    @contextmanager
    def temporary(**kwargs):
        with original(**kwargs) as stream:

            def write(document):
                stream.write(document[:3])
                return operation()

            yield (
                SimpleNamespace(name=stream.name, write=write)
                if stage == "write"
                else stream
            )
        if stage == "close":
            operation()

    if stage == "create":
        monkeypatch.setattr(export.tempfile, "NamedTemporaryFile", operation)
    elif stage in ("write", "close"):
        monkeypatch.setattr(export.tempfile, "NamedTemporaryFile", temporary)
    else:
        monkeypatch.setattr(export.os, "replace", operation)
    with pytest.raises(type(failure)) as caught:
        export.publish_transcript(
            "complete\n", database=source, output=destination, overwrite=True
        )
    assert caught.value is failure
    operation.assert_called_once()
    assert destination.read_bytes() == b"previous bytes"
    assert not list(source.parent.glob(".tuesday-transcript-*"))


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
def test_source_and_sidecars_are_protected_with_relative_and_absolute_aliases(
    files, monkeypatch, overwrite, suffix
):
    source, _ = files
    protected = Path(str(source) + suffix)
    protected.write_bytes(b"protected bytes")
    monkeypatch.chdir(source.parent)
    for destination in (
        protected,
        Path(protected.name),
        source.parent / ".." / source.parent.name / protected.name,
    ):
        with pytest.raises(OSError, match="protected"):
            export.publish_transcript(
                "replacement\n",
                database=source.name,
                output=destination,
                overwrite=overwrite,
            )
    assert protected.read_bytes() == b"protected bytes"
    assert not list(source.parent.glob(".tuesday-transcript-*"))


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
def test_hard_link_aliases_of_source_and_sidecars_are_protected(files, suffix):
    source, destination = files
    protected = Path(str(source) + suffix)
    protected.write_bytes(b"protected bytes")
    destination.hardlink_to(protected)
    with pytest.raises(OSError, match="protected"):
        export.publish_transcript(
            "replacement\n", database=source, output=destination, overwrite=True
        )
    assert destination.read_bytes() == protected.read_bytes() == b"protected bytes"


def create_symlink(link, target):
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"Native symlink creation unavailable: {exc.winerror}")


@pytest.mark.parametrize("overwrite", [False, True])
def test_real_symbolic_link_destinations_are_rejected(files, overwrite):
    source, destination = files
    target = source.parent / "target.txt"
    target.write_bytes(b"target bytes")
    create_symlink(destination, target)
    with pytest.raises(OSError, match="symbolic link"):
        export.publish_transcript(
            "replacement\n", database=source, output=destination, overwrite=overwrite
        )
    assert destination.is_symlink() and target.read_bytes() == b"target bytes"


def test_resolved_source_symlink_alias_protects_real_sidecars(files):
    source, _ = files
    alias = source.parent / "source-alias.sqlite3"
    create_symlink(alias, source)
    sidecar = Path(str(source) + "-wal")
    with pytest.raises(OSError, match="protected"):
        export.publish_transcript(
            "replacement\n", database=alias, output=sidecar, overwrite=True
        )
    assert not sidecar.exists()


def test_symlink_check_precedes_any_temporary_write(files, monkeypatch):
    source, destination = files
    monkeypatch.setattr(Path, "is_symlink", lambda self: self == destination)
    temporary = Mock(side_effect=AssertionError("No output may be opened"))
    monkeypatch.setattr(export.tempfile, "NamedTemporaryFile", temporary)
    with pytest.raises(OSError, match="symbolic link"):
        export.publish_transcript("document\n", database=source, output=destination)
    temporary.assert_not_called()


@pytest.mark.parametrize("destination", ["", " \t", "bad\x00name"])
def test_invalid_destination_paths_are_expected_output_errors(files, destination):
    source, _ = files
    with pytest.raises(OSError):
        export.publish_transcript("document\n", database=source, output=destination)


def test_missing_parent_is_not_created(files):
    source, _ = files
    destination = source.parent / "missing" / "file.txt"
    with pytest.raises(FileNotFoundError):
        export.publish_transcript("document\n", database=source, output=destination)
    assert not destination.parent.exists()


@pytest.mark.parametrize("stage", ["source", "destination"])
def test_invalid_path_resolution_is_scoped_to_filesystem_operations(
    files, monkeypatch, stage
):
    source, destination = files
    original = Path.resolve

    def resolve(path, **kwargs):
        if path == (source if stage == "source" else destination):
            raise ValueError("private invalid path")
        return original(path, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(OSError, match="Invalid transcript destination") as caught:
        export.publish_transcript("document\n", database=source, output=destination)
    assert isinstance(caught.value.__cause__, ValueError)
    assert not destination.exists()


def test_directory_destination_is_preserved(files):
    source, destination = files
    destination.mkdir()
    with pytest.raises(OSError):
        export.publish_transcript(
            "document\n", database=source, output=destination, overwrite=True
        )
    assert destination.is_dir() and list(destination.iterdir()) == []


def test_cleanup_failure_is_best_effort_and_does_not_delete_published_file(
    files, monkeypatch
):
    source, destination = files
    original = Path.unlink

    def fail_cleanup(path, **kwargs):
        if path.name.startswith(".tuesday-transcript-"):
            raise PermissionError("private cleanup failure")
        return original(path, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    export.publish_transcript("document\n", database=source, output=destination)
    assert destination.read_bytes() == b"document\n"
    leftovers = list(source.parent.glob(".tuesday-transcript-*"))
    assert len(leftovers) == 1
    original(leftovers[0])
