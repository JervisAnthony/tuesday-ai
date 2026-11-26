"""Collect, render, and publish transcripts through the read-only query boundary."""

import json
import os
import re
import tempfile
from contextlib import suppress
from pathlib import Path
from uuid import UUID

from tuesday.conversations import (
    BaseConversationHistoryReader,
    ConversationMessagePage,
    ConversationRepositoryError,
)
from tuesday.conversations.history import _validate_conversation_id
from tuesday.domain import ConversationContext

_PAGE_SIZE = 200


async def collect_transcript(
    reader: BaseConversationHistoryReader, conversation_id: UUID
) -> ConversationContext:
    """Follow current-data pages, rejecting mismatched or inconsistent pagination."""
    _validate_conversation_id(conversation_id)
    messages = []
    offset = 0
    while True:
        page = await reader.get_messages(
            conversation_id, limit=_PAGE_SIZE, offset=offset
        )
        if (
            not isinstance(page, ConversationMessagePage)
            or page.conversation_id != conversation_id
            or page.limit != _PAGE_SIZE
            or page.offset != offset
            or len(page.messages) > _PAGE_SIZE
            or (
                page.next_offset is not None
                and (
                    page.next_offset != offset + len(page.messages)
                    or not offset < page.next_offset <= 2**63 - 1
                )
            )
        ):
            raise ConversationRepositoryError("Inconsistent transcript history page.")
        messages.extend(page.messages)
        if page.next_offset is None:
            return ConversationContext(conversation_id, tuple(messages))
        offset = page.next_offset


def render_transcript(transcript: ConversationContext, format: str = "json") -> str:
    """Render a deterministic version-one document, including its final newline."""
    if format == "json":
        return (
            json.dumps(
                {
                    "format": "tuesday.conversation",
                    "version": 1,
                    "conversation_id": str(transcript.conversation_id),
                    "messages": [
                        {
                            "message_id": str(message.message_id),
                            "role": message.role.value,
                            "content": message.content,
                            "created_at": message.created_at.isoformat(),
                        }
                        for message in transcript.messages
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
    if format != "markdown":
        raise ValueError("Unsupported transcript format.")
    sections = [
        "# Conversation transcript\n\n"
        f"Conversation: {transcript.conversation_id}\n"
        "Format: tuesday.conversation\n"
        "Version: 1\n"
        f"Messages: {len(transcript.messages)}"
    ]
    for index, message in enumerate(transcript.messages, 1):
        fence = "`" * max(
            3,
            1
            + max((len(run) for run in re.findall(r"`+", message.content)), default=0),
        )
        sections.append(
            f"## Message {index}\n\n"
            f"Role: {message.role.value}\n"
            f"Message: {message.message_id}\n"
            f"Created at: {message.created_at.isoformat()}\n\n"
            f"{fence}text\n{message.content}\n{fence}"
        )
    return "\n\n".join(sections) + "\n"


def _check_destination(source: Path, destination: Path) -> None:
    """Check only output symlinks and aliases of the source and its sidecars."""
    try:
        if destination.is_symlink():
            raise OSError("Transcript destination must not be a symbolic link.")
        resolved = destination.resolve()
        for suffix in ("", "-wal", "-shm", "-journal"):
            protected = Path(str(source) + suffix)
            if resolved == protected or (
                destination.exists()
                and protected.exists()
                and destination.samefile(protected)
            ):
                raise OSError("Transcript destination identifies protected storage.")
    except ValueError as exc:
        raise OSError("Invalid transcript destination path.") from exc


def publish_transcript(
    document: str, *, database: str | Path, output: str | Path, overwrite: bool = False
) -> None:
    """Publish a closed same-directory temporary file; cleanup is best effort."""
    try:
        if not str(output).strip() or "\x00" in str(output):
            raise OSError("Invalid transcript destination path.")
        source = Path(database).resolve()
        destination = Path(output).absolute()
    except ValueError as exc:
        raise OSError("Invalid transcript destination path.") from exc
    _check_destination(source, destination)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=".tuesday-transcript-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(document)
        _check_destination(source, destination)
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
