"""Explicitly configured, file-backed SQLite conversation history."""

import asyncio
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from uuid import UUID

from tuesday.conversations._sqlite import decode_message, schema_matches
from tuesday.conversations.base import (
    BaseConversationRepository,
    ConversationRepositoryError,
)
from tuesday.domain import ConversationContext, ConversationMessage, MessageRole

__all__ = ["SQLiteConversationRepository"]

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS tuesday_conversation_messages (
    sequence INTEGER PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""
_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_tuesday_conversation_messages_conversation_sequence
ON tuesday_conversation_messages (conversation_id, sequence)
"""
_SELECT = """
SELECT message_id, role, content, created_at
FROM tuesday_conversation_messages
WHERE conversation_id = ? ORDER BY sequence ASC
"""
_INSERT = """
INSERT INTO tuesday_conversation_messages
    (conversation_id, message_id, role, content, created_at)
VALUES (?, ?, ?, ?, ?)
"""


class SQLiteConversationRepository(BaseConversationRepository):
    """Initialize synchronously; read and append in workers with owned connections.

    Cancelling an await does not necessarily stop an already-running transaction.
    Atomic appends do not coordinate the complete load/model/append service turn.
    """

    __slots__ = ("_database_path", "_database_uri")

    def __init__(self, database_path: str | Path) -> None:
        if not isinstance(database_path, (str, Path)):
            raise TypeError("database_path must be a string or Path.")
        path_text = str(database_path)
        if not path_text.strip():
            raise ValueError("database_path must not be empty.")
        if path_text == ":memory:" or path_text.lower().startswith("file:"):
            raise ValueError(
                "database_path must be a filesystem path, not a SQLite URI."
            )
        try:
            self._database_path = Path(database_path).resolve()
            self._database_uri = self._database_path.as_uri() + "?mode=rw"
            self._initialize()
        except (OSError, sqlite3.Error) as exc:
            raise ConversationRepositoryError(
                "Could not initialize SQLite history."
            ) from exc

    def _initialize(self) -> None:
        with (
            closing(sqlite3.connect(self._database_path, timeout=5.0)) as connection,
            connection,
        ):
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(_CREATE_TABLE)
            connection.execute(_CREATE_INDEX)
            columns = connection.execute(
                "PRAGMA table_info(tuesday_conversation_messages)"
            ).fetchall()
            if not schema_matches(columns):
                raise ConversationRepositoryError("Incompatible SQLite history schema.")

    def _connect(self) -> sqlite3.Connection:
        # mode=rw prevents silently recreating a database deleted after construction.
        return sqlite3.connect(self._database_uri, uri=True, timeout=5.0)

    async def get_context(self, conversation_id: UUID) -> ConversationContext:
        """Read an independent immutable snapshot without blocking the event loop."""
        self._validate_conversation_id(conversation_id)
        return await asyncio.to_thread(self._get_context_sync, conversation_id)

    async def append_messages(
        self,
        conversation_id: UUID,
        messages: tuple[ConversationMessage, ...],
    ) -> ConversationContext:
        """Validate the whole batch, then atomically append and return its snapshot."""
        self._validate_conversation_id(conversation_id)
        serialized = self._serialize_messages(messages)
        return await asyncio.to_thread(
            self._append_messages_sync, conversation_id, serialized
        )

    @staticmethod
    def _validate_conversation_id(conversation_id: UUID) -> None:
        if not isinstance(conversation_id, UUID):
            raise TypeError("conversation_id must be a UUID.")

    @staticmethod
    def _serialize_messages(
        messages: tuple[ConversationMessage, ...],
    ) -> tuple[tuple[str, str, str, str], ...]:
        if not isinstance(messages, tuple):
            raise TypeError(
                "messages must be a tuple of ConversationMessage instances."
            )
        if not messages:
            raise ValueError("messages must contain at least one ConversationMessage.")
        serialized = []
        for message in messages:
            if not isinstance(message, ConversationMessage):
                raise TypeError(
                    "messages must contain only ConversationMessage instances."
                )
            if not isinstance(message.role, MessageRole):
                raise TypeError("Message role must be a MessageRole.")
            if not isinstance(message.message_id, UUID):
                raise TypeError("Message message_id must be a UUID.")
            if not isinstance(message.content, str):
                raise TypeError("Message content must be a string.")
            if not message.content.strip():
                raise ValueError("Message content must not be empty.")
            if not isinstance(message.created_at, datetime):
                raise TypeError("Message created_at must be a datetime.")
            if (
                message.created_at.tzinfo is None
                or message.created_at.utcoffset() is None
            ):
                raise ValueError("Message created_at must be timezone-aware.")
            serialized.append(
                (
                    str(message.message_id),
                    message.role.value,
                    message.content,
                    message.created_at.isoformat(),
                )
            )
        return tuple(serialized)

    @staticmethod
    def _read_context(
        connection: sqlite3.Connection, conversation_id: UUID
    ) -> ConversationContext:
        rows = connection.execute(_SELECT, (str(conversation_id),)).fetchall()
        try:
            messages = tuple(decode_message(row) for row in rows)
        except (TypeError, ValueError) as exc:
            raise ConversationRepositoryError(
                "Invalid persisted conversation message."
            ) from exc
        return ConversationContext(conversation_id, tuple(messages))

    def _get_context_sync(self, conversation_id: UUID) -> ConversationContext:
        try:
            with closing(self._connect()) as connection:
                return self._read_context(connection, conversation_id)
        except sqlite3.Error as exc:
            raise ConversationRepositoryError("Could not read SQLite history.") from exc

    def _append_messages_sync(
        self,
        conversation_id: UUID,
        serialized: tuple[tuple[str, str, str, str], ...],
    ) -> ConversationContext:
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.executemany(
                    _INSERT, ((str(conversation_id), *row) for row in serialized)
                )
                context = self._read_context(connection, conversation_id)
            return context
        except sqlite3.Error as exc:
            raise ConversationRepositoryError(
                "Could not append SQLite history."
            ) from exc
