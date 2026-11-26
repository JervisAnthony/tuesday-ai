"""Existing-file, read-only SQLite conversation history queries."""

import asyncio
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from uuid import UUID

from tuesday.conversations._sqlite import decode_message, schema_matches
from tuesday.conversations.base import ConversationRepositoryError
from tuesday.conversations.history import (
    BaseConversationHistoryReader,
    ConversationMessagePage,
    ConversationSummary,
    ConversationSummaryPage,
    _validate_conversation_id,
    _validate_pagination,
)

__all__ = ["SQLiteConversationHistoryReader"]

_LIST = """
WITH conversations AS (
    SELECT conversation_id, COUNT(*) AS message_count,
           MIN(sequence) AS first_sequence, MAX(sequence) AS last_sequence
    FROM tuesday_conversation_messages GROUP BY conversation_id
)
SELECT conversations.conversation_id, conversations.message_count,
       first.created_at, last.created_at
FROM conversations
JOIN tuesday_conversation_messages AS first
    ON first.sequence = conversations.first_sequence
JOIN tuesday_conversation_messages AS last
    ON last.sequence = conversations.last_sequence
ORDER BY conversations.first_sequence ASC LIMIT ? OFFSET ?
"""
_MESSAGES = """
SELECT message_id, role, content, created_at FROM tuesday_conversation_messages
WHERE conversation_id = ? ORDER BY sequence ASC LIMIT ? OFFSET ?
"""


def _decode_summary(row) -> ConversationSummary:
    conversation_id, count, first, last = row
    if not all(isinstance(value, str) for value in (conversation_id, first, last)):
        raise TypeError("Stored summary identifiers and timestamps must be strings.")
    return ConversationSummary(
        UUID(conversation_id),
        count,
        datetime.fromisoformat(first),
        datetime.fromisoformat(last),
    )


class SQLiteConversationHistoryReader(BaseConversationHistoryReader):
    """Resolve an explicit path without connecting or initializing storage."""

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
            self._database_uri = self._database_path.as_uri() + "?mode=ro"
        except OSError as exc:
            raise ConversationRepositoryError(
                "Could not resolve SQLite history path."
            ) from exc

    async def list_conversations(
        self, *, limit: int = 50, offset: int = 0
    ) -> ConversationSummaryPage:
        _validate_pagination(limit, offset)
        return await asyncio.to_thread(self._list_sync, limit, offset)

    async def get_messages(
        self, conversation_id: UUID, *, limit: int = 50, offset: int = 0
    ) -> ConversationMessagePage:
        _validate_conversation_id(conversation_id)
        _validate_pagination(limit, offset)
        return await asyncio.to_thread(
            self._messages_sync, conversation_id, limit, offset
        )

    def _query(self, sql: str, parameters: tuple) -> list[tuple]:
        with closing(
            sqlite3.connect(self._database_uri, uri=True, timeout=5.0)
        ) as connection:
            columns = connection.execute(
                "PRAGMA table_info(tuesday_conversation_messages)"
            ).fetchall()
            if not schema_matches(columns):
                raise ValueError("Incompatible SQLite history schema.")
            return connection.execute(sql, parameters).fetchall()

    def _list_sync(self, limit: int, offset: int) -> ConversationSummaryPage:
        try:
            rows = self._query(_LIST, (limit + 1, offset))
            items = tuple(_decode_summary(row) for row in rows[:limit])
            return ConversationSummaryPage(
                items, limit, offset, offset + len(items) if len(rows) > limit else None
            )
        except (sqlite3.Error, TypeError, ValueError) as exc:
            raise ConversationRepositoryError(
                "Could not query SQLite conversation summaries."
            ) from exc

    def _messages_sync(
        self, conversation_id: UUID, limit: int, offset: int
    ) -> ConversationMessagePage:
        try:
            rows = self._query(_MESSAGES, (str(conversation_id), limit + 1, offset))
            messages = tuple(decode_message(row) for row in rows[:limit])
            return ConversationMessagePage(
                conversation_id,
                messages,
                limit,
                offset,
                offset + len(messages) if len(rows) > limit else None,
            )
        except (sqlite3.Error, TypeError, ValueError) as exc:
            raise ConversationRepositoryError(
                "Could not query SQLite conversation messages."
            ) from exc
