"""Pure SQLite schema comparison and message decoding shared by storage adapters."""

from datetime import datetime
from uuid import UUID

from tuesday.domain import ConversationMessage, MessageRole

_SCHEMA = (
    ("sequence", "INTEGER", 0, 1),
    ("conversation_id", "TEXT", 1, 0),
    ("message_id", "TEXT", 1, 0),
    ("role", "TEXT", 1, 0),
    ("content", "TEXT", 1, 0),
    ("created_at", "TEXT", 1, 0),
)


def schema_matches(columns) -> bool:
    return tuple((row[1], row[2].upper(), row[3], row[5]) for row in columns) == _SCHEMA


def decode_message(row) -> ConversationMessage:
    message_id, role, content, created_at = row
    if not all(isinstance(value, str) for value in row):
        raise TypeError("Stored message fields must be strings.")
    return ConversationMessage(
        message_id=UUID(message_id),
        role=MessageRole(role),
        content=content,
        created_at=datetime.fromisoformat(created_at),
    )
