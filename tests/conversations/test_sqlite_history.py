"""Read-only SQLite safety, bounded queries, malformed data, and worker ownership."""

import asyncio
import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock
from uuid import UUID

import pytest

import tuesday.conversations.sqlite_history as reader_module
from tuesday.conversations import (
    ConversationRepositoryError,
    SQLiteConversationHistoryReader,
)

A, B = UUID(int=471), UUID(int=472)
STAMP = "2026-11-26T12:30:45.123456+05:30"
_CREATE = """
CREATE TABLE tuesday_conversation_messages (
    sequence INTEGER PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""
_INSERT = """
INSERT INTO tuesday_conversation_messages
(conversation_id, message_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)
"""


def run(operation):
    return asyncio.run(operation)


def execute(path, sql, parameters=()):
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(sql, parameters)


@pytest.fixture
def database_path(tmp_path):
    """Create a compatible database directly, independently of the writable adapter."""
    path = tmp_path / "existing # ü.sqlite3"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(_CREATE)
        connection.executemany(
            _INSERT,
            [
                (str(A), str(UUID(int=1)), "user", "First body", STAMP),
                (str(A), str(UUID(int=1)), "assistant", "Second body", STAMP),
                (str(B), str(UUID(int=2)), "system", "Other body", STAMP),
            ],
        )
        connection.execute("CREATE TABLE unrelated (value TEXT)")
        connection.execute("INSERT INTO unrelated VALUES (?)", ("preserved",))
        connection.execute("PRAGMA user_version=47")
    return path


@pytest.fixture
def reader(database_path):
    return SQLiteConversationHistoryReader(database_path)


@pytest.fixture
def tracker(database_path, monkeypatch):
    original = sqlite3.connect
    state = {"connections": [], "queries": [], "rows": [], "uris": [], "failure": None}

    class TrackedConnection(sqlite3.Connection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.owner = threading.get_ident()
            self.closed = False

        def execute(self, sql, parameters=()):
            assert threading.get_ident() == self.owner
            state["queries"].append((sql, parameters))
            if state["failure"] is not None and sql.lstrip().upper().startswith(
                ("SELECT", "WITH")
            ):
                raise state["failure"]
            return super().execute(sql, parameters)

        def close(self):
            assert threading.get_ident() == self.owner
            super().close()
            self.closed = True

    def row_factory(cursor, row):
        if cursor.description[0][0] != "cid":
            state["rows"].append(row)
        return row

    def connect(database_uri, **kwargs):
        assert database_uri.endswith("?mode=ro") and "immutable" not in database_uri
        assert kwargs == {"uri": True, "timeout": 5.0}
        state["uris"].append(database_uri)
        connection = original(database_uri, **kwargs, factory=TrackedConnection)
        connection.row_factory = row_factory
        state["connections"].append(connection)
        return connection

    monkeypatch.setattr(reader_module.sqlite3, "connect", connect)
    return state


@pytest.mark.parametrize("path", [None, 47, True, object(), [], {}, b"history.db"])
def test_invalid_path_types_are_rejected(path):
    with pytest.raises(TypeError, match="string or Path"):
        SQLiteConversationHistoryReader(path)


@pytest.mark.parametrize(
    "path",
    [
        "",
        " \t\n",
        ":memory:",
        Path(":memory:"),
        "file:history.db",
        "FILE:/history.db?mode=ro",
    ],
)
def test_invalid_path_values_are_rejected(path):
    with pytest.raises(ValueError):
        SQLiteConversationHistoryReader(path)


def test_construction_resolves_without_connecting_or_initializing(
    tmp_path, monkeypatch
):
    connect = Mock(side_effect=AssertionError("No constructor database I/O"))
    monkeypatch.setattr(reader_module.sqlite3, "connect", connect)
    path = tmp_path / "missing" / "new.sqlite3"
    reader = SQLiteConversationHistoryReader(path)
    assert reader._database_path == path.resolve()
    assert not hasattr(reader, "__dict__")
    assert not path.parent.exists()
    connect.assert_not_called()


def test_path_resolution_errors_are_chained(tmp_path, monkeypatch):
    def fail(self):
        raise OSError("Cannot resolve filesystem location")

    monkeypatch.setattr(Path, "resolve", fail)
    with pytest.raises(ConversationRepositoryError) as caught:
        SQLiteConversationHistoryReader(tmp_path / "history.db")
    assert isinstance(caught.value.__cause__, OSError)


@pytest.mark.parametrize("as_string", [False, True])
def test_relative_path_resolution_survives_working_directory_changes(
    database_path, tmp_path, monkeypatch, as_string
):
    monkeypatch.chdir(tmp_path)
    value = database_path.name if as_string else Path(database_path.name)
    reader = SQLiteConversationHistoryReader(value)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert len(run(reader.get_messages(A)).messages) == 2
    assert list(elsewhere.iterdir()) == []


def test_queries_preserve_quiescent_database_bytes_schema_version_and_unrelated_data(
    database_path, reader
):
    before = database_path.read_bytes()
    with closing(sqlite3.connect(database_path)) as connection:
        schema = connection.execute(
            "SELECT name, sql FROM sqlite_master ORDER BY name"
        ).fetchall()
    assert len(run(reader.list_conversations()).items) == 2
    assert len(run(reader.get_messages(A)).messages) == 2
    assert run(reader.get_messages(UUID(int=999))).messages == ()
    assert database_path.read_bytes() == before
    with closing(sqlite3.connect(database_path)) as connection:
        assert (
            connection.execute(
                "SELECT name, sql FROM sqlite_master ORDER BY name"
            ).fetchall()
            == schema
        )
        assert connection.execute("PRAGMA user_version").fetchone() == (47,)
        assert connection.execute("SELECT value FROM unrelated").fetchall() == [
            ("preserved",)
        ]
        assert connection.execute(
            "SELECT COUNT(*) FROM tuesday_conversation_messages"
        ).fetchone() == (3,)


def test_connections_are_read_only_worker_owned_and_closed(reader, tracker):
    run(reader.list_conversations())
    run(reader.get_messages(A))
    assert len(tracker["connections"]) == 2
    assert all(connection.closed for connection in tracker["connections"])
    assert all(
        connection.owner != threading.get_ident()
        for connection in tracker["connections"]
    )
    assert all("%23" in uri and "%C3%BC" in uri for uri in tracker["uris"])
    assert all(
        sql.lstrip().upper().startswith(("PRAGMA TABLE_INFO", "SELECT", "WITH"))
        for sql, _ in tracker["queries"]
    )


def test_writes_through_reader_connections_are_rejected(database_path, monkeypatch):
    original = sqlite3.connect
    rejected = []

    def connect(database_uri, **kwargs):
        connection = original(database_uri, **kwargs)
        try:
            connection.execute("CREATE TABLE must_not_exist (value TEXT)")
        except sqlite3.OperationalError as error:
            rejected.append(str(error))
        return connection

    monkeypatch.setattr(reader_module.sqlite3, "connect", connect)
    assert (
        len(
            run(
                SQLiteConversationHistoryReader(database_path).list_conversations()
            ).items
        )
        == 2
    )
    assert len(rejected) == 1 and "readonly" in rejected[0].lower()


def test_message_pagination_is_bounded_and_bound_in_sql(database_path, reader, tracker):
    # Seed through the uninstrumented DB-API connection, then observe bounded reads.
    original_connect = sqlite3.dbapi2.connect
    with closing(original_connect(database_path)) as connection, connection:
        connection.executemany(
            _INSERT,
            [
                (str(A), str(UUID(int=index + 10)), "user", f"Body {index}", STAMP)
                for index in range(1000)
            ],
        )
    page = run(reader.get_messages(A, limit=2, offset=10))
    assert len(page.messages) == 2 and page.next_offset == 12
    assert len(tracker["rows"]) == 3
    sql, parameters = tracker["queries"][-1]
    assert "LIMIT ? OFFSET ?" in sql
    assert parameters == (str(A), 3, 10)
    assert "Body" not in sql
    assert all(connection.closed for connection in tracker["connections"])


def test_summaries_use_counts_and_first_last_metadata_without_message_bodies(
    reader, tracker
):
    page = run(reader.list_conversations(limit=1))
    assert page.items[0].message_count == 2 and page.next_offset == 1
    assert len(tracker["rows"]) == 2
    sql, parameters = tracker["queries"][-1]
    assert "content" not in sql and "message_id" not in sql
    assert "COUNT(*)" in sql and "MIN(sequence)" in sql and "MAX(sequence)" in sql
    assert "LIMIT ? OFFSET ?" in sql and parameters == (2, 0)


@pytest.mark.parametrize("operation", ["list", "messages"])
@pytest.mark.parametrize(
    "kind", ["missing", "removed", "corrupt", "no-table", "incompatible"]
)
def test_bad_storage_fails_without_recreation_or_repair(tmp_path, operation, kind):
    path = tmp_path / "missing-parent" / "history.sqlite3"
    if kind != "missing":
        path.parent.mkdir()
    if kind == "corrupt":
        path.write_bytes(b"not a database")
    elif kind in ("no-table", "incompatible", "removed"):
        with closing(sqlite3.connect(path)) as connection:
            if kind == "incompatible":
                connection.execute(
                    "CREATE TABLE tuesday_conversation_messages (sequence TEXT)"
                )
            else:
                connection.execute("CREATE TABLE unrelated (value TEXT)")
    reader = SQLiteConversationHistoryReader(path)
    if kind == "removed":
        path.unlink()
    before = path.read_bytes() if path.exists() else None
    with pytest.raises(ConversationRepositoryError) as caught:
        run(
            reader.list_conversations()
            if operation == "list"
            else reader.get_messages(A)
        )
    assert isinstance(caught.value.__cause__, (sqlite3.Error, ValueError))
    assert (path.read_bytes() if path.exists() else None) == before
    if kind == "missing":
        assert not path.parent.exists()


@pytest.mark.parametrize("operation", ["list", "messages"])
def test_incompatible_schema_errors_close_worker_connections(
    database_path, tracker, operation
):
    with closing(sqlite3.dbapi2.connect(database_path)) as connection:
        connection.execute("DROP TABLE tuesday_conversation_messages")
    reader = SQLiteConversationHistoryReader(database_path)
    with pytest.raises(ConversationRepositoryError) as caught:
        run(
            reader.list_conversations()
            if operation == "list"
            else reader.get_messages(A)
        )
    assert isinstance(caught.value.__cause__, ValueError)
    assert all(connection.closed for connection in tracker["connections"])


@pytest.mark.parametrize("operation", ["list", "messages"])
def test_connection_failures_chain_without_retry(reader, monkeypatch, operation):
    error = sqlite3.OperationalError("private filesystem detail")
    connect = Mock(side_effect=error)
    monkeypatch.setattr(reader_module.sqlite3, "connect", connect)
    with pytest.raises(ConversationRepositoryError) as caught:
        run(
            reader.list_conversations()
            if operation == "list"
            else reader.get_messages(A)
        )
    assert caught.value.__cause__ is error
    assert "private" not in str(caught.value)
    connect.assert_called_once()


@pytest.mark.parametrize(
    "update,value",
    [
        (
            "UPDATE tuesday_conversation_messages SET message_id=? WHERE sequence=1",
            "bad",
        ),
        ("UPDATE tuesday_conversation_messages SET role=? WHERE sequence=1", "tool"),
        ("UPDATE tuesday_conversation_messages SET content=? WHERE sequence=1", " \n"),
        (
            "UPDATE tuesday_conversation_messages SET created_at=? WHERE sequence=1",
            "bad",
        ),
        (
            "UPDATE tuesday_conversation_messages SET created_at=? WHERE sequence=1",
            "2026-11-26",
        ),
        (
            "UPDATE tuesday_conversation_messages SET message_id=? WHERE sequence=1",
            b"uuid",
        ),
        ("UPDATE tuesday_conversation_messages SET role=? WHERE sequence=1", b"user"),
        (
            "UPDATE tuesday_conversation_messages SET content=? WHERE sequence=1",
            b"text",
        ),
        (
            "UPDATE tuesday_conversation_messages SET created_at=? WHERE sequence=1",
            b"time",
        ),
    ],
)
def test_malformed_returned_messages_fail_closed(database_path, update, value):
    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.execute("PRAGMA ignore_check_constraints=ON")
        connection.execute(update, (value,))
    reader = SQLiteConversationHistoryReader(database_path)
    with pytest.raises(ConversationRepositoryError) as caught:
        run(reader.get_messages(A))
    assert isinstance(caught.value.__cause__, (TypeError, ValueError))
    assert "First body" not in str(caught.value)


@pytest.mark.parametrize(
    "update,value",
    [
        (
            "UPDATE tuesday_conversation_messages SET conversation_id=? "
            "WHERE conversation_id=?",
            "bad",
        ),
        (
            "UPDATE tuesday_conversation_messages SET conversation_id=? "
            "WHERE conversation_id=?",
            b"uuid",
        ),
        (
            "UPDATE tuesday_conversation_messages SET created_at=? WHERE sequence=1",
            "bad",
        ),
        (
            "UPDATE tuesday_conversation_messages SET created_at=? WHERE sequence=2",
            "2026-11-26",
        ),
        (
            "UPDATE tuesday_conversation_messages SET created_at=? WHERE sequence=1",
            b"date",
        ),
    ],
)
def test_malformed_returned_summary_fields_are_rejected(database_path, update, value):
    execute(
        database_path,
        update,
        (value, str(A)) if "conversation_id=?" in update else (value,),
    )
    with pytest.raises(ConversationRepositoryError) as caught:
        run(SQLiteConversationHistoryReader(database_path).list_conversations())
    assert isinstance(caught.value.__cause__, (TypeError, ValueError))


def test_validation_is_limited_to_returned_records_and_summary_fields(
    database_path, reader
):
    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.execute("PRAGMA ignore_check_constraints=ON")
        connection.execute(
            "UPDATE tuesday_conversation_messages SET role='invalid' WHERE sequence=2"
        )
        connection.execute(
            "UPDATE tuesday_conversation_messages SET created_at='bad' WHERE sequence=3"
        )
    # Lookahead does not decode message bodies; summaries do not decode roles.
    page = run(reader.get_messages(A, limit=1))
    assert len(page.messages) == 1 and page.next_offset == 1
    assert run(reader.list_conversations(limit=1)).items[0].message_count == 2
    with pytest.raises(ConversationRepositoryError):
        run(reader.get_messages(A, limit=1, offset=1))
    with pytest.raises(ConversationRepositoryError):
        run(reader.list_conversations(limit=1, offset=1))


@pytest.mark.parametrize("operation", ["list", "messages"])
def test_query_failures_close_connections(reader, tracker, operation):
    # Inject failure at SELECT execution on a real, validated read-only connection.
    error = sqlite3.OperationalError("query failure")
    tracker["failure"] = error
    with pytest.raises(ConversationRepositoryError) as caught:
        run(
            reader.list_conversations()
            if operation == "list"
            else reader.get_messages(A)
        )
    assert caught.value.__cause__ is error
    assert all(
        isinstance(connection, sqlite3.Connection) and connection.closed
        for connection in tracker["connections"]
    )


@pytest.mark.parametrize("operation", ["list", "messages"])
def test_decode_failures_leave_worker_connections_closed(
    database_path, reader, tracker, operation
):
    with closing(sqlite3.dbapi2.connect(database_path)) as connection, connection:
        connection.execute("UPDATE tuesday_conversation_messages SET created_at='bad'")
    with pytest.raises(ConversationRepositoryError):
        run(
            reader.list_conversations()
            if operation == "list"
            else reader.get_messages(A)
        )
    assert all(connection.closed for connection in tracker["connections"])
