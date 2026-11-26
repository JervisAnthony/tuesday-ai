"""Real temporary SQLite storage, durability, and failure boundary tests."""

import ast
import asyncio
import inspect
import json
import sqlite3
import subprocess
import sys
import threading
from contextlib import closing
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from uuid import UUID, uuid4

import pytest

import tuesday.conversations as conversations
import tuesday.conversations.sqlite as sqlite_module
from tuesday.conversations import (
    BaseConversationRepository,
    ConversationRepositoryError,
    SQLiteConversationRepository,
)
from tuesday.domain import ConversationContext, ConversationMessage, MessageRole

CONVERSATION_ID = UUID(int=43)
STAMP = datetime(
    2026, 11, 24, 12, 34, 56, 123456, tzinfo=timezone(timedelta(hours=5, minutes=30))
)


def read(repository, conversation_id=CONVERSATION_ID):
    return asyncio.run(repository.get_context(conversation_id))


def append(repository, messages, conversation_id=CONVERSATION_ID):
    return asyncio.run(repository.append_messages(conversation_id, messages))


def message(content="Hello", role=MessageRole.USER, index=1):
    return ConversationMessage(role, content, UUID(int=index), STAMP)


@pytest.fixture
def database_path(tmp_path):
    return tmp_path / "conversation #43 ü.sqlite3"


@pytest.fixture
def repository(database_path):
    return SQLiteConversationRepository(database_path)


@pytest.fixture
def batch():
    return (
        message("  System\ncontext  ", MessageRole.SYSTEM, 1),
        message(
            "  /chat Unicode café नमस्ते 🌍\n'; DROP TABLE "
            "tuesday_conversation_messages; --  ",
            MessageRole.USER,
            2,
        ),
        message("  Answer\nsecond line\t ", MessageRole.ASSISTANT, 3),
    )


@pytest.fixture
def tracked_connections(monkeypatch):
    """Keep real SQLite connections while observing closure and thread ownership."""
    original = sqlite3.connect
    connections = []

    class TrackedConnection(sqlite3.Connection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.owner = threading.get_ident()
            self.closed = False
            self.statements = []

        def execute(self, sql, *args, **kwargs):
            assert threading.get_ident() == self.owner
            self.statements.append(sql)
            return super().execute(sql, *args, **kwargs)

        def executemany(self, *args, **kwargs):
            assert threading.get_ident() == self.owner
            return super().executemany(*args, **kwargs)

        def close(self):
            assert threading.get_ident() == self.owner
            super().close()
            self.closed = True

    def connect(*args, **kwargs):
        assert kwargs["timeout"] == 5.0
        assert "check_same_thread" not in kwargs
        connection = original(*args, **kwargs, factory=TrackedConnection)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite_module.sqlite3, "connect", connect)
    return connections


def test_public_contract_and_exports(repository):
    assert isinstance(repository, BaseConversationRepository)
    assert not hasattr(repository, "__dict__")
    assert conversations.__all__ == [
        "BaseConversationRepository",
        "ConversationRepositoryError",
        "InMemoryConversationRepository",
        "SQLiteConversationRepository",
        "BaseConversationHistoryReader",
        "ConversationSummary",
        "ConversationSummaryPage",
        "ConversationMessagePage",
        "SQLiteConversationHistoryReader",
    ]
    assert sqlite_module.__all__ == ["SQLiteConversationRepository"]
    for name in ("get_context", "append_messages"):
        operation = getattr(SQLiteConversationRepository, name)
        assert inspect.iscoroutinefunction(operation)
        assert inspect.signature(operation) == inspect.signature(
            getattr(BaseConversationRepository, name)
        )
    assert tuple(inspect.signature(SQLiteConversationRepository).parameters) == (
        "database_path",
    )


@pytest.mark.parametrize("invalid", [None, 43, True, [], {}, object(), b"history.db"])
def test_constructor_rejects_non_path_objects(invalid):
    with pytest.raises(TypeError, match="database_path must be a string or Path"):
        SQLiteConversationRepository(invalid)


@pytest.mark.parametrize("invalid", ["", "  ", "\t\n"])
def test_constructor_rejects_blank_paths(invalid):
    with pytest.raises(ValueError, match="database_path must not be empty"):
        SQLiteConversationRepository(invalid)


@pytest.mark.parametrize(
    "invalid",
    [
        ":memory:",
        Path(":memory:"),
        "file:history.db",
        "file::memory:",
        "FILE:/history.db?mode=rw",
    ],
)
def test_constructor_rejects_special_sqlite_paths(invalid):
    with pytest.raises(ValueError, match="filesystem path"):
        SQLiteConversationRepository(invalid)


@pytest.mark.parametrize("as_string", [False, True])
def test_initialization_creates_table_index_and_role_constraint(
    database_path, as_string
):
    assert not database_path.exists()
    SQLiteConversationRepository(str(database_path) if as_string else database_path)
    with closing(sqlite3.connect(database_path)) as connection:
        columns = connection.execute(
            "PRAGMA table_info(tuesday_conversation_messages)"
        ).fetchall()
        assert [row[1] for row in columns] == [
            "sequence",
            "conversation_id",
            "message_id",
            "role",
            "content",
            "created_at",
        ]
        index = connection.execute(
            "PRAGMA index_info(idx_tuesday_conversation_messages_conversation_sequence)"
        ).fetchall()
        assert [row[2] for row in index] == ["conversation_id", "sequence"]
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO tuesday_conversation_messages "
                "(conversation_id,message_id,role,content,created_at) "
                "VALUES (?,?,?,?,?)",
                (
                    str(CONVERSATION_ID),
                    str(uuid4()),
                    "tool",
                    "Invalid role",
                    STAMP.isoformat(),
                ),
            )


def test_path_is_resolved_before_later_working_directory_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repository = SQLiteConversationRepository("relative.sqlite3")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert append(repository, (message(),)).messages == (message(),)
    assert read(repository).messages == (message(),)
    assert (tmp_path / "relative.sqlite3").is_file()
    assert not (elsewhere / "relative.sqlite3").exists()


@pytest.mark.parametrize("kind", ["missing-parent", "directory", "corrupt"])
def test_initialization_errors_preserve_existing_files(tmp_path, kind):
    path = tmp_path / "history.db"
    if kind == "missing-parent":
        path = tmp_path / "missing" / "history.db"
    elif kind == "directory":
        path.mkdir()
    else:
        path.write_bytes(b"This is not SQLite")
    with pytest.raises(ConversationRepositoryError, match="initialize") as caught:
        SQLiteConversationRepository(path)
    assert isinstance(caught.value.__cause__, sqlite3.Error)
    if kind == "missing-parent":
        assert not path.parent.exists()
    elif kind == "directory":
        assert path.is_dir()
    else:
        assert path.read_bytes() == b"This is not SQLite"


def test_path_resolution_errors_are_chained(tmp_path, monkeypatch):
    def fail(_):
        raise OSError("Cannot resolve path")

    monkeypatch.setattr(Path, "resolve", fail)
    with pytest.raises(ConversationRepositoryError) as caught:
        SQLiteConversationRepository(tmp_path / "history.db")
    assert isinstance(caught.value.__cause__, OSError)


def test_reopening_preserves_history_unrelated_tables_and_user_version(
    database_path, repository, batch
):
    append(repository, batch)
    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")
        connection.execute("INSERT INTO unrelated VALUES (?)", ("preserved",))
        connection.execute("PRAGMA user_version = 43")
    recreated = SQLiteConversationRepository(database_path)
    assert read(recreated).messages == batch
    with closing(sqlite3.connect(database_path)) as connection:
        assert connection.execute("SELECT value FROM unrelated").fetchall() == [
            ("preserved",)
        ]
        assert connection.execute("PRAGMA user_version").fetchone() == (43,)


@pytest.mark.parametrize(
    "schema",
    [
        "conversation_id TEXT, sequence INTEGER",
        "sequence INTEGER PRIMARY KEY, conversation_id TEXT, "
        "message_id TEXT, role TEXT, content TEXT, created_at TEXT",
    ],
)
def test_incompatible_existing_schema_is_rejected(database_path, schema):
    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute(
            "CREATE TABLE tuesday_conversation_messages (" + schema + ")"
        )
    with pytest.raises(ConversationRepositoryError):
        SQLiteConversationRepository(database_path)


def test_unknown_reads_return_independent_empty_snapshots_without_rows(
    database_path, repository
):
    first, second = read(repository), read(repository)
    assert first == second == ConversationContext(CONVERSATION_ID)
    assert first is not second
    assert isinstance(first.messages, tuple)
    with closing(sqlite3.connect(database_path)) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM tuesday_conversation_messages"
        ).fetchone() == (0,)


@pytest.mark.parametrize("operation", ["get_context", "append_messages"])
@pytest.mark.parametrize("invalid", [None, "conversation", 43, object()])
def test_operations_reject_invalid_conversation_ids(repository, operation, invalid):
    args = (invalid,) if operation == "get_context" else (invalid, (message(),))
    with pytest.raises(TypeError, match="conversation_id must be a UUID"):
        asyncio.run(getattr(repository, operation)(*args))
    assert read(repository).messages == ()


@pytest.mark.parametrize("invalid", [None, [], [message()], "messages", object()])
def test_append_rejects_non_tuple_batches(repository, invalid):
    with pytest.raises(TypeError, match="messages must be a tuple"):
        append(repository, invalid)
    assert read(repository).messages == ()


def test_append_rejects_empty_batch(repository):
    with pytest.raises(ValueError, match="at least one ConversationMessage"):
        append(repository, ())


class NoOffset(tzinfo):
    def utcoffset(self, dt):
        return None


@pytest.mark.parametrize(
    "field,value,error_type",
    [
        ("role", "user", TypeError),
        ("role", None, TypeError),
        ("message_id", "not-a-uuid", TypeError),
        ("content", 43, TypeError),
        ("content", None, TypeError),
        ("content", "", ValueError),
        ("content", " \t\n", ValueError),
        ("created_at", "not-a-datetime", TypeError),
        ("created_at", datetime(2026, 1, 1), ValueError),
        ("created_at", datetime(2026, 1, 1, tzinfo=NoOffset()), ValueError),
        (None, None, TypeError),
    ],
)
def test_entire_batch_validates_before_opening_a_write_connection(
    database_path, tracked_connections, field, value, error_type
):
    repository = SQLiteConversationRepository(database_path)
    first = message("Earlier committed history")
    append(repository, (first,))
    malformed = message("Malformed")
    if field is None:
        malformed = object()
    else:
        # The domain's permissive constructor is not the storage validation boundary.
        object.__setattr__(malformed, field, value)
    connection_count = len(tracked_connections)
    with pytest.raises(error_type):
        append(repository, (message("Valid first entry"), malformed))
    assert len(tracked_connections) == connection_count
    assert read(repository).messages == (first,)


@pytest.mark.parametrize("count", [1, 3])
def test_exact_value_roundtrip_and_immutable_independent_snapshots(
    repository, database_path, batch, count
):
    supplied = batch[:count]
    original = tuple(replace(item) for item in supplied)
    appended = append(repository, supplied)
    loaded = read(SQLiteConversationRepository(database_path))
    assert appended == loaded == ConversationContext(CONVERSATION_ID, original)
    assert appended is not loaded
    for source, stored in zip(supplied, loaded.messages, strict=True):
        assert source is not stored
        assert source == stored
        assert stored.created_at.utcoffset() == timedelta(hours=5, minutes=30)
        assert stored.created_at.microsecond == 123456
        with pytest.raises(FrozenInstanceError):
            stored.content = "Changed"
    with pytest.raises(FrozenInstanceError):
        loaded.messages = ()
    append(repository, (message("New"),))
    assert loaded.messages == original
    assert supplied == original


def test_append_sequence_overrides_timestamps_and_does_not_deduplicate(repository):
    first = message("First")
    earlier = replace(
        first,
        content="Later append, earlier timestamp",
        created_at=STAMP - timedelta(days=1),
    )
    same_time = replace(first, content="Later append, identical timestamp")
    initial = append(repository, (first, first))
    append(repository, (earlier, same_time))
    assert initial.messages == (first, first)
    assert read(repository).messages == (first, first, earlier, same_time)


def test_interleaved_conversations_shared_and_separate_databases(
    repository, database_path, tmp_path
):
    other_id = UUID(int=44)
    shared = SQLiteConversationRepository(database_path)
    separate = SQLiteConversationRepository(tmp_path / "other.sqlite3")
    first, second, third = message("A first"), message("B first"), message("A second")
    append(repository, (first,))
    assert append(shared, (second,), other_id).messages == (second,)
    assert append(shared, (third,)).messages == (first, third)
    assert read(repository, other_id).messages == (second,)
    assert read(separate).messages == ()
    append(separate, (second,))
    assert read(repository).messages == (first, third)


def test_clean_process_write_and_read_durability(database_path, tmp_path):
    writer = """
import asyncio, sys
from datetime import datetime
from uuid import UUID
from tuesday.conversations import SQLiteConversationRepository
from tuesday.domain import ConversationMessage, MessageRole
repository = SQLiteConversationRepository(sys.argv[1])
message = ConversationMessage(MessageRole.USER, "  /chat café\\nline two  ",
    UUID(int=430), datetime.fromisoformat("2026-11-24T12:34:56.123456+05:30"))
asyncio.run(repository.append_messages(UUID(int=43), (message,)))
"""
    reader = """
import asyncio, json, sys
from uuid import UUID
from tuesday.conversations import SQLiteConversationRepository
repository = SQLiteConversationRepository(sys.argv[1])
context = asyncio.run(repository.get_context(UUID(int=43)))
print(json.dumps([str(context.conversation_id), [
    [str(m.message_id), m.role.value, m.content, m.created_at.isoformat()]
    for m in context.messages]]))
"""
    for script in (writer, reader):
        result = subprocess.run(
            [sys.executable, "-c", script, str(database_path)],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
    assert json.loads(result.stdout) == [
        str(CONVERSATION_ID),
        [[str(UUID(int=430)), "user", "  /chat café\nline two  ", STAMP.isoformat()]],
    ]


def test_actual_second_insert_failure_rolls_back_entire_batch(
    repository, database_path
):
    existing = message("Committed earlier")
    append(repository, (existing,))
    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute("""
            CREATE TRIGGER reject_second BEFORE INSERT ON tuesday_conversation_messages
            WHEN NEW.content = 'reject second'
            BEGIN SELECT RAISE(ABORT, 'designated second insert rejected'); END
        """)
    with pytest.raises(ConversationRepositoryError, match="append") as caught:
        append(
            repository,
            (message("First inserted then rolled back"), message("reject second")),
        )
    assert isinstance(caught.value.__cause__, sqlite3.IntegrityError)
    assert read(SQLiteConversationRepository(database_path)).messages == (existing,)


@pytest.mark.parametrize(
    "update,value",
    [
        (
            "UPDATE tuesday_conversation_messages SET message_id=? WHERE sequence=2",
            "bad",
        ),
        ("UPDATE tuesday_conversation_messages SET role=? WHERE sequence=2", "tool"),
        ("UPDATE tuesday_conversation_messages SET content=? WHERE sequence=2", " \n"),
        (
            "UPDATE tuesday_conversation_messages SET created_at=? WHERE sequence=2",
            "bad",
        ),
        (
            "UPDATE tuesday_conversation_messages SET created_at=? WHERE sequence=2",
            "2026-11-24T12:00:00",
        ),
        (
            "UPDATE tuesday_conversation_messages SET message_id=? WHERE sequence=2",
            b"id",
        ),
        ("UPDATE tuesday_conversation_messages SET role=? WHERE sequence=2", b"user"),
        (
            "UPDATE tuesday_conversation_messages SET content=? WHERE sequence=2",
            b"text",
        ),
        (
            "UPDATE tuesday_conversation_messages SET created_at=? WHERE sequence=2",
            b"date",
        ),
    ],
)
@pytest.mark.parametrize("operation", ["read", "append"])
def test_malformed_persisted_data_fails_closed_and_rolls_back_new_appends(
    repository, database_path, update, value, operation
):
    append(repository, (message("Valid first"), message("Corrupted second")))
    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.execute("PRAGMA ignore_check_constraints=ON")
        connection.execute(update, (value,))
    with pytest.raises(
        ConversationRepositoryError, match="Invalid persisted"
    ) as caught:
        if operation == "read":
            read(repository)
        else:
            append(repository, (message("Must roll back"),))
    assert isinstance(caught.value.__cause__, (TypeError, ValueError))
    assert "Corrupted second" not in str(caught.value)
    with closing(sqlite3.connect(database_path)) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM tuesday_conversation_messages"
        ).fetchone() == (2,)


@pytest.mark.parametrize("operation", ["read", "append"])
@pytest.mark.parametrize("failure", ["deleted", "dropped-table", "corrupt"])
def test_database_failures_do_not_recreate_or_reset_storage(
    repository, database_path, operation, failure
):
    if failure == "deleted":
        database_path.unlink()
    elif failure == "dropped-table":
        with closing(sqlite3.connect(database_path)) as connection:
            connection.execute("DROP TABLE tuesday_conversation_messages")
    else:
        database_path.write_bytes(b"corrupted database")
    with pytest.raises(ConversationRepositoryError) as caught:
        read(repository) if operation == "read" else append(repository, (message(),))
    assert isinstance(caught.value.__cause__, sqlite3.Error)
    if failure == "deleted":
        assert not database_path.exists()
    elif failure == "corrupt":
        assert database_path.read_bytes() == b"corrupted database"


@pytest.mark.parametrize("operation", ["initialize", "read", "append"])
def test_connection_errors_are_chained(
    repository, database_path, monkeypatch, operation
):
    error = sqlite3.OperationalError("Connection unavailable")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(sqlite_module.sqlite3, "connect", fail)
    with pytest.raises(ConversationRepositoryError) as caught:
        if operation == "initialize":
            SQLiteConversationRepository(database_path)
        elif operation == "read":
            read(repository)
        else:
            append(repository, (message(),))
    assert caught.value.__cause__ is error


def test_operations_use_owned_worker_connections_and_always_close(
    database_path, tracked_connections
):
    repository = SQLiteConversationRepository(database_path)
    assert tracked_connections[0].owner == threading.get_ident()
    append(repository, (message(),))
    read(repository)
    assert len(tracked_connections) == 3
    assert all(item.closed for item in tracked_connections)
    assert all(item.owner != threading.get_ident() for item in tracked_connections[1:])


@pytest.mark.parametrize("operation", ["read", "append", "decode"])
def test_failed_queries_and_decoding_close_connections(
    database_path, tracked_connections, operation
):
    repository = SQLiteConversationRepository(database_path)
    append(repository, (message(),))
    with closing(sqlite3.connect(database_path, timeout=5.0)) as connection, connection:
        if operation == "decode":
            connection.execute(
                "UPDATE tuesday_conversation_messages SET message_id='bad'"
            )
        else:
            connection.execute("DROP TABLE tuesday_conversation_messages")
    with pytest.raises(ConversationRepositoryError):
        append(repository, (message(),)) if operation == "append" else read(repository)
    assert all(item.closed for item in tracked_connections)


def test_initialization_failure_closes_connection_and_rolls_back_schema(
    database_path, tracked_connections
):
    # A conflicting index name forces failure after CREATE TABLE has succeeded.
    with closing(sqlite3.connect(database_path, timeout=5.0)) as connection:
        connection.execute(
            "CREATE TABLE idx_tuesday_conversation_messages_conversation_sequence "
            "(x TEXT)"
        )
    with pytest.raises(ConversationRepositoryError):
        SQLiteConversationRepository(database_path)
    assert all(item.closed for item in tracked_connections)
    with closing(sqlite3.connect(database_path, timeout=5.0)) as connection:
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE name='tuesday_conversation_messages'"
            ).fetchall()
            == []
        )


def test_commit_failure_rolls_back_and_closes_connection(
    repository, database_path, monkeypatch, tracked_connections
):
    append(repository, (message("Earlier committed"),))
    with closing(sqlite3.connect(database_path, timeout=5.0)) as connection:
        connection.executescript("""
            CREATE TABLE parent (id INTEGER PRIMARY KEY);
            CREATE TABLE child (
                id INTEGER REFERENCES parent(id) DEFERRABLE INITIALLY DEFERRED
            );
            CREATE TRIGGER fail_commit AFTER INSERT ON tuesday_conversation_messages
            WHEN NEW.content = 'fail at commit'
            BEGIN INSERT INTO child VALUES (1); END;
        """)
    original = repository._connect

    def with_foreign_keys(self):
        connection = original()
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    monkeypatch.setattr(SQLiteConversationRepository, "_connect", with_foreign_keys)
    with pytest.raises(ConversationRepositoryError) as caught:
        append(repository, (message("First rolled back"), message("fail at commit")))
    assert isinstance(caught.value.__cause__, sqlite3.IntegrityError)
    assert all(connection.closed for connection in tracked_connections)
    assert read(repository).messages == (message("Earlier committed"),)


@pytest.mark.parametrize("shared_instance", [False, True])
def test_concurrent_batches_are_complete_contiguous_and_not_lost(
    repository, database_path, shared_instance
):
    other = (
        repository if shared_instance else SQLiteConversationRepository(database_path)
    )
    left = tuple(message(f"Left {index}", index=index + 1) for index in range(3))
    right = tuple(message(f"Right {index}", index=index + 4) for index in range(3))

    async def concurrent():
        return await asyncio.gather(
            repository.append_messages(CONVERSATION_ID, left),
            other.append_messages(CONVERSATION_ID, right),
        )

    snapshots = asyncio.run(concurrent())
    assert read(repository).messages in (left + right, right + left)
    for snapshot, own in zip(snapshots, (left, right), strict=True):
        assert snapshot.messages[-3:] == own
    assert sorted(len(snapshot.messages) for snapshot in snapshots) == [3, 6]


def test_storage_module_has_only_standard_library_and_domain_dependencies():
    tree = ast.parse(inspect.getsource(sqlite_module))
    imported = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported == {
        "asyncio",
        "sqlite3",
        "contextlib",
        "datetime",
        "pathlib",
        "uuid",
        "tuesday.conversations._sqlite",
        "tuesday.conversations.base",
        "tuesday.domain",
    }


def test_imports_never_connect_or_create_files(tmp_path):
    script = """
import sqlite3
def forbidden(*args, **kwargs):
    raise AssertionError("Imports must not connect to SQLite")
sqlite3.connect = forbidden
from tuesday.conversations import SQLiteConversationRepository
import tuesday.conversations.sqlite
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert list(tmp_path.iterdir()) == []
