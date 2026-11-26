"""Shared history-query contracts, pagination, and immutable result values."""

import asyncio
import inspect
from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime, timedelta, timezone, tzinfo
from unittest.mock import Mock
from uuid import UUID

import pytest

import tuesday.conversations as conversations
import tuesday.conversations.memory as memory_module
from tuesday.conversations import (
    BaseConversationHistoryReader,
    BaseConversationRepository,
    ConversationMessagePage,
    ConversationSummary,
    ConversationSummaryPage,
    InMemoryConversationRepository,
    SQLiteConversationHistoryReader,
    SQLiteConversationRepository,
)
from tuesday.domain import ConversationContext, ConversationMessage, MessageRole

A, B, C, UNKNOWN = (UUID(int=value) for value in (471, 472, 473, 499))
STAMP = datetime(
    2026, 11, 26, 12, 30, 45, 123456, tzinfo=timezone(timedelta(hours=5, minutes=30))
)
MAX_OFFSET = 2**63 - 1


def run(operation):
    return asyncio.run(operation)


@pytest.fixture(params=["memory", "sqlite"])
def backend(request, tmp_path):
    if request.param == "memory":
        writer = reader = InMemoryConversationRepository()
    else:
        path = tmp_path / "shared-history.sqlite3"
        writer = SQLiteConversationRepository(path)
        reader = SQLiteConversationHistoryReader(path)
    return writer, reader


@pytest.fixture
def messages():
    return (
        ConversationMessage(
            MessageRole.SYSTEM, "  First system\ncontext  ", UUID(int=1), STAMP
        ),
        ConversationMessage(
            MessageRole.USER,
            "  /chat café नमस्ते 🌍\n'; DROP TABLE tuesday_conversation_messages; --  ",
            UUID(int=1),
            STAMP,
        ),
        ConversationMessage(
            MessageRole.ASSISTANT,
            "  Last\nanswer  ",
            UUID(int=2),
            STAMP - timedelta(days=1),
        ),
    )


@pytest.fixture
def populated(backend, messages):
    writer, reader = backend
    run(writer.append_messages(A, messages[:1]))
    run(
        writer.append_messages(
            B, (replace(messages[0], created_at=STAMP - timedelta(days=2)),)
        )
    )
    run(writer.append_messages(A, messages[1:]))
    run(writer.append_messages(C, messages[:1]))
    return writer, reader


def test_separate_query_boundary_and_public_exports(tmp_path):
    assert BaseConversationRepository.__abstractmethods__ == {
        "get_context",
        "append_messages",
    }
    assert BaseConversationHistoryReader.__abstractmethods__ == {
        "list_conversations",
        "get_messages",
    }
    with pytest.raises(TypeError):
        BaseConversationHistoryReader()
    assert BaseConversationHistoryReader.__slots__ == ()

    class WriteOnly(BaseConversationRepository):
        async def get_context(self, conversation_id):
            return ConversationContext(conversation_id)

        async def append_messages(self, conversation_id, messages):
            return ConversationContext(conversation_id, messages)

    writer = WriteOnly()
    assert not isinstance(writer, BaseConversationHistoryReader)
    assert run(writer.get_context(A)).messages == ()
    assert run(writer.append_messages(A, ())).messages == ()
    assert isinstance(InMemoryConversationRepository(), BaseConversationHistoryReader)
    reader = SQLiteConversationHistoryReader(tmp_path / "not-created.db")
    assert isinstance(reader, BaseConversationHistoryReader)
    assert not isinstance(reader, BaseConversationRepository)
    assert not hasattr(reader, "append_messages") and not hasattr(reader, "get_context")
    for name in (
        "BaseConversationHistoryReader",
        "ConversationSummary",
        "ConversationSummaryPage",
        "ConversationMessagePage",
        "SQLiteConversationHistoryReader",
    ):
        assert getattr(conversations, name) is globals()[name]
        assert name in conversations.__all__


@pytest.mark.parametrize(
    "implementation",
    [
        BaseConversationHistoryReader,
        InMemoryConversationRepository,
        SQLiteConversationHistoryReader,
    ],
)
def test_async_signatures_and_keyword_only_pagination(implementation):
    for name, expected in (
        ("list_conversations", ("self", "limit", "offset")),
        ("get_messages", ("self", "conversation_id", "limit", "offset")),
    ):
        method = getattr(implementation, name)
        assert inspect.iscoroutinefunction(method)
        parameters = inspect.signature(method).parameters
        assert tuple(parameters) == expected
        assert parameters["limit"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["offset"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["limit"].default == 50 and parameters["offset"].default == 0


def test_empty_store_and_unknown_reads_do_not_create_listed_history(backend):
    writer, reader = backend
    assert run(reader.list_conversations()) == ConversationSummaryPage((), 50, 0, None)
    assert run(writer.get_context(UNKNOWN)).messages == ()
    assert run(reader.get_messages(UNKNOWN)) == ConversationMessagePage(
        UNKNOWN, (), 50, 0, None
    )
    assert run(reader.list_conversations()).items == ()


@pytest.mark.parametrize("limit", [1, 50, 200])
@pytest.mark.parametrize("offset", [0, 1, MAX_OFFSET])
def test_valid_pagination_including_maximum_offset(backend, limit, offset):
    _, reader = backend
    assert run(reader.list_conversations(limit=limit, offset=offset)).items == ()
    assert run(reader.get_messages(A, limit=limit, offset=offset)).messages == ()


@pytest.mark.parametrize(
    "field,value,error_type",
    [
        ("limit", 0, ValueError),
        ("limit", -1, ValueError),
        ("limit", 201, ValueError),
        ("offset", -1, ValueError),
        ("offset", 2**63, ValueError),
        *[
            (field, value, TypeError)
            for field in ("limit", "offset")
            for value in (True, False, None, 1.5, "2", [], {}, object())
        ],
    ],
)
@pytest.mark.parametrize("operation", ["list_conversations", "get_messages"])
def test_invalid_pagination_is_rejected_before_access(
    backend, monkeypatch, operation, field, value, error_type
):
    _, reader = backend
    if isinstance(reader, SQLiteConversationHistoryReader):
        monkeypatch.setattr(
            SQLiteConversationHistoryReader,
            "_query",
            Mock(side_effect=AssertionError("No database access")),
        )
    else:

        class NoAccess(dict):
            def items(self):
                raise AssertionError("No storage access")

            def get(self, *args):
                raise AssertionError("No storage access")

        reader._messages_by_conversation = NoAccess()
    args = () if operation == "list_conversations" else (A,)
    with pytest.raises(error_type):
        run(getattr(reader, operation)(*args, **{field: value}))


@pytest.mark.parametrize("invalid", [None, "not-a-uuid", 47, True, object()])
def test_invalid_conversation_id_is_rejected_before_query(
    backend, monkeypatch, invalid
):
    _, reader = backend
    if isinstance(reader, SQLiteConversationHistoryReader):
        monkeypatch.setattr(
            SQLiteConversationHistoryReader,
            "_query",
            Mock(side_effect=AssertionError("No database access")),
        )
    with pytest.raises(TypeError, match="conversation_id must be a UUID"):
        run(reader.get_messages(invalid))


def test_summary_counts_append_order_and_first_last_append_timestamps(
    populated, messages
):
    writer, reader = populated
    first = run(reader.list_conversations())
    assert [item.conversation_id for item in first.items] == [A, B, C]
    assert [item.message_count for item in first.items] == [3, 1, 1]
    assert first.items[0].first_message_at == STAMP
    assert first.items[0].last_message_at == STAMP - timedelta(days=1)
    later = replace(messages[0], created_at=STAMP - timedelta(days=3))
    run(writer.append_messages(A, (later,)))
    latest = run(reader.list_conversations())
    assert [item.conversation_id for item in latest.items] == [A, B, C]
    assert latest.items[0].message_count == 4
    assert latest.items[0].first_message_at == STAMP
    assert latest.items[0].last_message_at == later.created_at
    assert first.items[0].message_count == 3


@pytest.mark.parametrize(
    "limit,offset,indices,next_offset",
    [
        (1, 0, (0,), 1),
        (2, 0, (0, 1), 2),
        (3, 0, (0, 1, 2), None),
        (200, 0, (0, 1, 2), None),
        (1, 1, (1,), 2),
        (2, 1, (1, 2), None),
        (1, 2, (2,), None),
        (2, 3, (), None),
        (2, MAX_OFFSET, (), None),
    ],
)
def test_listing_and_message_page_boundaries(
    populated, messages, limit, offset, indices, next_offset
):
    _, reader = populated
    summaries = run(reader.list_conversations(limit=limit, offset=offset))
    page = run(reader.get_messages(A, limit=limit, offset=offset))
    assert [item.conversation_id for item in summaries.items] == [
        [A, B, C][index] for index in indices
    ]
    assert page.messages == tuple(messages[index] for index in indices)
    assert summaries.limit == page.limit == limit
    assert summaries.offset == page.offset == offset
    assert summaries.next_offset == page.next_offset == next_offset
    assert page.conversation_id == A


def test_following_next_offset_reads_all_results_without_lookahead_leaks(
    populated, messages
):
    _, reader = populated
    identifiers, transcript = [], []
    offset = 0
    while True:
        listing = run(reader.list_conversations(limit=1, offset=offset))
        identifiers.extend(item.conversation_id for item in listing.items)
        if listing.next_offset is None:
            break
        offset = listing.next_offset
    offset = 0
    while True:
        page = run(reader.get_messages(A, limit=2, offset=offset))
        transcript.extend(page.messages)
        if page.next_offset is None:
            break
        offset = page.next_offset
    assert identifiers == [A, B, C]
    assert tuple(transcript) == messages


def test_messages_preserve_values_isolation_duplicates_and_immutable_snapshots(
    populated, messages
):
    writer, reader = populated
    page = run(reader.get_messages(A))
    assert page.messages == messages
    assert page.messages[0].message_id == page.messages[1].message_id
    assert page.messages[1].created_at.utcoffset() == timedelta(hours=5, minutes=30)
    assert page.messages[1].created_at.microsecond == 123456
    assert run(reader.get_messages(B)).messages != page.messages
    run(writer.append_messages(A, (messages[0],)))
    assert page.messages == messages
    assert len(run(reader.get_messages(A)).messages) == 4
    assert not hasattr(page, "total_message_count")
    for value, field in (
        (page, "messages"),
        (page.messages[0], "content"),
        (run(reader.list_conversations()), "items"),
        (run(reader.list_conversations()).items[0], "message_count"),
    ):
        assert not hasattr(value, "__dict__")
        with pytest.raises(FrozenInstanceError):
            setattr(value, field, ())


class NoOffset(tzinfo):
    def utcoffset(self, dt):
        return None


@pytest.mark.parametrize(
    "field,value,error_type",
    [
        ("conversation_id", "id", TypeError),
        ("message_count", True, TypeError),
        ("message_count", 1.5, TypeError),
        ("message_count", -1, ValueError),
        ("first_message_at", "date", TypeError),
        ("last_message_at", None, TypeError),
        ("first_message_at", datetime(2026, 1, 1), ValueError),
        ("last_message_at", datetime(2026, 1, 1, tzinfo=NoOffset()), ValueError),
    ],
)
def test_summary_contract_rejects_invalid_types_and_naive_timestamps(
    field, value, error_type
):
    summary = ConversationSummary(A, 1, STAMP, STAMP)
    with pytest.raises(error_type):
        replace(summary, **{field: value})


@pytest.mark.parametrize("page_type", ["summary", "messages"])
@pytest.mark.parametrize("value", [[], [object()], (object(),)])
def test_page_containers_require_immutable_typed_tuples(page_type, value):
    with pytest.raises(TypeError):
        if page_type == "summary":
            ConversationSummaryPage(value, 50, 0, None)
        else:
            ConversationMessagePage(A, value, 50, 0, None)


@pytest.mark.parametrize(
    "value,error_type", [(True, TypeError), ("1", TypeError), (-1, ValueError)]
)
@pytest.mark.parametrize("page_type", ["summary", "messages"])
def test_next_offset_requires_nonnegative_integer_or_none(page_type, value, error_type):
    with pytest.raises(error_type):
        if page_type == "summary":
            ConversationSummaryPage((), 50, 0, value)
        else:
            ConversationMessagePage(A, (), 50, 0, value)


def test_result_field_sets_have_no_generated_metadata():
    assert [field.name for field in fields(ConversationSummary)] == [
        "conversation_id",
        "message_count",
        "first_message_at",
        "last_message_at",
    ]
    assert [field.name for field in fields(ConversationMessagePage)] == [
        "conversation_id",
        "messages",
        "limit",
        "offset",
        "next_offset",
    ]


def test_memory_builds_summaries_only_for_selected_entries(monkeypatch):
    repository = InMemoryConversationRepository()
    message = ConversationMessage(MessageRole.USER, "Stored", created_at=STAMP)
    for identifier in (A, B, C):
        run(repository.append_messages(identifier, (message,)))
    summary = Mock(wraps=ConversationSummary)
    monkeypatch.setattr(memory_module, "ConversationSummary", summary)
    page = run(repository.list_conversations(limit=1, offset=1))
    assert page.items[0].conversation_id == B
    summary.assert_called_once_with(B, 1, STAMP, STAMP)
