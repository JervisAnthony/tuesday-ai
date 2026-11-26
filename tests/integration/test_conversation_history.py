"""History queries across processes and existing stateful application writes."""

import ast
import asyncio
import inspect
import json
import subprocess
import sys
from unittest.mock import Mock
from uuid import UUID

import pytest

import tuesday.conversations._sqlite as sqlite_values
import tuesday.conversations.history as history_module
import tuesday.conversations.sqlite_history as reader_module
from tuesday.composition import create_default_orchestrator
from tuesday.conversations import (
    SQLiteConversationHistoryReader,
    SQLiteConversationRepository,
)
from tuesday.domain import TuesdayRequest
from tuesday.services.conversation import StatefulConversationService

IDENTIFIER = UUID(int=47)


def test_stateful_service_writes_are_visible_without_reconstructing_reader(tmp_path):
    path = tmp_path / "stateful.sqlite3"
    writer = SQLiteConversationRepository(path)
    reader = SQLiteConversationHistoryReader(path)
    service = StatefulConversationService(create_default_orchestrator(), writer)
    assert asyncio.run(reader.list_conversations()).items == ()
    first = TuesdayRequest("/chat First question", IDENTIFIER)
    first_response = asyncio.run(service.handle(first))
    snapshot = asyncio.run(reader.get_messages(IDENTIFIER, limit=1))
    assert snapshot.messages[0].content == first.content and snapshot.next_offset == 1
    second = TuesdayRequest("/conversation Second question", IDENTIFIER)
    second_response = asyncio.run(service.handle(second))
    page = asyncio.run(reader.get_messages(IDENTIFIER))
    assert [message.content for message in page.messages] == [
        first.content,
        first_response.content,
        second.content,
        second_response.content,
    ]
    assert len(page.messages) == 4
    assert asyncio.run(reader.list_conversations()).items[0].message_count == 4
    assert snapshot.messages[0].content == first.content
    assert len(snapshot.messages) == 1


def test_separate_process_reader_inspects_committed_stateful_history(tmp_path):
    path = tmp_path / "cross-process.sqlite3"
    writer_script = """
import asyncio, sys
from uuid import UUID
from tuesday.composition import create_default_orchestrator
from tuesday.conversations import SQLiteConversationRepository
from tuesday.domain import TuesdayRequest
from tuesday.services.conversation import StatefulConversationService
service = StatefulConversationService(create_default_orchestrator(),
    SQLiteConversationRepository(sys.argv[1]))
async def write():
    for content in ("/chat First", "/conversation Second"):
        await service.handle(TuesdayRequest(content, UUID(int=47)))
asyncio.run(write())
"""
    reader_script = """
import asyncio, json, sys
from uuid import UUID
from tuesday.conversations import (
    SQLiteConversationHistoryReader, SQLiteConversationRepository,
)
def forbidden(*args, **kwargs):
    raise AssertionError("A reader must not construct a writable repository")
SQLiteConversationRepository.__init__ = forbidden
reader = SQLiteConversationHistoryReader(sys.argv[1])
async def inspect_history():
    listing = await reader.list_conversations(limit=1)
    messages, offset = [], 0
    while True:
        page = await reader.get_messages(UUID(int=47), limit=1, offset=offset)
        messages.extend([str(m.message_id), m.role.value, m.content,
                         m.created_at.isoformat()] for m in page.messages)
        if page.next_offset is None:
            break
        offset = page.next_offset
    print(json.dumps({"id": str(listing.items[0].conversation_id),
        "count": listing.items[0].message_count, "messages": messages}))
asyncio.run(inspect_history())
"""
    subprocess.run(
        [sys.executable, "-c", writer_script, str(path)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    before = path.read_bytes()
    result = subprocess.run(
        [sys.executable, "-c", reader_script, str(path)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = json.loads(result.stdout)
    assert payload["id"] == str(IDENTIFIER) and payload["count"] == 4
    assert [item[1:3] for item in payload["messages"]] == [
        ["user", "/chat First"],
        ["assistant", "TUESDAY received: First"],
        ["user", "/conversation Second"],
        ["assistant", "TUESDAY received: Second (2 prior messages in context)"],
    ]
    original = asyncio.run(
        SQLiteConversationHistoryReader(path).get_messages(IDENTIFIER)
    )
    assert payload["messages"] == [
        [
            str(message.message_id),
            message.role.value,
            message.content,
            message.created_at.isoformat(),
        ]
        for message in original.messages
    ]
    assert path.read_bytes() == before


def test_reader_constructor_and_queries_never_construct_writer(tmp_path, monkeypatch):
    path = tmp_path / "existing.sqlite3"
    SQLiteConversationRepository(path)
    forbidden = Mock(side_effect=AssertionError("No writable repository construction"))
    monkeypatch.setattr(SQLiteConversationRepository, "__init__", forbidden)
    reader = SQLiteConversationHistoryReader(path)
    assert asyncio.run(reader.list_conversations()).items == ()
    assert asyncio.run(reader.get_messages(IDENTIFIER)).messages == ()
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "statement",
    [
        "import tuesday.conversations",
        "import tuesday.conversations.history; "
        "import tuesday.conversations.sqlite_history",
        "import tuesday.services; import tuesday.conversations",
        "import tuesday.cli; import tuesday.conversations.sqlite_history",
    ],
)
def test_clean_imports_and_reader_construction_never_open_or_create_storage(
    tmp_path, statement
):
    script = (
        """
import pathlib, sqlite3
def forbidden(*args, **kwargs):
    raise AssertionError("Imports and reader construction must not connect")
sqlite3.connect = forbidden
"""
        + statement
        + """
from tuesday.conversations import SQLiteConversationHistoryReader
SQLiteConversationHistoryReader(pathlib.Path("missing/never-created.sqlite3"))
"""
    )
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


@pytest.mark.parametrize("module", [history_module, reader_module, sqlite_values])
def test_query_modules_do_not_import_application_provider_agent_or_tool_layers(module):
    tree = ast.parse(inspect.getsource(module))
    imports = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert all(
        not name.startswith("tuesday.")
        or name == "tuesday.domain"
        or name.startswith("tuesday.conversations.")
        for name in imports
    )
    assert all(
        name.split(".")[0] not in ("openai", "httpx", "requests") for name in imports
    )
