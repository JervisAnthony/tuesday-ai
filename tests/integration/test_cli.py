"""Real module and installed-command processes with isolated temporary storage."""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from importlib.metadata import distribution
from uuid import UUID

import pytest

import tuesday
from tuesday.cli import main
from tuesday.conversations import SQLiteConversationRepository
from tuesday.domain import MessageRole

CONVERSATION_ID = "00000000-0000-0000-0000-000000000044"
OTHER_ID = "00000000-0000-0000-0000-000000000045"


def invoke(tmp_path, *arguments, env=None):
    return subprocess.run(
        [sys.executable, "-m", "tuesday", *arguments],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def history(path, conversation_id):
    return asyncio.run(
        SQLiteConversationRepository(path).get_context(UUID(conversation_id))
    )


@pytest.mark.parametrize("option", ["--help", "--version"])
def test_module_help_and_version_do_not_create_storage(tmp_path, option):
    result = invoke(tmp_path, option)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    if option == "--version":
        assert result.stdout == f"TUESDAY {tuesday.__version__}\n"
    else:
        assert "usage: tuesday" in result.stdout
        assert "/chat" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_module_text_and_json_success(tmp_path):
    plain = invoke(tmp_path, "/chat Hello TUESDAY")
    assert plain.returncode == 0, plain.stderr
    assert plain.stdout == "TUESDAY received: Hello TUESDAY\n"
    assert plain.stderr == ""
    structured = invoke(tmp_path, "--json", "/conversation Hello TUESDAY")
    assert structured.returncode == 0, structured.stderr
    assert structured.stderr == ""
    payload = json.loads(structured.stdout)
    assert set(payload) == {"content", "conversation_id", "request_id", "response_id"}
    assert payload["content"] == "TUESDAY received: Hello TUESDAY"
    for key in ("conversation_id", "request_id", "response_id"):
        assert str(UUID(payload[key])) == payload[key]


@pytest.mark.parametrize(
    "arguments,exit_code",
    [
        ([], 2),
        (["--conversation-id", "invalid", "/chat Hello"], 2),
        (["--unknown"], 2),
        (["/unknown Hello"], 1),
        (["/chat"], 1),
        (["/conversation"], 1),
    ],
)
def test_process_exit_codes_and_stderr(tmp_path, arguments, exit_code):
    result = invoke(tmp_path, *arguments)
    assert result.returncode == exit_code
    assert result.stdout == ""
    assert result.stderr and "Traceback" not in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_cross_process_durability_with_generated_conversation_id(tmp_path):
    path = tmp_path / "history.sqlite3"
    first_source = "/chat First question"
    first = invoke(tmp_path, "--json", "--database", str(path), first_source)
    assert first.returncode == 0, first.stderr
    one = json.loads(first.stdout)
    second_source = "/conversation Second question"
    second = invoke(
        tmp_path,
        "--json",
        "--database",
        str(path),
        "--conversation-id",
        one["conversation_id"],
        second_source,
    )
    assert second.returncode == 0, second.stderr
    assert first.stderr == second.stderr == ""
    two = json.loads(second.stdout)
    assert one["conversation_id"] == two["conversation_id"]
    assert one["request_id"] != two["request_id"]
    assert one["response_id"] != two["response_id"]
    assert one["content"] == "TUESDAY received: First question"
    assert (
        two["content"]
        == "TUESDAY received: Second question (2 prior messages in context)"
    )
    saved = history(path, one["conversation_id"])
    assert [(message.role, message.content) for message in saved.messages] == [
        (MessageRole.USER, first_source),
        (MessageRole.ASSISTANT, one["content"]),
        (MessageRole.USER, second_source),
        (MessageRole.ASSISTANT, two["content"]),
    ]
    assert len(saved.messages) == 4


def test_process_conversation_and_database_isolation(tmp_path):
    first_path, other_path = tmp_path / "first.db", tmp_path / "other.db"
    for path, identifier in (
        (first_path, CONVERSATION_ID),
        (first_path, OTHER_ID),
        (other_path, CONVERSATION_ID),
    ):
        result = invoke(
            tmp_path,
            "--database",
            str(path),
            "--conversation-id",
            identifier,
            "/chat Independent",
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == "TUESDAY received: Independent\n"
        assert len(history(path, identifier).messages) == 2


def test_default_memory_does_not_cross_process_boundaries(tmp_path):
    for _ in range(2):
        result = invoke(tmp_path, "--conversation-id", CONVERSATION_ID, "/chat Fresh")
        assert result.returncode == 0, result.stderr
        assert result.stdout == "TUESDAY received: Fresh\n"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("source", ["/unknown Hello", "/chat", "/conversation"])
def test_failed_process_does_not_append_to_existing_history(tmp_path, source):
    path = tmp_path / "history.db"
    options = ("--database", str(path), "--conversation-id", CONVERSATION_ID)
    first = invoke(tmp_path, *options, "/chat Previously committed")
    assert first.returncode == 0, first.stderr
    before = history(path, CONVERSATION_ID)
    failed = invoke(tmp_path, *options, source)
    assert failed.returncode == 1
    assert failed.stdout == "" and failed.stderr
    assert history(path, CONVERSATION_ID) == before


def test_model_environment_does_not_activate_live_execution(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "TUESDAY_MODEL_PROVIDER": "openai",
            "TUESDAY_MODEL_NAME": "unused-model",
            "TUESDAY_MODEL_API_KEY": "fake-not-a-real-key",
            "TUESDAY_MODEL_TIMEOUT_SECONDS": "invalid-and-must-not-be-read",
        }
    )
    result = invoke(tmp_path, "/chat Deterministic", env=env)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "TUESDAY received: Deterministic\n"
    assert result.stderr == ""


def test_imports_have_no_cli_runtime_database_or_network_side_effects(tmp_path):
    script = """
import argparse, socket, sqlite3, sys
def forbidden(*args, **kwargs):
    raise AssertionError("Import must not execute application logic")
sqlite3.connect = forbidden
socket.create_connection = forbidden
argparse.ArgumentParser.parse_args = forbidden
import tuesday.composition
import tuesday.config
import tuesday.services.conversation
tuesday.composition.create_default_orchestrator = forbidden
tuesday.config.load_settings = forbidden
tuesday.services.conversation.StatefulConversationService = forbidden
sys.argv = ["unrelated-program", "--unknown-option"]
import tuesday
import tuesday.cli
import tuesday.__main__
assert tuesday.__all__ == ["__version__"]
"""
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


def test_installed_entry_point_metadata_resolves_to_same_main():
    points = [
        point
        for point in distribution("tuesday-ai").entry_points
        if point.group == "console_scripts" and point.name == "tuesday"
    ]
    assert len(points) == 1
    assert points[0].value == "tuesday.cli:main"
    assert points[0].load() is main


@pytest.mark.parametrize(
    "arguments,expected,exit_code",
    [
        (["--version"], f"TUESDAY {tuesday.__version__}\n", 0),
        (["/chat Hello TUESDAY"], "TUESDAY received: Hello TUESDAY\n", 0),
        (["--help"], None, 0),
        ([], "", 2),
        (["/unknown Hello"], "", 1),
    ],
)
def test_actual_installed_console_script(tmp_path, arguments, expected, exit_code):
    # Discover the launcher for this interpreter, independently of the caller's PATH.
    executable = shutil.which("tuesday", path=sysconfig.get_path("scripts"))
    assert executable is not None, "Refresh this environment with pip install -e .[dev]"
    result = subprocess.run(
        [executable, *arguments],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == exit_code, result.stderr
    if expected is None:
        assert "usage: tuesday" in result.stdout
    else:
        assert result.stdout == expected
    assert bool(result.stderr) is (exit_code != 0)
