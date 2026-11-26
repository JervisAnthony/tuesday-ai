"""Interactive sessions over real application components and fake SDK I/O."""

import asyncio
import io
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest
from openai import OpenAIError

import tuesday.cli as cli
import tuesday.language_models.openai as openai_adapter
import tuesday.terminal_session as terminal
from tuesday.conversations import (
    ConversationRepositoryError,
    InMemoryConversationRepository,
    SQLiteConversationRepository,
)
from tuesday.services.conversation import StatefulConversationService

FIRST = UUID(int=46)
SECOND = UUID(int=47)
UNKNOWN = UUID(int=48)


class Input(io.StringIO):
    def __init__(self, text, *, tty=False, before_read=None):
        super().__init__(text)
        self.tty = tty
        self.before_read = before_read

    def isatty(self):
        return self.tty

    def readline(self):
        if self.before_read is not None:
            self.before_read()
        return super().readline()


def feed(monkeypatch, text, **kwargs):
    stream = Input(text, **kwargs)
    monkeypatch.setattr(sys, "stdin", stream)
    return stream


def options(*extra):
    return ["--interactive", "--conversation-id", str(FIRST), *extra]


def context(repository, identifier=FIRST):
    return asyncio.run(repository.get_context(identifier))


@pytest.fixture
def observed(monkeypatch):
    state = SimpleNamespace(
        requests=[], responses=[], loops=[], services=[], repositories=[]
    )
    original_handle = StatefulConversationService.handle
    original_service = cli.StatefulConversationService
    original_repository = cli.InMemoryConversationRepository

    async def handle(self, request):
        state.requests.append(request)
        state.loops.append(asyncio.get_running_loop())
        response = await original_handle(self, request)
        state.responses.append(response)
        return response

    def service(*args, **kwargs):
        value = original_service(*args, **kwargs)
        state.services.append(value)
        return value

    def repository():
        value = original_repository()
        state.repositories.append(value)
        return value

    monkeypatch.setattr(StatefulConversationService, "handle", handle)
    state.service = Mock(side_effect=service)
    state.repository = Mock(side_effect=repository)
    state.orchestrator = Mock(wraps=cli.create_default_orchestrator)
    state.model_orchestrator = Mock(wraps=cli.create_model_backed_orchestrator)
    state.runner = Mock(wraps=asyncio.Runner)
    monkeypatch.setattr(cli, "StatefulConversationService", state.service)
    monkeypatch.setattr(cli, "InMemoryConversationRepository", state.repository)
    monkeypatch.setattr(cli, "create_default_orchestrator", state.orchestrator)
    monkeypatch.setattr(
        cli, "create_model_backed_orchestrator", state.model_orchestrator
    )
    monkeypatch.setattr(cli.asyncio, "Runner", state.runner)
    return state


@pytest.fixture
def sdk(monkeypatch):
    for name in ("PROVIDER", "NAME", "API_KEY", "TIMEOUT_SECONDS", "TEMPERATURE"):
        monkeypatch.delenv("TUESDAY_MODEL_" + name, raising=False)
    monkeypatch.setenv("TUESDAY_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("TUESDAY_MODEL_NAME", "fake-model")
    monkeypatch.setenv("TUESDAY_MODEL_API_KEY", "fake-session-key")
    state = SimpleNamespace(
        outcomes=iter(["First answer", "Second answer", "Third answer"]),
        loops=[],
        cleanup_error=None,
    )

    async def generate(**kwargs):
        state.loops.append(asyncio.get_running_loop())
        outcome = next(state.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return SimpleNamespace(output=(), output_text=outcome, model="fake-model")

    async def close():
        state.loops.append(asyncio.get_running_loop())
        if state.cleanup_error is not None:
            raise state.cleanup_error

    state.create = AsyncMock(side_effect=generate)
    state.close = AsyncMock(side_effect=close)
    state.client = SimpleNamespace(
        responses=SimpleNamespace(create=state.create), close=state.close
    )

    def construct(**kwargs):
        state.loops.append(asyncio.get_running_loop())
        return state.client

    state.factory = Mock(side_effect=construct)
    state.settings = Mock(wraps=cli.load_settings)
    monkeypatch.setattr(openai_adapter, "AsyncOpenAI", state.factory)
    monkeypatch.setattr(cli, "load_settings", state.settings)
    return state


def test_session_switches_conversations_without_rebuilding_runtime(
    monkeypatch, capsys, observed
):
    monkeypatch.setattr(terminal, "uuid4", lambda: SECOND)
    source = (
        "/chat First\n:new\n/chat Separate\n"
        f":use {FIRST}\n/conversation Return\n:use {UNKNOWN}\n"
        "/chat Unknown conversation\n:id\n:exit\n/chat Never processed\n"
    )
    feed(monkeypatch, source)
    assert cli.main(options()) == 0
    output = capsys.readouterr()
    assert output.out == (
        "TUESDAY received: First\nTUESDAY received: Separate\n"
        "TUESDAY received: Return (2 prior messages in context)\n"
        "TUESDAY received: Unknown conversation\n"
    )
    assert "tuesday>" not in output.err
    assert [request.conversation_id for request in observed.requests] == [
        FIRST,
        SECOND,
        FIRST,
        UNKNOWN,
    ]
    assert len({request.request_id for request in observed.requests}) == 4
    observed.repository.assert_called_once_with()
    observed.service.assert_called_once()
    observed.orchestrator.assert_called_once_with()
    observed.runner.assert_called_once_with()
    assert all(loop is observed.loops[0] for loop in observed.loops)
    repository = observed.repositories[0]
    assert len(context(repository, FIRST).messages) == 4
    assert len(context(repository, SECOND).messages) == 2
    assert len(context(repository, UNKNOWN).messages) == 2


@pytest.mark.parametrize(
    "line",
    [
        ":",
        ":unknown private",
        ":exit extra",
        ":new extra",
        ":id extra",
        ":help extra",
        ":use",
        ":use bad",
        f":use {SECOND} extra",
    ],
)
def test_invalid_local_commands_preserve_selection_and_do_not_submit(
    monkeypatch, capsys, observed, line
):
    feed(monkeypatch, line + "\n:id\n/chat Still first\n")
    assert cli.main(options()) == 0
    output = capsys.readouterr()
    assert output.out == "TUESDAY received: Still first\n"
    assert output.err.count(str(FIRST)) == 2
    assert str(SECOND) not in output.err
    assert "private" not in output.err
    assert "tuesday:" in output.err
    assert len(observed.requests) == 1
    assert observed.requests[0].conversation_id == FIRST


def test_blank_lines_and_help_create_no_domain_request_or_messages(monkeypatch, capsys):
    feed(monkeypatch, "\r\n \t\n:help\n:id\n:exit\n")
    request = Mock(side_effect=AssertionError("No request for local commands"))
    monkeypatch.setattr(terminal, "TuesdayRequest", request)
    monkeypatch.setattr(cli, "TuesdayRequest", request)
    assert cli.main(options()) == 0
    output = capsys.readouterr()
    assert output.out == ""
    for value in (
        ":help",
        ":id",
        ":new",
        ":use UUID",
        ":exit",
        "/chat",
        "/conversation",
    ):
        assert value in output.err
    request.assert_not_called()


@pytest.mark.parametrize("tty", [False, True])
def test_prompts_only_use_stderr_for_terminal_input(monkeypatch, capsys, tty):
    feed(monkeypatch, "/chat Hello\n:exit\n", tty=tty)
    assert cli.main(options()) == 0
    output = capsys.readouterr()
    assert output.out == "TUESDAY received: Hello\n"
    assert output.err.count("tuesday> ") == (2 if tty else 0)


def test_generated_initial_id_and_exact_physical_request_lines(
    monkeypatch, capsys, observed
):
    monkeypatch.setattr(cli, "uuid4", lambda: FIRST)
    feed(monkeypatch, "/chat  café नमस्ते 🌍  \r\n/conversation Next\n")
    assert cli.main(["--interactive"]) == 0
    output = capsys.readouterr()
    assert output.out == (
        "TUESDAY received: café नमस्ते 🌍  \n"
        "TUESDAY received: Next (2 prior messages in context)\n"
    )
    assert [request.content for request in observed.requests] == [
        "/chat  café नमस्ते 🌍  ",
        "/conversation Next",
    ]
    assert all(request.conversation_id == FIRST for request in observed.requests)
    assert str(FIRST) in output.err
    assert (
        context(observed.repositories[0]).messages[0].content == "/chat  café नमस्ते 🌍  "
    )


def test_routing_and_preparation_failures_recover_without_rewriting_input(
    monkeypatch, capsys, observed
):
    feed(
        monkeypatch,
        " /chat private\n/unknown private\n/chat\n/conversation\n/chat Good\n",
    )
    assert cli.main(options()) == 0
    output = capsys.readouterr()
    assert output.out == "TUESDAY received: Good\n"
    assert "private" not in output.err
    assert output.err.count("tuesday:") == 4
    assert len(observed.requests) == 5
    assert observed.requests[0].content == " /chat private"
    assert all(request.conversation_id == FIRST for request in observed.requests)


def test_model_provider_is_reused_after_recoverable_failure_on_one_loop(
    monkeypatch, capsys, observed, sdk
):
    sdk.outcomes = iter(
        ["First answer", OpenAIError("private SDK fake-session-key"), "Later answer"]
    )

    def before_read():
        sdk.close.assert_not_awaited()

    feed(
        monkeypatch,
        "/chat First\n/chat Failed\n:help\n/chat Later\n:exit\n",
        before_read=before_read,
    )
    assert cli.main(options("--runtime", "openai")) == 0
    output = capsys.readouterr()
    assert output.out == "First answer\nLater answer\n"
    assert output.err.count("model operation failed") == 1
    assert "private" not in output.err and "fake-session-key" not in output.err
    sdk.settings.assert_called_once_with()
    sdk.factory.assert_called_once_with(
        api_key="fake-session-key", timeout=30.0, max_retries=0
    )
    sdk.close.assert_awaited_once()
    assert sdk.create.await_count == 3
    observed.model_orchestrator.assert_called_once()
    observed.repository.assert_called_once()
    observed.service.assert_called_once()
    observed.orchestrator.assert_not_called()
    observed.runner.assert_called_once()
    assert all(loop is sdk.loops[0] for loop in [*sdk.loops, *observed.loops])
    assert len(context(observed.repositories[0]).messages) == 4


def test_model_conversation_switching_reuses_provider_and_restores_selected_history(
    monkeypatch, capsys, observed, sdk
):
    monkeypatch.setattr(terminal, "uuid4", lambda: SECOND)
    sdk.outcomes = iter(["A first", "B first", "A second"])
    feed(
        monkeypatch,
        f"\n:help\n:id\n/chat A\n:new\n/chat B\n:use {FIRST}\n/chat A again\n",
    )
    assert cli.main(options("--runtime", "openai")) == 0
    assert capsys.readouterr().out == "A first\nB first\nA second\n"
    assert sdk.create.call_args_list[1].kwargs["input"][1:-1] == []
    assert sdk.create.call_args_list[2].kwargs["input"][1:-1] == [
        {"role": "user", "content": "/chat A"},
        {"role": "assistant", "content": "A first"},
    ]
    assert [request.conversation_id for request in observed.requests] == [
        FIRST,
        SECOND,
        FIRST,
    ]
    assert sdk.create.await_count == 3
    sdk.factory.assert_called_once()
    sdk.close.assert_awaited_once()
    observed.repository.assert_called_once()
    observed.service.assert_called_once()
    observed.model_orchestrator.assert_called_once()


@pytest.mark.parametrize("limit", [None, 0, 2])
@pytest.mark.parametrize("sqlite", [False, True])
def test_model_history_limit_preserves_all_successful_session_history(
    monkeypatch, capsys, observed, sdk, limit, sqlite, tmp_path
):
    feed(monkeypatch, "/chat One\n/conversation Two\n/chat Three\n")
    arguments = options("--runtime", "openai")
    database_path = tmp_path / "bounded-session.sqlite3"
    if sqlite:
        arguments += ["--database", str(database_path)]
    if limit is not None:
        arguments += ["--max-history-messages", str(limit)]
    assert cli.main(arguments) == 0
    capsys.readouterr()
    inputs = [call.kwargs["input"] for call in sdk.create.call_args_list]
    expected = [
        {"role": "user", "content": "/chat One"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "/conversation Two"},
        {"role": "assistant", "content": "Second answer"},
    ]
    selected = expected if limit is None else [] if limit == 0 else expected[-2:]
    assert inputs[2][1:-1] == selected
    assert inputs[2][0]["role"] == "system"
    assert inputs[2][-1] == {"role": "user", "content": "Three"}
    repository = observed.services[0]._repository
    assert len(context(repository).messages) == 6
    if sqlite:
        assert context(SQLiteConversationRepository(database_path)) == context(
            repository
        )
    assert (
        observed.model_orchestrator.call_args.kwargs["renderer"].max_history_messages
        == limit
    )
    sdk.close.assert_awaited_once()


@pytest.mark.parametrize("operation", ["get_context", "append_messages"])
def test_repository_turn_failure_recovers_with_same_provider_and_selection(
    monkeypatch, capsys, observed, sdk, operation
):
    original = getattr(InMemoryConversationRepository, operation)
    calls = 0

    async def fail_once(self, *args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConversationRepositoryError("private history")
        return await original(self, *args)

    monkeypatch.setattr(InMemoryConversationRepository, operation, fail_once)
    feed(monkeypatch, "/chat Fail once\n/chat Recover\n")
    assert cli.main(options("--runtime", "openai")) == 0
    output = capsys.readouterr()
    assert output.out == (
        "Second answer\n" if operation == "append_messages" else "First answer\n"
    )
    assert "conversation storage failed" in output.err and "private" not in output.err
    assert sdk.create.await_count == (2 if operation == "append_messages" else 1)
    assert len(context(observed.repositories[0]).messages) == 2
    assert all(request.conversation_id == FIRST for request in observed.requests)
    sdk.close.assert_awaited_once()


@pytest.mark.parametrize("ending", ["", ":exit\n"])
def test_eof_and_exit_close_owned_client_once(monkeypatch, capsys, sdk, ending):
    feed(monkeypatch, ending)
    assert cli.main(options("--runtime", "openai")) == 0
    assert capsys.readouterr().out == ""
    sdk.create.assert_not_awaited()
    sdk.close.assert_awaited_once()


@pytest.mark.parametrize("during", ["input", "execution"])
def test_interruption_closes_once_and_retains_earlier_success(
    monkeypatch, capsys, observed, sdk, during
):
    if during == "input":
        count = 0

        def interrupt():
            nonlocal count
            count += 1
            if count == 2:
                raise KeyboardInterrupt

        feed(monkeypatch, "/chat First\n", before_read=interrupt)
    else:
        sdk.outcomes = iter(["First answer", KeyboardInterrupt()])
        feed(monkeypatch, "/chat First\n/chat Interrupted\n/chat Not replayed\n")
    assert cli.main(options("--runtime", "openai")) == 130
    output = capsys.readouterr()
    assert output.out == "First answer\n"
    assert "interrupted" in output.err
    sdk.close.assert_awaited_once()
    assert len(context(observed.repositories[0]).messages) == 2


@pytest.mark.parametrize("stage", ["setup", "input", "execution"])
@pytest.mark.parametrize("secondary", [False, True])
def test_primary_unexpected_failures_survive_cleanup_without_retry(
    monkeypatch, capsys, sdk, stage, secondary
):
    primary = RuntimeError("primary programming defect")
    if secondary:
        sdk.cleanup_error = OpenAIError("secondary SDK detail")
    if stage == "setup":
        monkeypatch.setattr(
            cli, "create_model_backed_orchestrator", Mock(side_effect=primary)
        )
        feed(monkeypatch, "")
    elif stage == "input":

        def fail():
            raise primary

        feed(monkeypatch, "", before_read=fail)
    else:
        sdk.outcomes = iter([primary])
        feed(monkeypatch, "/chat Fails\n/chat Must not retry\n")
    with pytest.raises(RuntimeError) as caught:
        cli.main(options("--runtime", "openai"))
    assert caught.value is primary
    sdk.close.assert_awaited_once()
    assert sdk.create.await_count == int(stage == "execution")
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("handled_error", [False, True])
def test_shutdown_failure_is_visible_even_after_a_recoverable_error(
    monkeypatch, capsys, observed, sdk, handled_error
):
    sdk.cleanup_error = OpenAIError("private shutdown detail")
    sdk.outcomes = iter(
        [OpenAIError("private turn detail")] if handled_error else ["Saved answer"]
    )
    feed(monkeypatch, "/chat One turn\n")
    assert cli.main(options("--runtime", "openai")) == 1
    output = capsys.readouterr()
    assert output.err.count("model operation failed") == (2 if handled_error else 1)
    assert "private" not in output.err
    assert output.out == ("" if handled_error else "Saved answer\n")
    assert len(context(observed.repositories[0]).messages) == (
        0 if handled_error else 2
    )
    sdk.close.assert_awaited_once()


def test_injected_client_remains_caller_owned(monkeypatch, capsys, sdk):
    original = cli.OpenAILanguageModelProvider
    monkeypatch.setattr(
        cli,
        "OpenAILanguageModelProvider",
        lambda settings: original(settings, client=sdk.client),
    )
    feed(monkeypatch, "/chat Hello\n")
    assert cli.main(options("--runtime", "openai")) == 0
    assert capsys.readouterr().out == "First answer\n"
    sdk.close.assert_not_awaited()


def test_startup_configuration_error_constructs_no_provider(monkeypatch, capsys, sdk):
    monkeypatch.delenv("TUESDAY_MODEL_API_KEY")
    feed(monkeypatch, "/chat Not submitted\n")
    assert cli.main(options("--runtime", "openai")) == 1
    assert "invalid OpenAI model configuration" in capsys.readouterr().err
    sdk.factory.assert_not_called()


def test_startup_storage_failure_does_not_read_input_or_construct_provider(
    monkeypatch, capsys, tmp_path, sdk
):
    def fail():
        raise AssertionError("Startup failure must not read terminal input")

    feed(monkeypatch, "", before_read=fail)
    assert (
        cli.main(
            options(
                "--runtime",
                "openai",
                "--database",
                str(tmp_path / "missing" / "history.db"),
            )
        )
        == 1
    )
    assert capsys.readouterr() == ("", "tuesday: conversation storage failed.\n")
    sdk.factory.assert_not_called()
