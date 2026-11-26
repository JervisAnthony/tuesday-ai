"""Terminal entry point with explicit model-backed and interactive execution."""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from uuid import UUID, uuid4

from tuesday import __version__
from tuesday.composition import (
    create_default_orchestrator,
    create_model_backed_orchestrator,
)
from tuesday.config import ConfigurationError, LanguageModelSettings, load_settings
from tuesday.conversations import (
    BaseConversationRepository,
    InMemoryConversationRepository,
    SQLiteConversationRepository,
)
from tuesday.domain import TuesdayRequest, TuesdayResponse
from tuesday.history_cli import main as history_main
from tuesday.language_models import OpenAILanguageModelProvider
from tuesday.prompting import ConversationalPromptRenderer
from tuesday.services.conversation import StatefulConversationService
from tuesday.terminal_session import TURN_ERRORS, report_turn_error, run_session

__all__ = ["main"]


def _nonempty_request(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("request must not be empty.")
    return value


def _nonnegative_int(value: str) -> int:
    limit = int(value)
    if limit < 0:
        raise argparse.ArgumentTypeError("history limit must be nonnegative.")
    return limit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tuesday",
        description="Submit one request using /chat or /conversation.",
        allow_abbrev=False,
        epilog="Inspect stored history: tuesday history {list,show} --help",
    )
    parser.add_argument(
        "request",
        nargs="?",
        metavar="REQUEST",
        type=_nonempty_request,
        help="complete directive and content, quoted as one argument",
    )
    parser.add_argument("--version", action="version", version=f"TUESDAY {__version__}")
    parser.add_argument(
        "--conversation-id",
        type=UUID,
        metavar="UUID",
        help="conversation identifier to reuse; defaults to a fresh UUID",
    )
    parser.add_argument(
        "--database",
        metavar="PATH",
        help="explicit SQLite file for durable history; parent directory must exist",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="output JSON with response content and actual correlation identifiers",
    )
    parser.add_argument(
        "--runtime",
        choices=("deterministic", "openai"),
        default="deterministic",
        help="explicit runtime selection; defaults to deterministic",
    )
    parser.add_argument(
        "--max-history-messages",
        type=_nonnegative_int,
        metavar="N",
        help="OpenAI history message limit; zero omits history, default unlimited",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="read sequential requests and local session commands from stdin",
    )
    return parser


async def _openai_runtime(
    repository: BaseConversationRepository,
    settings: LanguageModelSettings,
    max_history_messages: int | None,
) -> tuple[StatefulConversationService, OpenAILanguageModelProvider]:
    """Construct on the execution loop and close the client if setup fails."""
    provider = OpenAILanguageModelProvider(settings)
    try:
        renderer = ConversationalPromptRenderer(
            max_history_messages=max_history_messages
        )
        orchestrator = create_model_backed_orchestrator(provider, renderer=renderer)
        return StatefulConversationService(orchestrator, repository), provider
    except BaseException:
        await _close_provider(provider, completed=False)
        raise


async def _close_provider(
    provider: OpenAILanguageModelProvider, *, completed: bool
) -> None:
    """Preserve a primary terminating failure; expose standalone cleanup failures."""
    try:
        await provider.aclose()
    except BaseException:
        if completed:
            raise


async def _run_openai(
    request: TuesdayRequest,
    repository: BaseConversationRepository,
    settings: LanguageModelSettings,
    max_history_messages: int | None,
) -> TuesdayResponse:
    service, provider = await _openai_runtime(
        repository, settings, max_history_messages
    )
    completed = False
    try:
        response = await service.handle(request)
        completed = True
        return response
    finally:
        await _close_provider(provider, completed=completed)


def _run_interactive(
    repository: BaseConversationRepository,
    settings: LanguageModelSettings | None,
    max_history_messages: int | None,
    conversation_id: UUID | None,
) -> int:
    with asyncio.Runner() as runner:
        provider = None
        if settings is None:
            service = StatefulConversationService(
                create_default_orchestrator(), repository
            )
        else:
            service, provider = runner.run(
                _openai_runtime(repository, settings, max_history_messages)
            )
        completed = False
        try:
            run_session(service, conversation_id or uuid4(), runner)
            completed = True
        finally:
            if provider is not None:
                runner.run(_close_provider(provider, completed=completed))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one terminal interaction and return its exit status.

    Async programmatic callers should use application services directly instead
    of invoking this function inside an already-running event loop.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "history":
        return history_main(arguments[1:])
    parser = _parser()
    args = parser.parse_args(arguments)
    if args.interactive:
        if args.request is not None:
            parser.error("--interactive does not accept a positional REQUEST.")
        if args.json:
            parser.error("--interactive does not support --json.")
    elif args.request is None:
        parser.error("REQUEST is required unless --interactive is selected.")
    if args.runtime == "deterministic" and args.max_history_messages is not None:
        parser.error("--max-history-messages requires --runtime openai.")
    try:
        model_settings = None
        if args.runtime == "openai":
            model_settings = load_settings().model
            if model_settings is None or model_settings.provider != "openai":
                raise ConfigurationError("OpenAI model settings are required.")
        if args.database is None:
            repository = InMemoryConversationRepository()
        else:
            try:
                repository = SQLiteConversationRepository(args.database)
            except (TypeError, ValueError):
                print("tuesday: invalid database path.", file=sys.stderr)
                return 1
        if args.interactive:
            return _run_interactive(
                repository,
                model_settings,
                args.max_history_messages,
                args.conversation_id,
            )
        request = (
            TuesdayRequest(args.request)
            if args.conversation_id is None
            else TuesdayRequest(args.request, conversation_id=args.conversation_id)
        )
        if model_settings is None:
            service = StatefulConversationService(
                create_default_orchestrator(), repository
            )
            response = asyncio.run(service.handle(request))
        else:
            response = asyncio.run(
                _run_openai(
                    request, repository, model_settings, args.max_history_messages
                )
            )
    except ConfigurationError:
        print("tuesday: invalid OpenAI model configuration.", file=sys.stderr)
        return 1
    except TURN_ERRORS as error:
        report_turn_error(error)
        return 1
    except KeyboardInterrupt:
        print("tuesday: interrupted.", file=sys.stderr)
        return 130

    if args.json:
        print(
            json.dumps(
                {
                    "content": response.content,
                    "conversation_id": str(response.conversation_id),
                    "request_id": str(response.request_id),
                    "response_id": str(response.response_id),
                }
            )
        )
    else:
        print(response.content)
    return 0
