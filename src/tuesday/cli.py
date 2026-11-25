"""One-shot terminal entry point with explicit model-backed execution."""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from uuid import UUID

from tuesday import __version__
from tuesday.composition import (
    create_default_orchestrator,
    create_model_backed_orchestrator,
)
from tuesday.config import ConfigurationError, LanguageModelSettings, load_settings
from tuesday.conversations import (
    BaseConversationRepository,
    ConversationRepositoryError,
    InMemoryConversationRepository,
    SQLiteConversationRepository,
)
from tuesday.domain import TuesdayRequest, TuesdayResponse
from tuesday.language_models import (
    LanguageModelProviderError,
    OpenAILanguageModelProvider,
)
from tuesday.preparation import RequestPreparationError
from tuesday.prompting import ConversationalPromptRenderer
from tuesday.routing import NoRouteFoundError
from tuesday.services.conversation import StatefulConversationService

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
    )
    parser.add_argument(
        "request",
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
    return parser


async def _run_openai(
    request: TuesdayRequest,
    repository: BaseConversationRepository,
    settings: LanguageModelSettings,
    max_history_messages: int | None,
) -> TuesdayResponse:
    """Execute and close the owned client on one loop, preserving primary errors."""
    provider = OpenAILanguageModelProvider(settings)
    completed = False
    try:
        renderer = ConversationalPromptRenderer(
            max_history_messages=max_history_messages
        )
        orchestrator = create_model_backed_orchestrator(provider, renderer=renderer)
        response = await StatefulConversationService(orchestrator, repository).handle(
            request
        )
        completed = True
        return response
    finally:
        try:
            await provider.aclose()
        except BaseException:
            # Cleanup must never replace an execution error, including interruption.
            # With no primary error, cleanup failures propagate normally.
            if completed:
                raise


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one terminal interaction and return its exit status.

    Async programmatic callers should use application services directly instead
    of invoking this function inside an already-running event loop.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if args.runtime == "deterministic" and args.max_history_messages is not None:
        parser.error("--max-history-messages requires --runtime openai.")
    try:
        request = (
            TuesdayRequest(args.request)
            if args.conversation_id is None
            else TuesdayRequest(args.request, conversation_id=args.conversation_id)
        )
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
    except LanguageModelProviderError:
        print("tuesday: model operation failed.", file=sys.stderr)
        return 1
    except NoRouteFoundError:
        print(
            "tuesday: use an explicit /chat or /conversation directive.",
            file=sys.stderr,
        )
        return 1
    except RequestPreparationError:
        print("tuesday: include content after the routing directive.", file=sys.stderr)
        return 1
    except ConversationRepositoryError:
        print("tuesday: conversation storage failed.", file=sys.stderr)
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
