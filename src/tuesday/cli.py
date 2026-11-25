"""One-shot deterministic terminal entry point for TUESDAY."""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from uuid import UUID

from tuesday import __version__
from tuesday.composition import create_default_orchestrator
from tuesday.conversations import (
    ConversationRepositoryError,
    InMemoryConversationRepository,
    SQLiteConversationRepository,
)
from tuesday.domain import TuesdayRequest
from tuesday.preparation import RequestPreparationError
from tuesday.routing import NoRouteFoundError
from tuesday.services.conversation import StatefulConversationService

__all__ = ["main"]


def _nonempty_request(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("request must not be empty.")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tuesday",
        description="Submit one deterministic request using /chat or /conversation.",
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one terminal interaction and return its exit status.

    Async programmatic callers should use application services directly instead
    of invoking this function inside an already-running event loop.
    """
    args = _parser().parse_args(argv)
    try:
        request = (
            TuesdayRequest(args.request)
            if args.conversation_id is None
            else TuesdayRequest(args.request, conversation_id=args.conversation_id)
        )
        if args.database is None:
            repository = InMemoryConversationRepository()
        else:
            try:
                repository = SQLiteConversationRepository(args.database)
            except (TypeError, ValueError):
                print("tuesday: invalid database path.", file=sys.stderr)
                return 1
        service = StatefulConversationService(create_default_orchestrator(), repository)
        response = asyncio.run(service.handle(request))
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
