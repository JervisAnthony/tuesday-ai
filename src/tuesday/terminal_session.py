"""Sequential terminal input and local conversation controls for the CLI."""

import asyncio
import sys
from uuid import UUID, uuid4

# Initialize the existing service graph before routing for direct module imports.
import tuesday.services.conversation as conversation_service
from tuesday.conversations import ConversationRepositoryError
from tuesday.domain import TuesdayRequest
from tuesday.language_models import LanguageModelProviderError
from tuesday.preparation import RequestPreparationError
from tuesday.routing import NoRouteFoundError

TURN_ERRORS = (
    NoRouteFoundError,
    RequestPreparationError,
    LanguageModelProviderError,
    ConversationRepositoryError,
)
_HELP = (
    "Session commands: :help, :id, :new, :use UUID, :exit.\n"
    "Requests require an explicit /chat or /conversation directive and content."
)


def report_turn_error(
    error: NoRouteFoundError
    | RequestPreparationError
    | LanguageModelProviderError
    | ConversationRepositoryError,
) -> None:
    """Report expected failures without disclosing request or exception details."""
    if isinstance(error, NoRouteFoundError):
        message = "use an explicit /chat or /conversation directive."
    elif isinstance(error, RequestPreparationError):
        message = "include content after the routing directive."
    elif isinstance(error, LanguageModelProviderError):
        message = "model operation failed."
    else:
        message = "conversation storage failed."
    print("tuesday: " + message, file=sys.stderr)


def _select(conversation_id: UUID) -> None:
    print(f"Conversation: {conversation_id}", file=sys.stderr)


def _command(line: str, conversation_id: UUID) -> tuple[UUID, bool]:
    parts = line.split()
    if parts == [":exit"]:
        return conversation_id, True
    if parts == [":help"]:
        print(_HELP, file=sys.stderr)
    elif parts == [":id"]:
        _select(conversation_id)
    elif parts == [":new"]:
        conversation_id = uuid4()
        _select(conversation_id)
    elif len(parts) == 2 and parts[0] == ":use":
        try:
            selected = UUID(parts[1])
        except ValueError:
            print("tuesday: :use requires a valid UUID.", file=sys.stderr)
        else:
            conversation_id = selected
            _select(conversation_id)
    else:
        print("tuesday: invalid session command; use :help.", file=sys.stderr)
    return conversation_id, False


def run_session(
    service: conversation_service.StatefulConversationService,
    conversation_id: UUID,
    runner: asyncio.Runner,
) -> None:
    """Read interruptible physical lines and run one turn at a time on one loop."""
    _select(conversation_id)
    while True:
        if sys.stdin.isatty():
            print("tuesday> ", end="", file=sys.stderr, flush=True)
        line = sys.stdin.readline()
        if line == "":
            return
        line = line.removesuffix("\n").removesuffix("\r")
        if not line.strip():
            continue
        if line.startswith(":"):
            conversation_id, exiting = _command(line, conversation_id)
            if exiting:
                return
            continue
        request = TuesdayRequest(line, conversation_id=conversation_id)
        try:
            response = runner.run(service.handle(request))
        except TURN_ERRORS as error:
            report_turn_error(error)
        else:
            print(response.content)
