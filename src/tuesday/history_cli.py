"""Read-only history command parsing, query invocation, and page formatting."""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from uuid import UUID

from tuesday.conversation_export import (
    collect_transcript,
    publish_transcript,
    render_transcript,
)
from tuesday.conversations import (
    ConversationMessagePage,
    ConversationRepositoryError,
    ConversationSummaryPage,
    SQLiteConversationHistoryReader,
)
from tuesday.conversations.history import _validate_pagination

__all__ = ["main"]


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.exit(2, "tuesday: invalid history arguments; use --help.\n")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="tuesday history",
        description="Inspect pages or export an existing SQLite conversation.",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command, description in (
        ("list", "List conversations in first-append order."),
        ("show", "Show one conversation's messages in append order."),
        ("export", "Export a complete conversation as JSON or Markdown."),
    ):
        child = commands.add_parser(
            command, help=description, description=description, allow_abbrev=False
        )
        if command != "list":
            child.add_argument("conversation_id", type=UUID, metavar="UUID")
        child.add_argument(
            "--database", required=True, metavar="PATH", help="existing SQLite file"
        )
        if command == "export":
            child.add_argument(
                "--format",
                choices=("json", "markdown"),
                default="json",
                help="export representation; default json",
            )
            child.add_argument(
                "--output", metavar="PATH", help="explicit destination file"
            )
            child.add_argument(
                "--overwrite", action="store_true", help="replace --output if it exists"
            )
            continue
        child.add_argument(
            "--limit", type=int, default=50, metavar="N", help="1–200; default 50"
        )
        child.add_argument(
            "--offset",
            type=int,
            default=0,
            metavar="N",
            help="0–2**63-1; default 0",
        )
        child.add_argument("--json", action="store_true", help="output one JSON object")
    return parser


def _format_page(
    page: ConversationSummaryPage | ConversationMessagePage, *, as_json: bool
) -> str:
    metadata = {
        "limit": page.limit,
        "offset": page.offset,
        "next_offset": page.next_offset,
    }
    if isinstance(page, ConversationSummaryPage):
        items = [
            {
                "conversation_id": str(item.conversation_id),
                "message_count": item.message_count,
                "first_message_at": item.first_message_at.isoformat(),
                "last_message_at": item.last_message_at.isoformat(),
            }
            for item in page.items
        ]
        payload = {"items": items, **metadata}
        sections = [
            f"Conversation: {item['conversation_id']}\n"
            f"Messages: {item['message_count']}\n"
            f"First appended: {item['first_message_at']}\n"
            f"Last appended: {item['last_message_at']}"
            for item in items
        ] or ["No conversations on this page."]
    else:
        messages = [
            {
                "message_id": str(message.message_id),
                "role": message.role.value,
                "content": message.content,
                "created_at": message.created_at.isoformat(),
            }
            for message in page.messages
        ]
        payload = {
            "conversation_id": str(page.conversation_id),
            "messages": messages,
            **metadata,
        }
        sections = [f"Conversation: {page.conversation_id}"]
        sections.extend(
            [
                f"Role: {message['role']}\n"
                f"Created at: {message['created_at']}\n"
                f"Message: {message['message_id']}\n"
                f"Content:\n{message['content']}"
                for message in messages
            ]
            or ["No messages on this page."]
        )
    if as_json:
        return json.dumps(payload)
    sections.append(
        f"Limit: {page.limit}\nOffset: {page.offset}\nNext offset: {page.next_offset}"
    )
    return "\n\n".join(sections)


def _export(reader: SQLiteConversationHistoryReader, args: argparse.Namespace) -> int:
    transcript = asyncio.run(collect_transcript(reader, args.conversation_id))
    document = render_transcript(transcript, args.format)
    try:
        if args.output is None:
            sys.stdout.write(document)
            sys.stdout.flush()
        else:
            publish_transcript(
                document,
                database=args.database,
                output=args.output,
                overwrite=args.overwrite,
            )
    except (OSError, UnicodeError):
        print("tuesday: transcript output failed.", file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect a page or export a fully prepared conversation document."""
    try:
        parser = _parser()
        args = parser.parse_args(argv)
        if args.command == "export":
            if args.overwrite and args.output is None:
                parser.error("--overwrite requires --output.")
        else:
            try:
                _validate_pagination(args.limit, args.offset)
            except ValueError:
                parser.error("Invalid pagination.")
        try:
            reader = SQLiteConversationHistoryReader(args.database)
        except (TypeError, ValueError):
            print("tuesday: invalid database path.", file=sys.stderr)
            return 1
        if args.command == "export":
            return _export(reader, args)
        if args.command == "list":
            page = asyncio.run(
                reader.list_conversations(limit=args.limit, offset=args.offset)
            )
        else:
            page = asyncio.run(
                reader.get_messages(
                    args.conversation_id, limit=args.limit, offset=args.offset
                )
            )
        result = _format_page(page, as_json=args.json)
    except ConversationRepositoryError:
        print("tuesday: conversation storage failed.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("tuesday: interrupted.", file=sys.stderr)
        return 130
    print(result)
    return 0
