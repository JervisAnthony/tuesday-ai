"""Tests for directive-aware request preparation."""

import ast
import inspect

import pytest

import tuesday.preparation.directive as directive_module
from tuesday.domain import TuesdayRequest
from tuesday.preparation import (
    BaseRequestPreparer,
    DirectiveRequestPreparer,
    PreparedRequest,
    RequestPreparationError,
)


def test_directive_preparer_implements_synchronous_contract() -> None:
    assert issubclass(DirectiveRequestPreparer, BaseRequestPreparer)
    assert not inspect.iscoroutinefunction(DirectiveRequestPreparer.prepare)


def test_chat_directive_produces_prepared_view() -> None:
    source = TuesdayRequest(content="/chat Hello")

    prepared = DirectiveRequestPreparer().prepare(source)

    assert isinstance(prepared, PreparedRequest)
    assert prepared.source_request is source
    assert prepared.content == "Hello"
    assert prepared.route_directive == "chat"


def test_conversation_directive_is_extracted() -> None:
    prepared = DirectiveRequestPreparer().prepare(
        TuesdayRequest(content="/conversation Hello")
    )

    assert prepared.route_directive == "conversation"


def test_directive_case_is_preserved() -> None:
    prepared = DirectiveRequestPreparer().prepare(
        TuesdayRequest(content="/Chat Hello")
    )

    assert prepared.route_directive == "Chat"


def test_preparer_does_not_validate_route_availability() -> None:
    prepared = DirectiveRequestPreparer().prepare(
        TuesdayRequest(content="/unconfigured Hello")
    )

    assert prepared.route_directive == "unconfigured"
    assert prepared.content == "Hello"


@pytest.mark.parametrize(
    "source_content",
    ["/chat   Hello", "/chat\tHello", "/chat\nHello"],
)
def test_separator_whitespace_is_consumed(source_content: str) -> None:
    prepared = DirectiveRequestPreparer().prepare(
        TuesdayRequest(content=source_content)
    )

    assert prepared.content == "Hello"


def test_internal_and_trailing_content_whitespace_is_preserved() -> None:
    prepared = DirectiveRequestPreparer().prepare(
        TuesdayRequest(content="/chat   Hello  world  ")
    )

    assert prepared.content == "Hello  world  "


def test_source_request_remains_the_exact_unchanged_object() -> None:
    source = TuesdayRequest(content="/chat Hello")
    original = source

    prepared = DirectiveRequestPreparer().prepare(source)

    assert prepared.source_request is original
    assert source.content == "/chat Hello"


def test_derived_request_is_distinct_and_preserves_correlation() -> None:
    source = TuesdayRequest(content="/chat Hello")

    derived = DirectiveRequestPreparer().prepare(source).to_request()

    assert derived is not source
    assert derived.content == "Hello"
    assert derived.conversation_id == source.conversation_id
    assert derived.request_id == source.request_id


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("Hello", "must begin with a routing directive"),
        (" /chat Hello", "must begin with a routing directive"),
        ("/", "Routing directive must not be empty"),
        ("/ Hello", "Routing directive must not be empty"),
        ("/chat", "must include content after the routing directive"),
        ("/chat   ", "must include content after the routing directive"),
    ],
)
def test_malformed_request_raises_preparation_error(
    content: str,
    message: str,
) -> None:
    with pytest.raises(RequestPreparationError, match=message):
        DirectiveRequestPreparer().prepare(TuesdayRequest(content=content))


def test_preparer_has_only_preparation_domain_dependencies() -> None:
    tree = ast.parse(inspect.getsource(directive_module))
    imported_modules = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert imported_modules == {
        "tuesday.domain",
        "tuesday.preparation.base",
    }


def test_preparer_stores_no_per_request_state() -> None:
    preparer = DirectiveRequestPreparer()

    preparer.prepare(TuesdayRequest(content="/chat Hello"))

    assert vars(preparer) == {}


def test_sequential_preparations_do_not_leak_state() -> None:
    preparer = DirectiveRequestPreparer()
    first_source = TuesdayRequest(content="/chat First")
    second_source = TuesdayRequest(content="/planner Second")

    first = preparer.prepare(first_source)
    second = preparer.prepare(second_source)

    assert first.source_request is first_source
    assert first.content == "First"
    assert first.route_directive == "chat"
    assert second.source_request is second_source
    assert second.content == "Second"
    assert second.route_directive == "planner"


def test_repeated_equivalent_preparations_are_equivalent() -> None:
    source = TuesdayRequest(content="/chat Hello")
    preparer = DirectiveRequestPreparer()

    first = preparer.prepare(source)
    second = preparer.prepare(source)

    assert first is not second
    assert first == second
