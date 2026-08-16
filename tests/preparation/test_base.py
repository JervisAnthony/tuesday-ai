"""Tests for the framework-independent request preparation contracts."""

import ast
import inspect
from dataclasses import FrozenInstanceError

import pytest

import tuesday.preparation.base as preparation_base
from tuesday.domain import TuesdayRequest
from tuesday.preparation import BaseRequestPreparer, PreparedRequest


class StubRequestPreparer(BaseRequestPreparer):
    """Minimal test-only implementation of the preparation contract."""

    def __init__(self) -> None:
        self.received_request: TuesdayRequest | None = None

    def prepare(self, request: TuesdayRequest) -> PreparedRequest:
        self.received_request = request
        return PreparedRequest(source_request=request, content=request.content)


class IncompleteRequestPreparer(BaseRequestPreparer):
    """Test subclass intentionally missing the required prepare method."""


@pytest.fixture
def source_request() -> TuesdayRequest:
    return TuesdayRequest(content="/chat Hello")


@pytest.mark.parametrize("field", ["source_request", "content", "route_directive"])
def test_prepared_request_is_frozen(
    source_request: TuesdayRequest,
    field: str,
) -> None:
    prepared = PreparedRequest(source_request=source_request, content="Hello")

    with pytest.raises(FrozenInstanceError):
        setattr(prepared, field, None)


def test_prepared_request_uses_slots(source_request: TuesdayRequest) -> None:
    prepared = PreparedRequest(source_request=source_request, content="Hello")

    assert not hasattr(prepared, "__dict__")


def test_source_request_preserves_exact_identity(
    source_request: TuesdayRequest,
) -> None:
    prepared = PreparedRequest(source_request=source_request, content="Hello")

    assert prepared.source_request is source_request


def test_non_request_source_is_rejected() -> None:
    with pytest.raises(TypeError, match="source_request must be a TuesdayRequest"):
        PreparedRequest(source_request=object(), content="Hello")  # type: ignore[arg-type]


@pytest.mark.parametrize("content", ["Hello", "  Hello  ", "/chat Hello"])
def test_meaningful_content_is_accepted_and_preserved(content: str) -> None:
    prepared = PreparedRequest(
        source_request=TuesdayRequest(content="source"),
        content=content,
    )

    assert prepared.content == content


@pytest.mark.parametrize("content", ["", "   ", "\t\n"])
def test_empty_or_whitespace_content_is_rejected(content: str) -> None:
    with pytest.raises(ValueError, match="content must not be empty"):
        PreparedRequest(
            source_request=TuesdayRequest(content="source"),
            content=content,
        )


@pytest.mark.parametrize("content", [None, 42, object()])
def test_non_string_content_is_rejected(content: object) -> None:
    with pytest.raises(TypeError, match="content must be a string"):
        PreparedRequest(
            source_request=TuesdayRequest(content="source"),
            content=content,  # type: ignore[arg-type]
        )


def test_route_directive_defaults_to_none(source_request: TuesdayRequest) -> None:
    prepared = PreparedRequest(source_request=source_request, content="Hello")

    assert prepared.route_directive is None


@pytest.mark.parametrize("directive", ["chat", "Chat", "CHAT"])
def test_route_directive_is_accepted_and_preserved(
    source_request: TuesdayRequest,
    directive: str,
) -> None:
    prepared = PreparedRequest(
        source_request=source_request,
        content="Hello",
        route_directive=directive,
    )

    assert prepared.route_directive == directive


@pytest.mark.parametrize("directive", ["", "   ", "\t\n"])
def test_empty_or_whitespace_route_directive_is_rejected(
    source_request: TuesdayRequest,
    directive: str,
) -> None:
    with pytest.raises(ValueError, match="directive must not be empty"):
        PreparedRequest(
            source_request=source_request,
            content="Hello",
            route_directive=directive,
        )


def test_route_directive_with_leading_slash_is_rejected(
    source_request: TuesdayRequest,
) -> None:
    with pytest.raises(ValueError, match="must not begin with '/'"):
        PreparedRequest(
            source_request=source_request,
            content="Hello",
            route_directive="/chat",
        )


@pytest.mark.parametrize("directive", ["chat now", "chat\tnow", "chat\nnow"])
def test_route_directive_containing_whitespace_is_rejected(
    source_request: TuesdayRequest,
    directive: str,
) -> None:
    with pytest.raises(ValueError, match="must not contain whitespace"):
        PreparedRequest(
            source_request=source_request,
            content="Hello",
            route_directive=directive,
        )


@pytest.mark.parametrize("directive", [42, object()])
def test_non_string_route_directive_is_rejected(
    source_request: TuesdayRequest,
    directive: object,
) -> None:
    with pytest.raises(TypeError, match="must be a string or None"):
        PreparedRequest(
            source_request=source_request,
            content="Hello",
            route_directive=directive,  # type: ignore[arg-type]
        )


def test_to_request_materializes_prepared_content_and_correlation_ids(
    source_request: TuesdayRequest,
) -> None:
    prepared = PreparedRequest(
        source_request=source_request,
        content="Hello",
        route_directive="chat",
    )

    derived = prepared.to_request()

    assert isinstance(derived, TuesdayRequest)
    assert derived is not source_request
    assert derived.content == "Hello"
    assert derived.conversation_id == source_request.conversation_id
    assert derived.request_id == source_request.request_id


def test_to_request_does_not_mutate_source_request(
    source_request: TuesdayRequest,
) -> None:
    original = source_request
    prepared = PreparedRequest(source_request=source_request, content="Hello")

    prepared.to_request()

    assert source_request is original
    assert source_request.content == "/chat Hello"


def test_repeated_to_request_calls_produce_equivalent_new_requests(
    source_request: TuesdayRequest,
) -> None:
    prepared = PreparedRequest(source_request=source_request, content="Hello")

    first = prepared.to_request()
    second = prepared.to_request()

    assert first is not second
    assert first == second
    assert first.conversation_id == source_request.conversation_id
    assert first.request_id == source_request.request_id


def test_base_request_preparer_is_abstract_and_cannot_be_instantiated() -> None:
    assert inspect.isabstract(BaseRequestPreparer)
    with pytest.raises(TypeError):
        BaseRequestPreparer()
    with pytest.raises(TypeError):
        IncompleteRequestPreparer()


def test_prepare_receives_exact_request_and_returns_prepared_request(
    source_request: TuesdayRequest,
) -> None:
    preparer = StubRequestPreparer()

    prepared = preparer.prepare(source_request)

    assert preparer.received_request is source_request
    assert isinstance(prepared, PreparedRequest)
    assert prepared.source_request is source_request


def test_prepare_contract_is_synchronous() -> None:
    assert not inspect.iscoroutinefunction(BaseRequestPreparer.prepare)
    assert not inspect.iscoroutinefunction(StubRequestPreparer.prepare)


def test_preparation_module_has_only_allowed_dependencies() -> None:
    tree = ast.parse(inspect.getsource(preparation_base))
    imported_modules = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert imported_modules == {"abc", "dataclasses", "tuesday.domain"}
