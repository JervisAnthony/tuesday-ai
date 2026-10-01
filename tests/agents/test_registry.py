"""Tests for explicit TUESDAY agent registration and lookup."""

import pytest

from tuesday.agents import (
    AgentNotFoundError,
    AgentRegistrationError,
    AgentRegistry,
    BaseAgent,
)
from tuesday.domain import ConversationContext, TuesdayRequest, TuesdayResponse


class StubAgent(BaseAgent):
    """Configurable test-only agent satisfying the base contract."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "A registry test agent."

    async def handle(
        self,
        request: TuesdayRequest,
        context: ConversationContext,
    ) -> TuesdayResponse:
        self._validate_context(request, context)
        return TuesdayResponse(
            content="stub response",
            conversation_id=request.conversation_id,
            request_id=request.request_id,
        )


def test_empty_registry_has_length_zero() -> None:
    assert len(AgentRegistry()) == 0


def test_registering_agent_increases_registry_size() -> None:
    registry = AgentRegistry()

    registry.register(StubAgent("planner"))

    assert len(registry) == 1


def test_registered_agent_is_retrieved_by_exact_identity() -> None:
    registry = AgentRegistry()
    agent = StubAgent("planner")

    registry.register(agent)

    assert registry.get("planner") is agent


def test_multiple_agents_preserve_registration_order() -> None:
    registry = AgentRegistry()
    registry.register(StubAgent("research"))
    registry.register(StubAgent("planner"))

    assert len(registry) == 2
    assert registry.names == ("research", "planner")


def test_duplicate_name_is_rejected_without_replacing_original() -> None:
    registry = AgentRegistry()
    original = StubAgent("planner")
    duplicate = StubAgent("planner")
    registry.register(original)

    with pytest.raises(
        AgentRegistrationError,
        match="An agent named 'planner' is already registered",
    ):
        registry.register(duplicate)

    assert len(registry) == 1
    assert registry.names == ("planner",)
    assert registry.get("planner") is original


def test_missing_agent_raises_error_containing_requested_name() -> None:
    registry = AgentRegistry()

    with pytest.raises(AgentNotFoundError, match="missing"):
        registry.get("missing")


@pytest.mark.parametrize("name", ["", " ", "\t", "\n"])
def test_meaningless_agent_name_is_rejected_without_changing_state(
    name: str,
) -> None:
    registry = AgentRegistry()

    with pytest.raises(
        AgentRegistrationError,
        match="Agent name must be a non-empty string",
    ):
        registry.register(StubAgent(name))

    assert len(registry) == 0
    assert registry.names == ()


def test_meaningful_agent_name_is_preserved_exactly() -> None:
    registry = AgentRegistry()
    agent = StubAgent("  planner  ")

    registry.register(agent)

    assert registry.names == ("  planner  ",)
    assert registry.get("  planner  ") is agent


def test_agent_names_are_case_sensitive() -> None:
    registry = AgentRegistry()
    lowercase_agent = StubAgent("planner")
    titlecase_agent = StubAgent("Planner")

    registry.register(lowercase_agent)
    registry.register(titlecase_agent)

    assert registry.names == ("planner", "Planner")
    assert registry.get("planner") is lowercase_agent
    assert registry.get("Planner") is titlecase_agent


def test_names_are_an_immutable_snapshot_not_internal_storage() -> None:
    registry = AgentRegistry()
    registry.register(StubAgent("planner"))

    names = registry.names
    names += ("external",)

    assert isinstance(registry.names, tuple)
    assert registry.names == ("planner",)
    assert "external" not in registry


def test_membership_reports_registered_and_unknown_names() -> None:
    registry = AgentRegistry()
    registry.register(StubAgent("planner"))

    assert "planner" in registry
    assert "missing" not in registry
