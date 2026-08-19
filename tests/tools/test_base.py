"""Tests for framework-independent TUESDAY tool contracts."""

import asyncio
import inspect
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from math import inf, nan
from types import MappingProxyType
from uuid import UUID, uuid4

import pytest

import tuesday.tools as tools
import tuesday.tools.base as tool_base_module
from tuesday.tools import (
    BaseTool,
    ToolExecutionError,
    ToolInvocation,
    ToolResult,
)


class EchoTool(BaseTool):
    """Small deterministic tool used to exercise the base contract."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Return the supplied arguments."

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        self._validate_invocation(invocation)
        return ToolResult(
            tool_name=self.name,
            invocation_id=invocation.invocation_id,
            output=invocation.arguments,
        )


def test_tool_execution_error_is_runtime_error() -> None:
    assert issubclass(ToolExecutionError, RuntimeError)


def test_base_tool_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseTool()


def test_base_tool_declares_exact_abstract_members() -> None:
    assert BaseTool.__abstractmethods__ == {
        "description",
        "execute",
        "name",
    }


def test_base_tool_execution_contract_is_asynchronous() -> None:
    assert inspect.iscoroutinefunction(BaseTool.execute)
    assert tuple(inspect.signature(BaseTool.execute).parameters) == (
        "self",
        "invocation",
    )


def test_base_tool_has_no_instance_dictionary_contract() -> None:
    assert BaseTool.__slots__ == ()


def test_concrete_tool_exposes_stable_metadata() -> None:
    tool = EchoTool()

    assert tool.name == "echo"
    assert tool.description == "Return the supplied arguments."


def test_tool_invocation_defaults_to_uuid_and_empty_read_only_arguments() -> None:
    invocation = ToolInvocation(tool_name="echo")

    assert isinstance(invocation.invocation_id, UUID)
    assert invocation.tool_name == "echo"
    assert invocation.arguments == {}
    assert isinstance(invocation.arguments, MappingProxyType)
    with pytest.raises(TypeError):
        invocation.arguments["new"] = "value"  # type: ignore[index]


def test_tool_invocation_preserves_supplied_uuid_identity() -> None:
    invocation_id = uuid4()

    invocation = ToolInvocation(
        tool_name="echo",
        invocation_id=invocation_id,
    )

    assert invocation.invocation_id is invocation_id


@pytest.mark.parametrize("invalid_name", [None, 1, object()])
def test_tool_invocation_name_must_be_string(invalid_name: object) -> None:
    with pytest.raises(TypeError, match="Tool invocation name must be a string"):
        ToolInvocation(tool_name=invalid_name)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_name", ["", "   ", "\t\n"])
def test_tool_invocation_name_must_be_meaningful(invalid_name: str) -> None:
    with pytest.raises(ValueError, match="Tool invocation name must not be empty"):
        ToolInvocation(tool_name=invalid_name)


@pytest.mark.parametrize("invalid_name", [" echo", "echo ", " echo "])
def test_tool_invocation_name_rejects_surrounding_whitespace(
    invalid_name: str,
) -> None:
    with pytest.raises(ValueError, match="must not have surrounding whitespace"):
        ToolInvocation(tool_name=invalid_name)


@pytest.mark.parametrize("invalid_arguments", [None, [], (), "arguments", 3])
def test_tool_invocation_arguments_must_be_mapping(
    invalid_arguments: object,
) -> None:
    with pytest.raises(TypeError, match="arguments must be a mapping"):
        ToolInvocation(
            tool_name="echo",
            arguments=invalid_arguments,  # type: ignore[arg-type]
        )


def test_tool_invocation_rejects_non_uuid_identifier() -> None:
    with pytest.raises(TypeError, match="invocation_id must be a UUID"):
        ToolInvocation(
            tool_name="echo",
            invocation_id="not-a-uuid",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("invalid_key", [None, 1, object()])
def test_tool_argument_names_must_be_strings(invalid_key: object) -> None:
    with pytest.raises(TypeError, match="Tool argument name must be a string"):
        ToolInvocation(
            tool_name="echo",
            arguments={invalid_key: "value"},  # type: ignore[dict-item]
        )


@pytest.mark.parametrize("invalid_key", ["", "  ", "\t"])
def test_tool_argument_names_must_be_meaningful(invalid_key: str) -> None:
    with pytest.raises(ValueError, match="Tool argument name must not be empty"):
        ToolInvocation(tool_name="echo", arguments={invalid_key: "value"})


@pytest.mark.parametrize("invalid_key", [" query", "query ", " query "])
def test_tool_argument_names_reject_surrounding_whitespace(
    invalid_key: str,
) -> None:
    with pytest.raises(ValueError, match="must not have surrounding whitespace"):
        ToolInvocation(tool_name="echo", arguments={invalid_key: "value"})


@pytest.mark.parametrize(
    "value",
    [None, "text", "", True, False, 0, 42, -8, 1.5, -0.25],
)
def test_tool_invocation_accepts_json_scalar_values(value: object) -> None:
    invocation = ToolInvocation(tool_name="echo", arguments={"value": value})

    assert invocation.arguments["value"] == value
    assert type(invocation.arguments["value"]) is type(value)


@pytest.mark.parametrize("invalid_float", [inf, -inf, nan])
def test_tool_invocation_rejects_non_finite_float_values(
    invalid_float: float,
) -> None:
    with pytest.raises(ValueError, match="floating-point values must be finite"):
        ToolInvocation(
            tool_name="echo",
            arguments={"value": invalid_float},
        )


@pytest.mark.parametrize(
    "unsupported_value",
    [object(), b"bytes", {1, 2, 3}, uuid4()],
)
def test_tool_invocation_rejects_non_json_values(
    unsupported_value: object,
) -> None:
    with pytest.raises(TypeError, match="Tool values must be JSON-compatible"):
        ToolInvocation(
            tool_name="echo",
            arguments={"value": unsupported_value},
        )


def test_tool_invocation_rejects_non_string_nested_object_keys() -> None:
    with pytest.raises(TypeError, match="Tool object keys must be strings"):
        ToolInvocation(
            tool_name="echo",
            arguments={
                "payload": {1: "value"},  # type: ignore[dict-item]
            },
        )


def test_tool_invocation_deeply_freezes_lists_and_mappings() -> None:
    source = {
        "query": "calendar",
        "options": {
            "include_declined": False,
            "labels": ["work", "important"],
        },
        "limits": [1, 2, 3],
    }

    invocation = ToolInvocation(tool_name="search", arguments=source)

    options = invocation.arguments["options"]
    assert isinstance(options, Mapping)
    assert isinstance(options, MappingProxyType)
    assert options["labels"] == ("work", "important")
    assert invocation.arguments["limits"] == (1, 2, 3)

    source["query"] = "changed"
    source["options"]["labels"].append("later")  # type: ignore[index,union-attr]
    source["limits"].append(4)  # type: ignore[union-attr]

    assert invocation.arguments["query"] == "calendar"
    assert options["labels"] == ("work", "important")
    assert invocation.arguments["limits"] == (1, 2, 3)

    with pytest.raises(TypeError):
        options["new"] = "blocked"  # type: ignore[index]


def test_tool_invocation_normalizes_tuple_values_recursively() -> None:
    invocation = ToolInvocation(
        tool_name="echo",
        arguments={
            "items": (
                "one",
                {"nested": ["two", "three"]},
            )
        },
    )

    items = invocation.arguments["items"]
    assert isinstance(items, tuple)
    assert items[0] == "one"
    assert isinstance(items[1], MappingProxyType)
    assert items[1]["nested"] == ("two", "three")


def test_tool_invocation_is_frozen_and_slotted() -> None:
    invocation = ToolInvocation(tool_name="echo")

    assert not hasattr(invocation, "__dict__")
    with pytest.raises(FrozenInstanceError):
        invocation.tool_name = "changed"  # type: ignore[misc]


def test_tool_result_preserves_name_and_invocation_correlation() -> None:
    invocation_id = uuid4()
    result = ToolResult(
        tool_name="echo",
        invocation_id=invocation_id,
        output="done",
    )

    assert result.tool_name == "echo"
    assert result.invocation_id is invocation_id
    assert result.output == "done"


@pytest.mark.parametrize("invalid_name", [None, 7, object()])
def test_tool_result_name_must_be_string(invalid_name: object) -> None:
    with pytest.raises(TypeError, match="Tool result name must be a string"):
        ToolResult(
            tool_name=invalid_name,  # type: ignore[arg-type]
            invocation_id=uuid4(),
            output=None,
        )


@pytest.mark.parametrize("invalid_name", ["", " ", "\n"])
def test_tool_result_name_must_be_meaningful(invalid_name: str) -> None:
    with pytest.raises(ValueError, match="Tool result name must not be empty"):
        ToolResult(
            tool_name=invalid_name,
            invocation_id=uuid4(),
            output=None,
        )


def test_tool_result_name_rejects_surrounding_whitespace() -> None:
    with pytest.raises(ValueError, match="must not have surrounding whitespace"):
        ToolResult(
            tool_name=" echo ",
            invocation_id=uuid4(),
            output=None,
        )


def test_tool_result_rejects_non_uuid_identifier() -> None:
    with pytest.raises(TypeError, match="invocation_id must be a UUID"):
        ToolResult(
            tool_name="echo",
            invocation_id="not-a-uuid",  # type: ignore[arg-type]
            output=None,
        )


def test_tool_result_deeply_freezes_structured_output() -> None:
    source = {
        "items": [
            {"id": 1, "name": "first"},
            {"id": 2, "name": "second"},
        ]
    }

    result = ToolResult(
        tool_name="search",
        invocation_id=uuid4(),
        output=source,
    )

    assert isinstance(result.output, MappingProxyType)
    items = result.output["items"]
    assert isinstance(items, tuple)
    assert isinstance(items[0], MappingProxyType)
    assert items[0]["name"] == "first"

    source["items"].append({"id": 3, "name": "later"})
    assert len(items) == 2


def test_tool_result_rejects_invalid_structured_output() -> None:
    with pytest.raises(TypeError, match="Tool values must be JSON-compatible"):
        ToolResult(
            tool_name="echo",
            invocation_id=uuid4(),
            output=object(),
        )


def test_tool_result_rejects_non_finite_nested_output() -> None:
    with pytest.raises(ValueError, match="floating-point values must be finite"):
        ToolResult(
            tool_name="echo",
            invocation_id=uuid4(),
            output={"nested": [1.0, inf]},
        )


def test_tool_result_rejects_non_string_nested_output_key() -> None:
    with pytest.raises(TypeError, match="Tool object keys must be strings"):
        ToolResult(
            tool_name="echo",
            invocation_id=uuid4(),
            output={1: "value"},  # type: ignore[dict-item]
        )


def test_tool_result_is_frozen_and_slotted() -> None:
    result = ToolResult(
        tool_name="echo",
        invocation_id=uuid4(),
        output=None,
    )

    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.output = "changed"  # type: ignore[misc]


def test_base_tool_validates_matching_invocation() -> None:
    tool = EchoTool()
    invocation = ToolInvocation(tool_name="echo", arguments={"value": "hello"})

    tool._validate_invocation(invocation)


def test_base_tool_rejects_non_invocation() -> None:
    tool = EchoTool()

    with pytest.raises(TypeError, match="invocation must be a ToolInvocation"):
        tool._validate_invocation(object())  # type: ignore[arg-type]


def test_base_tool_rejects_invocation_for_different_tool() -> None:
    tool = EchoTool()
    invocation = ToolInvocation(tool_name="calendar")

    with pytest.raises(ValueError, match="must target this tool"):
        tool._validate_invocation(invocation)


def test_concrete_tool_executes_and_preserves_correlation() -> None:
    tool = EchoTool()
    invocation = ToolInvocation(
        tool_name="echo",
        arguments={"message": "Hello", "count": 2},
    )

    result = asyncio.run(tool.execute(invocation))

    assert isinstance(result, ToolResult)
    assert result.tool_name == tool.name
    assert result.invocation_id == invocation.invocation_id
    assert result.output == invocation.arguments
    assert result.output is not invocation.arguments


def test_tool_contract_module_has_no_runtime_system_dependencies() -> None:
    source = inspect.getsource(tool_base_module)

    for forbidden in (
        "openai",
        "tuesday.agents",
        "tuesday.composition",
        "tuesday.config",
        "tuesday.conversations",
        "tuesday.language_models",
        "tuesday.orchestration",
        "tuesday.preparation",
        "tuesday.routing",
        "tuesday.services",
    ):
        assert forbidden not in source


def test_public_tools_package_exports_contracts() -> None:
    assert tools.BaseTool is BaseTool
    assert tools.ToolExecutionError is ToolExecutionError
    assert tools.ToolInvocation is ToolInvocation
    assert tools.ToolResult is ToolResult
    assert tools.__all__ == [
        "BaseTool",
        "BaseToolAuthorizationPolicy",
        "DeterministicToolExecutor",
        "DuplicateToolError",
        "InvalidToolResultError",
        "ToolAuthorizationDecision",
        "ToolAuthorizationOutcome",
        "ToolExecutionError",
        "ToolInvocation",
        "ToolNotFoundError",
        "ToolRegistry",
        "ToolRegistryError",
        "ToolResult",
        "ToolScalar",
        "ToolValue",
    ]
