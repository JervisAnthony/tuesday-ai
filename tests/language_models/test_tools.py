"""Tests for provider-neutral language-model tool contracts."""

import ast
import inspect
from collections.abc import Iterator, Mapping
from dataclasses import FrozenInstanceError, fields
from math import inf, nan
from types import MappingProxyType
from uuid import uuid4

import pytest

import tuesday.language_models as language_models
import tuesday.language_models.tools as model_tools
from tuesday.language_models import (
    LanguageModelToolCall,
    LanguageModelToolDefinition,
    LanguageModelToolScalar,
    LanguageModelToolValue,
)


class CustomMapping(Mapping[str, object]):
    """Small non-dict mapping used to verify the abstract input contract."""

    def __init__(self, values: dict[str, object]) -> None:
        self._values = values

    def __getitem__(self, key: str) -> object:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def make_definition(**changes: object) -> LanguageModelToolDefinition:
    values = {
        "name": "calculator.basic",
        "description": "Perform basic arithmetic.",
        "parameters": {},
    }
    values.update(changes)
    return LanguageModelToolDefinition(**values)  # type: ignore[arg-type]


def make_call(**changes: object) -> LanguageModelToolCall:
    values = {
        "call_id": "call_abc123",
        "name": "calculator.basic",
        "arguments": {},
    }
    values.update(changes)
    return LanguageModelToolCall(**values)  # type: ignore[arg-type]


def test_tool_definition_is_frozen_slotted_and_has_exact_fields() -> None:
    definition = make_definition()

    assert not hasattr(definition, "__dict__")
    assert tuple(field.name for field in fields(definition)) == (
        "name",
        "description",
        "parameters",
    )
    with pytest.raises(FrozenInstanceError):
        definition.name = "changed"  # type: ignore[misc]


def test_equal_tool_definitions_are_independent_and_compare_equal() -> None:
    first = make_definition(parameters={"required": ["left"]})
    second = make_definition(parameters={"required": ["left"]})

    assert first == second
    assert first is not second
    assert first.parameters is not second.parameters


@pytest.mark.parametrize("invalid_name", [None, 1, True, object()])
def test_definition_name_must_be_string(invalid_name: object) -> None:
    with pytest.raises(TypeError, match="tool name must be a string"):
        make_definition(name=invalid_name)


@pytest.mark.parametrize("invalid_name", ["", " ", "\t", "\n"])
def test_definition_name_must_be_meaningful(invalid_name: str) -> None:
    with pytest.raises(ValueError, match="tool name must not be empty"):
        make_definition(name=invalid_name)


@pytest.mark.parametrize(
    "invalid_name",
    [" calculator.basic", "calculator.basic ", "\tcalculator.basic"],
)
def test_definition_name_rejects_surrounding_whitespace(
    invalid_name: str,
) -> None:
    with pytest.raises(ValueError, match="must not have surrounding whitespace"):
        make_definition(name=invalid_name)


@pytest.mark.parametrize("name", ["calculator.basic", "Calculator.Basic"])
def test_definition_preserves_exact_valid_name(name: str) -> None:
    assert make_definition(name=name).name == name


@pytest.mark.parametrize("invalid_description", [None, 123, True])
def test_definition_description_must_be_string(
    invalid_description: object,
) -> None:
    with pytest.raises(TypeError, match="description must be a string"):
        make_definition(description=invalid_description)


@pytest.mark.parametrize("invalid_description", ["", " ", "\t"])
def test_definition_description_must_be_meaningful(
    invalid_description: str,
) -> None:
    with pytest.raises(ValueError, match="description must not be empty"):
        make_definition(description=invalid_description)


@pytest.mark.parametrize("invalid_description", [" Leading", "Trailing "])
def test_definition_description_rejects_surrounding_whitespace(
    invalid_description: str,
) -> None:
    with pytest.raises(ValueError, match="must not have surrounding whitespace"):
        make_definition(description=invalid_description)


def test_definition_preserves_exact_description_with_internal_whitespace() -> None:
    description = "Perform basic\n  arithmetic."

    assert make_definition(description=description).description == description


@pytest.mark.parametrize(
    "invalid_parameters",
    [None, [], (), "schema", 123, object()],
)
def test_definition_parameters_must_be_mapping(
    invalid_parameters: object,
) -> None:
    with pytest.raises(TypeError, match="parameters must be a mapping"):
        make_definition(parameters=invalid_parameters)


@pytest.mark.parametrize(
    "parameters",
    [
        {"type": "object"},
        MappingProxyType({"type": "object"}),
        CustomMapping({"type": "object"}),
    ],
)
def test_definition_accepts_mapping_implementations(
    parameters: Mapping[str, object],
) -> None:
    definition = make_definition(parameters=parameters)

    assert definition.parameters == {"type": "object"}
    assert isinstance(definition.parameters, MappingProxyType)


def test_definition_accepts_recursive_json_compatible_values() -> None:
    parameters = {
        "nullable": None,
        "title": "number",
        "enabled": True,
        "count": 2,
        "minimum": 1.5,
        "required": ["left", "right"],
        "enum": ("add", "subtract"),
        "properties": {"left": {"type": "number"}},
    }

    definition = make_definition(parameters=parameters)

    for key in ("nullable", "title", "enabled", "count", "minimum"):
        assert definition.parameters[key] == parameters[key]
        assert type(definition.parameters[key]) is type(parameters[key])
    assert definition.parameters["required"] == ("left", "right")
    assert definition.parameters["enum"] == ("add", "subtract")
    assert isinstance(definition.parameters["properties"], MappingProxyType)


def test_definition_deeply_snapshots_and_freezes_parameters() -> None:
    required = ["left", "right"]
    properties = {"left": {"type": "number"}}
    parameters = {
        "type": "object",
        "properties": properties,
        "required": required,
    }

    definition = make_definition(parameters=parameters)
    properties["left"]["type"] = "string"
    parameters["type"] = "array"
    required.append("later")

    assert definition.parameters["type"] == "object"
    assert definition.parameters["required"] == ("left", "right")
    stored_properties = definition.parameters["properties"]
    assert stored_properties["left"]["type"] == "number"
    with pytest.raises(TypeError):
        definition.parameters["new"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        stored_properties["new"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        stored_properties["left"]["type"] = "changed"  # type: ignore[index]


@pytest.mark.parametrize("invalid_float", [nan, inf, -inf])
@pytest.mark.parametrize("nested", [False, True])
def test_definition_rejects_non_finite_floats(
    invalid_float: float,
    nested: bool,
) -> None:
    parameters = (
        {"value": {"nested": [invalid_float]}}
        if nested
        else {"value": invalid_float}
    )

    with pytest.raises(
        ValueError,
        match="Language model tool floating-point values must be finite",
    ):
        make_definition(parameters=parameters)


@pytest.mark.parametrize(
    "invalid_value",
    [set(), object(), b"", bytearray(), uuid4()],
)
def test_definition_rejects_unsupported_recursive_values(
    invalid_value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="Language model tool values must be JSON-compatible",
    ):
        make_definition(parameters={"nested": [invalid_value]})


@pytest.mark.parametrize("invalid_key", [1, True, None, object()])
def test_definition_rejects_non_string_mapping_keys(
    invalid_key: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="Language model tool object keys must be strings",
    ):
        make_definition(
            parameters={"nested": {invalid_key: "value"}},  # type: ignore[dict-item]
        )


def test_definition_allows_arbitrary_string_metadata_keys() -> None:
    definition = make_definition(parameters={"": 1, " ": 2})

    assert definition.parameters == {"": 1, " ": 2}


def test_tool_call_is_frozen_slotted_and_has_exact_fields() -> None:
    call = make_call()

    assert not hasattr(call, "__dict__")
    assert tuple(field.name for field in fields(call)) == (
        "call_id",
        "name",
        "arguments",
    )
    with pytest.raises(FrozenInstanceError):
        call.call_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("invalid_call_id", [None, 1, True, object()])
def test_call_id_must_be_string(invalid_call_id: object) -> None:
    with pytest.raises(TypeError, match="tool call_id must be a string"):
        make_call(call_id=invalid_call_id)


@pytest.mark.parametrize("invalid_call_id", ["", " ", "\t"])
def test_call_id_must_be_meaningful(invalid_call_id: str) -> None:
    with pytest.raises(ValueError, match="tool call_id must not be empty"):
        make_call(call_id=invalid_call_id)


@pytest.mark.parametrize("invalid_call_id", [" call_1", "call_1 "])
def test_call_id_rejects_surrounding_whitespace(
    invalid_call_id: str,
) -> None:
    with pytest.raises(ValueError, match="must not have surrounding whitespace"):
        make_call(call_id=invalid_call_id)


@pytest.mark.parametrize("call_id", ["call_abc123", "toolu_01XYZ"])
def test_call_id_is_opaque_and_preserved_without_uuid_requirement(
    call_id: str,
) -> None:
    assert make_call(call_id=call_id).call_id == call_id


@pytest.mark.parametrize("invalid_name", [None, 1, True, object()])
def test_call_name_must_be_string(invalid_name: object) -> None:
    with pytest.raises(TypeError, match="tool call name must be a string"):
        make_call(name=invalid_name)


@pytest.mark.parametrize("invalid_name", ["", " ", "\t"])
def test_call_name_must_be_meaningful(invalid_name: str) -> None:
    with pytest.raises(ValueError, match="tool call name must not be empty"):
        make_call(name=invalid_name)


@pytest.mark.parametrize("invalid_name", [" calculator.basic", "name "])
def test_call_name_rejects_surrounding_whitespace(invalid_name: str) -> None:
    with pytest.raises(ValueError, match="must not have surrounding whitespace"):
        make_call(name=invalid_name)


def test_call_names_are_exact_case_sensitive_values() -> None:
    lower = make_call(name="calculator.basic")
    upper = make_call(name="Calculator.Basic")

    assert lower.name == "calculator.basic"
    assert upper.name == "Calculator.Basic"
    assert lower.name != upper.name


@pytest.mark.parametrize(
    "invalid_arguments",
    [None, [], (), "arguments", 123, object()],
)
def test_call_arguments_must_be_mapping(invalid_arguments: object) -> None:
    with pytest.raises(TypeError, match="call arguments must be a mapping"):
        make_call(arguments=invalid_arguments)


def test_call_accepts_empty_arguments_and_unknown_tool() -> None:
    call = make_call(name="does.not.exist", arguments={})

    assert call.name == "does.not.exist"
    assert call.arguments == {}
    assert isinstance(call.arguments, MappingProxyType)


def test_call_preserves_semantically_invalid_arguments() -> None:
    arguments = {
        "operation": "teleport",
        "left": "banana",
        "unexpected": True,
    }

    assert make_call(arguments=arguments).arguments == arguments


def test_call_recursively_snapshots_and_freezes_mapping_arguments() -> None:
    labels = ["work"]
    options = {"labels": labels}
    arguments = CustomMapping({"options": options, "items": (1, {"ok": True})})

    call = make_call(arguments=arguments)
    labels.append("later")
    options["new"] = False

    stored_options = call.arguments["options"]
    assert stored_options["labels"] == ("work",)
    assert "new" not in stored_options
    assert isinstance(call.arguments["items"], tuple)
    assert isinstance(call.arguments["items"][1], MappingProxyType)
    with pytest.raises(TypeError):
        call.arguments["new"] = None  # type: ignore[index]
    with pytest.raises(TypeError):
        stored_options["new"] = None  # type: ignore[index]


def test_call_uses_recursive_value_validation() -> None:
    with pytest.raises(ValueError, match="floating-point values must be finite"):
        make_call(arguments={"nested": {"value": inf}})
    with pytest.raises(TypeError, match="values must be JSON-compatible"):
        make_call(arguments={"nested": object()})
    with pytest.raises(TypeError, match="object keys must be strings"):
        make_call(arguments={"nested": {1: "value"}})  # type: ignore[dict-item]


def test_public_language_models_package_exports_tool_contracts() -> None:
    assert language_models.LanguageModelToolCall is LanguageModelToolCall
    assert language_models.LanguageModelToolDefinition is LanguageModelToolDefinition
    assert language_models.LanguageModelToolScalar is LanguageModelToolScalar
    assert language_models.LanguageModelToolValue is LanguageModelToolValue
    assert language_models.__all__ == [
        "BaseLanguageModelProvider",
        "LanguageModelMessage",
        "LanguageModelProviderError",
        "LanguageModelRequest",
        "LanguageModelResponse",
        "LanguageModelToolCall",
        "LanguageModelToolDefinition",
        "LanguageModelToolScalar",
        "LanguageModelToolValue",
        "OpenAILanguageModelProvider",
    ]


def test_tool_contract_module_has_only_standard_library_dependencies() -> None:
    tree = ast.parse(inspect.getsource(model_tools))
    imported_modules = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert imported_modules == {
        "__future__",
        "collections.abc",
        "dataclasses",
        "math",
        "types",
        "typing",
    }
    assert all(not module.startswith("tuesday") for module in imported_modules)
