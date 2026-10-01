"""Provider-neutral language-model tool representation contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import TypeAlias

LanguageModelToolScalar: TypeAlias = str | int | float | bool | None
LanguageModelToolValue: TypeAlias = (
    LanguageModelToolScalar
    | tuple["LanguageModelToolValue", ...]
    | Mapping[str, "LanguageModelToolValue"]
)

__all__ = [
    "LanguageModelToolCall",
    "LanguageModelToolDefinition",
    "LanguageModelToolScalar",
    "LanguageModelToolValue",
]


def _require_meaningful_text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    if not value.strip():
        raise ValueError(f"{label} must not be empty.")
    if value != value.strip():
        raise ValueError(f"{label} must not have surrounding whitespace.")
    return value


def _freeze_value(value: object) -> LanguageModelToolValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(
                "Language model tool floating-point values must be finite."
            )
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, Mapping):
        frozen: dict[str, LanguageModelToolValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "Language model tool object keys must be strings."
                )
            frozen[key] = _freeze_value(item)
        return MappingProxyType(frozen)
    raise TypeError("Language model tool values must be JSON-compatible.")


@dataclass(frozen=True, slots=True)
class LanguageModelToolDefinition:
    """An immutable model-visible description of a tool capability."""

    name: str
    description: str
    parameters: Mapping[str, LanguageModelToolValue]

    def __post_init__(self) -> None:
        _require_meaningful_text(self.name, "Language model tool name")
        _require_meaningful_text(
            self.description,
            "Language model tool description",
        )
        if not isinstance(self.parameters, Mapping):
            raise TypeError("Language model tool parameters must be a mapping.")
        object.__setattr__(self, "parameters", _freeze_value(self.parameters))


@dataclass(frozen=True, slots=True)
class LanguageModelToolCall:
    """An immutable structured tool-use request emitted by a model."""

    call_id: str
    name: str
    arguments: Mapping[str, LanguageModelToolValue]

    def __post_init__(self) -> None:
        _require_meaningful_text(
            self.call_id,
            "Language model tool call_id",
        )
        _require_meaningful_text(self.name, "Language model tool call name")
        if not isinstance(self.arguments, Mapping):
            raise TypeError(
                "Language model tool call arguments must be a mapping."
            )
        object.__setattr__(self, "arguments", _freeze_value(self.arguments))
