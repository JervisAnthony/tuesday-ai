"""Framework-independent tool execution contracts for TUESDAY."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import TypeAlias
from uuid import UUID, uuid4

ToolScalar: TypeAlias = str | int | float | bool | None
ToolValue: TypeAlias = (
    ToolScalar | tuple["ToolValue", ...] | Mapping[str, "ToolValue"]
)

__all__ = [
    "BaseTool",
    "ToolExecutionError",
    "ToolInvocation",
    "ToolResult",
    "ToolScalar",
    "ToolValue",
]


def _require_name(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    if not value.strip():
        raise ValueError(f"{label} must not be empty.")
    if value != value.strip():
        raise ValueError(f"{label} must not have surrounding whitespace.")
    return value


def _freeze_value(value: object) -> ToolValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("Tool floating-point values must be finite.")
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, Mapping):
        frozen: dict[str, ToolValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Tool object keys must be strings.")
            frozen[key] = _freeze_value(item)
        return MappingProxyType(frozen)
    raise TypeError("Tool values must be JSON-compatible.")


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    """An immutable request to execute one named tool."""

    tool_name: str
    arguments: Mapping[str, ToolValue] = field(default_factory=dict)
    invocation_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        _require_name(self.tool_name, "Tool invocation name")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("Tool invocation arguments must be a mapping.")
        if not isinstance(self.invocation_id, UUID):
            raise TypeError("Tool invocation_id must be a UUID.")

        frozen_arguments: dict[str, ToolValue] = {}
        for key, value in self.arguments.items():
            _require_name(key, "Tool argument name")
            frozen_arguments[key] = _freeze_value(value)
        object.__setattr__(
            self,
            "arguments",
            MappingProxyType(frozen_arguments),
        )


@dataclass(frozen=True, slots=True)
class ToolResult:
    """An immutable successful result from one tool invocation."""

    tool_name: str
    invocation_id: UUID
    output: ToolValue

    def __post_init__(self) -> None:
        _require_name(self.tool_name, "Tool result name")
        if not isinstance(self.invocation_id, UUID):
            raise TypeError("Tool result invocation_id must be a UUID.")
        object.__setattr__(self, "output", _freeze_value(self.output))


class ToolExecutionError(RuntimeError):
    """Raised when a tool cannot complete its requested operation."""


class BaseTool(ABC):
    """Abstract asynchronous execution boundary for one TUESDAY tool."""

    __slots__ = ()

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the tool's stable machine-friendly name."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Return the tool's human-readable purpose."""

    @abstractmethod
    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        """Execute one invocation and return its successful result."""

    def _validate_invocation(self, invocation: ToolInvocation) -> None:
        if not isinstance(invocation, ToolInvocation):
            raise TypeError("invocation must be a ToolInvocation.")
        if invocation.tool_name != self.name:
            raise ValueError("Tool invocation must target this tool.")
