"""Explicit registration and lookup for TUESDAY tools."""

from tuesday.tools.base import BaseTool

__all__ = [
    "DuplicateToolError",
    "ToolNotFoundError",
    "ToolRegistry",
    "ToolRegistryError",
]


class ToolRegistryError(RuntimeError):
    """Base error for tool registry operations."""


class DuplicateToolError(ToolRegistryError):
    """Raised when an exact tool name is already registered."""


class ToolNotFoundError(ToolRegistryError):
    """Raised when a requested tool is not registered."""


class ToolRegistry:
    """A deterministic collection of explicitly registered tool instances."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a concrete tool instance under its exact stable name."""
        if not isinstance(tool, BaseTool):
            raise TypeError("tool must be a BaseTool instance.")

        name = tool.name
        if not isinstance(name, str):
            raise TypeError("Tool name must be a string.")
        if not name.strip():
            raise ValueError("Tool name must not be empty.")
        if name != name.strip():
            raise ValueError("Tool name must not have surrounding whitespace.")
        if name in self._tools:
            raise DuplicateToolError(
                f"A tool named {name!r} is already registered."
            )

        self._tools[name] = tool

    def get(self, name: str) -> BaseTool:
        """Return the registered tool with the requested exact name."""
        if not isinstance(name, str):
            raise TypeError("Tool lookup name must be a string.")

        try:
            return self._tools[name]
        except KeyError as error:
            raise ToolNotFoundError(
                f"No tool named {name!r} is registered."
            ) from error

    @property
    def names(self) -> tuple[str, ...]:
        """Return an immutable snapshot of names in registration order."""
        return tuple(self._tools)
