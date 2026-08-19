"""Tool execution contracts for TUESDAY."""

from tuesday.tools.authorization import (
    BaseToolAuthorizationPolicy,
    ToolAuthorizationDecision,
    ToolAuthorizationOutcome,
)
from tuesday.tools.base import (
    BaseTool,
    ToolExecutionError,
    ToolInvocation,
    ToolResult,
    ToolScalar,
    ToolValue,
)
from tuesday.tools.executor import (
    DeterministicToolExecutor,
    InvalidToolResultError,
)
from tuesday.tools.guarded import (
    GuardedToolExecutor,
    InvalidToolAuthorizationDecisionError,
    ToolAuthorizationBlockedError,
    ToolAuthorizationDeniedError,
    ToolConfirmationRequiredError,
)
from tuesday.tools.registry import (
    DuplicateToolError,
    ToolNotFoundError,
    ToolRegistry,
    ToolRegistryError,
)

__all__ = [
    "BaseTool",
    "BaseToolAuthorizationPolicy",
    "DeterministicToolExecutor",
    "DuplicateToolError",
    "GuardedToolExecutor",
    "InvalidToolAuthorizationDecisionError",
    "InvalidToolResultError",
    "ToolAuthorizationDecision",
    "ToolAuthorizationBlockedError",
    "ToolAuthorizationDeniedError",
    "ToolAuthorizationOutcome",
    "ToolConfirmationRequiredError",
    "ToolExecutionError",
    "ToolInvocation",
    "ToolNotFoundError",
    "ToolRegistry",
    "ToolRegistryError",
    "ToolResult",
    "ToolScalar",
    "ToolValue",
]
