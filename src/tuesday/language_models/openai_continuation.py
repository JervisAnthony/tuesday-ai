"""Immutable OpenAI-specific contracts for manual stateless continuation."""

import json
from dataclasses import dataclass
from math import isfinite

from tuesday.language_models.base import LanguageModelRequest, LanguageModelResponse

__all__ = ["OpenAIContinuationState", "OpenAIGenerationResult"]


def _require_text(value: object, label: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    if not value.strip():
        raise ValueError(f"{label} must not be empty.")
    if value != value.strip():
        raise ValueError(f"{label} must not have surrounding whitespace.")


@dataclass(frozen=True, slots=True)
class OpenAIContinuationState:
    """Configured request and serialized provider items awaiting tool results."""

    request: LanguageModelRequest
    model: str
    temperature: float | None
    replay_items_json: tuple[str, ...]
    pending_call_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, LanguageModelRequest):
            raise TypeError("request must be a LanguageModelRequest.")
        _require_text(self.model, "OpenAI continuation model")
        if self.temperature is not None:
            if isinstance(self.temperature, bool) or not isinstance(
                self.temperature, (int, float)
            ):
                raise TypeError(
                    "OpenAI continuation temperature must be numeric or None."
                )
            if not isfinite(self.temperature) or not 0 <= self.temperature <= 2:
                raise ValueError(
                    "OpenAI continuation temperature must be finite and in [0, 2]."
                )
        if not isinstance(self.replay_items_json, tuple):
            raise TypeError("OpenAI continuation replay items must be a tuple.")
        if not self.replay_items_json:
            raise ValueError("OpenAI continuation requires at least one replay item.")
        for item in self.replay_items_json:
            if not isinstance(item, str):
                raise TypeError("OpenAI continuation replay items must be strings.")
            try:
                decoded = json.loads(item)
                # Reject NaN/Infinity, including numeric overflow in JSON literals.
                json.dumps(decoded, allow_nan=False)
            except ValueError as error:
                raise ValueError(
                    "OpenAI continuation replay items must be valid JSON."
                ) from error
            if not isinstance(decoded, dict):
                raise ValueError(
                    "OpenAI continuation replay items must be JSON objects."
                )
        if not isinstance(self.pending_call_ids, tuple):
            raise TypeError("OpenAI continuation pending call IDs must be a tuple.")
        if not self.pending_call_ids:
            raise ValueError(
                "OpenAI continuation requires at least one pending call ID."
            )
        for call_id in self.pending_call_ids:
            _require_text(call_id, "OpenAI continuation pending call ID")
        if len(self.pending_call_ids) != len(set(self.pending_call_ids)):
            raise ValueError("OpenAI continuation pending call IDs must be unique.")


@dataclass(frozen=True, slots=True)
class OpenAIGenerationResult:
    """Provider-neutral response with optional correlated OpenAI replay state."""

    response: LanguageModelResponse
    continuation: OpenAIContinuationState | None

    def __post_init__(self) -> None:
        if not isinstance(self.response, LanguageModelResponse):
            raise TypeError("response must be a LanguageModelResponse.")
        if self.continuation is not None and not isinstance(
            self.continuation, OpenAIContinuationState
        ):
            raise TypeError("continuation must be an OpenAIContinuationState or None.")
        if self.response.provider != "openai":
            raise ValueError("OpenAI generation response provider must be openai.")
        if self.response.tool_calls:
            if self.continuation is None:
                raise ValueError(
                    "OpenAI tool-call response requires continuation state."
                )
            if (
                tuple(call.call_id for call in self.response.tool_calls)
                != self.continuation.pending_call_ids
            ):
                raise ValueError(
                    "OpenAI generation tool call IDs must match pending calls."
                )
        elif self.continuation is not None:
            raise ValueError(
                "OpenAI text-only response must not have continuation state."
            )
