"""Environment-based application configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite

__all__ = [
    "AppSettings",
    "ConfigurationError",
    "LanguageModelSettings",
    "RuntimeEnvironment",
    "load_settings",
]

_ENVIRONMENT_VARIABLE = "TUESDAY_ENV"
_DEBUG_VARIABLE = "TUESDAY_DEBUG"
_MODEL_PROVIDER_VARIABLE = "TUESDAY_MODEL_PROVIDER"
_MODEL_NAME_VARIABLE = "TUESDAY_MODEL_NAME"
_MODEL_API_KEY_VARIABLE = "TUESDAY_MODEL_API_KEY"
_MODEL_TIMEOUT_VARIABLE = "TUESDAY_MODEL_TIMEOUT_SECONDS"
_MODEL_TEMPERATURE_VARIABLE = "TUESDAY_MODEL_TEMPERATURE"
_MODEL_VARIABLES = (
    _MODEL_PROVIDER_VARIABLE,
    _MODEL_NAME_VARIABLE,
    _MODEL_API_KEY_VARIABLE,
    _MODEL_TIMEOUT_VARIABLE,
    _MODEL_TEMPERATURE_VARIABLE,
)
_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSEY_VALUES = frozenset({"0", "false", "no", "off"})


class ConfigurationError(ValueError):
    """Raised when application configuration is invalid."""


class RuntimeEnvironment(StrEnum):
    """Supported TUESDAY runtime environments."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


@dataclass(frozen=True, slots=True)
class LanguageModelSettings:
    """Immutable runtime settings for a future language-model provider."""

    provider: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    timeout_seconds: float = 30.0
    temperature: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str):
            raise TypeError("Model provider must be a string.")
        if not self.provider.strip():
            raise ConfigurationError("Model provider must not be empty.")
        if self.provider != self.provider.strip():
            raise ConfigurationError(
                "Model provider must not have leading or trailing whitespace."
            )

        if not isinstance(self.model, str):
            raise TypeError("Model name must be a string.")
        if not self.model.strip():
            raise ConfigurationError("Model name must not be empty.")
        if self.model != self.model.strip():
            raise ConfigurationError(
                "Model name must not have leading or trailing whitespace."
            )

        if self.api_key is not None:
            if not isinstance(self.api_key, str):
                raise TypeError("Model API key must be a string or None.")
            if not self.api_key.strip():
                raise ConfigurationError("Model API key must not be empty.")

        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
        ):
            raise TypeError("Model timeout must be numeric.")
        if not isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ConfigurationError(
                "Model timeout must be a finite number greater than zero."
            )

        if self.temperature is not None:
            if isinstance(self.temperature, bool) or not isinstance(
                self.temperature, (int, float)
            ):
                raise TypeError("Model temperature must be numeric or None.")
            if not isfinite(self.temperature) or self.temperature < 0:
                raise ConfigurationError(
                    "Model temperature must be a finite non-negative number."
                )


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Immutable foundational application settings."""

    environment: RuntimeEnvironment
    debug: bool
    app_name: str = "TUESDAY"
    model: LanguageModelSettings | None = None


def _parse_environment(value: str) -> RuntimeEnvironment:
    normalized_value = value.strip().lower()
    try:
        return RuntimeEnvironment(normalized_value)
    except ValueError as error:
        supported_values = ", ".join(
            environment.value for environment in RuntimeEnvironment
        )
        raise ConfigurationError(
            f"Invalid {_ENVIRONMENT_VARIABLE} value {value!r}. "
            f"Expected one of: {supported_values}."
        ) from error


def _parse_debug(value: str) -> bool:
    normalized_value = value.strip().lower()
    if normalized_value in _TRUTHY_VALUES:
        return True
    if normalized_value in _FALSEY_VALUES:
        return False

    accepted_values = ", ".join(sorted(_TRUTHY_VALUES | _FALSEY_VALUES))
    raise ConfigurationError(
        f"Invalid {_DEBUG_VARIABLE} value {value!r}. "
        f"Expected one of: {accepted_values}."
    )


def _parse_model_float(value: str, error_message: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(error_message) from error


def _load_model_settings(
    source: Mapping[str, str],
) -> LanguageModelSettings | None:
    if not any(variable in source for variable in _MODEL_VARIABLES):
        return None

    provider = source.get(_MODEL_PROVIDER_VARIABLE)
    model = source.get(_MODEL_NAME_VARIABLE)
    if provider is None:
        if model is not None:
            raise ConfigurationError(
                f"{_MODEL_NAME_VARIABLE} requires {_MODEL_PROVIDER_VARIABLE}."
            )
        raise ConfigurationError(
            "Model configuration requires TUESDAY_MODEL_PROVIDER and "
            "TUESDAY_MODEL_NAME."
        )
    if model is None:
        raise ConfigurationError(
            f"{_MODEL_PROVIDER_VARIABLE} requires {_MODEL_NAME_VARIABLE}."
        )

    timeout_value = source.get(_MODEL_TIMEOUT_VARIABLE)
    timeout_seconds = (
        30.0
        if timeout_value is None
        else _parse_model_float(
            timeout_value,
            "Model timeout must be a finite number greater than zero.",
        )
    )
    temperature_value = source.get(_MODEL_TEMPERATURE_VARIABLE)
    temperature = (
        None
        if temperature_value is None
        else _parse_model_float(
            temperature_value,
            "Model temperature must be a finite non-negative number.",
        )
    )

    return LanguageModelSettings(
        provider=provider,
        model=model,
        api_key=source.get(_MODEL_API_KEY_VARIABLE),
        timeout_seconds=timeout_seconds,
        temperature=temperature,
    )


def load_settings(environ: Mapping[str, str] | None = None) -> AppSettings:
    """Load and validate settings from an environment-variable mapping."""
    source = os.environ if environ is None else environ
    environment = _parse_environment(
        source.get(_ENVIRONMENT_VARIABLE, RuntimeEnvironment.DEVELOPMENT.value)
    )

    debug_value = source.get(_DEBUG_VARIABLE)
    debug = (
        environment is RuntimeEnvironment.DEVELOPMENT
        if debug_value is None
        else _parse_debug(debug_value)
    )

    return AppSettings(
        environment=environment,
        debug=debug,
        model=_load_model_settings(source),
    )
