"""Environment-based application configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "AppSettings",
    "ConfigurationError",
    "RuntimeEnvironment",
    "load_settings",
]

_ENVIRONMENT_VARIABLE = "TUESDAY_ENV"
_DEBUG_VARIABLE = "TUESDAY_DEBUG"
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
class AppSettings:
    """Immutable foundational application settings."""

    environment: RuntimeEnvironment
    debug: bool
    app_name: str = "TUESDAY"


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

    return AppSettings(environment=environment, debug=debug)
