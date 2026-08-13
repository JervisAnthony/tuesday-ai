"""Tests for environment-based application configuration."""

from dataclasses import FrozenInstanceError

import pytest

from tuesday.config import (
    ConfigurationError,
    RuntimeEnvironment,
    load_settings,
)


def test_default_settings_are_suitable_for_development() -> None:
    settings = load_settings({})

    assert settings.app_name == "TUESDAY"
    assert settings.environment is RuntimeEnvironment.DEVELOPMENT
    assert settings.debug is True


@pytest.mark.parametrize(
    ("environment_value", "expected_environment", "expected_debug"),
    [
        ("testing", RuntimeEnvironment.TESTING, False),
        ("production", RuntimeEnvironment.PRODUCTION, False),
    ],
)
def test_non_development_environments_disable_debug_by_default(
    environment_value: str,
    expected_environment: RuntimeEnvironment,
    expected_debug: bool,
) -> None:
    settings = load_settings({"TUESDAY_ENV": environment_value})

    assert settings.environment is expected_environment
    assert settings.debug is expected_debug


@pytest.mark.parametrize(
    ("environment_value", "expected_environment"),
    [
        ("DEVELOPMENT", RuntimeEnvironment.DEVELOPMENT),
        ("Production", RuntimeEnvironment.PRODUCTION),
        ("testing", RuntimeEnvironment.TESTING),
    ],
)
def test_environment_values_are_case_insensitive(
    environment_value: str,
    expected_environment: RuntimeEnvironment,
) -> None:
    settings = load_settings({"TUESDAY_ENV": environment_value})

    assert settings.environment is expected_environment


def test_invalid_environment_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="Invalid TUESDAY_ENV"):
        load_settings({"TUESDAY_ENV": "staging"})


@pytest.mark.parametrize("debug_value", ["1", "true", "TRUE", "yes", "on"])
def test_truthy_debug_override_values(debug_value: str) -> None:
    settings = load_settings(
        {"TUESDAY_ENV": "production", "TUESDAY_DEBUG": debug_value}
    )

    assert settings.debug is True


@pytest.mark.parametrize("debug_value", ["0", "false", "FALSE", "no", "off"])
def test_falsey_debug_override_values(debug_value: str) -> None:
    settings = load_settings(
        {"TUESDAY_ENV": "development", "TUESDAY_DEBUG": debug_value}
    )

    assert settings.debug is False


def test_invalid_debug_override_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="Invalid TUESDAY_DEBUG"):
        load_settings({"TUESDAY_DEBUG": "sometimes"})


def test_settings_are_immutable() -> None:
    settings = load_settings({})

    with pytest.raises(FrozenInstanceError):
        settings.debug = False  # type: ignore[misc]
