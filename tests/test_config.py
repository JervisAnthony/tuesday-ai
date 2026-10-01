"""Tests for environment-based application configuration."""

import ast
import inspect
from dataclasses import FrozenInstanceError

import pytest

import tuesday.config as config_module
from tuesday.config import (
    AppSettings,
    ConfigurationError,
    LanguageModelSettings,
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


def test_language_model_settings_are_frozen_and_slotted() -> None:
    settings = LanguageModelSettings(provider="openai", model="example-model")

    assert not hasattr(settings, "__dict__")
    with pytest.raises(FrozenInstanceError):
        settings.model = "other"  # type: ignore[misc]


def test_model_settings_default_to_none_without_changing_existing_defaults() -> None:
    direct = AppSettings(
        environment=RuntimeEnvironment.DEVELOPMENT,
        debug=True,
    )
    loaded = load_settings({})

    assert direct.model is None
    assert loaded.model is None
    assert loaded.app_name == "TUESDAY"
    assert loaded.environment is RuntimeEnvironment.DEVELOPMENT
    assert loaded.debug is True


def test_provider_and_model_activate_model_configuration() -> None:
    settings = load_settings(
        {
            "TUESDAY_MODEL_PROVIDER": "OpenAI",
            "TUESDAY_MODEL_NAME": "organisation/custom-model",
        }
    )

    assert isinstance(settings.model, LanguageModelSettings)
    assert settings.model.provider == "OpenAI"
    assert settings.model.model == "organisation/custom-model"
    assert settings.model.api_key is None
    assert settings.model.timeout_seconds == 30.0
    assert settings.model.temperature is None


def test_explicit_api_key_is_preserved_but_hidden_from_representations() -> None:
    secret = " test-secret-value "
    settings = load_settings(
        {
            "TUESDAY_MODEL_PROVIDER": "openai",
            "TUESDAY_MODEL_NAME": "example-model",
            "TUESDAY_MODEL_API_KEY": secret,
        }
    )

    assert settings.model is not None
    assert settings.model.api_key == secret
    assert secret not in repr(settings.model)
    assert secret not in repr(settings)


@pytest.mark.parametrize("provider", ["", " ", "\t"])
def test_empty_or_whitespace_model_provider_is_rejected(provider: str) -> None:
    with pytest.raises(ConfigurationError, match="provider must not be empty"):
        load_settings(
            {
                "TUESDAY_MODEL_PROVIDER": provider,
                "TUESDAY_MODEL_NAME": "example-model",
            }
        )


@pytest.mark.parametrize("provider", [" openai", "openai "])
def test_model_provider_surrounding_whitespace_is_rejected(
    provider: str,
) -> None:
    with pytest.raises(
        ConfigurationError,
        match="provider must not have leading or trailing whitespace",
    ):
        load_settings(
            {
                "TUESDAY_MODEL_PROVIDER": provider,
                "TUESDAY_MODEL_NAME": "example-model",
            }
        )


@pytest.mark.parametrize("model", ["", " ", "\t"])
def test_empty_or_whitespace_model_name_is_rejected(model: str) -> None:
    with pytest.raises(ConfigurationError, match="Model name must not be empty"):
        load_settings(
            {
                "TUESDAY_MODEL_PROVIDER": "openai",
                "TUESDAY_MODEL_NAME": model,
            }
        )


@pytest.mark.parametrize("model", [" example-model", "example-model "])
def test_model_name_surrounding_whitespace_is_rejected(model: str) -> None:
    with pytest.raises(
        ConfigurationError,
        match="Model name must not have leading or trailing whitespace",
    ):
        load_settings(
            {
                "TUESDAY_MODEL_PROVIDER": "openai",
                "TUESDAY_MODEL_NAME": model,
            }
        )


def test_provider_without_model_name_is_rejected() -> None:
    with pytest.raises(
        ConfigurationError,
        match="TUESDAY_MODEL_PROVIDER requires TUESDAY_MODEL_NAME",
    ):
        load_settings({"TUESDAY_MODEL_PROVIDER": "openai"})


def test_model_name_without_provider_is_rejected() -> None:
    with pytest.raises(
        ConfigurationError,
        match="TUESDAY_MODEL_NAME requires TUESDAY_MODEL_PROVIDER",
    ):
        load_settings({"TUESDAY_MODEL_NAME": "example-model"})


@pytest.mark.parametrize(
    "partial_setting",
    [
        {"TUESDAY_MODEL_API_KEY": "test-secret-value"},
        {"TUESDAY_MODEL_TIMEOUT_SECONDS": "20"},
        {"TUESDAY_MODEL_TEMPERATURE": "0.2"},
    ],
)
def test_model_option_without_provider_and_name_is_rejected(
    partial_setting: dict[str, str],
) -> None:
    with pytest.raises(
        ConfigurationError,
        match="requires TUESDAY_MODEL_PROVIDER and TUESDAY_MODEL_NAME",
    ):
        load_settings(partial_setting)


@pytest.mark.parametrize("api_key", ["", " ", "\t"])
def test_explicit_blank_api_key_is_rejected(api_key: str) -> None:
    with pytest.raises(ConfigurationError, match="API key must not be empty"):
        load_settings(
            {
                "TUESDAY_MODEL_PROVIDER": "openai",
                "TUESDAY_MODEL_NAME": "example-model",
                "TUESDAY_MODEL_API_KEY": api_key,
            }
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [("30", 30.0), ("45.5", 45.5)],
)
def test_model_timeout_is_parsed_as_float(value: str, expected: float) -> None:
    settings = load_settings(
        {
            "TUESDAY_MODEL_PROVIDER": "openai",
            "TUESDAY_MODEL_NAME": "example-model",
            "TUESDAY_MODEL_TIMEOUT_SECONDS": value,
        }
    )

    assert settings.model is not None
    assert settings.model.timeout_seconds == expected
    assert isinstance(settings.model.timeout_seconds, float)


@pytest.mark.parametrize("timeout", ["0", "-1", "abc", "NaN", "inf", "-inf"])
def test_invalid_model_timeout_is_rejected(timeout: str) -> None:
    with pytest.raises(
        ConfigurationError,
        match="timeout must be a finite number greater than zero",
    ):
        load_settings(
            {
                "TUESDAY_MODEL_PROVIDER": "openai",
                "TUESDAY_MODEL_NAME": "example-model",
                "TUESDAY_MODEL_TIMEOUT_SECONDS": timeout,
            }
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [("0", 0.0), ("0.2", 0.2), ("3.75", 3.75)],
)
def test_model_temperature_is_parsed_and_preserved(
    value: str,
    expected: float,
) -> None:
    settings = load_settings(
        {
            "TUESDAY_MODEL_PROVIDER": "openai",
            "TUESDAY_MODEL_NAME": "example-model",
            "TUESDAY_MODEL_TEMPERATURE": value,
        }
    )

    assert settings.model is not None
    assert settings.model.temperature == expected


@pytest.mark.parametrize("temperature", ["-1", "abc", "NaN", "inf", "-inf"])
def test_invalid_model_temperature_is_rejected(temperature: str) -> None:
    with pytest.raises(
        ConfigurationError,
        match="temperature must be a finite non-negative number",
    ):
        load_settings(
            {
                "TUESDAY_MODEL_PROVIDER": "openai",
                "TUESDAY_MODEL_NAME": "example-model",
                "TUESDAY_MODEL_TEMPERATURE": temperature,
            }
        )


@pytest.mark.parametrize(
    ("arguments", "error_type", "message"),
    [
        (
            {"provider": 1, "model": "example"},
            TypeError,
            "provider must be a string",
        ),
        (
            {"provider": "openai", "model": 1},
            TypeError,
            "name must be a string",
        ),
        (
            {"provider": "openai", "model": "example", "api_key": 1},
            TypeError,
            "API key must be a string or None",
        ),
        (
            {"provider": "openai", "model": "example", "timeout_seconds": True},
            TypeError,
            "timeout must be numeric",
        ),
        (
            {"provider": "openai", "model": "example", "temperature": True},
            TypeError,
            "temperature must be numeric or None",
        ),
    ],
)
def test_direct_construction_rejects_invalid_python_types(
    arguments: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        LanguageModelSettings(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"provider": "", "model": "example"}, "provider must not be empty"),
        ({"provider": "openai", "model": ""}, "name must not be empty"),
        (
            {"provider": "openai", "model": "example", "api_key": " "},
            "API key must not be empty",
        ),
        (
            {"provider": "openai", "model": "example", "timeout_seconds": 0},
            "timeout must be a finite number greater than zero",
        ),
        (
            {"provider": "openai", "model": "example", "temperature": -1},
            "temperature must be a finite non-negative number",
        ),
    ],
)
def test_direct_construction_enforces_semantic_invariants(
    arguments: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        LanguageModelSettings(**arguments)  # type: ignore[arg-type]


def test_unknown_provider_and_arbitrary_model_are_structurally_accepted() -> None:
    settings = load_settings(
        {
            "TUESDAY_MODEL_PROVIDER": "FutureProvider",
            "TUESDAY_MODEL_NAME": "organisation/some-future-model",
        }
    )

    assert settings.model is not None
    assert settings.model.provider == "FutureProvider"
    assert settings.model.model == "organisation/some-future-model"


def test_supplied_environment_mapping_is_not_mutated() -> None:
    environ = {
        "TUESDAY_ENV": "testing",
        "TUESDAY_MODEL_PROVIDER": "openai",
        "TUESDAY_MODEL_NAME": "example-model",
    }
    original = environ.copy()

    load_settings(environ)

    assert environ == original


def test_separate_calls_do_not_leak_model_settings() -> None:
    configured = load_settings(
        {
            "TUESDAY_MODEL_PROVIDER": "openai",
            "TUESDAY_MODEL_NAME": "example-model",
        }
    )
    unconfigured = load_settings({})

    assert configured.model is not None
    assert unconfigured.model is None


def test_process_environment_is_read_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    model_variables = (
        "TUESDAY_MODEL_PROVIDER",
        "TUESDAY_MODEL_NAME",
        "TUESDAY_MODEL_API_KEY",
        "TUESDAY_MODEL_TIMEOUT_SECONDS",
        "TUESDAY_MODEL_TEMPERATURE",
    )
    monkeypatch.delenv("TUESDAY_ENV", raising=False)
    monkeypatch.delenv("TUESDAY_DEBUG", raising=False)
    for variable in model_variables:
        monkeypatch.delenv(variable, raising=False)

    assert load_settings().model is None

    monkeypatch.setenv("TUESDAY_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("TUESDAY_MODEL_NAME", "example-model")

    assert load_settings().model == LanguageModelSettings(
        provider="openai",
        model="example-model",
    )


def test_configuration_module_has_no_provider_or_network_dependencies() -> None:
    tree = ast.parse(inspect.getsource(config_module))
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
        "enum",
        "math",
        "os",
    }
