"""Repository-level tests for collaboration and CI quality gates."""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
CI_WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
CONTRIBUTING = PROJECT_ROOT / "CONTRIBUTING.md"
PR_TEMPLATE = PROJECT_ROOT / ".github" / "pull_request_template.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_ci_workflow_exists() -> None:
    assert CI_WORKFLOW.is_file()


def test_ci_runs_for_pull_requests_and_integration_branch_updates() -> None:
    workflow = _read(CI_WORKFLOW)

    expected_triggers = """on:
  pull_request:
    branches:
      - Dev
      - production
  push:
    branches:
      - Dev
      - production
  workflow_dispatch:
"""
    assert expected_triggers in workflow
    assert "pull_request_target:" not in workflow


def test_ci_uses_least_privilege_and_no_secrets() -> None:
    workflow = _read(CI_WORKFLOW)

    assert "permissions:\n  contents: read\n" in workflow
    assert "contents: write" not in workflow
    assert "write-all" not in workflow
    assert "id-token: write" not in workflow
    assert "secrets." not in workflow


def test_ci_cancels_superseded_runs() -> None:
    workflow = _read(CI_WORKFLOW)

    assert "group: ci-${{ github.workflow }}-${{ github.ref }}" in workflow
    assert "cancel-in-progress: true" in workflow


def test_ci_covers_every_supported_python_version() -> None:
    workflow = _read(CI_WORKFLOW)

    expected_matrix = (
        'python-version:\n'
        '          - "3.11"\n'
        '          - "3.12"\n'
        '          - "3.13"'
    )
    assert expected_matrix in workflow
    assert "fail-fast: false" in workflow
    assert "timeout-minutes: 15" in workflow


def test_ci_uses_only_expected_official_actions() -> None:
    workflow = _read(CI_WORKFLOW)
    actions = re.findall(r"^\s*uses:\s*([^\s]+)\s*$", workflow, flags=re.MULTILINE)

    assert actions == ["actions/checkout@v6", "actions/setup-python@v6"]
    assert "persist-credentials: false" in workflow


def test_ci_enforces_install_dependency_lint_test_and_coverage_gates() -> None:
    workflow = _read(CI_WORKFLOW)

    required_commands = (
        'python -m pip install -e ".[dev]"',
        "python -m pip check",
        "python -m ruff check .",
        "python -m pytest",
        "--cov=tuesday",
        "--cov-report=term-missing",
        "--cov-fail-under=100",
    )
    for command in required_commands:
        assert command in workflow

    assert "continue-on-error: true" not in workflow


def test_collaboration_docs_preserve_branch_and_validation_policy() -> None:
    contributing = _read(CONTRIBUTING)
    template = _read(PR_TEMPLATE)

    assert "Feature pull requests target `Dev`, never `production`." in contributing
    assert "git pull --ff-only origin Dev" in contributing
    assert "--cov-fail-under=100" in contributing
    assert "CI is intentionally read-only" in contributing

    assert "This PR targets `Dev`" in template
    assert "python -m pip check" in template
    assert "python -m ruff check ." in template
    assert "--cov-fail-under=100" in template
    assert "No credentials, secrets" in template
