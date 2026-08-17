# Contributing to TUESDAY

Thanks for contributing to T.U.E.S.D.A.Y. — Task-Unifying Engine for Smart Decisions, Actions & You.

## Branch model

- `production` is the protected stable/release branch.
- `Dev` is the integration branch for validated development work.
- Normal work starts from the latest `Dev` on a focused `feature/*` branch.
- Feature pull requests target `Dev`, never `production`.
- Promotion from `Dev` to `production` is a deliberate release operation.

## Before starting work

Synchronize your local integration branch before creating a feature branch:

```text
git fetch origin
git switch Dev
git pull --ff-only origin Dev
git switch -c feature/<short-purpose>
```

Keep each pull request focused on one coherent change and avoid unrelated refactors.

## Local validation

Create or activate your virtual environment, then run:

```text
python -m pip install -e ".[dev]"
python -m pip check
python -m ruff check .
python -m pytest --cov=tuesday --cov-report=term-missing --cov-fail-under=100
```

The repository currently requires 100% coverage for production code.

## Pull requests

Before requesting review:

- target `Dev` unless the change is an explicitly coordinated release promotion;
- describe the problem, scope, and architectural impact clearly;
- include the validation you performed;
- update tests and documentation when behavior changes;
- never commit credentials, API keys, local environment files, caches, or virtual environments; and
- resolve review conversations before merge.

GitHub Actions validates pull requests to `Dev` and `production` across all supported Python versions. CI is intentionally read-only and does not require repository secrets.

Changes that weaken or bypass CI checks should be isolated, explicitly justified, and reviewed as CI/infrastructure work rather than bundled into product features.
