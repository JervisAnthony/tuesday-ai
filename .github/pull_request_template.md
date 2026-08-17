## Summary

Describe what this PR changes and why.

## Changes

- List the main implementation changes.

## Validation

- [ ] `python -m pip check`
- [ ] `python -m ruff check .`
- [ ] `python -m pytest --cov=tuesday --cov-report=term-missing --cov-fail-under=100`

## Checklist

- [ ] This PR targets `Dev` unless it is an explicitly coordinated release promotion.
- [ ] The change is focused and contains no unrelated refactors.
- [ ] Tests cover new or changed behavior.
- [ ] Documentation is updated where needed.
- [ ] No credentials, secrets, caches, generated artifacts, or virtual environments are included.
- [ ] GitHub Actions CI is expected to pass across all supported Python versions.
