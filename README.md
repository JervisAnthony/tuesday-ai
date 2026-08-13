# T.U.E.S.D.A.Y.

**Task-Unifying Engine for Smart Decisions, Actions & You**

TUESDAY is intended to become a modular, context-aware personal AI operating
system. The project is currently at its foundation stage: it provides packaging,
development tooling, and an importable Python package, but no application
capabilities yet.

Over time, TUESDAY aims to:

- unify tasks and contextual information;
- assist with intelligent decisions;
- coordinate specialised capabilities;
- perform user-authorised actions; and
- integrate with external tools and services.

## Development setup

Python 3.13 is recommended for local development. The package supports Python
3.11 and newer.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Run the validation suite:

```powershell
python -m pytest
python -m ruff check .
python -m pytest --cov=tuesday
```

## Configuration

TUESDAY reads its foundational runtime settings from environment variables:

- `TUESDAY_ENV` selects `development`, `testing`, or `production`. Its default
  is `development`.
- `TUESDAY_DEBUG` optionally overrides the environment-derived debug setting.
  Accepted true values are `1`, `true`, `yes`, and `on`; accepted false values
  are `0`, `false`, `no`, and `off`. Values are case-insensitive.

Debug mode defaults to enabled in development and disabled in testing and
production. Invalid environment or debug values produce a configuration error.

## License

TUESDAY is licensed under the MIT License. See [LICENSE](LICENSE).
