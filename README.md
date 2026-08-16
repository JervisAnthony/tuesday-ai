# T.U.E.S.D.A.Y.

**Task-Unifying Engine for Smart Decisions, Actions & You**

<p align="center">
  <img
    src="docs/assets/tuesday-readme.png"
    alt="TUESDAY — Task-Unifying Engine for Smart Decisions, Actions & You"
    width="900"
  />
</p>

TUESDAY is intended to become a modular, context-aware personal AI operating
system. It currently includes a deterministic conversational baseline agent used
to validate the production agent execution path. LLM-backed conversational
behavior will be introduced later.

Over time, TUESDAY aims to:

- unify tasks and contextual information;
- assist with intelligent decisions;
- coordinate specialised capabilities;
- perform user-authorised actions; and
- integrate with external tools and services.

## Default application composition

TUESDAY now has a deterministic end-to-end interaction path that composes its
router, agent registry, conversational baseline agent, and orchestrator. The
default composition supports the explicit `/chat` and `/conversation`
directives; it does not infer intent from unrestricted text.

```python
import asyncio
from uuid import uuid4

from tuesday.composition import create_default_orchestrator
from tuesday.domain import ConversationContext, TuesdayRequest

conversation_id = uuid4()
request = TuesdayRequest(
    content="/chat Hello",
    conversation_id=conversation_id,
)
context = ConversationContext(conversation_id=conversation_id)

orchestrator = create_default_orchestrator()
response = asyncio.run(orchestrator.handle(request, context))

print(response.content)  # TUESDAY received: Hello
```

The `/chat` directive is used for deterministic routing and removed before the
conversational agent receives the user-facing content. The original
`TuesdayRequest` remains unchanged.

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

## Model runtime configuration

TUESDAY includes optional runtime configuration contracts for future
language-model providers, but the current default composition performs no LLM
requests. No provider or model is selected by default. Configuration uses:

- `TUESDAY_MODEL_PROVIDER`
- `TUESDAY_MODEL_NAME`
- `TUESDAY_MODEL_API_KEY`
- `TUESDAY_MODEL_TIMEOUT_SECONDS`
- `TUESDAY_MODEL_TEMPERATURE`

Provider and model name must be configured together. Credentials should be
supplied through environment configuration and are excluded from settings
representations.

Provider-neutral language-model message, request, response, and asynchronous
provider contracts are also available. TUESDAY's first concrete adapter uses the
official OpenAI SDK and implements those provider-neutral contracts, but it is
not wired into the default application. `/chat` remains deterministic, and
configuring model settings alone performs no provider call; model-backed
conversation is forthcoming.

The OpenAI adapter uses the existing TUESDAY-owned runtime variables:

```text
TUESDAY_MODEL_PROVIDER=openai
TUESDAY_MODEL_NAME=<model-name>
TUESDAY_MODEL_API_KEY=<credential>
TUESDAY_MODEL_TIMEOUT_SECONDS=30
TUESDAY_MODEL_TEMPERATURE=0.2
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
