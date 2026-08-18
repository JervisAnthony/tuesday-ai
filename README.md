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
system. It includes a deterministic default conversational runtime and an
explicit provider-neutral model-backed composition path. Model execution remains
opt-in; the default application does not contact a language-model provider.

Over time, TUESDAY aims to:

- unify tasks and contextual information;
- assist with intelligent decisions;
- coordinate specialised capabilities;
- perform user-authorised actions; and
- integrate with external tools and services.

## Default application composition

TUESDAY has a deterministic end-to-end interaction path that composes its
router, agent registry, conversational baseline agent, request preparer, and
orchestrator. The default composition supports the explicit `/chat` and
`/conversation` directives; it does not infer intent from unrestricted text.

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

## Model-backed application composition

`create_model_backed_orchestrator(...)` composes any
`BaseLanguageModelProvider` behind the same `conversation` routes, request
preparation, registry, and orchestration path used by the deterministic runtime.
The factory itself does not read environment variables, select a provider, or
construct an OpenAI client.

A caller can opt into the existing OpenAI adapter explicitly:

```python
import asyncio
from uuid import uuid4

from tuesday.composition import create_model_backed_orchestrator
from tuesday.config import load_settings
from tuesday.domain import ConversationContext, TuesdayRequest
from tuesday.language_models import OpenAILanguageModelProvider

settings = load_settings()
if settings.model is None:
    raise RuntimeError("Model settings are required for model-backed execution.")

provider = OpenAILanguageModelProvider(settings.model)
orchestrator = create_model_backed_orchestrator(provider)

conversation_id = uuid4()
request = TuesdayRequest(
    content="/chat Hello",
    conversation_id=conversation_id,
)
context = ConversationContext(conversation_id=conversation_id)

response = asyncio.run(orchestrator.handle(request, context))
print(response.content)
```

This path remains explicit by design: configuring model settings alone does not
change the default application or trigger provider execution.

## Conversation history repository

TUESDAY now includes an asynchronous `BaseConversationRepository` boundary for
conversation history and a deterministic `InMemoryConversationRepository`
implementation. Repositories return immutable `ConversationContext` snapshots
and append immutable batches of `ConversationMessage` objects in supplied order.

An unknown conversation reads as an empty context with the requested
`conversation_id`. Appending the first message batch creates that conversation's
in-memory history implicitly:

```python
import asyncio
from uuid import uuid4

from tuesday.conversations import InMemoryConversationRepository
from tuesday.domain import ConversationMessage, MessageRole

repository = InMemoryConversationRepository()
conversation_id = uuid4()
messages = (
    ConversationMessage(MessageRole.USER, "Hello"),
    ConversationMessage(MessageRole.ASSISTANT, "Hi there"),
)

asyncio.run(repository.append_messages(conversation_id, messages))
context = asyncio.run(repository.get_context(conversation_id))
```

The in-memory repository is process-local and intentionally has no database,
filesystem, model-provider, routing, or orchestration dependency. Application
execution still receives `ConversationContext` explicitly; automatically loading
and persisting history belongs to a separate stateful conversation-service
layer.

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
python -m pip check
python -m ruff check .
python -m pytest --cov=tuesday --cov-report=term-missing --cov-fail-under=100
```

## Continuous integration

GitHub Actions validates pull requests targeting `Dev` or `production` and
pushes to those branches across Python 3.11, 3.12, and 3.13. CI performs an
editable development install, dependency consistency checks, Ruff linting, and
the full pytest suite with a 100% production-code coverage gate.

The CI workflow uses read-only repository permissions, does not require secrets,
and cancels superseded runs for the same ref. Contributor branch, validation,
and pull-request expectations are documented in [CONTRIBUTING.md](CONTRIBUTING.md).

## Model runtime configuration

TUESDAY includes optional runtime configuration contracts for language-model
providers, but the default composition performs no LLM requests. No provider or
model is selected by default. Configuration uses:

- `TUESDAY_MODEL_PROVIDER`
- `TUESDAY_MODEL_NAME`
- `TUESDAY_MODEL_API_KEY`
- `TUESDAY_MODEL_TIMEOUT_SECONDS`
- `TUESDAY_MODEL_TEMPERATURE`

Provider and model name must be configured together. Credentials should be
supplied through environment configuration and are excluded from settings
representations.

Provider-neutral language-model message, request, response, and asynchronous
provider contracts are available. TUESDAY's first concrete adapter uses the
official OpenAI SDK and implements those contracts. It can now be supplied
explicitly to the model-backed application composition while `/chat` remains
deterministic in the default composition.

The OpenAI adapter uses the existing TUESDAY-owned runtime variables:

```text
TUESDAY_MODEL_PROVIDER=openai
TUESDAY_MODEL_NAME=<model-name>
TUESDAY_MODEL_API_KEY=<credential>
TUESDAY_MODEL_TIMEOUT_SECONDS=30
TUESDAY_MODEL_TEMPERATURE=0.2
```

## Conversational prompt rendering

TUESDAY includes a provider-neutral `ConversationalPromptRenderer` that converts
an immutable `TuesdayRequest` and its prior `ConversationContext` into a
`LanguageModelRequest`. Rendering prepends TUESDAY's conversational system
instruction, preserves prior history in order, and appends the current request
as the final user message.

The renderer does not call OpenAI or any other provider, does not include
conversation/request identifiers in model input, and does not mutate the source
request or context.

## Model-backed conversational agent

`ModelBackedConversationalAgent` composes the provider-neutral prompt renderer
with any `BaseLanguageModelProvider`. For each request it validates conversation
correlation, renders the current request and prior history, performs exactly one
provider generation, and returns a correlated `TuesdayResponse` containing the
model-generated text.

The agent keeps the stable registry name `conversation`, allowing deterministic
and model-backed compositions to share routing semantics. It does not depend on
OpenAI directly, does not implement retries or fallbacks, and does not retain
conversation history or per-request state.

The default application still uses `ConversationalAgent`; model-backed execution
must be selected explicitly through `create_model_backed_orchestrator(...)`.

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
