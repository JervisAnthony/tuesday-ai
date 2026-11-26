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

## Command-line interface

Install the package for local development, then submit one quoted request with
an explicit `/chat` or `/conversation` directive:

```text
python -m pip install -e ".[dev]"
python -m tuesday "/chat Hello TUESDAY"
tuesday "/chat Hello TUESDAY"
```

Both entry points default to the existing deterministic conversational agent. The
example prints `TUESDAY received: Hello TUESDAY`. Each one-shot invocation handles
one request through the existing default orchestrator and stateful conversation
service. Plain-text output contains only the completed response and a terminal
newline. The original request reaches the service unchanged; existing routing
and preparation remove the current directive before agent execution.

Help and version work without a request or application execution:

```text
python -m tuesday --help
python -m tuesday --version
tuesday --help
tuesday --version
```

Use `--json` to receive one object containing `content`, `conversation_id`,
`request_id`, and `response_id`, all taken from the actual response. Without an
explicit conversation ID, the domain generates a fresh one:

```text
python -m tuesday --json "/chat Hello TUESDAY"
python -m tuesday --conversation-id 00000000-0000-0000-0000-000000000044 "/chat Hello"
```

Storage defaults to a fresh in-memory repository for every invocation. Reusing a
conversation ID alone does not preserve history across processes. Select SQLite
explicitly and reuse both its path and the conversation ID for durable history:

```text
python -m tuesday --database /caller-selected/location/history.sqlite3 --conversation-id 00000000-0000-0000-0000-000000000044 "/chat Hello"
python -m tuesday --database /caller-selected/location/history.sqlite3 --conversation-id 00000000-0000-0000-0000-000000000044 "/conversation Continue"
```

Replace the example database path with a suitable caller-selected location whose
parent directory already exists. The second command prints
`TUESDAY received: Continue (2 prior messages in context)`. Both successful pairs
remain stored, including the original routing directives. Different conversation
IDs or database files have independent histories.

You can also use `--database` without `--conversation-id`. Add `--json` to retrieve
the generated identifier and reuse its `conversation_id` value in a later command:

```text
python -m tuesday --json --database /caller-selected/location/history.sqlite3 "/chat Start a new conversation"
```

Exit status `0` means success, help, or version; `1` means an expected routing,
preparation, storage, model configuration, or provider failure; `2` means invalid
command-line arguments; and
`130` means interruption. Errors go to stderr without a successful stdout result.
Unexpected internal programming errors propagate. Supported options are `-h`,
`--help`, `--version`, `--conversation-id UUID`, `--database PATH`, `--json`,
`--runtime {deterministic,openai}`, `--max-history-messages N`, and `--interactive`.
Invalid arguments are rejected before runtime or database construction.

The default CLI provides deterministic acknowledgments. Model environment
configuration alone does not activate a provider or get read by that path;
`--runtime deterministic` selects it explicitly. There is no guarded tool
execution, streaming, automatic database activation,
conversation discovery, or request idempotency
guarantee. Programmatic asynchronous callers should use application services
directly; one-shot execution uses `asyncio.run`, while interactive mode owns an
`asyncio.Runner`. CLI `main` must not be invoked inside an already-running event loop.

Choose a protected database location and manage permissions and backups. SQLite
adds no encryption-at-rest feature. Cancellation or interruption does not
guarantee that an already-running database transaction stopped or rolled back;
the CLI does not replay interrupted requests. See the SQLite repository section
for its transaction and concurrency limitations.

Command-line requests may remain in shell history or be visible through operating
system process inspection. Avoid secrets or sensitive personal information in
arguments. Neither shell history nor database history is automatically securely
deleted.

### Opt-in OpenAI runtime

Select model-backed, text-only execution explicitly:

```text
python -m tuesday --runtime openai "/chat Hello"
tuesday --runtime openai --json "/conversation Explain the next step"
```

The existing configuration loader reads `TUESDAY_MODEL_PROVIDER`,
`TUESDAY_MODEL_NAME`, and `TUESDAY_MODEL_API_KEY`. Set the provider to exactly
`openai`, select a model available to your account, and supply its credential
through your environment rather than command-line arguments. Optional
`TUESDAY_MODEL_TIMEOUT_SECONDS` and `TUESDAY_MODEL_TEMPERATURE` retain their
existing validation; OpenAI temperature must be between zero and two. Missing,
unsupported, or invalid model settings fail with exit status `1`. Help and version
require no model settings and construct no provider.

This path reuses `OpenAILanguageModelProvider`, the model-backed orchestrator
factory, and the stateful conversation service. Each valid successful interaction
makes one text-generation request with SDK retries disabled. It advertises no
tools and executes none; unexpected model tool calls retain the text-only agent's
`UnsupportedModelToolCallError` and never produce a successful persisted pair.
Provider failures report a concise stderr message and status `1`, without raw SDK
details, credentials, or request content. There is no deterministic fallback or
automatic retry. Unexpected internal failures propagate.

Use SQLite and a message limit together when desired:

```text
python -m tuesday --runtime openai --json --database /caller-selected/location/history.sqlite3 --conversation-id 00000000-0000-0000-0000-000000000044 --max-history-messages 2 "/chat Continue"
```

Omitting `--max-history-messages` sends all available history. Zero excludes
historical messages; positive N selects the last N individual messages in order.
The system prompt and current prepared user request are always included. Odd
limits may split a prior user/assistant pair. This is a message count, not a token,
context-window, or cost guarantee, and SQLite retains complete successful history,
including original user directives. The option is rejected as an argument error
with deterministic runtime, before configuration or runtime construction.

Explicit model execution sends the selected conversation content and current
request to the configured provider. Review your history and provider settings
before opting in. JSON retains the same four response fields; it adds no runtime
or provider metadata.

The one-shot CLI creates, executes, and asynchronously closes its provider-owned
client within one `asyncio.run` lifecycle, including after composition, request, or
storage failures. `OpenAILanguageModelProvider.aclose()` leaves injected clients
under their caller's ownership; it adds no cleanup requirement to the abstract
provider contract. If execution and cleanup both fail, the execution failure wins.
Cleanup failures after successful execution propagate normally; SDK cleanup errors
report a model failure with status `1`. Such cleanup can fail after the successful
pair has already been committed, so failure status does not guarantee that nothing
was persisted and is not a reason to replay automatically. Cleanup never undoes
committed history, model requests, or external effects.

### Interactive terminal conversations

Select interactive mode explicitly to submit sequential requests in one session:

```text
python -m tuesday --interactive
tuesday --interactive
python -m tuesday --interactive --runtime openai --max-history-messages 2
```

One-shot execution still requires a positional request. Interactive mode accepts
no positional request and cannot be combined with `--json`. These invalid
combinations fail before configuration loading or runtime construction. Existing
`--runtime`, `--conversation-id`, and `--database` options remain available;
`--max-history-messages` continues to require the OpenAI runtime.

Enter one physical line per request, beginning with `/chat` or `/conversation`
and including content. Turns execute sequentially without automatic directive
insertion, queuing, or retries. Blank lines are ignored. Submitted lines retain
their content except for the terminal line ending; the existing router and
preparer determine whether a request is valid. This is not multiline input or
streaming output.

Local commands begin with a colon and are never sent to the application or model:

| Command | Behavior |
|---|---|
| `:help` | Show session commands and explicit directive requirements. |
| `:id` | Show the selected conversation UUID. |
| `:new` | Select a fresh UUID while retaining the repository and earlier history. |
| `:use UUID` | Select an existing or unknown UUID; unknown IDs start with empty history. |
| `:exit` | End the session normally. |

Commands require the displayed argument counts and valid UUID syntax. Invalid or
unknown commands report an error without changing the selected conversation.
They do not create conversation messages. The initial UUID comes from
`--conversation-id` when supplied, otherwise it is generated. Startup and `:id`
show it on stderr so it can be reused explicitly later; no conversation is
discovered automatically.

A session retains one repository, orchestrator, and stateful service. In-memory
history survives `:new` and `:use` switching within that session. For recovery
across process restarts, choose SQLite and reuse the same path and UUID:

```text
python -m tuesday --interactive --database /caller-selected/location/history.sqlite3 --conversation-id 00000000-0000-0000-0000-000000000046
```

The database parent must already exist. Only completed successful turns are
persisted through the existing service, with original user directives intact.
OpenAI history limits affect model-visible input only, leaving stored history
complete.

Only successful response content and its terminal newline go to stdout. Session
IDs, command help, selection confirmations, and errors go to stderr. A prompt is
emitted on stderr only when stdin is a terminal; piped/scripted stdin is supported
without prompts or input echo.

`:exit` or EOF ends normally with status `0`. Expected routing, preparation,
provider, and repository errors during a turn are reported and return to input;
a later normal exit still returns `0`. Startup/configuration/storage-initialization
errors return `1`, invalid CLI arguments return `2`, and keyboard interruption
during input or execution terminates with `130`. Unexpected programming failures
terminate and propagate. No failed or interrupted request is replayed.

In OpenAI mode, settings load once and one provider and renderer serve the entire
session, including recoverable failed turns and conversation switches. Async
construction, every turn, and final cleanup share one event loop owned by a single
`asyncio.Runner`. Terminal input is synchronous and interruptible; no background
input worker remains blocked after exit. The owned provider closes once on EOF,
explicit exit, interruption, fatal failure, or failure after provider construction.
Injected clients retain caller ownership.

Primary terminating failures take precedence over cleanup failures. A previously
handled turn error does not hide a later cleanup-only failure, which remains
visible under the existing CLI error rules. Already-emitted responses and committed
history remain valid if a later turn or shutdown fails. Cleanup and interruption
do not guarantee rollback or secure deletion, and external effects are not undone.

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

## Guarded OpenAI application composition

`create_guarded_openai_orchestrator(tool_turn_service, tools, renderer=None)`
returns a `TuesdayOrchestrator` with one `GuardedOpenAIConversationalAgent`.
This factory is explicitly opt-in: construct the `OpenAIToolTurnService` and
non-empty tuple of `LanguageModelToolDefinition` objects separately, then pass
those exact dependencies to the factory:

```python
from tuesday.composition import create_guarded_openai_orchestrator

orchestrator = create_guarded_openai_orchestrator(
    tool_turn_service,
    tools,
)
```

Pass an existing `ConversationalPromptRenderer` with `renderer=renderer` to
use a custom system prompt; omitting it retains the agent's default renderer.
The factory preserves the supplied service, tools tuple, definition objects,
and optional renderer by identity. Validation remains in the guarded agent.

The shared composition helper creates a fresh agent registry, deterministic
router, and directive request preparer. Both `/chat` and `/conversation` select
the guarded agent. Existing directive handling removes the route before prompt
rendering while preserving the original request and its correlation IDs.

The caller remains responsible for OpenAI clients, concrete tool registration,
and authorization policies. The factory constructs none of these, reads no
environment configuration, and does not discover or expand advertised tools.
Advertisement still requires guarded authorization before execution.
Construction alone does not execute a model or tool, and neither existing
factory automatically enables guarded execution. `create_default_orchestrator`
remains deterministic; `create_model_backed_orchestrator` remains text-only.

Each interaction remains bounded to one initial generation, at most one guarded
execution, and at most one continuation. Further tool calls fail closed through
the guarded agent. There are no automatic multi-step loops, retries, or fallbacks.
The factory adds application wiring rather than autonomous execution.

EXPLICIT COMPOSITION != AUTOMATIC MODEL EXECUTION.
EXISTING DEFAULT COMPOSITION MUST REMAIN DETERMINISTIC.

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
filesystem, model-provider, routing, or orchestration dependency.

## SQLite conversation history repository

`SQLiteConversationRepository` implements the same asynchronous repository
contract using Python's standard-library `sqlite3`, with no third-party database
dependency. Select a file-backed database explicitly:

```python
from pathlib import Path

from tuesday.composition import create_stateful_guarded_openai_service
from tuesday.conversations import SQLiteConversationRepository

repository = SQLiteConversationRepository(
    Path("/caller-selected/location/conversations.sqlite3")
)
service = create_stateful_guarded_openai_service(
    tool_turn_service,
    tools,
    repository,
)
```

Replace the example path with a suitable location whose parent directory already
exists. The caller owns database selection; missing parent directories are not
created. Strings and `Path` values are accepted, while SQLite `:memory:` and
caller-supplied file URIs are rejected. The path is resolved once so later working
directory changes do not redirect storage. Importing the package creates no
database. Construction synchronously initializes the conversation table and its
conversation/sequence index in one transaction, preserving existing records,
unrelated tables, and the database's `user_version`. Incompatible schemas surface
an error; there is no migration framework or automatic database replacement.

Reads and appends run through `asyncio.to_thread`. Each operation creates, uses,
and closes its own connection in the owning thread, with a five-second SQLite
lock timeout. Ordinary operations require the initialized file to exist, so a
deleted database is not silently recreated. Expected SQLite and malformed stored
data errors raise `ConversationRepositoryError` with the original cause; invalid
caller inputs raise `TypeError` or `ValueError`. Failed operations are not retried.

Snapshots reconstruct immutable domain messages with the original UUIDs, roles,
exact content, and ISO 8601 timestamps, preserving represented instants, offsets,
and microseconds. Named timezone objects and Python object identities are not
preserved. History follows database append sequence, including within each batch,
regardless of timestamps. Duplicate message IDs are retained. Unknown
conversations return empty contexts, and different conversation IDs remain
isolated. Reopening the same file recovers committed history; separate files keep
independent histories.

An append validates and serializes the entire batch before writing, acquires a
write transaction with `BEGIN IMMEDIATE`, inserts in supplied order, and reads the
resulting snapshot before committing. A failed batch rolls back all its inserts
and preserves earlier committed history. Concurrent successful appends retain
each complete batch contiguously; their acquisition order is unspecified.

Existing stateful services and factories accept this repository unchanged. The
in-memory implementation remains available and no backend activates
automatically. An explicitly supplied `ConversationalPromptRenderer` with
`max_history_messages=2` or zero bounds only model input: SQLite retains every
successfully persisted message, including original user routing directives.

Database files can contain sensitive conversation history. Choose a protected
storage location and manage file permissions and backups. This backend adds no
encryption-at-rest feature, backup or retention scheduler, request deduplication,
or exactly-once tool execution. Durability relies on SQLite's ordinary transaction
guarantees; it does not promise protection from disk failure or corruption.

Atomic message batches do not coordinate complete conversation turns or roll back
external tools. A tool can execute before a later append fails. Concurrent service
turns can use overlapping history snapshots, and cancelling an awaiting caller
does not necessarily interrupt an already-running database worker or transaction.
No automatic cancellation rollback or request-level exactly-once persistence is
provided.

## Stateful conversation service

`StatefulConversationService` owns the application-level interaction lifecycle
around an existing `TuesdayOrchestrator` and `BaseConversationRepository`:

`load stored context → execute TUESDAY → persist the successful turn → return`

```python
import asyncio
from uuid import uuid4

from tuesday.composition import create_default_orchestrator
from tuesday.conversations import InMemoryConversationRepository
from tuesday.domain import TuesdayRequest
from tuesday.services import StatefulConversationService

repository = InMemoryConversationRepository()
service = StatefulConversationService(
    create_default_orchestrator(),
    repository,
)
conversation_id = uuid4()

response = asyncio.run(
    service.handle(
        TuesdayRequest(
            content="/chat Hello",
            conversation_id=conversation_id,
        )
    )
)
```

The service loads history before orchestration and appends exactly one user and
one assistant message only after successful execution. The persisted user
message preserves the original incoming request content exactly, including any
routing directive. Repository and orchestration failures propagate unchanged;
there is no retry, rollback, or persistence fallback in this layer.

## Stateful guarded OpenAI conversation composition

`create_stateful_guarded_openai_service` combines the existing guarded
orchestrator factory and `StatefulConversationService` with a caller-supplied
`BaseConversationRepository`. Configure `tool_turn_service` and the non-empty
model-visible `tools` tuple separately, then choose a repository explicitly:

```python
import asyncio
from uuid import uuid4

from tuesday.composition import create_stateful_guarded_openai_service
from tuesday.conversations import InMemoryConversationRepository
from tuesday.domain import TuesdayRequest

repository = InMemoryConversationRepository()
service = create_stateful_guarded_openai_service(
    tool_turn_service,
    tools,
    repository,
)
conversation_id = uuid4()


async def conversation():
    first = await service.handle(
        TuesdayRequest("/chat Hello TUESDAY", conversation_id=conversation_id)
    )
    second = await service.handle(
        TuesdayRequest(
            "/conversation What did I say earlier?",
            conversation_id=conversation_id,
        )
    )
    return first, second


first, second = asyncio.run(conversation())
```

An optional `renderer=renderer` preserves an existing custom
`ConversationalPromptRenderer`. Construction performs no model execution,
authorization, repository reads, or writes, and creates no persistence backend.
All three existing orchestrator factories retain their behavior.

To bound model-visible history explicitly, pass a configured renderer through
the same stateful factory:

```python
from tuesday.prompting import ConversationalPromptRenderer

renderer = ConversationalPromptRenderer(max_history_messages=4)
service = create_stateful_guarded_openai_service(
    tool_turn_service,
    tools,
    repository,
    renderer=renderer,
)
```

The model-backed, guarded OpenAI, and stateful guarded factories already accept
`renderer=renderer` and preserve the exact instance; no new factory signatures
are required. No default cap is imposed and no summarization occurs. The
repository retains all successfully persisted history, even when the renderer
includes only its most recent messages in model input.

Each explicit `handle` call loads the current history before interaction and
appends one original user message followed by one final assistant message only
after a successful, completed response. Stored user messages retain their
routing directives. Earlier messages pass through unchanged; only the current
request has its leading `/chat` or `/conversation` directive removed before
rendering. Responses preserve the request's conversation and request IDs.
Reusing a conversation ID reuses its history; distinct IDs keep histories
isolated. The in-memory repository shown here is process-local, not permanent
storage. Repository selection always belongs to the caller.

Tool advertisement still requires guarded authorization. Unadvertised tools,
denials, confirmation requirements, multiple initial calls, and unresolved
continuations fail closed without appending a successful pair. Existing history
remains intact. Initial provider and execution failures also prevent appending;
repository read failures prevent execution. An append failure after completion
propagates without retrying the model or tool.

There is no automatic agent loop, retry, rollback, or incomplete-turn recovery.
A failed continuation or append can follow a completed tool execution; withheld
conversation persistence does not undo tool side effects. This factory provides
no built-in concurrency control, request idempotency, exactly-once side effects,
or transaction isolation. Callers must handle concurrency and side-effect
recovery at higher layers when those guarantees are needed.

EXPLICIT REPOSITORY != AUTOMATIC PERSISTENCE BACKEND.
FAILED INTERACTION != SUCCESSFUL PERSISTED CONVERSATION TURN.
CONVERSATION PERSISTENCE != TOOL SIDE-EFFECT ROLLBACK.
SAME CONVERSATION_ID REUSES HISTORY.
DIFFERENT CONVERSATION_ID MUST NOT SHARE HISTORY.
STATEFUL FACTORY != TRANSACTION OR IDEMPOTENCY MANAGER.

## Tool execution contracts

TUESDAY now includes provider- and integration-neutral contracts for future
tool execution. `ToolInvocation` carries a stable tool name, immutable structured
arguments, and a correlation UUID. `ToolResult` carries the originating tool
name and invocation UUID together with immutable structured output.

Tool argument and result values are restricted to JSON-compatible data. Mutable
lists and mappings supplied by callers are recursively snapshotted into tuples
and read-only mappings so an invocation or result cannot be changed indirectly
after construction.

`BaseTool` defines a stable machine-friendly name, human-readable description,
and one asynchronous `execute(...)` operation. `ToolExecutionError` provides the
common failure boundary for concrete tool adapters.

This contract layer does not execute tools or introduce any external side
effect.

## Provider-neutral model tool contracts

`LanguageModelToolDefinition` is a model-visible description of a capability,
while `LanguageModelToolCall` is structured intent emitted by a language model.
Neither is executable: `BaseTool` is the executable implementation, and
`ToolInvocation` is an authorized-execution candidate in the tool domain.

`LanguageModelRequest` can carry an ordered
`tools: tuple[LanguageModelToolDefinition, ...]`. The tuple defaults to empty
for text-only requests, preserves caller-defined order, and rejects exact
duplicate names so model-visible capabilities are unambiguous.

`LanguageModelResponse` supports text-only, tool-call-only, and text + tool-call
responses through `tool_calls: tuple[LanguageModelToolCall, ...]`, defaulting to
`()` and preserving tuple identity, call identities, and order. `content=None`
is valid only with at least one tool call; empty or whitespace strings remain
invalid. Correlation IDs must be unique within a response using exact,
case-sensitive comparison; repeated tool names are allowed.

`ModelBackedConversationalAgent` remains a text-only consumer. Any model tool
calls cause `UnsupportedModelToolCallError`, including responses that also have
text. MODEL TOOL CALLS MUST NOT BE SILENTLY DISCARDED.

`OpenAILanguageModelProvider` serializes explicit `LanguageModelToolDefinition`
values as Responses API custom function tools, copying frozen schemas into
plain dictionaries and lists. Empty request tools preserve the text-only payload.
OpenAI `function_call` output items are decoded into immutable
`LanguageModelToolCall` values in `LanguageModelResponse.tool_calls`, preserving
call IDs and output order. Tool-only output becomes `content=None`; mixed output
preserves both exact text and tool calls. Malformed arguments and call metadata
fail with safe provider errors.

This is representation, not execution:
`LanguageModelToolCall != ToolInvocation`,
MODEL TOOL REQUEST != EXECUTION AUTHORIZATION, and
MODEL TOOL CALL != TOOL EXECUTION. Commit 32 does not create `ToolInvocation`,
authorize, confirm, execute, send `function_call_output`, make a second model
request, or continue reasoning/tool interaction. The conversational agent still
raises `UnsupportedModelToolCallError` for tool calls; prompt rendering and
application composition remain unchanged.

## Explicit model tool invocation bridge

`LanguageModelToolCall → ModelToolInvocationBridge → ToolInvocation` is an
explicit, provider-neutral translation from model intent to an execution candidate:

```python
from tuesday.bridges import ModelToolInvocationBridge
from tuesday.language_models.tools import LanguageModelToolCall

tool_call = LanguageModelToolCall("call_abc123", "calculator.basic", {"left": 6})
invocation = ModelToolInvocationBridge().to_invocation(tool_call)
```

Each synchronous conversion preserves the exact tool name and passes arguments
through `ToolInvocation` to create a fresh, independent immutable snapshot.
Unknown tool names remain bridgeable. The target's stricter top-level argument
name validation still applies; names and keys are never normalized, and nested
keys receive no additional bridge validation. Target failures propagate unchanged.

The bridge generates a new execution UUID using `uuid4` by default, or an injected
`invocation_id_factory`, called exactly once after source-type validation. The
original model call remains unchanged: its opaque `call_id` is never parsed or
reused as `invocation_id`. The original call must remain available to a future
tool-result continuation layer for provider/model correlation.

`LanguageModelToolCall != ToolInvocation`.
MODEL CALL ID != EXECUTION INVOCATION ID.
MODEL TOOL CALL != TOOL EXECUTION.
TOOL INVOCATION CANDIDATE != AUTHORIZED EXECUTION.
The bridge does not check registry membership, authorize, confirm, or execute.
It does not handle results, continue model interaction, or change composition.

## Guarded model-requested tool execution

`GuardedModelToolExecutor` accepts exactly one `LanguageModelToolCall` and calls
`ModelToolInvocationBridge` once to create a `ToolInvocation`. It then delegates
exclusively through `GuardedToolExecutor`: authorization precedes deterministic
execution, registry lookup, and tool execution. A successful `ToolResult` becomes
a frozen `ModelToolExecution` retaining the original model call, exact invocation,
and exact result, with validated names, arguments, and execution UUID correlation.

`ALLOW` may execute. `REQUIRE_CONFIRMATION` blocks through the existing
`ToolConfirmationRequiredError`; `DENY` blocks through
`ToolAuthorizationDeniedError`. Unconfigured tools are denied by static policy
before registry lookup. An explicitly allowed unknown tool can reach the registry
and fail with `ToolNotFoundError`. Failures propagate without retries or fallbacks.
This API does not request, store, or resume confirmations.

MODEL CALL ID != EXECUTION INVOCATION ID.
MODEL TOOL CALL != TOOL EXECUTION.
TOOL INVOCATION CANDIDATE != AUTHORIZED EXECUTION.
MODEL TOOL EXECUTION MUST PASS THROUGH AUTHORIZATION.
The original `call_id` remains available through `execution.tool_call.call_id`;
it is never passed into tool execution as the invocation UUID.

`ModelToolExecution` is the terminal successful result of this stage. A future
continuation layer may use `execution.tool_call.call_id` and `execution.result`.
No provider continuation, second model request, or agent loop occurs here.
Caller composition remains explicit: no agent is automatically wired to this API,
and conversation handling still rejects model tool calls.

## Provider-neutral model tool result bridge

`ModelToolExecution → ModelToolResultBridge → LanguageModelToolResult` translates
one successful execution into a provider-neutral result. The result contains only
the original opaque model/provider `call_id`, taken exactly from
`execution.tool_call.call_id`, and immutable `output` from
`execution.result.output`. Containers receive a fresh recursive snapshot using
the existing model-tool value contract; scalars, including `None`, retain their
values. The source execution objects remain unchanged.

The model result carries no tool name, execution invocation UUID, or
provider-specific continuation data.
EXECUTION INVOCATION ID != MODEL CALL ID.
EXECUTION TOOL RESULT != MODEL TOOL RESULT.
MODEL TOOL RESULT != PROVIDER CONTINUATION.
MODEL TOOL RESULT MUST PRESERVE MODEL CALL CORRELATION.

TUESDAY continues to use `store=False`. A later Responses API continuation layer
must construct provider-native `function_call_output` correlated by `call_id`;
stateless/reasoning-capable continuation may also require preserving and replaying
prior provider output items. This stage neither models nor sends that continuation
state, serializes output for a provider, or makes a second model request.

## Stateless OpenAI tool continuation

`OpenAILanguageModelProvider.generate_with_continuation(request)` returns an
`OpenAIGenerationResult`: the existing `LanguageModelResponse` plus an optional
`OpenAIContinuationState` for pending function calls. Ordinary `generate()` keeps
its existing behavior and does not capture replay state.

The caller executes tools through the guarded application path outside the
provider and obtains `LanguageModelToolResult` values. An explicit
`continue_with_tool_results(state, results)` call requires all pending call IDs
to match exactly in order, then performs one Responses request. Its input contains
the original messages, all accumulated provider items, and new
`function_call_output` items using original model call IDs and JSON-text outputs
(including JSON `null` for `None`). Original tools, configured model, and optional
temperature come from the saved state.

`store=False` and `stream=False` remain unchanged; `previous_response_id` is not
used. Every provider output item is snapshotted using public SDK serialization.
Reasoning and encrypted fields are preserved opaquely when returned, with no raw
SDK objects retained in state. This follows the
[OpenAI manual replay guidance](https://developers.openai.com/api/docs/guides/reasoning).
A continued response with more calls yields a new state accumulating prior
items, submitted outputs, and new provider items in order. Final text returns
`continuation=None`.

MODEL CALL ID != EXECUTION INVOCATION ID.
MODEL TOOL RESULT != PROVIDER CONTINUATION.
PROVIDER CONTINUATION STATE != LANGUAGE MODEL RESPONSE.
STATELESS CONTINUATION MUST REPLAY PROVIDER OUTPUT ITEMS.
FUNCTION CALL OUTPUT MUST USE THE ORIGINAL MODEL CALL ID.
TOOL EXECUTION MUST REMAIN OUTSIDE THE OPENAI ADAPTER.

Commit 36 exposes explicit provider continuation primitives. It does not connect
`ModelBackedConversationalAgent` to `GuardedModelToolExecutor` or automatically
iterate model/tool/model cycles. Each additional continuation requires a caller
action; no automatic loop, confirmation completion, or composition is added.

## Guarded OpenAI tool turn

`OpenAIToolTurnService(provider, executor, result_bridge)` explicitly composes an
`OpenAILanguageModelProvider`, `GuardedModelToolExecutor`, and
`ModelToolResultBridge`. Callers must construct and inject these dependencies;
the service does not read configuration or create clients, registries, or policies.

`await service.run_once(request)` accepts a `LanguageModelRequest` and performs
one initial generation through `generate_with_continuation`. A text-only response
returns immediately, including when `request.tools` is empty, without inspecting
continuation state. A single model tool call requires three separate checks:

1. Its exact name must appear in the original `request.tools` definitions.
2. Continuation state must reference that exact original request object.
3. The existing guarded authorization path must permit execution.

Before invoking the guarded executor, the service first checks request identity
using `initial.continuation.request is request`, then checks tool exposure by
exact, case-sensitive name. An equal but distinct request fails with `ValueError`;
the service does not copy or repair continuation state. An unadvertised name
raises `UnadvertisedModelToolCallError`, even when the tool is registered and
explicitly allowed by policy. Empty `request.tools` rejects every initial tool
call. Names receive no aliases, wildcard matching, or normalization. These errors
occur before invocation creation, authorization, lookup, execution, result
bridging, or provider continuation.

An in-scope call passes through the existing guarded executor, then the result
bridge, then exactly one provider continuation using a one-element tuple of model
results. There are zero or one guarded executions and zero or one continuation
requests per invocation. These checks do not implement a complete identity or
RBAC system.

The frozen `OpenAIToolTurnResult` retains `initial`, `execution`, `model_result`,
and `continued` separately. Text-only turns have `None` for the last three fields.
Tool turns preserve the exact original model call object, its opaque `call_id`,
and the correlated output; the execution invocation UUID remains separate.
Initial content and continued content are retained without concatenation.

Authorization remains mandatory inside the guarded pipeline. `ALLOW` may execute;
`DENY` raises `ToolAuthorizationDeniedError`, and `REQUIRE_CONFIRMATION` raises
`ToolConfirmationRequiredError` without obtaining or inferring confirmation.
Unconfigured tools are denied before registry lookup. An explicitly allowed
advertised unknown tool raises `ToolNotFoundError` at lookup. Authorization and
execution failures propagate without result bridging or continuation.

Multiple initial calls raise `MultipleModelToolCallsUnsupportedError` before
invocation bridging, authorization, lookup, or execution. A continuation may
return further tool calls: its complete response and continuation state remain
available, but those calls are not executed and no third provider request occurs.
One tool turn does not enable an automatic agent loop, iteration, retries, or
fallbacks. Provider, executor, bridge, and continuation failures propagate.
A continuation failure or invalid continuation result after successful execution
does not undo tool side effects; this service provides no transactional rollback.

This service introduces no automatic application composition, conversational
agent wiring, confirmation handling, conversation persistence, or persistent
execution history. The existing conversational agent still rejects model tool
calls through its ordinary `generate` path. Tool execution remains outside the
OpenAI adapter.

MODEL TOOL CALL MUST BE WITHIN THE REQUEST'S ADVERTISED TOOL SCOPE.
ADVERTISED TOOL != AUTHORIZED TOOL.
AUTHORIZED TOOL != EXECUTED TOOL.
REQUEST CONTINUATION STATE MUST BELONG TO THE ORIGINAL REQUEST.
MODEL CALL ID != EXECUTION INVOCATION ID.
MODEL TOOL EXECUTION MUST PASS THROUGH AUTHORIZATION.
REQUIRE CONFIRMATION != CONFIRMATION RECEIVED.
MODEL TOOL RESULT != PROVIDER CONTINUATION.
ONE TOOL TURN != AUTOMATIC AGENT LOOP.

## Tool registry

TUESDAY now has an explicit `ToolRegistry`. Concrete `BaseTool` instances are
registered under their exact, case-sensitive names, and lookup returns the same
instances. Registration order is preserved in immutable name snapshots,
duplicate names are rejected, and separate registry instances remain isolated.

Tool Registry != Tool Executor. The registry performs no execution. This change
does not introduce tool execution orchestration, authorization, user
confirmation, retries, model tool-calling, or external integrations.

## Deterministic tool executor

TUESDAY now has a deterministic executor for an explicitly constructed
`ToolInvocation`: it performs an exact, case-sensitive `ToolRegistry` lookup,
passes the same invocation object to one `BaseTool.execute(...)` call, validates
the returned `ToolResult` name and invocation UUID, and returns that exact
result object. One executor call executes at most one tool. Lookup and execution
failures propagate unchanged, invalid result correlation fails explicitly, and
there are no retries or fallbacks.

Tool Executor != Tool Authorization. This layer does not provide confirmation
policy, permissions, side-effect classification, model tool-calling, automatic
tool selection, multi-tool planning, integrations, or application composition.

## Tool authorization policy contracts

TUESDAY now defines a provider-neutral authorization boundary from an explicit
`ToolInvocation`, through `BaseToolAuthorizationPolicy`, to a
`ToolAuthorizationDecision`. A policy decision has exactly one outcome:
`ALLOW`, `REQUIRE_CONFIRMATION`, or `DENY`. Every decision includes a required
human-readable reason and is correlated to its invocation by both tool name and
invocation UUID.

Authorization Decision != Tool Execution, and Require Confirmation !=
Confirmation Received. These contracts do not execute tools, modify the
deterministic executor, obtain or store confirmation, define users or roles,
classify side effects, provide concrete policies, call models, add integrations,
or change application composition.

## Guarded tool execution

TUESDAY now composes an explicit `ToolInvocation`, one
`BaseToolAuthorizationPolicy` evaluation, and a correlated
`ToolAuthorizationDecision` through `GuardedToolExecutor`. `ALLOW` delegates the
same invocation to `DeterministicToolExecutor`; `REQUIRE_CONFIRMATION` blocks
with `ToolConfirmationRequiredError`, while `DENY` blocks with
`ToolAuthorizationDeniedError`. Malformed or mismatched decisions fail closed
with `InvalidToolAuthorizationDecisionError`.

Guarded Execution != Confirmation Handling. A denial remains a normal policy
decision; its exception exists only because execution was attempted. The
guarded boundary preserves exact invocation and result identities, evaluates
policy exactly once, executes at most one tool, and provides no retries or
fallbacks.

## Static tool authorization policy

TUESDAY now includes `StaticToolAuthorizationPolicy`, a deterministic production
policy that maps exact, case-sensitive tool names to `ALLOW`,
`REQUIRE_CONFIRMATION`, or `DENY`. Unconfigured names are denied. Matching uses
no aliases, wildcards, or normalization, and the rule configuration is
snapshotted when the policy is created.

Static Policy != Tool Registry, and Static Policy != Tool Execution. The policy
only returns correlated authorization decisions. `REQUIRE_CONFIRMATION` remains
a policy outcome; this layer does not obtain or store confirmation.

## Basic calculator tool

TUESDAY now includes its first concrete executable production tool,
`BasicCalculatorTool`, with the machine name `calculator.basic`. An invocation
must supply exactly `operation`, `left`, and `right`; supported operations are
`add`, `subtract`, `multiply`, and `divide`. Operation matching is exact,
booleans are not numeric operands, division by zero raises `ToolExecutionError`,
and non-finite results are rejected.

The calculator performs no arbitrary expression evaluation, external service
calls, or side effects. Callers may explicitly register it with `ToolRegistry`
and execute it through `StaticToolAuthorizationPolicy`, `GuardedToolExecutor`,
and `DeterministicToolExecutor`; it is not automatically enabled.

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

Callers can explicitly limit historical messages:

```python
from tuesday.prompting import ConversationalPromptRenderer

renderer = ConversationalPromptRenderer(max_history_messages=4)
```

The keyword-only `max_history_messages` defaults to `None`, which sends the full
available history. Zero excludes all historical messages; a positive integer N
selects the last N historical messages in their original tuple order. Booleans
and other non-integer values are rejected, as are negative integers.

The configured system prompt always appears first and the current user request
always appears last; neither counts toward the history limit. Selected messages
preserve their original roles and exact text, including whitespace and newlines.
Historical system messages count toward the same cap as other history. The
entire context is validated before selection, including omitted messages.

Limits count individual messages, not complete turns: odd limits may split a
prior user/assistant pair. This changes only model-visible history; stored
repository history remains unchanged. It performs no summarization or token
counting and does not guarantee that input fits a model's context window.
Existing callers retain unlimited history unless they opt in explicitly.

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

## Opt-in guarded OpenAI conversational agent

`GuardedOpenAIConversationalAgent` adapts TUESDAY requests and conversation
context to the bounded `OpenAIToolTurnService`. Construct it explicitly with an
already-configured service, a non-empty tuple of exact model-visible
`LanguageModelToolDefinition` values, and an optional
`ConversationalPromptRenderer`:

```python
from tuesday.agents import GuardedOpenAIConversationalAgent
from tuesday.language_models import LanguageModelToolDefinition
from tuesday.services.openai_tool_turn import OpenAIToolTurnService


def conversational_agent(tool_turn_service: OpenAIToolTurnService):
    tools = (
        LanguageModelToolDefinition(
            name="calculator.basic",
            description="Perform one basic arithmetic operation.",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                    },
                    "left": {"type": "number"},
                    "right": {"type": "number"},
                },
                "required": ["operation", "left", "right"],
                "additionalProperties": False,
            },
        ),
    )
    return GuardedOpenAIConversationalAgent(tool_turn_service, tools)
```

The caller supplies the service's provider, registry, authorization policy, and
guarded execution stack separately. The agent advertises only its configured
tools, preserving their order and definition objects; it never discovers tools
or infers permissions. Advertised tools still require guarded authorization.
The rendered message tuple is preserved in a new model request with that exact
tools tuple. A renderer that already supplies tools is rejected rather than
merged. Continuation state must still belong to this exact model request.

Each interaction renders once and calls `run_once` once. Text-only generations
complete normally. A successful tool turn returns only the final continued model
text as a correlated `TuesdayResponse`; initial and continued text are never
concatenated, and raw tool output is not substituted for a model answer.

Further continuation tool calls raise `IncompleteGuardedToolTurnError`, even
when accompanied by text. The agent does not return a partial answer, call the
service again, retry, fall back, or run an automatic multi-step model/tool loop.
Existing provider, scope, authorization, execution, and correlation errors
propagate unchanged.

An incomplete turn may follow a successful tool execution. Completed tool side
effects remain: there is no rollback, compensation, transaction management, or
persistent pending-turn record. Through `StatefulConversationService`, a failed
or incomplete interaction does not append its user/assistant message pair;
that persistence behavior does not undo earlier tool side effects.

The agent uses the registry name `conversation`. Callers can select it through
`create_guarded_openai_orchestrator` or manually wire one chosen conversational
implementation into the existing registry, router, request preparer, and
orchestrator. The deterministic and text-only agents and their composition
factories retain their behavior; neither existing factory selects this agent
automatically. This opt-in adapter does not provide an autonomous assistant.

ONE GUARDED TOOL TURN != AUTOMATIC AGENT LOOP.
UNRESOLVED MODEL TOOL CALLS != COMPLETED ASSISTANT RESPONSE.

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
