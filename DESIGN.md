# agentloop design

agentloop runs a language model in a loop with tools. It is a Go library
first. The `agentloop` command is one small program built on it.

It is written for running agents whose access has to be controlled exactly.
An agent can do only what it was explicitly given, every decision it causes
is written down before it takes effect, and a crash or a kill at any point
leaves a record that can be resumed without repeating an action.

## Principles

1. **Nothing ambient.** The library reads no environment variables, opens no
   files and has no tools of its own. A model adapter gets its credentials
   when it is built. A tool gets exactly the environment it is built with.
   The only network traffic is the adapter's requests to its provider.
2. **One gate for every effect.** A tool call reaches its tool only after
   three checks, in order: the tool exists, the input matches its schema,
   and the policy allows it. The policy sees the call and the session's
   state, including its labels, and can allow, deny or hold the call.
3. **The log is the session.** A session is an append-only list of entries,
   each written durably before the loop goes on. The conversation, the
   labels and every pending call are derived from it. Nothing in a session
   is ever edited or removed.
4. **No vendor in the core.** The core knows no provider. An adapter turns
   the neutral conversation into one provider's requests and back.
   Assistant turns keep the provider's own record (`Native`) next to the
   neutral parts, and the same provider is always given its own record
   back unchanged (the same JSON, though whitespace may differ).
   Reasoning state survives, and history is never rewritten, which
   current models require.
5. **Failures are classified, not parsed.** A failed model call comes back
   as a `ModelError` with a kind (rate limited, overloaded, unreachable,
   unauthorized, billing, not found, invalid request) and the times the
   provider gave for retrying. The adapters do this from typed SDK errors
   and response headers.
6. **Retries and limits belong to the loop.** SDK retries are turned off.
   The agent retries transient failures under a policy you set, logs each
   retry, and stops at the turn, tool-call and token limits you set.

## Packages

| Package | What it is |
|---|---|
| `agentloop` | The core: messages, the `Model` and `Tool` interfaces, policies, sessions, stores, the agent loop |
| `agentloop/anthropic` | The `Model` for Anthropic's Messages API |
| `agentloop/openai` | The `Model` for OpenAI's Responses API |
| `agentloop/trifecta` | The trifecta rule, pure, and a `Policy` built on it |
| `agentloop/otel` | A pure converter from a session log to OpenTelemetry GenAI traces, as OTLP/JSON |
| `agentloop/quarantine` | Pure checks for a quarantined reader: whether an answer schema admits free text, and validating an answer |
| `agentloop/command` | A `Tool` that runs a program, with the plain or the envelope protocol |
| `agentloop/mcp` | The tools of one MCP server over stdio, as `Tool` values |
| `agentloop/fake` | A scripted `Model` for tests |
| `cmd/agentloop` | The command: runs one agent from a JSON request |

## The core

### Messages

`Message`, `Part`, `ToolCall`, `ToolResult` and `Native` are in
`message.go`. A message is a role and a list of parts (text, reasoning,
tool call, tool result). An assistant message also carries `Native`: the
provider name, the serving model and the provider's raw JSON for the turn.

### Models

`Model`, `Request`, `Response`, `Stop`, `Usage`, `Delta`, `Effort` and
`ModelError` are in `model.go`. `Generate` makes one request, streams
deltas to `observe` (which may be nil), and returns the assistant message
with `Native` set.

An adapter, given `Request.Messages`:

- replays an assistant message from `Native` when `Native.Provider` is its
  own, whatever the model, and otherwise renders it from its parts, leaving
  out reasoning parts;
- renders user messages from their parts, merging consecutive messages of
  the same role where its API needs that;
- maps `Effort` to the provider's nearest level;
- returns `*ModelError` for every failure, with `Kind` from the status and
  the provider's error type, `RetryAfter` from `retry-after`, and
  `ResetsAt` from the provider's rate-limit reset headers;
- turns its SDK's own retries off.

A model can also implement `Describer`, which is optional. `Describe`
asks the provider about a model with one request and returns a
`ModelInfo`: the id, the display name and the context window, where empty
or zero means the provider did not say. The window always comes from the
provider and never from a table in agentloop. Anthropic's adapter asks
`GET /v1/models/{id}` under its base URL, with the Messages API's key,
and takes the window from `max_input_tokens` (`max_tokens` is the output
cap). OpenAI's Models API returns only the id, owner and creation time,
so its adapter returns the id and leaves the name and window unknown.
`fake` does not describe its models.

### Tools

`Tool`, `Result` and `Func` are in `tool.go`. A tool's parameters are a
JSON Schema, compiled when the agent starts. A schema that does not compile
is an error then, not at the first call. Schemas cannot load remote
references.

A `Result` can carry labels, which are added to the session when the
result is recorded. Its `Log` is for the operator: it is kept in the
`tool_result` entry and never shown to the model.

A tool that also implements `CallTool` is called through `CallWith`,
which gives it the call's id and the session's labels as the call runs
(those include the labels the call's own decision added).

### Policies

```go
type Verdict string // "allow", "deny", "hold"

type Decision struct {
    Verdict Verdict
    Reason  string   // shown to the model on deny; recorded always
    Labels  []string // added to the session when the call is allowed
}

type Policy interface {
    Decide(ctx context.Context, call ToolCall, view View) Decision
}

type PolicyFunc func(ctx context.Context, call ToolCall, view View) Decision

// View is the session as a policy sees it.
type View struct {
    Labels    []string // sorted
    ToolCalls int      // tool calls decided so far in the session
}
```

A nil policy allows every call to a tool the agent was given. The tools
themselves are the grant. The policy adds conditions on top.

Labels are the information-flow primitive. A policy can add labels when it
allows a call, before the result exists (cleanroom's rule). A tool can add
labels with its result. Labels are never removed, so a policy can refuse
an `egress` call once the session holds `private`, for example.

`Agent.Labels` are added to the session whenever a run starts, in a
`labels` entry written only when one of them is new. `Session.AddLabels`
does the same from outside a run.

**Forbidden labels.** `Agent.Forbid` names labels the session may never
hold, for an agent that must stay clear of something (relay's public
roles never hold `private`). The loop enforces it whatever the policy
says, with `Forbidden(labels, forbid)`, the pure intersection:

- A run on a session that holds a forbidden label, or would from
  `Agent.Labels`, does not start. It writes nothing, calls no model, and
  ends with reason `forbidden`, the labels in `Outcome.Forbidden`.
- A decision that would add a forbidden label is a deny, with the reason
  `ForbiddenDecision`, and `Resolve` refuses a person's allow of such a
  call with `ErrPolicyDenies`.
- A tool result that carries a forbidden label is refused. The model is
  told only `RefusedResult` ("This result was refused: it carries a
  label this session may not hold."), the session does not take the
  label, and nothing the tool returned is kept, not even in `log`, which
  says only that the result was refused and why. `Outcome.Refused`
  lists each refused result of the run as `{"call_id", "tool",
  "labels"}`, whatever reason the run ended with, so whoever runs the
  agent can tell the operator.

**Holds.** A held call waits for a person. A hold adds no labels; its
decision entry records `input_sha256`, the SHA-256 in hex of the call's
input as canonical JSON (keys sorted, no whitespace, no HTML escaping,
numbers as written; `InputDigest`). The run ends with reason `held` and
lists the held calls, each with its `input_sha256`, and `label_sources`:
for each label the session holds, the entry that first added it, as
`{"kind": "seed"|"prompt"|"tool", "tool", "call_id", "seq"}` (`tool` and
`call_id` for a tool's decision or result). `Session.LabelSources` gives
the same, so whoever decides can be told why the session holds a label.

`Agent.Resolve(ctx, session, Resolution)` records the person's decision:
a call id, `allow` or `deny`, an optional reason, and the input digest,
which must match the call's input. An allow takes its labels from the
agent's policy now, as an allow from the policy would have; the person
never supplies labels. Resolving a call again with the verdict it was
resolved with does nothing, and any other call that is not held is an
error. An allow is refused when the agent has no policy (nothing could
give the call its labels) or when its policy now denies the call: a
person can approve what the policy held, not what it refuses.
`CheckResolve` says whether `Resolve` would accept a resolution, without
writing anything. `Session.Policed` reports whether a session holds a
label or ever held a call. The next `Run` carries on: it runs the call or reports the denial,
then goes back to the model. `Session.Resolve(callID, Decision)` is the
lower-level form, which records the decision as given.

### The trifecta

The trifecta is private data, untrusted content and egress held by one
session at once. `agentloop/trifecta` keeps a session from using all
three without a person's approval.

There are two labels and no others: `private` (the session has seen the
operator's private data) and `contaminated` (it has seen something an
outsider could have written). Any other name is an error wherever a
label is read, so a typo cannot drop one. A tool has a class: `read`
(returns content, changes nothing, sends nothing out), `write` (changes
the operator's own state), `egress` (can put data the model chose where
an outsider sees it) or `act` (does something in the operator's name).

`trifecta.Decide(labels, tool)` is the rule, a pure function of the
session's labels when the call is decided and the tool's entry:

- `always_hold`: hold
- `egress`, and the session holds both labels: hold
- `act`, and the session holds `contaminated`: hold
- an unknown class: hold
- otherwise: allow

It never denies; a person does. `trifecta.Policy` applies it to a map of
tool entries. Each decision carries the tool's static labels, which the
session takes on when the call is allowed, before it runs, and keeps
whatever becomes of the call. A tool with no entry is denied, and so is
a tool whose static labels include one of `Policy.Forbid`.

### Quarantined answers

`agentloop/quarantine` is the pure part of a quarantined reader, which
nothing uses yet. A reader is a model with no tools that reads content
an outsider may have written and answers the caller's question as JSON
matching the caller's schema. `quarantine.Check` says whether a JSON
Schema 2020-12 admits free text: a string or object key the schema does
not fix. Strings fixed by `enum` or `const`, or closed by the `date`,
`time` and `date-time` formats, are not free text, and neither are
booleans, numbers and null. Arrays and objects are free of text when
their items, keys and values are, and an array must also say
`maxItems` of at most 100. A fixed string or property name of one code
point or none (after NFC) is free text, since an array of them spells.
Numbers are not bounded. `patternProperties`,
`unevaluatedProperties`, `unevaluatedItems`, `$id`, `$anchor`,
`$dynamicRef`, the content keywords and every keyword `Check` does not
know count as free text, wherever they appear. `$ref` may point only at
the root or at the root's `$defs`. `Compile` compiles a schema with
format assertion on and nothing loaded, and `Answer` validates an answer
and returns it as canonical JSON, which is what a caller must be given.

### Sessions and stores

```go
type Store interface {
    Load() ([]Entry, error)
    Append(Entry) error // durable before it returns
}

func NewFileStore(path string) *FileStore // JSON lines, fsync on append
func NewMemoryStore() *MemoryStore

func OpenSession(store Store) (*Session, error)
```

An `Entry` has a sequence number, a time, a type and one payload:

| Type | Payload | Written |
|---|---|---|
| `run_started` | the prompt, if any | when `Run` starts |
| `labels` | labels added from outside any call, and their `source`: `seed` or `prompt` | when a run's starting labels, or `AddLabels`, add one the session lacks |
| `message` | a `Message`, and for an assistant turn its `Stop`, `Usage` and serving model | when a user prompt is added or a response arrives |
| `decision` | call id, tool name, verdict, reason, labels; `resolved` for a person's; `input_sha256` for a hold | when a call is checked, or when a held call is resolved |
| `call_started` | call id | immediately before the tool runs |
| `tool_result` | the `ToolResult`, its labels, and the operator's `log` | when a call finishes, or is refused |
| `model_error` | the `ModelError`, and whether the loop will retry | on each failed model call |
| `run_ended` | the `Outcome` | when `Run` returns |

A session rebuilds its state from the entries: the conversation, the
labels, the usage, and the state of every tool call (decided, started,
finished). Tool results are logged one by one as they finish, so a crash
between two calls loses neither. In the conversation, results that follow
an assistant turn are gathered into one user message.

`FileStore.Load` ignores a final line that is not a complete entry, which
is what a write cut short by a crash leaves, and the next append cuts it
off. A bad line anywhere else is an error, and so are sequence numbers that
do not run 1, 2, 3 and so on.

Every entry goes through JSON before the session applies it, so the live
state and the state rebuilt from the same log are always equal, and a
session shares no memory with its callers.

### The loop

```go
type Agent struct {
    Model           Model
    ModelID         string
    System          string
    Tools           []Tool
    Policy          Policy
    Effort          Effort
    MaxOutputTokens int
    Limits          Limits
    Retry           *RetryPolicy  // nil means DefaultRetry
    ToolTimeout     time.Duration // per call; zero means none
    Observe         func(Event)   // every entry as it is appended, and every delta
}

type Limits struct {
    MaxTurns     int   // model calls per run; zero means no limit
    MaxToolCalls int   // per run
    MaxTokens    int64 // Usage.Total per run
}

type RetryPolicy struct {
    Retries int           // further attempts after the first
    Backoff time.Duration // first wait for overloaded or unreachable; doubles each time
    MaxWait time.Duration // the longest single wait, including for a rate limit
}

func (a *Agent) Run(ctx context.Context, s *Session, prompt string) (*Outcome, error)
```

`Run` returns an error only when the agent is set up wrongly, the store
fails, or another `Run` is already in progress on the same session.
Everything else is an `Outcome`:

```go
type Outcome struct {
    Reason Reason      // completed, max_tokens, refused, held, limit, model_error, canceled
    Text   string      // the last assistant text
    Error  *ModelError // for model_error
    Held   []ToolCall  // for held
    Limit  string      // which limit, for limit
    Turns  int
    Usage  Usage
}
```

A run:

1. Writes `run_started`, and a `labels` entry if `Agent.Labels` adds any.
2. Settles the calls left open by the last assistant turn (below). If any
   is still held, ends with `held`.
3. Adds the prompt as a user message, if there is one. With no prompt and
   nothing waiting for the model, it ends with the last turn's outcome,
   usually `completed`, and the last assistant text. A run that ends
   `held` or `canceled` before this step does not add its prompt, so the
   caller passes it again.
4. Repeats: checks the limits, calls the model (retrying under the retry
   policy), and records the response. On `tool_use` it settles the turn's
   calls and goes on. On `pause` it settles them as cut off and goes on.
   On `end_turn`, `max_tokens` or `refusal` it settles them as cut off and
   ends with `completed`, `max_tokens` or `refused`. So at rest every call
   has a result, unless it is held.
5. Writes `run_ended`.

A response that repeats a tool call id, or has a call with no id, is not
recorded: the run ends with `model_error`, kind `invalid_request`. So is
a response whose tool call input, or the provider's record of the turn,
nests deeper than 64 levels or has a number longer than 100 characters:
every line agentloop writes stays readable by strict JSON parsers.

A call refused because the run reached `MaxToolCalls` keeps the labels its
decision added, though it never ran. Labels err toward caution.

Settling takes the calls of the last assistant turn that have no result,
in order:

- If the turn did not end in `tool_use`, its calls were cut off, and each
  gets an error result saying so. Nothing runs. (Normally these were
  settled when the turn ended; this covers a crash in between.)
- A call that was started and never finished gets an error result saying
  the run stopped while it was running, so it may or may not have taken
  effect. It is never run again.
- A call with no decision is checked (tool exists, input is JSON that
  names no key twice in any object, nests no deeper than 64 levels, has
  no number longer than 100 characters and matches the schema, policy) and the decision is recorded. A failed check is a deny
  with the reason. So is a call the policy allowed in an earlier run that
  never started: that allow is stale, since the session may hold more
  labels now, and the call is decided again. A person's allow stands.
- An allowed call is recorded as started, run under the tool timeout, and
  its result recorded. A Go error from the tool becomes an error result.
- A denied call gets an error result with the reason.
- A held call is left for later. The other calls of the turn still settle.

Retries: `overloaded` and `unreachable` wait `Backoff`, doubling each time,
or the provider's `RetryAfter` if it is later. `rate_limited` waits until
`RetryAfter`. A wait longer than `MaxWait`, or a retry beyond `Retries`,
ends the run with `model_error`. Other kinds are not retried.
`DefaultRetry` is 3 retries, a 2-second backoff and a 60-second maximum
wait.

When `ctx` is cancelled, the model call or the tool stops, and the run
ends with `canceled`. A call that was running gets the same result as
after a crash: it may or may not have taken effect, and it is not run
again.

### MCP servers

`agentloop/mcp` gives an agent the tools of one MCP server, which it runs
as a child process and speaks to over stdio. It uses the official Go SDK
(`github.com/modelcontextprotocol/go-sdk`, v1.8.0), which speaks MCP
2026-07-28 and falls back to older revisions. The request does not use it
yet.

```go
c, err := mcp.Connect(ctx, mcp.Server{
    Name:  "cleanroom",
    Argv:  []string{"/usr/local/bin/cleanroom", "mcp"},
    Env:   []string{"PATH=/usr/bin:/bin"},  // the whole environment
    Dir:   "/path/to/work",
    Stderr: operatorLog,
    SendLabels: false,
    Tools: map[string]mcp.ToolConfig{
        "mail_read": {Policy: trifecta.Tool{Class: "read", Labels: []string{"contaminated"}},
                      Digest: "9f2c…"},
    },
})
tools := c.Tools()       // []agentloop.Tool, by name
pins := c.Digests()      // keep these, and pass them next time
warnings := c.Warnings() // annotations that contradict the config
```

The configuration decides everything. `Tools` is an allowlist, and each
entry has the tool's trifecta class, labels and `always_hold`. A tool the
server lists that is not on the allowlist is ignored, and an allowlisted
tool the server does not list is an error. The server's annotations are
untrusted (the spec says so) and only linted: `Warnings` reports, for
example, a `read` tool with `destructiveHint` true, a `read` tool with
`openWorldHint` true whose labels lack `contaminated`, a `write` tool with
`openWorldHint` true, or a tool of any other class with `readOnlyHint`
true. The SDK reads an absent `readOnlyHint` as false, so false is never
linted.

Each allowed tool's definition is pinned when the client connects:
`Digest` is the SHA-256 in hex of its name, description, `inputSchema`
and `outputSchema` as canonical JSON (as for `InputDigest`). A tool
whose definition differs from the `Digest` its entry gives is a
`DriftError`, and an empty `Digest` accepts what the server lists.
`Reconnect` starts the server again and requires every definition to
match what the first connect pinned. To accept a changed definition the
caller connects again with the new digest. The model sees the pinned
description and `inputSchema` as the tool's spec.

The client declares no capabilities (an empty `ClientCapabilities`; a nil
one would advertise roots), sets no sampling, elicitation or roots
handler, negotiates no extension, and turns off the SDK's handling of
input requests. A result with `input_required` fails the call. Any
request from the server other than `ping` breaks the connection. The SDK
handles a server's requests apart from the results of calls, so the
order between the two is lost: the call in flight when the client
handles the request fails, a call that already had its result keeps it,
and no call runs after it until `Reconnect`.

A call is one `tools/call`. With `SendLabels`, its `_meta` carries the
session's labels in `com.kirahowe.agentloop/labels` (a sorted array,
never null) and the call id in `com.kirahowe.agentloop/call-id`. Without
it, the server learns neither. A result's `_meta` may carry
`com.kirahowe.agentloop/labels`, whatever `SendLabels` says, and those
labels are added to the configured ones; a result can never take one
away. A value that is not an array of known labels fails the call.

A result is text: its text blocks, joined by newlines. Image, audio,
`resource_link` and embedded resource content fail the call unless the
tool's `AllowContent` names the kind, and then the model sees a
placeholder such as `[image not shown: image/*, 1234 bytes]` instead. A
placeholder names only the top-level media type, from the registered
set, and nothing else of the server's text.
When the tool has an `outputSchema`, the model sees only the result's
`structuredContent`, validated against the schema and written again as
canonical JSON. The server's text is never shown, and missing or invalid
`structuredContent` fails the call. `isError` makes an error result; for
a tool with an `outputSchema` its text goes to `Log` and the model is
told only that the tool returned an error. A call that fails is shown to
the model as "The tool <name> failed.", with the reason in `Log`, as with
the envelope protocol. A JSON-RPC error from the server is such a
failure.

The server gets exactly `Argv`, `Env` and `Dir`, as a command tool does:
a nil `Env` is an empty environment, and nothing is inherited. Secrets
the server needs come from its own configuration. Its stderr goes to
`Stderr`, and one message from it may be at most `MaxMessageBytes` (1 MiB
by default). A listing may have at most 100 pages and 1000 tools, and
no cursor twice. The server runs in a process group whose leader is a
sentinel, as for a command tool. `Close` closes the server's stdin,
waits two seconds, sends SIGTERM to the group and the server, waits two
more, and kills the group and then the server by its pid (it is still
unreaped, so the pid is its own), which catches a server that left the
group. It returns within about eight seconds whatever the server does.
A call that is waiting for a `Reconnect` or `Close` to finish stops
waiting when its context ends.

A call runs under the agent's tool timeout. On a timeout or a cancel the
SDK sends `notifications/cancelled`, and the server stays connected for
the next call. A call that gets any error other than a JSON-RPC error
or its own context's end has lost the connection: the server died, its
stdout ended, or a message could not be read (an over-long one, say).
Its result says the server stopped while it ran, so it may or may not
have taken effect. A call also ends this way 200 milliseconds after the
server exits, even if a process it left behind holds its stdout. The
call is never retried. Every later call is not sent, and the model is
told the tool was not run because its server must be reconnected, until
the caller calls `Reconnect`. Nothing reconnects on its own.

## Telemetry

`agentloop/otel` turns a session's entries into OTLP/JSON traces with
`otel.Traces(entries, otel.Options{...})`. It is a pure function: no files,
no network, no OpenTelemetry SDK, and the same log gives the same bytes.
The log is not changed, so any old session can be exported.

It follows the OpenTelemetry GenAI semantic conventions at commit
`f3c6d0c9bb66ebd474668a3a0be25c61226f02c2` of
`open-telemetry/semantic-conventions-genai`. Every `gen_ai.*` attribute in
them is still at stability "development", so the commit is pinned in the
package documentation.

Each run is one trace. The root is `invoke_agent {agent}`, from
`run_started` to `run_ended` (a run with no end is closed at its last
entry, as an error). Under it sit `chat {model}` spans, one per assistant
message and one per failed model call, and `execute_tool {tool}` spans, one
per call, from its first decision to its `tool_result`. Holds, denials and
resolutions are `agentloop.policy.decision` events on the tool span.
`gen_ai.usage.input_tokens` counts cached tokens, which the log's
`InputTokens` does not, and the cache counts are added as their own
attributes. Labels go under `agentloop.labels`, never under `gen_ai`.

The log holds only append times, so a chat span starts at the time of the
entry before it. That includes the loop's own overhead.

Prompts, messages, tool inputs and results, policy reasons and error
messages are left out unless `Options.IncludeContent` is set. The caller
supplies the conversation id, the agent name and the provider, which the
log does not carry. Ids come from a seed, the run's index and the entry's
`seq`.

## The command

```
agentloop run REQUEST.json   run one agent to an outcome
agentloop acp REQUEST.json   serve it to an ACP client on stdin and stdout
```

The request:

```jsonc
{
  "provider": "anthropic",            // or "openai"
  "model": "claude-opus-5-5",
  "effort": "high",
  "base_url": "",                     // the provider's API by default
  "api_key_command": ["security", "find-generic-password", "-s", "anthropic", "-w"],
                                      // run to get the key; required
  "system": "…",
  "prompt": "…",
  "session": "/path/to/session.jsonl",
  "tools": [{"name": "…", "description": "…", "parameters": {}}],
  "tool_command": ["relay", "tool"],  // a call runs tool_command + [name]
  "tool_protocol": "envelope",        // or leave out for the plain protocol
  "policy": {                         // required when there are tools
    "labels": ["private"],            // added to the session at every run start
    "forbid_labels": [],              // labels the session may never hold
    "tools": {                        // an entry for every tool, and no others
      "memo_send": {"class": "write"},
      "web_fetch": {"class": "egress", "labels": ["contaminated"]},
      "rem":       {"class": "act", "labels": ["private"], "always_hold": false}
    }
  },
  "resolve": [                        // run only: decisions on held calls
    {"call_id": "toolu_1", "verdict": "allow", "input_sha256": "9f2c…"},
    {"call_id": "toolu_2", "verdict": "deny", "reason": "not now", "input_sha256": "41ab…"}
  ],
  "tool_env": {"RELAY_ROLE": "clerk"},// the whole environment a tool gets
  "tool_dir": "/path/to/work",
  "tool_timeout_seconds": 300,
  "timeout_seconds": 1800,            // the run ends with canceled at this deadline
  "limits": {"max_turns": 0, "max_tool_calls": 0, "max_tokens": 0},
  "retry": {"retries": 3, "backoff_seconds": 2, "max_wait_seconds": 60}
}
```

`policy` is required whenever `tools` is not empty. `{"unrestricted":
true}` turns the trifecta off, for examples and tests, and nothing else
may sit beside it; `"unrestricted": false` is refused. A real policy
needs `"tool_protocol": "envelope"`, an entry with a known class for
every tool and none for anything else, and only known labels.
`forbid_labels` becomes `Agent.Forbid` and `trifecta.Policy.Forbid`. A
request whose `policy.labels`, or any tool's labels, include a label it
forbids is unusable, since the session could not start or the tool
could never be called; so is `forbid_labels` beside `unrestricted`. Unknown
fields anywhere are refused, and so are fields spelled other than exactly
(`"POLICY"`) or given twice in one object, which encoding/json would
otherwise accept, the last one winning. `resolve` needs a real policy, and
a request without one refuses a session that `Session.Policed` reports:
an unrestricted request never carries on a policed session. With a real policy the agent's starting
labels are `policy.labels` and its policy is `trifecta.Policy`.

**Tool protocols.** A call runs `tool_command` with the tool's name
appended, in `tool_dir`, with exactly `tool_env`. With the plain
protocol, stdin is the call's input and stdout is the result (stderr when
stdout is empty; both, as an error, when the program fails). The plain
protocol shows stderr to the model, which is why a real policy cannot use
it. With the envelope protocol, stdin is one object:

```json
{"input": {"url": "…"}, "labels": ["contaminated", "private"], "call_id": "toolu_1"}
```

`labels` are the session's labels as the call runs, which include the
labels its decision added and everything the session held when it was
decided. Stdout must be exactly one object, with nothing after it but
whitespace:

```json
{"content": "text for the model", "is_error": false, "labels": ["contaminated"]}
```

`content` and `labels` are required and `is_error` defaults to false. The
result's labels are added to the session. If the program exits non-zero,
writes more than the output cap (1 MiB) to stdout, or writes anything
(a field spelled otherwise or given twice included)
but one such envelope with known labels, the model gets only "The tool
<name> failed." and the call is an error. The reason, with the program's
stderr, goes to the result entry's `log`, for the operator. A timeout or
a cancel is reported by the loop as for any tool.

`run` writes every session entry of the run to stdout as one JSON line,
and then a last line `{"type": "outcome", ...}` with the outcome, the
session path and the session's `labels` when the run ended (left out
when an outcome that never opened the session cannot read it) and, when
every model call of the run was priced, `cost_usd`: the run's cost in US
dollars, each call at the price of the model that served it. `cost_usd` is
left out, never 0, when the run made no call, a call's model has no price,
or the provider is not Anthropic. It is on this line only, not in the
`run_ended` entry. Nothing else goes to stdout. For a held run the outcome's `held` lists each call as
`{"id", "name", "input", "input_sha256"}`. A run with a refused result
lists it in `refused`, and one that did not start for holding a
forbidden label ends with reason `forbidden` and the labels in
`forbidden`.

`resolve` is applied before the run starts, through `Agent.Resolve`, and
the decision entries it writes come first on stdout. The whole list is
checked against the session first: a call the session does not have, a
digest that does not match, or a call that is not held (other than one
already resolved the same way) makes the request unusable, and nothing is
written. An allow the policy now denies outright (its tool is gone, say)
is recorded as a deny with the reason "The tool is no longer allowed.",
and the run carries on; the same allow sent again is that deny again. `acp` refuses `resolve`.

It exits 0 whenever it reaches an outcome, whatever the reason. It exits 2
when the request is unusable, before writing anything: bad JSON, an unknown
field, a schema that does not compile, a tool name the providers would
reject, a missing or invalid policy, a session path it cannot use, a
`resolve` entry the session cannot take. It exits 1 when the session log is
corrupt or another process holds it. On SIGTERM or SIGINT it cancels the
run, so the outcome line is still written.

`acp` reads the same request with different rules. It needs
`session_dir`, and each session the client opens is the log
`<session_dir>/<id>.jsonl`. It refuses `session`, `prompt` and
`timeout_seconds`: the client opens sessions and sends prompts, and a
deadline would end a turn as cancelled, which ACP keeps for the client.
Three fields are for `acp` alone, and `run` refuses them: `tool_kinds`
(`{"tool name": "read"}`, using ACP's tool kinds, so a client can show
each call), `title` (the agent's display name in `agentInfo.title`,
which a client can show; omitted when empty) and `new_sessions_only`.
With `"new_sessions_only": true` every session starts empty: initialize
advertises closing sessions but not loading, resuming or listing them,
and `session/load`, `session/resume` and `session/list` are answered
`-32601` (method not found) without reading the session directory. A
conversation then never takes on the history or the labels of an
earlier one. `false` is refused; leave the field out to allow
reopening. The client's `cwd` is
ignored, and so are its MCP servers, with a line on stderr: tools come
only from the request.

The request does not give the model's context window. When the first
session is opened, `acp` asks the provider to describe the model, once
per process, and keeps the answer. If that fails it writes one line to
stderr and carries on: the model is shown by its id, and since the window
is unknown no `usage_update` is sent. Every `session/new`, `session/load`
and `session/resume` response carries the model as ACP's session config
option in the `model` category: a select with id `model` whose only
value, and current value, is the request's model id, named by the
provider's display name (or the id when there is none). A client such as
acp-tui can show it. `session/set_config_option` that sets the model to
itself answers the options unchanged. Anything else, including switching
to another model, is refused with `-32602`. The model is fixed by the
request. Only JSON-RPC goes to
stdout. When the client closes stdin, stdout fails, or a SIGTERM or
SIGINT arrives, agentloop cancels every turn, waits for the runs to end
and release their sessions, and exits 0. A turn cancelled this way is
still answered `cancelled` when stdout works: closing stdin does not stop
agentloop writing. A second signal exits at once.

`session/new` creates the log file straight away, so a session can be
listed and loaded before anything has been said in it. `session/new`,
`session/load` and `session/resume` write the request's `policy.labels`
into the session's log as they open it. Without a policy, load and
resume refuse a session `Session.Policed` reports (`-32603`). A prompt with an embedded
`resource` or a `resource_link` adds `contaminated` to the session, since
its content could have come from anyone. Nothing a client sends removes a
label or changes the policy. `forbid_labels` holds in a conversation as
in a run: a prompt to a session that holds a forbidden label is answered
with `-32603` and the labels, without calling the model; a refused result
reaches the client only as the refusal, and stderr names the call; and
`session/new` refuses to seed a forbidden label.

**Held calls.** When a run ends `held`, the server asks the client about
each held call with `session/request_permission`: the tool call's id,
title, name, kind, its whole input as `rawInput`, and the session's labels
in `_meta["agentloop/labels"]` and their sources (as in a held outcome)
in `_meta["agentloop/label_sources"]`, with two options, `allow_once` and
`reject_once`. The turn waits for the answers. Only the `allow_once`
option allows; a reject, an option that was not offered, a `cancelled`
outcome, an error or a lost connection is a deny, recorded through
`Agent.Resolve` with the reason "A person did not approve this call.".
An allow the policy now denies is recorded as a deny. Then the run
carries on. A `session/cancel` while a request is open denies it, even if
an allow arrives after the cancel, and the turn is answered `cancelled`. A held call found in a
loaded session is asked about at the next prompt, never run on its own. `session/load`
sends the session's history and then answers; `session/resume` answers
without it. Both take the session's lock, so a session open in another
process, or on another connection in the same process, is refused with
`-32603` "session in use". A session already open on the same
connection is taken up again as it is when it is idle, and refused the
same way while a turn runs. An id that agentloop could not have made (a
lowercase UUID) is refused as unknown (`-32002`) before any path is built
from it, as is a valid id with no log. `session/close` cancels the turn
in flight, if any, and answers once the session is released.
`session/list` lists every log in `session_dir`, newest first, with the
time of its last entry. ACP requires a cwd for each session, and since
the client's is ignored, agentloop reports the directory the tools run
in: `tool_dir`, made absolute, or agentloop's own working directory when
`tool_dir` is empty. A `cwd` filter therefore matches every session or
none.

What a client sees of a run comes from the run's events, translated by
pure functions in `acp/` (`Translator.Step` for a live turn, `Replay` for
`session/load`):

| agentloop | ACP `session/update` |
|---|---|
| text delta, or text not streamed | `agent_message_chunk`, id `a<seq>` |
| reasoning delta | `agent_thought_chunk`, id `a<seq>` |
| a refusal the provider explained outside the text | one more `agent_message_chunk` with the explanation |
| a tool call in an assistant message | `tool_call`, `pending`, with the request's kind for the tool (`other` if none) |
| a held call | `session/request_permission` (above) |
| `call_started` | `tool_call_update`, `in_progress` |
| `tool_result` | `tool_call_update`, `completed` or `failed`, the content as text, `rawOutput` `{content, is_error}` |
| an assistant message, when the provider gave the window | `usage_update`, used = the turn's input, cache and output tokens, size = the window |
| `model_error` | `notice`, only to a client that advertises notices |
| a user message | nothing live (the client showed it); `user_message_chunk`, id `u<seq>`, on load |

The outcome of the turn answers `session/prompt`:

| outcome | answer |
|---|---|
| `completed` | `end_turn` |
| `max_tokens`, or `limit` on tokens | `max_tokens` |
| `limit`, any other | `max_turn_requests` |
| `refused` | `refusal` |
| `canceled` | `cancelled` |
| `model_error` | error `-32603`, data: the error's kind, status, message and times |
| `held` | error `-32603`, data: the held calls (only when the client could not be asked) |
| `forbidden` | error `-32603`, data: the labels the session may not hold |

Two `_meta` keys carry what ACP has no field for:
`_meta["agentloop/labels"]` on every `tool_call`, `tool_call_update` and
`usage_update` holds the session's labels as of that update (a replayed
call shows them as of its last entry). When the labels change and nothing
else sent for the change carries them (a seed, a resource in a prompt, a
resolution), a `session_info_update` with only that `_meta` follows; one
also follows `session/new` and `session/resume` of a labelled session,
and ends the replay of a labelled one. And `_meta["agentloop/retry"]` on a notice for a
model call that will be retried holds `kind`, `wait_ms`, `retry_at`,
`attempt` and, when there was one, the HTTP `status`, so a client can
count down to the retry.

After `refusal`, ACP says the refused prompt and what follows it are left
out of the next prompt. agentloop's log is append-only and keeps the
refused turn, so the model sees it next time.

A session is held under an exclusive lock on `<session>.lock` while a
process uses it. The OS releases the lock when the process dies, however it
dies.

The key comes from `api_key_command`, which agentloop runs itself: no
shell, an empty stdin, and an environment of only `PATH` and `HOME`, taken
from agentloop's own. The command has 30 seconds. Its stdout, trimmed, is
the key. Nothing is read from the environment, so there is no variable to
set; a keychain helper like the one above works as it is.

If the command cannot start, exits non-zero, times out or prints nothing,
`run` calls nothing and reports `model_error` with kind `unauthorized`
and a message such as `The api_key_command exited 1: item not found.`
The message may carry the end of the command's stderr, never its stdout.
`acp` prints the same message and exits 2 before reading anything from
the client. A request with no
`api_key_command`, an empty one, or one with an empty program name or a NUL
character is unusable and exits 2.

The key reaches the model and nothing else. It is not logged, not written
to the session and not passed on: tools get only `tool_env`.

## What it does not guarantee

- The trifecta rule is only as good as what it is told. agentloop trusts
  the request's classes and static labels, and the labels a tool's
  envelope returns; a tool that under-labels its output, or an egress
  tool classed `read`, gets through. It also treats the model provider as
  an accepted destination: everything a session sees is sent to it.
- A tool is killed with its process group. The group's leader is a
  sentinel (`/bin/cat` reading a pipe) that agentloop reaps only after it
  has swept the group, so the sweep can never reach a group that reused
  the id. The program itself is also killed by its pid, so it cannot
  escape by leaving the group, but a process it starts that leaves the
  group (with `setsid`, say) can outlive its call. On Linux a cgroup could close
  that gap; macOS has no equivalent.
- When agentloop itself is killed with SIGKILL, a tool that is running
  carries on as an orphan. Its call is still never run a second time.
- A signal that arrives before the Go runtime has started kills the
  process with nothing written. Running the same request again is safe.
- Replay is checked against recorded wire formats and the SDKs' types.
  Whether each provider accepts every replayed shape is confirmed only by
  the live tests (`make live`), which spend money and do not run by
  default.
