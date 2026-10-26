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
   The model clients never take a proxy from the environment: with no
   `HTTPClient` given, they use a client whose transport has `Proxy` set to
   nil, so `HTTPS_PROXY` and its relatives change nothing.
2. **One gate for every effect.** A tool call reaches its tool only after
   three checks, in order: the tool exists, the input matches its schema,
   and the policy, if there is one, allows it. The policy sees the call and
   the session's labels, and can allow or deny the call and add labels.
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
| `agentloop/otel` | A pure converter from a session log to OpenTelemetry GenAI traces, as OTLP/JSON |
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
`tool_result` entry and never shown to the model. Its `Usage` lists the
model calls the tool made for the run, each with the model that served
it. They count toward the run's usage and its `MaxTokens` limit, and the
`tool_result` entry keeps them, so the session's usage rebuilds from the
log. A refused result keeps its usage.

A tool that also implements `CallTool` is called through `CallWith`,
which gives it the call's id and the session's labels as the call runs
(those include the labels the call's own decision added). A tool that implements `InputChecker`
checks input of its own after the schema and before the policy; a
reason it returns denies the call. `CompileInput` compiles a tool's
parameters as the agent does, so a tool can check input meant for
another.

### Policies

```go
type Verdict string // "allow" or "deny"

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
    Labels []string // sorted
}
```

A nil policy allows every call to a tool the agent was given. The tools
themselves are the grant. The policy adds conditions on top. A verdict
that is neither `allow` nor `deny` is a deny.

Labels are short provenance names such as `private`, `contaminated` or
`mail:stranger@example.com`. `CheckLabel` accepts 1 to 100 bytes that start
with a lower-case letter or digit and continue with lower-case letters,
digits and `_ . : @ / + -`. Case, spacing, invisible, bidi and lookalike
variants of a name therefore cannot exist, and a forbid list cannot miss a
label by one byte. A label can be shown, logged and compared safely. The core gives no label a meaning. A policy
can add labels when it allows a call, before the result exists. A tool
can add labels with its result. Labels are never removed. Where they came
from is recorded: `Session.LabelSources` gives, for each label the
session holds, the entry that first added it, as `{"kind":
"seed"|"tool", "tool", "call_id", "seq"}` (`tool` and `call_id` for a
tool's decision or result).

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
  `ForbiddenDecision`.
- A tool result that carries a forbidden label is refused. The model is
  told only `RefusedResult` ("This result was refused: it carries a
  label this session may not hold."), the session does not take the
  label, and nothing the tool returned is kept, not even in `log`, which
  says only that the result was refused and why. `Outcome.Refused`
  lists each refused result of the run as `{"call_id", "tool",
  "labels"}`, whatever reason the run ended with, so whoever runs the
  agent can tell the operator.

### Displays

A `Result` may carry a `Display`: a title, a Markdown body, labels and
sources (each a title and a ref such as `web:https://...`), for the
operator and never the model. The loop keeps it in the `tool_result`
entry's `display`, and the conversation never holds any of it:
`Messages` is built from the result alone. Its labels become the union of
its own, the result's and the session's as the call
ran, so a display is never labelled less than the call that made it. The
session does not take a display's labels. A display that would carry a
label the agent forbids is dropped, and the entry's log says so. A
refused result keeps no display.

With `Agent.DisplayRefPrefix` set (the request's `display_ref_prefix`),
the display gets a ref, the prefix and `DisplayID(call id)`: the call id
in lower case when it is a short run of letters, digits, `_` and `-`, and
otherwise `call-` and 16 hex digits of its sha256. The model's content
gains one line, `The operator was also shown [[<ref>]].` (without a
prefix, `The operator was also shown more than this.`), which holds
nothing the display said. relay keeps each display as a doc under that
ref, and an ACP client gets it as tool call content (see the command).

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
| `labels` | labels added from outside any call, and their `source`, `seed` | when a run's starting labels, or `AddLabels`, add one the session lacks |
| `message` | a `Message`, and for an assistant turn its `Stop`, `Usage` and serving model | when a user prompt is added or a response arrives |
| `decision` | call id, tool name, verdict, reason, and the labels an allow adds | when a call is checked |
| `call_started` | call id | immediately before the tool runs |
| `tool_result` | the `ToolResult`, its labels, the operator's `log`, and the tool's model `usage` | when a call finishes, or is refused |
| `model_error` | the `ModelError`, and whether the loop will retry | on each failed model call |
| `run_ended` | the `Outcome` | when `Run` returns |
| `pins` | digests of tool definitions, by tool name | when a run starts with `Agent.Pins` the session lacks |

A session rebuilds its state from the entries: the conversation, the
labels, the usage, and the state of every tool call (decided, started,
finished). Tool results are logged one by one as they finish, so a crash
between two calls loses neither. In the conversation, results that follow
an assistant turn are gathered into one user message.

`FileStore.Load` ignores a final line that is not a complete entry, which
is what a write cut short by a crash leaves, and the next append cuts it
off. A bad line anywhere else is an error, and so are sequence numbers that
do not run 1, 2, 3 and so on.

`Agent.Pins` maps tool names to digests of their definitions, which are
opaque to the core (the `mcp` package makes them). A run records the
pins the session lacks in one `pins` entry. A run whose pin for a tool
differs from the session's does not start: `Run` returns a `*PinError`
and writes nothing. A pin is never changed, so a log that pins one tool
to two digests does not open. To accept a changed definition, start a
new session.

Every entry goes through JSON before the session applies it, so the live
state and the state rebuilt from the same log are always equal, and a
session shares no memory with its callers. Both that round trip and
reading entries back from a store use `encoding/json`, not the strict
reader: the log is agentloop's own (see The loop).

### Reading JSON

relay, agentloop and relay-tui read every message one of them gets from
another by the same rules, so a message means the same thing to all
three or none of them takes it (relay#58). In agentloop that reader is
`internal/strictjson`, built on `encoding/json/v2`. It accepts:

- RFC 8259 JSON in valid UTF-8, with no byte order mark and no unpaired
  surrogate escape;
- exactly one value, with only whitespace after it;
- no member name twice in one object, at any depth, compared after
  unescaping;
- member names matched exactly and case-sensitively, so `"Labels"` is
  another member, not `labels`;
- nesting of at most 128 levels, number literals of at most 1000
  characters, and exponents between -9999 and 9999 by value (leading
  zeros allowed).

Unknown members are ignored, except in a tool envelope, a request and
the `mcp-tools` input, which refuse them. A value of no fixed shape keeps
each number as written.

`encoding/json` (v1) does none of this: it folds case when it matches a
name to a struct field (`ſ` reaches a field named `s`), takes one of two
repeated names, and replaces invalid UTF-8. agentloop's own code does not
use it to read what another process wrote. The strict reader covers the ACP
client's lines (see The command), tool envelopes, request files,
`mcp-tools` input, MCP servers' stdio messages, and the canonical JSON that tool definition digests are taken of. The table of hostile lines that all three
components test against is `testdata/strict-json/cases.jsonl`, a copy
of the one in relay's `testdata/strict-json`, which is the source.

Provider responses are read by the providers' SDKs, which keep their
own decoders: relay's threat model trusts the provider. What a provider
returns for a turn must still pass the reader, with tighter bounds,
before it is logged (see The loop). Remote MCP servers are read by the
MCP SDK's HTTP transport, which decodes each message before agentloop
sees it, so their messages are not screened by the strict reader. Only
agentloop reads that traffic, so no other component can read it
differently.

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
    MaxTokens    int64 // Usage.Total per run, tools' model calls included
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
    Reason    Reason      // completed, max_tokens, refused, limit, model_error, canceled, forbidden
    Text      string      // the last assistant text
    Error     *ModelError // for model_error
    Limit     string      // which limit, for limit
    Forbidden []string    // the labels, for forbidden
    Refused   []Refusal   // results refused for carrying a forbidden label
    Turns  int
    Usage  Usage
}
```

A run:

1. Writes `run_started`, and a `labels` entry if `Agent.Labels` adds any.
2. Settles the calls left open by the last assistant turn (below).
3. Adds the prompt as a user message, if there is one. With no prompt and
   nothing waiting for the model, it ends with the last turn's outcome,
   usually `completed`, and the last assistant text. A run that ends
   `canceled` before this step does not add its prompt, so the caller
   passes it again.
4. Repeats: checks the limits, calls the model (retrying under the retry
   policy), and records the response. On `tool_use` it settles the turn's
   calls and goes on. On `pause` it settles them as cut off and goes on.
   On `end_turn`, `max_tokens` or `refusal` it settles them as cut off and
   ends with `completed`, `max_tokens` or `refused`. So at rest every call
   has a result.
5. Writes `run_ended`.

A response that repeats a tool call id, or has a call with no id, is not
recorded: the run ends with `model_error`, kind `invalid_request`. So is
a response whose tool call input, or the provider's record of the turn,
is not JSON in the strict dialect (a name given twice, invalid UTF-8),
nests deeper than 64 levels or has a number longer than 100 characters:
relay reads the session log with its strict reader, so every line
agentloop writes there must pass it. agentloop reads its own log back
with `encoding/json`. Logs written before relay#58 can
hold a complete line with a repeated name or invalid UTF-8 inside a tool
call's input or the provider's data. Such a line still loads, and an
append never cuts it off as a torn write: only a line that fails
`encoding/json` counts as torn.

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
  labels now, and the call is decided again.
- An allowed call is recorded as started, run under the tool timeout, and
  its result recorded. A Go error from the tool becomes an error result.
  Before it runs, the tool-call and token limits are checked: a tool can
  spend tokens, so the token limit can be
  reached in the middle of a turn. A call past a limit gets an error
  result saying so, keeps its decision's labels, and the run ends with
  `limit`.
- A denied call gets an error result with the reason.

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
2026-07-28 and falls back to older revisions. The command uses it for a
request's `mcp_servers` (see [MCP tools](#the-command)).

```go
c, err := mcp.Connect(ctx, mcp.Server{
    Name:  "cleanroom",
    Argv:  []string{"/usr/local/bin/cleanroom", "mcp"},
    Env:   []string{"PATH=/usr/bin:/bin"},  // the whole environment
    Dir:   "/path/to/work",
    Stderr: operatorLog,
    SendLabels: false,
    Restart: true,
    Tools: map[string]mcp.ToolConfig{
        "read_mail": {Alias: "mail_read",
                      Labels: []string{"contaminated"},
                      Digest: "9f2c…"},
    },
})
tools := c.Tools()       // []agentloop.Tool, by alias
pins := c.Digests()      // by alias; keep these, and pass them next time
```

The configuration decides everything. `Tools` is an allowlist keyed by
the server's names, and each entry has the labels every result of the tool
carries, and the `Alias` the agent knows it by (the server's name when
empty). The model and the policy see only the alias. A tool the server
lists that is not on the allowlist is ignored, and an allowlisted tool the
server does not list is an error. The server's annotations are untrusted
(the spec says so) and are not used at all.

Each allowed tool's definition is pinned when the client connects:
`Digest` is the SHA-256 in hex of its name, description, `inputSchema`
and `outputSchema` as canonical JSON. Annotations
are left out: they decide nothing, and the SDK would have to encode them
again, which would tie the digest to the SDK's version. A tool
whose definition differs from the `Digest` its entry gives is a
`DriftError`, and an empty `Digest` accepts what the server lists.
`Reconnect` starts the server again and requires every definition to
match what the first connect pinned. To accept a changed definition the
caller connects again with the new digest. The model sees the pinned
description and `inputSchema` as the tool's spec.

The client declares no capabilities (an empty `ClientCapabilities`; a nil
one would advertise roots), sets no sampling, elicitation or roots
handler, negotiates no extension, and turns off the SDK's handling of
input requests. A result with `input_required` fails the call.

The client reads the server's stdout itself, as one stream of JSON
values, and decodes each with the SDK's own decoder, so what it checks
is what the SDK acts on. It checks each message before the SDK sees it,
because the SDK handles a server's requests on a goroutine of its own
and would lose their order with the results. These break the
connection: a request from the server other than `ping`, a notification
whose method does not start with `notifications/`, a batch, a frame that
is not a JSON-RPC message, a message the strict reader refuses (see
Reading JSON), and a message that shares a line with the one before. The call
whose result follows such a message fails, and no call runs after it
until `Reconnect`. `Restart` does not start such a server again.

A call is one `tools/call`. Its `_meta` always carries the call id in
`com.kirahowe.agentloop/call-id`. With `SendLabels` it also carries the
session's labels in `com.kirahowe.agentloop/labels` (a sorted array,
`[]` when there are none). Without it, the server does not learn them:
they say the session holds private data. A result's `_meta` may carry
`com.kirahowe.agentloop/labels`, whatever `SendLabels` says, and those
labels are added to the configured ones; a result can never take one
away. A value that is not an array of valid labels fails the call.

A result is text: its text blocks, joined by newlines. Image, audio,
`resource_link` and embedded resource content fail the call unless the
tool's `AllowContent` names the kind, and then the model sees a
placeholder such as `[image not shown: image/*, 1234 bytes]` instead. A
placeholder names only the top-level media type, from the registered
set, and nothing else of the server's text. The one exception is an
allowed embedded resource that holds text rather than a blob: the model
sees its text, as if it were a text block, and nothing of its URI or
MIME type.
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
call is never retried. Without `Restart`, every later call is not sent,
and the model is told the tool was not run, until the caller calls
`Reconnect`. With `Restart`, the next new call starts the server again
first, within `StartTimeout` (30 seconds by default), and every allowed
definition must match its pin. A restart that
finds a changed definition fails that call and is not tried again; one
that fails otherwise (the program will not start, say) fails that call
and is tried again for the next.

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
per call, from its first decision to its `tool_result`. Policy decisions are `agentloop.policy.decision` events on the tool span.
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
  "tools": [
    {"name": "…", "description": "…", "parameters": {}},
    {"name": "mail_read", "protocol": "mcp-stdio", "server": "cleanroom", "tool": "read_mail",
     "allow_content": ["resource"]}   // an MCP tool, by its alias
  ],
  "tool_command": ["relay", "tool"],  // a call runs tool_command + [name]
  "mcp_servers": {
    "cleanroom": {
      "command": ["/usr/local/bin/cleanroom", "mcp", "--token-file", "{credentials_file}"],
      "credentials_command": ["security", "find-generic-password", "-s", "cleanroom", "-w"],
      "send_labels": true             // false by default
    }
  },
  "tool_protocol": "envelope",        // or leave out for the plain protocol
  "policy": {                         // optional; without it every tool is allowed
    "labels": ["private"],            // added to the session at every run start
    "forbid_labels": [],              // labels the session may never hold
    "tools": {                        // labels a call to the tool adds; entries are optional
      "web_fetch": {"labels": ["contaminated"]},
      "mail_read": {"labels": ["private", "from a mail"]}
    }
  },
  "tool_env": {"RELAY_ROLE": "clerk"},// the whole environment a tool gets
  "tool_dir": "/path/to/work",
  "tool_timeout_seconds": 300,
  "timeout_seconds": 1800,            // the run ends with canceled at this deadline
  "limits": {"max_turns": 0, "max_tool_calls": 0, "max_tokens": 0},
  "retry": {"retries": 3, "backoff_seconds": 2, "max_wait_seconds": 60},
  "display_ref_prefix": "shown:2026-10-18/" // where displays are kept; see Displays
}
```

**The policy.** `policy` is optional. Without one every tool the request
gives is allowed and the session starts with no labels. With one, the
agent's starting labels are `policy.labels`, `Agent.Forbid` is
`forbid_labels`, and the agent's policy allows every call and adds the
labels of the tool it calls (`policy.tools`, where an entry is optional
and must name one of the tools). Labels use a restricted character set (see Policies). A
request whose `policy.labels`, or any tool's labels, include a label it
forbids is unusable, since the session could not start or the tool could
never be called. With command tools a policy also needs `"tool_protocol":
"envelope"`, since only the envelope returns the labels of a result.

The request is read by the strict reader (see Reading JSON): unknown
fields anywhere are refused, and so are fields spelled other than exactly
(`"POLICY"`) and fields given twice in one object. The fields that older
versions took are refused with a message that says they were removed:
`resolve`, `quarantine`, `workbench`, `policy.unrestricted`, and `class`,
`always_hold` and `consent` in a tool's policy entry. agentloop no longer
holds a call for a person, classes tools, or runs a reader or a
workbench. What an agent may do is decided by the tools it is given and
the walls around them.

**MCP tools.** A tool with `"protocol": "mcp-stdio"` comes from the MCP
server `server` in `mcp_servers`, where it is called `tool`. Its `name`
is the alias the model and the policy see. It gives no `description` or
`parameters`: the server gives them. Every server must give at least
one tool, and no server tool may be given twice. `digest`, when given,
is the tool's definition digest (lowercase hex, as on stderr and in the
`pins` entry), and a server that lists another definition fails the
start, at a restart too. `allow_content` may
name `image`, `audio`, `resource_link` and `resource`; see [MCP
servers](#mcp-servers) for what each lets through. Without it, a result
with any non-text content fails the call. An MCP tool's labels come from
the policy's entry for it. `tool_protocol` is needed only when there are
command tools too. A request with MCP
servers and no `tool_timeout_seconds` gets 120 seconds, for every tool,
since a server can leave a call unanswered for ever.

The servers start, in name order, before the session is run, since the
model needs their tools' definitions for its first turn. Each runs once
for the whole run with `Restart` on, in `tool_dir`, with PATH and HOME
from agentloop's environment and nothing else, and writes its stderr to
agentloop's. A server whose start takes over 30 seconds, lacks a tool
the request gives, or whose schema does not compile fails the start.
The digest of each tool's definition goes to stderr and into the
session as a `pins` entry. A run on a session that pinned a different
digest stops before anything is written: a changed definition needs a new
session. In `acp` a refused prompt leaves the session as it was.

When the servers cannot be set up, `run` writes no entry and ends with
reason `tools_unavailable` and a `tools_error` that says why:

```json
{"type": "outcome", "reason": "tools_unavailable", "turns": 0, "usage": {…},
 "session": "/path/to/session.jsonl", "labels": ["private"],
 "tools_error": {"server": "files", "tool": "file_read", "pinned": "f11c…", "got": "36c6…",
                 "message": "The definition of the tool file_read is not the one this session pinned (…). The session cannot go on with it: use a new session."}}
```

`server`, `tool`, `pinned` and `got` are there when they apply: a
server that did not start gives `server` and `message`. `labels` are
the session's. A signal before or while the servers start ends the run
with reason `canceled` in the same way, and stops them.

A server never gets a secret through its environment. A
`credentials_command`, run like `api_key_command` (PATH and HOME, empty
stdin, 30 seconds), prints the secret. agentloop writes its stdout,
unchanged, to a file of mode 0600 in a directory of mode 0700 made for
the run, and removes both when it exits, also on the second signal that
makes `acp` exit at once. The directory, `agentloop-mcp-<digits>` in
the temp directory, holds a `pid` file with agentloop's pid. A later
start removes such a directory left behind by a process that is gone
(one killed with SIGKILL, say), but only when its name, owner, files and
pid file all have that shape and no process has the pid. `{credentials_file}` in any
argument of `command` but the first is replaced by that file's path. A
`credentials_command` needs the placeholder and the placeholder needs a
`credentials_command`. The secret is never in an argument or on stderr;
a failing command is reported with its exit status and the end of its
stderr only.

**Remote MCP servers.** A server with a `url` in place of `command` is
reached over MCP's streamable HTTP transport, with the Go SDK's client,
and its tools have `"protocol": "mcp-http"` (an `mcp-stdio` tool must
name a server with a command, and an `mcp-http` tool one with a url). The
URL is https, or http to a loopback host (`localhost`, `127.0.0.0/8`,
`::1`), with no user name, password or fragment. Its
`credentials_command` needs no placeholder: what it prints, trimmed of
white space, is the bearer token, which stays in memory and goes in the
`Authorization` header of requests to the server's origin and nowhere
else. A token with a space, a control character or a character outside
ASCII fails the start. When the server answers 401, the command is run
again (at most once every 10 seconds) and the request, which the server
did not act on, is sent once more with the new token. A redirect is
followed only within the server's origin. No proxy is used, whatever the
environment says. agentloop opens no standalone event stream and never
resumes a broken one, and a response body that is not an event stream,
like each event of one, is cut at the message cap (1 MiB). The SDK keeps
`Mcp-Session-Id` and sends `MCP-Protocol-Version`. A call the server
refuses because it ended the session was not run, and the model is told
so; the next call connects again, with every definition as pinned. The
checks a stdio frame gets for duplicate keys and for sharing a line do
not apply: the SDK reads each HTTP message once, and what it read is what
agentloop acts on. A request from the server still breaks the connection.

**Narrowed arguments.** An MCP tool may give `fixed_args`, an object of
argument names and the JSON values each call gets, and `hidden_args`, a
list of names. Each must be a property of the tool's input schema, and a
hidden argument cannot be required (fix it instead) or fixed as well; a
server that lists a schema breaking these fails the start. The model is
shown the schema without them. A call that names a fixed or hidden
argument, or a key that differs from one only in case (Go's
encoding/json, for one, matches field names that way), is refused with a
message saying so, before the policy decides, and the server is sent
nothing. Otherwise the fixed values are added and the whole is checked
against the server's own schema before it is sent. The pinned digest is
the server's definition's, not the narrowed one's.

`agentloop mcp-tools` reads `{"url": "...", "token": "..."}` (token
optional) on stdin, lists every tool the remote server has, and prints
`{"tools": [...]}`: each tool's `name`, `title`, `description`,
`annotations`, `input_schema`, `output_schema` and `digest`, sorted by
name. It is how an operator sees what a server offers before allowing
any of it.

**Tool protocols.** A call runs `tool_command` with the tool's name
appended, in `tool_dir`, with exactly `tool_env`. With the plain
protocol, stdin is the call's input and stdout is the result (stderr when
stdout is empty; both, as an error, when the program fails). The plain
protocol shows stderr to the model, which is why a request with a policy
cannot use it. With the envelope protocol, stdin is one object:

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
result's labels are added to the session. An envelope may also carry
`display`, something for the operator alone (see Displays):

```json
{"content": "...", "labels": [], "display": {"title": "Venues", "body": "markdown",
  "labels": ["contaminated"], "sources": [{"title": "List", "ref": "web:https://..."}]}}
```

`title`, `body` and `labels` are required, and `sources` may be left out.
A display that breaks the bounds of `agentloop.CheckDisplay` (a title of 1
to 200 characters, a body of at most 200,000 bytes, at most 50 sources,
each with a title and a one-line ref) fails the call like any other bad
envelope. If the program exits non-zero,
writes more than the output cap (1 MiB) to stdout, or writes anything
but one such envelope with valid labels, read by the strict reader (see
Reading JSON; a field spelled otherwise or given twice fails it), the
model gets only "The tool
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
`run_ended` entry. Nothing else goes to stdout. A run with a refused result
lists it in `refused`, and one that did not start for holding a
forbidden label ends with reason `forbidden` and the labels in
`forbidden`.

It exits 0 whenever it reaches an outcome, whatever the reason. It exits 2
when the request is unusable, before writing anything: bad JSON, an unknown
field, a schema that does not compile, a tool name the providers would
reject, an invalid policy, a session path it cannot use. It exits 1 when the session log is
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
relay-tui can show it. `session/set_config_option` that sets the model to
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
into the session's log as they open it. Nothing a client sends adds or
removes a label or changes the policy: a prompt's content brings no
labels. `forbid_labels` holds in a conversation as
in a run: a prompt to a session that holds a forbidden label is answered
with `-32603` and the labels, without calling the model; a refused result
reaches the client only as the refusal, and stderr names the call; and
`session/new` refuses to seed a forbidden label.

A session opened by `session/load` sends the session's history and then
answers; `session/resume` answers without it. Both take the session's
lock, so a session open in another process, or on another connection in the same process, is refused with
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
none. agentloop never sends `session/request_permission`: no call waits
for a client.

Every line the client writes must pass the strict reader (see Reading
JSON) as one object before the ACP SDK reads it, since the SDK decodes
with `encoding/json` v1. Passing the strict reader is not enough on its
own, because v1 also matches member names without regard to case: with
no exact match, `"Method"`, `"ID"` or `"paramſ"` reach the fields named
`method`, `id` and `params`, and `{"cwd":"a","Cwd":"b"}` reads as cwd
`b`, where relay and relay-tui see unknown members. So the screen also
removes every member, at any depth, whose name folds to a name the SDK
decodes without being spelled exactly as one. To relay and relay-tui such
a member is unknown and ignored, so once it is gone the SDK reads the
same members they do, and a line all three accept means the same to all
three (the shared table's case variants are tested through the screen
against the method the table expects). Those names come from reflection
over the SDK's message types, and a test checks them against every json
tag in the SDK's source. The check goes by name, not by position, with
one exception: it does not look inside the value of a member named
exactly `_meta` or `data`. The SDK keeps `_meta` as raw JSON and decodes
`data` into `any` or a string wherever they appear, so v1 folds no names
there, and a client's `{"_meta":{"Title":"x"}}` passes whole (a test
checks those field types in the SDK). Elsewhere a folded name is removed
even from free-form data the SDK keeps in a map, such as an
elicitation's content, which agentloop never reads. A line with nothing
to remove reaches the SDK byte for byte; a line with something removed
keeps every other member and value as written, and loses only the
whitespace between tokens. The removal is noted in the log.
A line that fails the strict reader never reaches the SDK. A request that fails, if it has one `id` (a string or a number) and a
`method` that can still be read, gets a JSON-RPC parse error (`-32700`)
for that id. Any other line that fails is dropped and noted in the log,
a response from the client included: its id is one the agent chose, so
an error with it would read to the client as the answer to a request of
its own. A line longer than 10 MiB ends the connection, as it would in
the SDK.

What a client sees of a run comes from the run's events, translated by
pure functions in `acp/` (`Translator.Step` for a live turn, `Replay` for
`session/load`):

| agentloop | ACP `session/update` |
|---|---|
| text delta, or text not streamed | `agent_message_chunk`, id `a<seq>` |
| reasoning delta | `agent_thought_chunk`, id `a<seq>` |
| a refusal the provider explained outside the text | one more `agent_message_chunk` with the explanation |
| a tool call in an assistant message | `tool_call`, `pending`, with the request's kind for the tool (`other` if none) |
| `call_started` | `tool_call_update`, `in_progress` |
| `tool_result` | `tool_call_update`, `completed` or `failed`, the content as text, `rawOutput` `{content, is_error}` |
| an assistant message, when the provider gave the window or the model is priced | `usage_update`, used = the turn's input, cache and output tokens, size = the window, and the session's cost |
| a `tool_result` with model usage (a tool's own model calls) | `usage_update` as of the last assistant message, with the new cost |

The cost on a `usage_update` is the session's, from its first entry:
each assistant message at the model's price, and each model call a tool
recorded at the price of the model that served it (`Options.PriceOf`,
the provider adapter's table). If any of those models has no price, no
cost is sent from then on, never a partial one. A reopened session
counts its log the same way, so live, replay and reopen agree.
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
| `forbidden` | error `-32603`, data: the labels the session may not hold |

Two `_meta` keys carry what ACP has no field for:
`_meta["agentloop/labels"]` on every `tool_call`, `tool_call_update` and
`usage_update` holds the session's labels as of that update (a replayed
call shows them as of its last entry). When the labels change and nothing
else sent for the change carries them (a seed, say), a `session_info_update` with only that `_meta` follows; one
also follows `session/new` and `session/resume` of a labelled session,
and ends the replay of a labelled one. A result with a display has one more content item after its text:
a text block holding the display's title and body, with the whole
display, `{"title", "body", "labels", "sources", "ref"}`, in its
`_meta["relay/display"]`, so a client can show it apart from the
result. The block's own text is inert (`Inert`): characters that draw as
nothing show as `\u{XXXX}`, links are defanged, and the whole is in a
code fence nothing in it can close. A client that renders it as
markdown fetches no image and shows no live link. And `_meta["agentloop/retry"]` on a notice for a
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

- Labels are only as good as what agentloop is told. It trusts the
  request's labels and the labels a tool's envelope returns; a tool that
  under-labels its output gets through `forbid_labels`. It also treats the
  model provider as an accepted destination: everything a session sees is
  sent to it.
- agentloop does not hold a call for a person, and it does not stop a
  tool from doing what it can do. Whatever walls keep a tool from going
  where it should not belong to whoever runs agentloop.
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
