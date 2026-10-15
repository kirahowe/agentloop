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
| `agentloop/command` | A `Tool` that runs a program: the input on stdin, the output as the result |
| `agentloop/fake` | A scripted `Model` for tests |
| `cmd/agentloop` | The command: runs one agent from a JSON request |

An MCP client package, giving the agent the tools of one MCP server and
nothing else, comes next. It is how an agent gets put behind a gateway
such as cleanroom.

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
result is recorded.

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

**Holds.** A held call waits for a person. The run ends with reason `held`
and lists the held calls. `Session.Resolve(callID, Decision)` records the
person's decision, and the next `Run` carries on: it runs the call or
reports the denial, then goes back to the model.

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
| `message` | a `Message`, and for an assistant turn its `Stop`, `Usage` and serving model | when a user prompt is added or a response arrives |
| `decision` | call id, tool name, verdict, reason, labels | when a call is checked, or when a held call is resolved |
| `call_started` | call id | immediately before the tool runs |
| `tool_result` | the `ToolResult` and its labels | when a call finishes, or is refused |
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

1. Writes `run_started`.
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
recorded: the run ends with `model_error`, kind `invalid_request`.

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
- A call with no decision is checked (tool exists, input matches the
  schema, policy) and the decision is recorded. A failed check is a deny
  with the reason.
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
  "tool_command": ["relay", "tool"],  // a call runs tool_command + [name], input on stdin
  "tool_env": {"RELAY_ROLE": "clerk"},// the whole environment a tool gets
  "tool_dir": "/path/to/work",
  "tool_timeout_seconds": 300,
  "timeout_seconds": 1800,            // the run ends with canceled at this deadline
  "limits": {"max_turns": 0, "max_tool_calls": 0, "max_tokens": 0},
  "retry": {"retries": 3, "backoff_seconds": 2, "max_wait_seconds": 60}
}
```

`run` writes every session entry of the run to stdout as one JSON line,
and then a last line `{"type": "outcome", ...}` with the outcome and the
session path. Nothing else goes to stdout.

It exits 0 whenever it reaches an outcome, whatever the reason. It exits 2
when the request is unusable, before writing anything: bad JSON, an unknown
field, a schema that does not compile, a tool name the providers would
reject, a session path it cannot use. It exits 1 when the session log is
corrupt or another process holds it. On SIGTERM or SIGINT it cancels the
run, so the outcome line is still written.

`acp` reads the same request with different rules. It needs
`session_dir`, and each session the client opens is the log
`<session_dir>/<id>.jsonl`. It refuses `session`, `prompt` and
`timeout_seconds`: the client opens sessions and sends prompts, and a
deadline would end a turn as cancelled, which ACP keeps for the client.
Two fields are for `acp` alone, and `run` refuses them: `tool_kinds`
(`{"tool name": "read"}`, using ACP's tool kinds, so a client can show
each call) and `title` (the agent's display name in `agentInfo.title`,
which a client can show; omitted when empty). The client's `cwd` is
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
listed and loaded before anything has been said in it. `session/load`
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
| `held` | error `-32603`, data: the held calls |

Two `_meta` keys carry what ACP has no field for:
`_meta["agentloop/labels"]` on a tool call's result holds the policy
labels of the call, and `_meta["agentloop/retry"]` on a notice for a
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

- A tool is killed with its process group. A tool that leaves the group
  (with `setsid`, say) can outlive its call. On Linux a cgroup could close
  that gap; macOS has no equivalent.
- When agentloop itself is killed with SIGKILL, a tool that is running
  carries on as an orphan. Its call is still never run a second time.
- A signal that arrives before the Go runtime has started kills the
  process with nothing written. Running the same request again is safe.
- Replay is checked against recorded wire formats and the SDKs' types.
  Whether each provider accepts every replayed shape is confirmed only by
  the live tests (`AGENTLOOP_LIVE=1`), which spend money and do not run by
  default.
