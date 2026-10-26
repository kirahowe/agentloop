# agentloop

agentloop runs a language model in a loop with tools, for agents whose
access has to be controlled exactly. It is a Go library with adapters for
Anthropic and OpenAI, and a small command, `agentloop`, built on it.

An agent can do only what it was given. It has no tools, files or
environment of its own. Every tool call is checked before it runs: the tool
must exist, the input must match its schema, and a policy you write, if
there is one, must allow it. A policy can attach labels to the session that
are never removed. Labels are short provenance names such as `private` or
`mail:stranger@example.com`. They record where the session's
material came from, and a session can forbid a label outright: one that
may not hold `private` never starts holding it, and a tool result that
carries it is refused before the model sees it.

agentloop never holds a call for a person to approve. What an agent can
reach is decided by the tools it is given and by the walls around them.

A session is an append-only log, written to disk entry by entry. If the
process crashes or is killed at any point, the next run carries on from the
log and never runs a tool call twice. A call that was running when the
process died is reported to the model as possibly having taken effect.

The core knows no vendor. Each provider gets its own earlier turns back
unchanged, so reasoning carries across turns, and failures come back sorted
by what to do about them: rate limited, overloaded, unauthorized and so on.

[DESIGN.md](DESIGN.md) has the whole design, including what it does not
guarantee.

## The library

```sh
go get github.com/kirahowe/agentloop
```

```go
clock := agentloop.Func(agentloop.ToolSpec{
    Name:        "clock",
    Description: "The current time.",
    Parameters:  json.RawMessage(`{"type":"object","properties":{},"additionalProperties":false}`),
}, func(ctx context.Context, input json.RawMessage) (agentloop.Result, error) {
    return agentloop.Result{Content: "09:30"}, nil
})

agent := &agentloop.Agent{
    Model:   anthropic.New(anthropic.Config{APIKey: key}),
    ModelID: "claude-opus-5-5",
    System:  "You answer briefly.",
    Tools:   []agentloop.Tool{clock},
    Limits:  agentloop.Limits{MaxTurns: 20},
}

session, err := agentloop.OpenSession(agentloop.NewFileStore("session.jsonl"))
// ...
outcome, err := agent.Run(ctx, session, "What time is it?")
```

The packages:

| Package | What it is |
|---|---|
| `agentloop` | Messages, the `Model` and `Tool` interfaces, policies, sessions, the loop |
| `agentloop/anthropic` | Anthropic's Messages API |
| `agentloop/openai` | OpenAI's Responses API |
| `agentloop/command` | A tool that runs a program, with exactly the environment you give it |
| `agentloop/mcp` | The tools of one MCP server, run as a child process over stdio or reached over streamable HTTP |
| `agentloop/fake` | A scripted model for tests |

## The command

`agentloop run request.json` runs one agent to an outcome. The request
names the provider and model, the system prompt and prompt, the session
file, the tools, and the command each tool call runs. It prints every
session entry as a JSON line and then an outcome line, which also gives
the run's cost in US dollars as `cost_usd` when the model is priced. `agentloop acp
request.json` serves the same agent to an [Agent Client Protocol](https://agentclientprotocol.com)
client over stdin and stdout, where a client can also list, load, resume
and close the sessions kept in the request's `session_dir` (with
`"new_sessions_only": true`, only open new ones and close them). It asks the
provider for the model's display name and context window, and shows the
model to the client as a session config option, so nothing in the request
can drift from the model. The request format is in
[DESIGN.md](DESIGN.md#the-command).

A request may give a policy: the session's starting labels, the labels
each tool adds when it is called, and the labels the session may never
hold (`forbid_labels`). Without a policy every tool the request gives is
allowed. With a policy, command tools speak the envelope protocol: JSON in
with the session's labels, JSON out with the labels of what the tool
returned. A request that uses a field from an older version (`resolve`,
`quarantine`, `workbench`, `policy.unrestricted`, or `class`,
`always_hold` or `consent` in a tool's policy entry) is refused with a
message that says the field was removed. `agentloop acp` never asks the
client for permission, and shows the session's labels on every tool call.

What agentloop reads from another process (the ACP client, tool
envelopes, local MCP servers, request files) goes through one
strict JSON reader, the same dialect relay and relay-tui read: valid
UTF-8, one value, no name given twice, names matched exactly. See
[Reading JSON](DESIGN.md#reading-json).

A tool result can carry a display: a title, a Markdown body, labels and
sources for the operator. agentloop keeps it in the session file and
sends it to an ACP client, and never gives the model any of it. The
model is told only that the operator was shown something, under the ref
`display_ref_prefix` gives it.

A request can also take tools from [MCP](https://modelcontextprotocol.io)
servers it runs over stdio. The request is the allowlist: it names each
tool it wants from a server, under its own name for it, and the policy
gives the tool its labels, as for any other tool. The server
gives the description and schema. agentloop pins each definition by its
SHA-256 in the session, and a later run on that session refuses a
definition that has changed, ending with reason `tools_unavailable`. A
tool's entry may also give the digest it must have. A server gets only PATH and HOME. A secret
it needs comes from a `credentials_command`, whose output is written to
a private file that the server's arguments can name as
`{credentials_file}`. A server sees the session's labels only with
`send_labels`, and a result can add labels through `_meta` but never
remove one. Images and other non-text content fail the call unless the
tool allows them.

```json
"mcp_servers": {"files": {
  "command": ["/usr/local/bin/files-mcp", "--token-file", "{credentials_file}"],
  "credentials_command": ["security", "find-generic-password", "-s", "files", "-w"]
}},
"tools": [{"name": "file_read", "protocol": "mcp-stdio", "server": "files", "tool": "read"}],
"policy": {"tools": {"file_read": {"class": "read", "labels": ["private"]}}}
```

A server can also be remote, reached over MCP's streamable HTTP transport.
It gives a `url` in place of `command`, and its tools have protocol
`mcp-http`. The URL must be https, or http to this machine for tests. What
the `credentials_command` prints is the bearer token, kept in memory only,
and sent only to the server's origin. When the server answers 401, the
command is run again and the request sent once more, so a command that
refreshes an OAuth token keeps a long session going. A redirect to another
origin fails the request. No proxy is used.

A tool's entry can narrow its arguments. `fixed_args` sets arguments to
the given values on every call, and `hidden_args` names arguments the
model may not give. Both are taken out of the schema the model sees, and a
call that names one is refused before the server is sent anything. The
fixed values are added to the call before it is sent.

```json
"mcp_servers": {"mail": {
  "url": "https://mcp.example.com/mcp",
  "credentials_command": ["relay", "mcp-token", "mail"]
}},
"tools": [{"name": "mail_send", "protocol": "mcp-http", "server": "mail", "tool": "send_campaign",
           "fixed_args": {"send_now": false}, "hidden_args": ["reply_to"]}]
```

`agentloop mcp-tools` lists every tool of a remote server, with the
digest each definition would be pinned at. It reads `{"url": ..., "token":
...}` on stdin, so the token never shows in a process list, and prints
`{"tools": [...]}`, each with its name, title, description, annotations,
schemas and digest. It is for an operator choosing which tools to allow.

The API key is fetched by a command named in the request, such as a
keychain lookup, and reaches only the model. The model clients ignore the
proxy environment variables (`HTTPS_PROXY` and the rest), so the key and
the prompts go to the provider directly or not at all.

### Traces

`agentloop otel SESSION.jsonl` turns a session file into OpenTelemetry
traces, offline. It prints one OTLP/JSON `ExportTraceServiceRequest` on
one line, or writes it to `--out FILE`. Nothing is sent anywhere, and
the session is not changed, so old sessions can be exported at any time
and the same file always gives the same bytes.

`--conversation-id ID` sets `gen_ai.conversation.id`; the default is the
file's name without its extension. `--parent-trace` and `--parent-span`
(hex ids) put every run under a span made elsewhere, such as an eval's
workflow span. The file may also be a run's transcript (what `agentloop run`
prints); its closing outcome line is skipped. Flags may come before or after the file. A usage error
exits 2.

Each run is one `invoke_agent` span, with a `chat` span per model call
and an `execute_tool` span per tool call. The model calls a tool
made are `chat` spans under the tool span. A run with no `run_ended`
(a crash) ends with status ERROR and reason `crashed`. The export has
no content: no messages, tool inputs or results, policy reasons or
tool logs. Labels are exported, so treat the telemetry backend as
inside the policy.

The OpenTelemetry GenAI conventions are at Development stability and
nothing in them is stable. The command follows the pinned commit and
schema URL named in the [otel package documentation](otel/doc.go),
which also lists every span and attribute. Expect renames when the
pin moves.

## Building

Go 1.27 or later.

```sh
make build    # bin/agentloop
make test     # gofmt, vet, and every test with the race detector
make update   # move every dependency to its latest version, then test
make tck      # the ACP conformance kit against agentloop acp (needs uv)
```

The tests need no network and no key. Live tests against the real APIs spend a
little money and run only with `make live`, which passes `-live` and a key
command to the anthropic and openai test packages. The OpenAI test skips
unless you add `-openai-key-command`.

## License

MIT
