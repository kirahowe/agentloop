# agentloop

agentloop runs a language model in a loop with tools, for agents whose
access has to be controlled exactly. It is a Go library with adapters for
Anthropic and OpenAI, and a small command, `agentloop`, built on it.

An agent can do only what it was given. It has no tools, files or
environment of its own. Every tool call is checked before it runs: the tool
must exist, the input must match its schema, and a policy you write must
allow it. A policy can also hold a call until a person decides, and attach
labels to the session that are never removed, so a rule like "no network
calls once this session has read private data" is a few lines.

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
| `agentloop/fake` | A scripted model for tests |

## The command

`agentloop run request.json` runs one agent to an outcome. The request
names the provider and model, the system prompt and prompt, the session
file, the tools, and the command each tool call runs. It prints every
session entry as a JSON line and then an outcome line. `agentloop acp
request.json` serves the same agent to an [Agent Client Protocol](https://agentclientprotocol.com)
client over stdin and stdout, where a client can also list, load, resume
and close the sessions kept in the request's `session_dir`. The request
format is in
[DESIGN.md](DESIGN.md#the-command).

The API key is fetched by a command named in the request, such as a
keychain lookup, and reaches only the model.

## Building

Go 1.27 or later.

```sh
make build    # bin/agentloop
make test     # gofmt, vet, and every test with the race detector
make update   # move every dependency to its latest version, then test
make tck      # the ACP conformance kit against agentloop acp (needs uv)
```

The tests need no network and no key. Live tests against the real APIs run
only with `AGENTLOOP_LIVE=1` and spend a little money.

## License

MIT
