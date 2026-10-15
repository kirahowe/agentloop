# Client traces

Each `*.jsonl` file here is the client side of one real ACP session: the
messages a client sent to the agent, one JSON-RPC message per line.
`TestClientTraces` in `cmd/agentloop` replays every file in this directory
against the real `agentloop acp` binary and checks the agent's answers
against the vendored schema. A new file is picked up without any code change.

## Capturing a trace

Point the client at `relay acp <role> --trace FILE`, use it for a short
session (initialize, a new session, one prompt), then close it. The trace
holds both directions. Keep only the client-to-agent messages; the agent's
side is regenerated on every replay.

## Before committing

Traces come from real sessions. Replace personal text before committing:
prompt text, working directories, project names, user names, and any other
identifying value. The replay substitutes the session id, so the recorded one
can stay or be made up.
