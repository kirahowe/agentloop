// Package otel turns an agentloop session log into OpenTelemetry traces.
//
// Traces is a pure function: entries in, OTLP/JSON out. It reads no files,
// makes no requests and needs no OpenTelemetry SDK. The log is not changed,
// so old sessions can be exported at any time, and the same log gives the
// same bytes.
//
// # Conventions followed
//
// The GenAI semantic conventions live in
// https://github.com/open-telemetry/semantic-conventions-genai. This
// package follows commit f3c6d0c9bb66ebd474668a3a0be25c61226f02c2
// (2026-10-14), schema URL
// https://opentelemetry.io/schemas/gen-ai-dev/1.42.0-dev, which builds on
// core semantic conventions v1.44.0. Nothing in those conventions is
// stable. Every gen_ai.* attribute below has stability "development" and
// may be renamed; the commit is pinned for that reason.
//
// # Spans
//
// One trace per run, from run_started to run_ended. A run that has no
// run_ended (a crash) is closed at the time of the last entry before the
// next run, or before the end of the log, with status ERROR and
// agentloop.run.reason "crashed".
//
//   - "invoke_agent {agent}" (INTERNAL), the root of the trace. It has
//     gen_ai.operation.name, gen_ai.provider.name, gen_ai.agent.name,
//     gen_ai.request.model, gen_ai.conversation.id and the gen_ai.usage.*
//     totals.
//   - "chat {model}" (CLIENT), one per assistant message and one per
//     failed model call, a child of the run. It has
//     gen_ai.operation.name, gen_ai.provider.name, gen_ai.request.model,
//     gen_ai.response.model, gen_ai.response.finish_reasons,
//     gen_ai.conversation.id and gen_ai.usage.*. A failed call has
//     error.type and a gen_ai.client.operation.exception event.
//   - "execute_tool {tool}" (INTERNAL), one per tool call, a child of the
//     run that finished the call. It has gen_ai.operation.name,
//     gen_ai.tool.name, gen_ai.tool.call.id, gen_ai.tool.type,
//     gen_ai.agent.name and gen_ai.conversation.id. It has error.type when
//     the call was denied, was not run, failed or never finished.
//
// gen_ai.usage.input_tokens counts every input token, cached ones
// included: agentloop's InputTokens + CacheReadTokens + CacheWriteTokens.
// gen_ai.usage.cache_read.input_tokens and
// gen_ai.usage.cache_write.input_tokens are added when they are not zero.
//
// Stop reasons map to gen_ai.response.finish_reasons as end_turn to stop,
// tool_use to tool_call, max_tokens to length and refusal to
// content_filter. pause has no counterpart and is kept as it is. The
// agentloop value is also recorded as agentloop.stop.
//
// Everything that has no place in the conventions is under the agentloop.
// prefix: agentloop.labels (the session's labels, as a string array),
// agentloop.run.reason, agentloop.run.turns, agentloop.run.limit,
// agentloop.stop, agentloop.call.state,
// agentloop.policy.verdict, agentloop.retry and
// agentloop.retry.wait_ms. Policy decisions, call starts and label changes
// are span events: agentloop.policy.decision, agentloop.call.started and
// agentloop.labels.
//
// Labels say that a session held private or contaminated data, so they are
// metadata, not content, and are always exported. Treat the telemetry
// backend as inside the policy.
//
// Two more entry types have a place. A pins entry is an event
// agentloop.pins on the run span, with the pinned tool names in
// agentloop.pins.tools (the digests stay out). A tool_result's
// usage, the model calls the tool made, becomes one
// "chat {model}" span (CLIENT) per call under the execute_tool span, with
// gen_ai.usage.*. The convention's "SHOULD NOT record two spans for one
// call" is about the MCP client span, not about a tool that calls a
// model, so a child inference span is the right shape here. The log has no
// times for those calls, so each spans its tool call. Those calls
// also count in a crashed run's usage, as they do in an outcome's. The
// outcome reason tools_unavailable is not in the log (the run never
// starts), so nothing converts it.
//
// Options.ParentTraceID and ParentSpanID hang every run under a span made
// elsewhere; the runs then share that trace.
//
// # Timing
//
// An entry carries only the time it was appended. A chat span ends at its
// entry's time and starts at the time of the entry before it (the user
// message, the last tool result or the previous model error). That includes
// the loop's own overhead, so it overstates the model's latency a little.
// A tool span starts at its first decision and ends at its tool_result. A
// call decided in one run and finished in a later one belongs to the later
// run: its start is moved to that run's start, and its decision events keep
// their real times. Times are never invented: a span whose clock went
// backwards is given a zero length.
//
// # Content
//
// By default no session content is exported: no prompts, messages, tool
// inputs or results, policy reasons, tool logs or error
// messages. Options.IncludeContent turns on the convention's opt-in
// attributes: gen_ai.input.messages and gen_ai.output.messages on chat
// spans, gen_ai.output.messages on the run span, gen_ai.tool.call.arguments
// and gen_ai.tool.call.result on tool spans, and exception.message, with
// agentloop.policy.reason, agentloop.refusal,
// agentloop.tool.error and agentloop.tool.log. Structured values are
// recorded as JSON strings, which the conventions allow on spans.
//
// # Ids
//
// Trace id: the first 16 bytes of SHA-256 over the seed and the run's
// index. Span id: the first 8 bytes of SHA-256 over the trace id, the kind
// of span and the entry's seq. The seed is Options.TraceSeed, or the
// conversation id when that is empty.
package otel
