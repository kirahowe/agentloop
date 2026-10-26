package otel_test

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/otel"
)

func attrOf(m map[string]any, key string) (map[string]any, bool) {
	as, _ := m["attributes"].([]any)
	for _, a := range as {
		kv := a.(map[string]any)
		if kv["key"] == key {
			return kv["value"].(map[string]any), true
		}
	}
	return nil, false
}

func spanNamed(spans []map[string]any, prefix string) []map[string]any {
	var out []map[string]any
	for _, s := range spans {
		if strings.HasPrefix(s["name"].(string), prefix) {
			out = append(out, s)
		}
	}
	return out
}

func eventNamed(span map[string]any, name string) map[string]any {
	evs, _ := span["events"].([]any)
	for _, e := range evs {
		if m := e.(map[string]any); m["name"] == name {
			return m
		}
	}
	return nil
}

// toolModelLog is a run in which a tool made two model calls.
func toolModelLog(ended bool) []agentloop.Entry {
	t0 := time.Date(2026, 10, 17, 9, 0, 0, 0, time.UTC)
	es := []agentloop.Entry{
		{Type: agentloop.EntryRunStarted, RunStarted: &agentloop.RunStartedEntry{Prompt: "p"}},
		{Type: agentloop.EntryPins, Pins: &agentloop.PinsEntry{Pins: map[string]string{"b": "d2", "a": "d1"}}},
		{Type: agentloop.EntryMessage, Message: &agentloop.MessageEntry{
			Message: agentloop.Message{Role: agentloop.Assistant, Parts: []agentloop.Part{{Type: agentloop.ToolCallPart,
				ToolCall: &agentloop.ToolCall{ID: "c1", Name: "summarize", Input: json.RawMessage(`{}`)}}}},
			Stop: agentloop.StopToolUse, Usage: agentloop.Usage{InputTokens: 5, OutputTokens: 6}, Model: "big"}},
		{Type: agentloop.EntryDecision, Decision: &agentloop.DecisionEntry{CallID: "c1", Tool: "summarize", Verdict: agentloop.Allow}},
		{Type: agentloop.EntryCallStarted, CallStarted: &agentloop.CallStartedEntry{CallID: "c1"}},
		{Type: agentloop.EntryToolResult, ToolResult: &agentloop.ToolResultEntry{
			Result: agentloop.ToolResult{CallID: "c1", Content: "SECRET"},
			Usage: []agentloop.ModelUsage{
				{Model: "small", Usage: agentloop.Usage{InputTokens: 10, OutputTokens: 2, CacheReadTokens: 7}},
				{Model: "small", Usage: agentloop.Usage{InputTokens: 1, OutputTokens: 1}}}}},
	}
	if ended {
		es = append(es, agentloop.Entry{Type: agentloop.EntryRunEnded, RunEnded: &agentloop.Outcome{Reason: agentloop.ReasonCompleted, Turns: 1}})
	}
	for i := range es {
		es[i].Seq, es[i].Time = i+1, t0.Add(time.Duration(i)*time.Second)
	}
	return es
}

func TestPinsAndToolModelCalls(t *testing.T) {
	raw, err := otel.Marshal(toolModelLog(true), otel.Options{ConversationID: "c"})
	if err != nil {
		t.Fatal(err)
	}
	checkOTLP(t, raw)
	spans := spansOf(t, raw)

	run := spanNamed(spans, "invoke_agent")[0]
	pins := eventNamed(run, "agentloop.pins")
	if pins == nil || !strings.Contains(string(mustJSON(pins)), `"a"`) {
		t.Fatalf("no pins event on the run span: %v", pins)
	}

	tool := spanNamed(spans, "execute_tool")[0]
	if dec := eventNamed(tool, "agentloop.policy.decision"); dec == nil {
		t.Fatal("tool span lacks the decision event")
	}

	var calls []map[string]any
	for _, s := range spanNamed(spans, "chat small") {
		if s["parentSpanId"] == tool["spanId"] {
			calls = append(calls, s)
		}
	}
	if len(calls) != 2 {
		t.Fatalf("want 2 tool model spans under the tool span, got %d", len(calls))
	}
	if v, _ := attrOf(calls[0], "gen_ai.usage.input_tokens"); v["intValue"] != "17" {
		t.Fatalf("tool model input tokens (cache included): %v", v)
	}
	if calls[0]["spanId"] == calls[1]["spanId"] {
		t.Fatal("tool model spans share an id")
	}
	if strings.Contains(string(raw), "SECRET") {
		t.Fatal("content leaked")
	}
}

func mustJSON(v any) []byte { b, _ := json.Marshal(v); return b }

// A run with no run_ended ends ERROR with reason crashed, and its usage
// includes the tool's model calls.
func TestCrashedRunEndsError(t *testing.T) {
	raw, _ := otel.Marshal(toolModelLog(false), otel.Options{ConversationID: "c"})
	checkOTLP(t, raw)
	run := spanNamed(spansOf(t, raw), "invoke_agent")[0]
	st := run["status"].(map[string]any)
	if st["code"].(float64) != otel.StatusError {
		t.Fatalf("status %v", st)
	}
	if v, _ := attrOf(run, "agentloop.run.reason"); v["stringValue"] != "crashed" {
		t.Fatalf("reason %v", v)
	}
	if v, _ := attrOf(run, "gen_ai.usage.input_tokens"); v["intValue"] != "23" {
		t.Fatalf("crashed run input tokens: %v", v)
	}
}
