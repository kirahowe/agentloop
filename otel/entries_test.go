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

// readerLog is a run in which a quarantined reader's call was deferred,
// made two model calls and was labelled.
func readerLog(ended bool) []agentloop.Entry {
	t0 := time.Date(2026, 10, 17, 9, 0, 0, 0, time.UTC)
	es := []agentloop.Entry{
		{Type: agentloop.EntryRunStarted, RunStarted: &agentloop.RunStartedEntry{Prompt: "p"}},
		{Type: agentloop.EntryPins, Pins: &agentloop.PinsEntry{Pins: map[string]string{"b": "d2", "a": "d1"}}},
		{Type: agentloop.EntryMessage, Message: &agentloop.MessageEntry{
			Message: agentloop.Message{Role: agentloop.Assistant, Parts: []agentloop.Part{{Type: agentloop.ToolCallPart,
				ToolCall: &agentloop.ToolCall{ID: "c1", Name: "read_quarantined", Input: json.RawMessage(`{}`)}}}},
			Stop: agentloop.StopToolUse, Usage: agentloop.Usage{InputTokens: 5, OutputTokens: 6}, Model: "big"}},
		{Type: agentloop.EntryDecision, Decision: &agentloop.DecisionEntry{CallID: "c1", Tool: "read_quarantined", Verdict: agentloop.Allow, Deferred: []string{"contaminated"}}},
		{Type: agentloop.EntryCallStarted, CallStarted: &agentloop.CallStartedEntry{CallID: "c1"}},
		{Type: agentloop.EntryToolResult, ToolResult: &agentloop.ToolResultEntry{
			Result: agentloop.ToolResult{CallID: "c1", Content: "SECRET"},
			Usage: []agentloop.ModelUsage{
				{Model: "small", Usage: agentloop.Usage{InputTokens: 10, OutputTokens: 2, CacheReadTokens: 7}},
				{Model: "small", Usage: agentloop.Usage{InputTokens: 1, OutputTokens: 1}}},
			Quarantine: &agentloop.QuarantineReport{Tool: "mail_read", FreeText: false, Dropped: []string{"contaminated"}}}},
	}
	if ended {
		es = append(es, agentloop.Entry{Type: agentloop.EntryRunEnded, RunEnded: &agentloop.Outcome{Reason: agentloop.ReasonCompleted, Turns: 1}})
	}
	for i := range es {
		es[i].Seq, es[i].Time = i+1, t0.Add(time.Duration(i)*time.Second)
	}
	return es
}

func TestPinsDeferredReaderQuarantine(t *testing.T) {
	raw, err := otel.Marshal(readerLog(true), otel.Options{ConversationID: "c"})
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
	dec := eventNamed(tool, "agentloop.policy.decision")
	if dec == nil || !strings.Contains(string(mustJSON(dec)), "agentloop.policy.deferred") {
		t.Fatalf("decision event lacks deferred labels: %v", dec)
	}
	if _, ok := attrOf(tool, "agentloop.quarantine.tool"); !ok {
		t.Fatal("tool span lacks the quarantine report")
	}

	var readers []map[string]any
	for _, s := range spanNamed(spans, "chat small") {
		if s["parentSpanId"] == tool["spanId"] {
			readers = append(readers, s)
		}
	}
	if len(readers) != 2 {
		t.Fatalf("want 2 reader spans under the tool span, got %d", len(readers))
	}
	if v, _ := attrOf(readers[0], "gen_ai.usage.input_tokens"); v["intValue"] != "17" {
		t.Fatalf("reader input tokens (cache included): %v", v)
	}
	if readers[0]["spanId"] == readers[1]["spanId"] {
		t.Fatal("reader spans share an id")
	}
	if strings.Contains(string(raw), "SECRET") {
		t.Fatal("content leaked")
	}
}

func mustJSON(v any) []byte { b, _ := json.Marshal(v); return b }

// A run with no run_ended ends ERROR with reason crashed, and its usage
// includes the reader's calls.
func TestCrashedRunEndsError(t *testing.T) {
	raw, _ := otel.Marshal(readerLog(false), otel.Options{ConversationID: "c"})
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
