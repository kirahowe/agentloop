package acpagent

import (
	"encoding/json"
	"slices"
	"testing"

	"github.com/kirahowe/agentloop"
)

func TestQuarantineMeta(t *testing.T) {
	call := agentloop.ToolCall{ID: "c1", Name: "read_quarantined", Input: json.RawMessage(`{"tool":"mail_read"}`)}
	answer := agentloop.ToolResultEntry{
		Result:     agentloop.ToolResult{CallID: "c1", Content: `{"answer":true,"free_text":false}`},
		Labels:     []string{"private"},
		Quarantine: &agentloop.QuarantineReport{Tool: "mail_read", FreeText: false, Dropped: []string{"contaminated"}},
	}
	plain := agentloop.ToolResultEntry{Result: agentloop.ToolResult{CallID: "c2", Content: "x"}}
	const want = `{"tool":"mail_read","free_text":false,"dropped":["contaminated"]}`

	// Live: the completed update carries the report beside the labels.
	ups := translate(Options{},
		entry(1, runStarted),
		assistant(2, agentloop.StopToolUse, "", agentloop.Part{Type: agentloop.ToolCallPart, ToolCall: &call}),
		entry(3, agentloop.Entry{Type: agentloop.EntryToolResult, ToolResult: &answer}),
		entry(4, agentloop.Entry{Type: agentloop.EntryToolResult, ToolResult: &plain}),
	)
	var got []string
	for _, u := range ups {
		if tu := u.ToolCallUpdate; tu != nil && tu.Status != nil {
			got = append(got, string(tu.Meta["agentloop/quarantine"]))
			if string(tu.Meta["agentloop/labels"]) == "" {
				t.Errorf("update without labels: %+v", tu)
			}
		}
	}
	if len(got) != 2 || got[0] != want || got[1] != "" {
		t.Errorf("quarantine meta = %q", got)
	}

	// Replay shows it on the call.
	es := []agentloop.Entry{
		{Seq: 1, Type: agentloop.EntryMessage, Message: &agentloop.MessageEntry{Message: agentloop.Message{Role: agentloop.Assistant, Parts: []agentloop.Part{{Type: agentloop.ToolCallPart, ToolCall: &call}}}, Stop: agentloop.StopToolUse}},
		{Seq: 2, Type: agentloop.EntryToolResult, ToolResult: &answer},
	}
	found := false
	for _, u := range Replay(Options{}, es) {
		if tc := u.ToolCall; tc != nil {
			found = string(tc.Meta["agentloop/quarantine"]) == want
		}
	}
	if !found {
		t.Error("replay lost the quarantine meta")
	}
}

func readerResult(seq int, id string, uses ...agentloop.ModelUsage) agentloop.Event {
	return entry(seq, agentloop.Entry{Type: agentloop.EntryToolResult, ToolResult: &agentloop.ToolResultEntry{
		Result: agentloop.ToolResult{CallID: id, Content: "{}"}, Usage: uses,
	}})
}

// A reader's calls are priced at the reader model's price, and the cost
// is the same live, on replay and on reopen. An unpriced reader model
// leaves the cost out from then on.
func TestCostIncludesReaderCalls(t *testing.T) {
	small := agentloop.Prices{Input: 1}
	priceOf := func(m string) *agentloop.Prices {
		if m == "small" {
			return &small
		}
		return nil
	}
	opts := Options{ContextWindow: 1000, Prices: &agentloop.Prices{Input: 10}, PriceOf: priceOf}
	call := agentloop.ToolCall{ID: "c1", Name: "read_quarantined", Input: json.RawMessage(`{}`)}
	evs := []agentloop.Event{
		entry(1, runStarted),
		assistant(2, agentloop.StopToolUse, "", agentloop.Part{Type: agentloop.ToolCallPart, ToolCall: &call}),
		readerResult(3, "c1", agentloop.ModelUsage{Model: "small", Usage: agentloop.Usage{InputTokens: 1_000_000}},
			agentloop.ModelUsage{Model: "small", Usage: agentloop.Usage{InputTokens: 2_000_000}}),
	}
	// The assistant message used no tokens; the reader used 3M at $1.
	live := usageUpdates(translate(opts, evs...))
	if len(live) != 2 || live[1].Cost == nil || live[1].Cost.Amount != 3 {
		t.Fatalf("live %+v", live)
	}
	var entries []agentloop.Entry
	for _, ev := range evs {
		entries = append(entries, *ev.Entry)
	}
	replay := usageUpdates(Replay(opts, entries))
	if len(replay) != 1 || replay[0].Cost == nil || replay[0].Cost.Amount != 3 {
		t.Fatalf("replay %+v", replay)
	}
	_, ups := NewTranslator(opts).WithHistory(entries).Step(usageAssistant(4, agentloop.Usage{InputTokens: 1_000_000}))
	if r := usageUpdates(ups); len(r) != 1 || r[0].Cost == nil || r[0].Cost.Amount != 13 {
		t.Fatalf("reopened %+v", r)
	}

	// An unpriced reader model: no cost from then on, never a partial one.
	unpriced := append(slices.Clone(evs), readerResult(4, "c1", agentloop.ModelUsage{Model: "mystery", Usage: agentloop.Usage{InputTokens: 1}}),
		usageAssistant(5, agentloop.Usage{InputTokens: 1}))
	for _, u := range usageUpdates(translate(opts, unpriced...))[2:] {
		if u.Cost != nil {
			t.Fatalf("cost %v after an unpriced call", u.Cost.Amount)
		}
	}
}
