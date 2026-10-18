package acpagent

import (
	"encoding/json"
	"testing"

	acp "github.com/caelis-labs/acp-go-sdk"

	"github.com/kirahowe/agentloop"
)

// displayOf finds the display block in a call's content, and returns its
// meta and text.
func displayOf(content []acp.ToolCallContent) (agentloop.Display, string, bool) {
	for _, c := range content {
		if c.Content == nil || c.Content.Content.Text == nil {
			continue
		}
		tb := c.Content.Content.Text
		if raw, ok := tb.Meta[DisplayMeta]; ok {
			var d agentloop.Display
			if err := json.Unmarshal(raw, &d); err != nil {
				return d, "", false
			}
			return d, tb.Text, true
		}
	}
	return agentloop.Display{}, "", false
}

func TestDisplayGoesToTheClientAsToolContent(t *testing.T) {
	call := agentloop.ToolCall{ID: "c1", Name: "web_ask", Input: json.RawMessage(`{}`)}
	d := &agentloop.Display{Title: "Venues", Body: "body", Labels: []string{"contaminated"},
		Sources: []agentloop.Source{{Title: "List", Ref: "web:https://example.org/"}}, Ref: "shown:2026-10-18/c1"}
	res := agentloop.ToolResultEntry{Result: agentloop.ToolResult{CallID: "c1", Content: "closed"}, Display: d}
	ups := translate(Options{},
		entry(1, runStarted),
		assistant(2, agentloop.StopToolUse, "", agentloop.Part{Type: agentloop.ToolCallPart, ToolCall: &call}),
		entry(3, agentloop.Entry{Type: agentloop.EntryToolResult, ToolResult: &res}),
	)
	found := false
	for _, u := range ups {
		if tu := u.ToolCallUpdate; tu != nil && tu.Status != nil {
			got, text, ok := displayOf(tu.Content)
			if !ok {
				t.Fatalf("no display in %+v", tu.Content)
			}
			found = true
			if got.Title != "Venues" || got.Ref != d.Ref || len(got.Sources) != 1 || got.Labels[0] != "contaminated" {
				t.Errorf("display %+v", got)
			}
			if text != "Venues\n\nbody" {
				t.Errorf("text %q", text)
			}
			if tu.Content[0].Content.Content.Text.Text != "closed" {
				t.Errorf("the result's own content moved: %+v", tu.Content[0])
			}
		}
	}
	if !found {
		t.Fatal("no completed update")
	}
	// Replay sends it too.
	es := []agentloop.Entry{
		{Seq: 1, Type: agentloop.EntryMessage, Message: &agentloop.MessageEntry{Message: agentloop.Message{Role: agentloop.Assistant,
			Parts: []agentloop.Part{{Type: agentloop.ToolCallPart, ToolCall: &call}}}, Stop: agentloop.StopToolUse}},
		{Seq: 2, Type: agentloop.EntryToolResult, ToolResult: &res},
	}
	replayed := false
	for _, u := range Replay(Options{}, es) {
		if tc := u.ToolCall; tc != nil {
			_, _, replayed = displayOf(tc.Content)
		}
	}
	if !replayed {
		t.Error("replay lost the display")
	}
}

// A display with no labels or sources is sent with empty arrays, never
// null, so a client can read it without special cases.
func TestDisplayContentHasArrays(t *testing.T) {
	c := DisplayContent(agentloop.Display{Title: "t", Body: "b"})
	raw := string(c.Content.Content.Text.Meta[DisplayMeta])
	if raw != `{"title":"t","body":"b","labels":[],"sources":[]}` {
		t.Errorf("meta %s", raw)
	}
}
