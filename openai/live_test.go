package openai

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/kirahowe/agentloop"
)

// TestLiveToolUseWithReasoning is a smoke test against the real API. It
// costs money, so it runs only when AGENTLOOP_LIVE=1 and a key is in
// AGENTLOOP_OPENAI_KEY. AGENTLOOP_OPENAI_MODEL picks the model, which must
// be a reasoning model because the test sets an effort.
//
// It runs a two-turn tool-use conversation and checks that the API accepts
// the first turn replayed from Native, encrypted reasoning included.
func TestLiveToolUseWithReasoning(t *testing.T) {
	key := os.Getenv("AGENTLOOP_OPENAI_KEY")
	if os.Getenv("AGENTLOOP_LIVE") != "1" || key == "" {
		t.Skip("set AGENTLOOP_LIVE=1 and AGENTLOOP_OPENAI_KEY to run")
	}
	model := os.Getenv("AGENTLOOP_OPENAI_MODEL")
	if model == "" {
		model = "gpt-5"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	m := New(Config{APIKey: key})

	req := &agentloop.Request{
		Model:           model,
		System:          "You are terse. Always use the tool when it applies.",
		Effort:          agentloop.EffortLow,
		MaxOutputTokens: 8000,
		Messages:        []agentloop.Message{agentloop.Text(agentloop.User, "What is the weather in Paris right now? Use the get_weather tool.")},
		Tools: []agentloop.ToolSpec{{
			Name:        "get_weather",
			Description: "Current weather for a city.",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}`),
		}},
	}
	first, err := m.Generate(ctx, req, nil)
	if err != nil {
		t.Fatalf("first turn: %v", err)
	}
	calls := first.Message.ToolCalls()
	if first.Stop != agentloop.StopToolUse || len(calls) == 0 {
		t.Fatalf("expected a tool call, got stop %q parts %+v", first.Stop, first.Message.Parts)
	}
	var thought bool
	for _, p := range first.Message.Parts {
		thought = thought || p.Type == agentloop.ReasoningPart
	}
	t.Logf("first turn: stop=%s reasoning=%v calls=%d usage=%+v", first.Stop, thought, len(calls), first.Usage)

	req.Messages = append(req.Messages, first.Message)
	var results []agentloop.Part
	for _, c := range calls {
		results = append(results, agentloop.Part{Type: agentloop.ToolResultPart, ToolResult: &agentloop.ToolResult{CallID: c.ID, Content: "14 C, light rain"}})
	}
	req.Messages = append(req.Messages, agentloop.Message{Role: agentloop.User, Parts: results})

	second, err := m.Generate(ctx, req, nil)
	if err != nil {
		t.Fatalf("second turn (replay of the first from Native): %v", err)
	}
	if second.Message.Text() == "" {
		t.Fatalf("second turn has no text: stop %q parts %+v", second.Stop, second.Message.Parts)
	}
	t.Logf("second turn: stop=%s text=%q", second.Stop, second.Message.Text())
}
