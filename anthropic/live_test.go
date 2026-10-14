package anthropic

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	"agentloop"
)

// TestLiveToolUseWithThinking is a smoke test against the real API. It
// costs money, so it runs only when AGENTLOOP_LIVE=1 and a key is in
// AGENTLOOP_ANTHROPIC_KEY. AGENTLOOP_ANTHROPIC_MODEL picks the model.
//
// It runs a two-turn tool-use conversation and checks that the API accepts
// the first turn replayed from Native, thinking blocks and signatures
// included.
func TestLiveToolUseWithThinking(t *testing.T) {
	key := os.Getenv("AGENTLOOP_ANTHROPIC_KEY")
	if os.Getenv("AGENTLOOP_LIVE") != "1" || key == "" {
		t.Skip("set AGENTLOOP_LIVE=1 and AGENTLOOP_ANTHROPIC_KEY to run")
	}
	model := os.Getenv("AGENTLOOP_ANTHROPIC_MODEL")
	if model == "" {
		model = "claude-sonnet-5-5"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	m := New(Config{APIKey: key})

	req := &agentloop.Request{
		Model:           model,
		System:          "You are terse. Always use the tool when it applies.",
		Effort:          agentloop.EffortMedium,
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
	t.Logf("first turn: stop=%s thinking=%v calls=%d usage=%+v", first.Stop, thought, len(calls), first.Usage)

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
