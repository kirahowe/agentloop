package anthropic_test

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/anthropic"
	"github.com/kirahowe/agentloop/internal/livetest"
)

// TestLiveToolUseWithThinking is a smoke test against the real API. It
// costs money, so it runs only with -live and -anthropic-key-command (see
// internal/livetest). -anthropic-model picks the model.
//
// It runs a two-turn tool-use conversation and checks that the API accepts
// the first turn replayed from Native, thinking blocks and signatures
// included.
func TestLiveToolUseWithThinking(t *testing.T) {
	key, model := livetest.Anthropic(t)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	m := anthropic.New(anthropic.Config{APIKey: key})

	req := &agentloop.Request{
		Model:  model,
		System: "You are terse. Always use the tool when it applies.",
		// Adaptive thinking skips trivial prompts, and a turn without a
		// thinking block would pass without testing its replay. High
		// effort and a question that needs a moment's thought make the
		// model think.
		Effort:          agentloop.EffortHigh,
		MaxOutputTokens: 8000,
		Messages:        []agentloop.Message{agentloop.Text(agentloop.User, "Which European capital lies furthest north among Paris, Oslo, Helsinki and Reykjavik? Think it through, then get its current weather with the get_weather tool.")},
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
	if !thought {
		t.Fatalf("the first turn has no thinking block, so its replay would test nothing; try a model that thinks (-anthropic-model)")
	}

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
