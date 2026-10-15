package agentloop_test

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/anthropic"
)

// An agent with one tool and a policy, on a session kept in a file.
func Example() {
	clock := agentloop.Func(agentloop.ToolSpec{
		Name:        "clock",
		Description: "The current time.",
		Parameters:  json.RawMessage(`{"type":"object","properties":{},"additionalProperties":false}`),
	}, func(ctx context.Context, input json.RawMessage) (agentloop.Result, error) {
		return agentloop.Result{Content: "09:30"}, nil
	})

	agent := &agentloop.Agent{
		Model:   anthropic.New(anthropic.Config{APIKey: "sk-..."}),
		ModelID: "claude-opus-5-5",
		System:  "You answer briefly.",
		Tools:   []agentloop.Tool{clock},
		Policy: agentloop.PolicyFunc(func(ctx context.Context, call agentloop.ToolCall, view agentloop.View) agentloop.Decision {
			return agentloop.Decision{Verdict: agentloop.Allow}
		}),
		Limits: agentloop.Limits{MaxTurns: 20},
	}

	session, err := agentloop.OpenSession(agentloop.NewFileStore("session.jsonl"))
	if err != nil {
		log.Fatal(err)
	}
	outcome, err := agent.Run(context.Background(), session, "What time is it?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(outcome.Reason, outcome.Text)
}
