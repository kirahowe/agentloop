package scenario

import (
	"context"
	"testing"
	"time"

	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop"
)

// Regression: an iteration armed to cancel left its cancel function on
// the scenario. The next run outside iterate (an agent from Agent, run by
// an ACP server after HandOver) met a tool that blocks until cancelled,
// fired the stale cancel, and waited on its own context forever.
func TestAgentAfterACancelledIterationDoesNotHang(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		call := Step{Kind: StToolUse, Stop: agentloop.StopToolUse, Calls: []CallSpec{{Name: "echo"}}}
		cfg := Config{
			Steps:     []Step{call, call, {Kind: StText, Stop: agentloop.StopEndTurn, Text: "done"}},
			Behaviors: map[string]Behavior{"echo": {Kind: BBlockUntilCancel}, "any": {}, "lab": {}},
			NilPolicy: true,
		}
		sc := New(t, cfg, Faults{Cancel: true})
		if out := sc.iterate("go", 0, false, 2); out == nil || out.Reason != agentloop.ReasonCanceled {
			t.Fatalf("the armed iteration ended %+v", out)
		}
		s := sc.open()
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		out, err := sc.Agent().Run(ctx, s, "continue")
		if err != nil || out.Reason != agentloop.ReasonCompleted {
			t.Fatalf("the next run: %+v, %v", out, err)
		}
	})
}
