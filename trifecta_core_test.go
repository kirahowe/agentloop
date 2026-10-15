package agentloop_test

import (
	"context"
	"encoding/json"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/fake"
)

// callTool is a CallTool that records the calls it is given.
type callTool struct {
	spec  agentloop.ToolSpec
	calls []agentloop.Call
	res   agentloop.Result
}

func (c *callTool) Spec() agentloop.ToolSpec { return c.spec }
func (c *callTool) Call(context.Context, json.RawMessage) (agentloop.Result, error) {
	panic("the agent calls a CallTool through CallWith")
}
func (c *callTool) CallWith(_ context.Context, call agentloop.Call) (agentloop.Result, error) {
	c.calls = append(c.calls, call)
	return c.res, nil
}

func TestAgentLabelsAreAddedAtEveryRunStart(t *testing.T) {
	store := agentloop.NewMemoryStore()
	s := reopen(t, store)
	a := &agentloop.Agent{Model: fake.Script(fake.Text("hi"), fake.Text("again")), Labels: []string{"private"}}
	wantReason(t, run(t, a, s, "go"), agentloop.ReasonCompleted)
	if got := s.Labels(); !reflect.DeepEqual(got, []string{"private"}) {
		t.Fatalf("labels %v", got)
	}
	// A later run, resuming, adds its own starting labels, and keeps the old.
	s = reopen(t, store)
	a.Labels = []string{"contaminated"}
	wantReason(t, run(t, a, s, "more"), agentloop.ReasonCompleted)
	if got := reopen(t, store).Labels(); !reflect.DeepEqual(got, []string{"contaminated", "private"}) {
		t.Fatalf("labels %v", got)
	}
}

func TestCallToolSeesIDAndLabelsAfterItsDecision(t *testing.T) {
	tool := &callTool{
		spec: agentloop.ToolSpec{Name: "t"},
		res:  agentloop.Result{Content: "out", Labels: []string{"contaminated"}, Log: "stderr text"},
	}
	m := fake.Script(fake.ToolUse("", fake.Call("c1", "t", `{"a":1}`)), fake.Text("done"))
	a := &agentloop.Agent{
		Model: m, Tools: []agentloop.Tool{tool}, Labels: []string{"seed"},
		Policy: agentloop.PolicyFunc(func(context.Context, agentloop.ToolCall, agentloop.View) agentloop.Decision {
			return agentloop.Decision{Verdict: agentloop.Allow, Labels: []string{"private"}}
		}),
	}
	s, _ := newSession(t)
	wantReason(t, run(t, a, s, "go"), agentloop.ReasonCompleted)
	if len(tool.calls) != 1 {
		t.Fatalf("calls %+v", tool.calls)
	}
	c := tool.calls[0]
	if c.ID != "c1" || string(c.Input) != `{"a":1}` || !reflect.DeepEqual(c.Labels, []string{"private", "seed"}) {
		t.Fatalf("call %+v", c)
	}
	// The log text is in the entry and nowhere in the conversation.
	var entry *agentloop.ToolResultEntry
	for _, e := range s.Entries() {
		if e.Type == agentloop.EntryToolResult {
			entry = e.ToolResult
		}
	}
	if entry == nil || entry.Log != "stderr text" {
		t.Fatalf("result entry %+v", entry)
	}
	b, _ := json.Marshal(s.Messages())
	if strings.Contains(string(b), "stderr text") {
		t.Fatal("the operator's log reached the conversation")
	}
	if got := s.Labels(); !reflect.DeepEqual(got, []string{"contaminated", "private", "seed"}) {
		t.Fatalf("labels %v", got)
	}
}

// A person allowing a held call gives the session the labels the policy
// put on the hold, as an allow at once would have.
func TestResolveAllowCarriesTheHoldsLabels(t *testing.T) {
	c := newCounter()
	m := fake.New(callsThenDone(fake.Call("a", "echo", input("x")), fake.Call("b", "echo", input("y"))))
	a := &agentloop.Agent{
		Model: m, Tools: []agentloop.Tool{echoTool("echo", c)},
		Policy: agentloop.PolicyFunc(func(context.Context, agentloop.ToolCall, agentloop.View) agentloop.Decision {
			return agentloop.Decision{Verdict: agentloop.Hold, Labels: []string{"private"}}
		}),
	}
	store := agentloop.NewMemoryStore()
	s := reopen(t, store)
	wantReason(t, run(t, a, s, "go"), agentloop.ReasonHeld)
	if len(s.Labels()) != 0 {
		t.Fatalf("a held call added labels %v", s.Labels())
	}
	if err := s.Resolve("a", agentloop.Decision{Verdict: agentloop.Allow, Labels: []string{"extra"}}); err != nil {
		t.Fatal(err)
	}
	if err := s.Resolve("b", agentloop.Decision{Verdict: agentloop.Deny, Reason: "no"}); err != nil {
		t.Fatal(err)
	}
	s = reopen(t, store)
	if got := s.Labels(); !reflect.DeepEqual(got, []string{"extra", "private"}) {
		t.Fatalf("labels %v", got)
	}
	wantReason(t, run(t, a, s, ""), agentloop.ReasonCompleted)
	if c.count("x") != 1 || c.count("y") != 0 {
		t.Fatalf("runs %v", c.runs)
	}
	if !slices.Contains(s.Labels(), "private") {
		t.Fatal("lost a label")
	}
}
