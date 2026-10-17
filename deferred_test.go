package agentloop_test

import (
	"context"
	"encoding/json"
	"reflect"
	"slices"
	"testing"

	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/fake"
)

// deferTool records the Call it was given and returns the deferred
// labels as its result's, with usage and a report, as a reader does.
type deferTool struct {
	got   []agentloop.Call
	check string
}

func (t *deferTool) Spec() agentloop.ToolSpec {
	return agentloop.ToolSpec{Name: "reader", Parameters: json.RawMessage(`{"type":"object"}`)}
}

func (t *deferTool) Call(ctx context.Context, in json.RawMessage) (agentloop.Result, error) {
	return t.CallWith(ctx, agentloop.Call{Input: in})
}

func (t *deferTool) CallWith(_ context.Context, c agentloop.Call) (agentloop.Result, error) {
	t.got = append(t.got, c)
	return agentloop.Result{
		Content: "answer", Labels: c.Deferred,
		Usage:      []agentloop.ModelUsage{{Model: "small", Usage: agentloop.Usage{InputTokens: 1000, OutputTokens: 20}}},
		Quarantine: &agentloop.QuarantineReport{Tool: "inner", Dropped: []string{}},
	}, nil
}

func (t *deferTool) CheckInput(json.RawMessage) string { return t.check }

func deferPolicy(verdict agentloop.Verdict) agentloop.Policy {
	return agentloop.PolicyFunc(func(context.Context, agentloop.ToolCall, agentloop.View) agentloop.Decision {
		return agentloop.Decision{Verdict: verdict, Labels: []string{"private"}, Deferred: []string{"contaminated"}}
	})
}

func TestDeferredLabelsWaitForTheResult(t *testing.T) {
	s, store := newSession(t)
	tool := &deferTool{}
	var labelsAtCall []string
	a := &agentloop.Agent{
		Model:  fake.Script(fake.ToolUse("", fake.Call("c1", "reader", "")), fake.Text("done")),
		Tools:  []agentloop.Tool{tool},
		Policy: deferPolicy(agentloop.Allow),
		Observe: func(ev agentloop.Event) {
			if ev.Entry != nil && ev.Entry.Type == agentloop.EntryCallStarted {
				labelsAtCall = s.Labels()
			}
		},
	}
	out := run(t, a, s, "go")
	wantReason(t, out, agentloop.ReasonCompleted)
	// At decision time the session takes the labels, not the deferred ones.
	if !reflect.DeepEqual(labelsAtCall, []string{"private"}) {
		t.Errorf("labels as the call ran = %v", labelsAtCall)
	}
	if len(tool.got) != 1 || !reflect.DeepEqual(tool.got[0].Deferred, []string{"contaminated"}) || !reflect.DeepEqual(tool.got[0].Labels, []string{"private"}) {
		t.Errorf("tool got %+v", tool.got)
	}
	// The result carries them on.
	if !reflect.DeepEqual(s.Labels(), []string{"contaminated", "private"}) {
		t.Errorf("labels = %v", s.Labels())
	}
	var dec *agentloop.DecisionEntry
	var res *agentloop.ToolResultEntry
	for _, e := range reopen(t, store).Entries() {
		if e.Decision != nil {
			dec = e.Decision
		}
		if e.ToolResult != nil {
			res = e.ToolResult
		}
	}
	if dec == nil || !reflect.DeepEqual(dec.Deferred, []string{"contaminated"}) {
		t.Errorf("decision = %+v", dec)
	}
	if res == nil || len(res.Usage) != 1 || res.Quarantine == nil || res.Quarantine.Tool != "inner" {
		t.Errorf("result = %+v", res)
	}
	// The tool's model call counts toward the run's usage and the session's.
	want := agentloop.Usage{InputTokens: 2*fake.DefaultUsage.InputTokens + 1000, OutputTokens: 2*fake.DefaultUsage.OutputTokens + 20}
	if out.Usage != want || s.Usage() != want || reopen(t, store).Usage() != want {
		t.Errorf("usage = %+v, session %+v, want %+v", out.Usage, s.Usage(), want)
	}
}

func TestToolUsageCountsTowardMaxTokens(t *testing.T) {
	s, _ := newSession(t)
	a := &agentloop.Agent{
		Model:  fake.Script(fake.ToolUse("", fake.Call("c1", "reader", "")), fake.Text("never")),
		Tools:  []agentloop.Tool{&deferTool{}},
		Limits: agentloop.Limits{MaxTokens: 500},
	}
	out := run(t, a, s, "go")
	wantReason(t, out, agentloop.ReasonLimit)
	if out.Limit != agentloop.LimitTokens || out.Turns != 1 {
		t.Errorf("outcome = %+v", out)
	}
}

func TestForbidAppliesToDeferredLabels(t *testing.T) {
	s, _ := newSession(t)
	tool := &deferTool{}
	a := &agentloop.Agent{
		Model:  fake.Script(fake.ToolUse("", fake.Call("c1", "reader", "")), fake.Text("done")),
		Tools:  []agentloop.Tool{tool},
		Policy: deferPolicy(agentloop.Allow),
		Forbid: []string{"contaminated"},
	}
	run(t, a, s, "go")
	info, _ := s.Call("c1")
	if info.Decision.Verdict != agentloop.Deny || info.Decision.Reason != agentloop.ForbiddenDecision || len(tool.got) != 0 {
		t.Errorf("decision = %+v, calls %d", info.Decision, len(tool.got))
	}
	if len(s.Labels()) != 0 {
		t.Errorf("labels = %v", s.Labels())
	}
}

func TestResolveTakesDeferredFromThePolicy(t *testing.T) {
	s, _ := newSession(t)
	tool := &deferTool{}
	a := &agentloop.Agent{
		Model:  fake.Script(fake.ToolUse("", fake.Call("c1", "reader", "")), fake.Text("done")),
		Tools:  []agentloop.Tool{tool},
		Policy: deferPolicy(agentloop.Hold),
	}
	out := run(t, a, s, "go")
	wantReason(t, out, agentloop.ReasonHeld)
	held, _ := s.Call("c1")
	if held.Decision.Deferred != nil {
		t.Errorf("a hold recorded deferred labels: %+v", held.Decision)
	}
	res := agentloop.Resolution{CallID: "c1", Verdict: agentloop.Allow, InputSHA256: out.Held[0].InputSHA256}
	// With the deferred label forbidden, a person cannot allow it.
	b := *a
	b.Forbid = []string{"contaminated"}
	if err := b.CheckResolve(context.Background(), s, res); err == nil {
		t.Error("a forbidden deferred label was allowed")
	}
	a.Policy = deferPolicy(agentloop.Allow)
	if err := a.Resolve(context.Background(), s, res); err != nil {
		t.Fatal(err)
	}
	run(t, a, s, "")
	if len(tool.got) != 1 || !reflect.DeepEqual(tool.got[0].Deferred, []string{"contaminated"}) {
		t.Errorf("tool got %+v", tool.got)
	}
}

func TestRefusedResultKeepsUsageNotReport(t *testing.T) {
	s, _ := newSession(t)
	a := &agentloop.Agent{
		Model: fake.Script(fake.ToolUse("", fake.Call("c1", "reader", "")), fake.Text("done")),
		Tools: []agentloop.Tool{&deferTool{}},
		Policy: agentloop.PolicyFunc(func(context.Context, agentloop.ToolCall, agentloop.View) agentloop.Decision {
			return agentloop.Decision{Verdict: agentloop.Allow}
		}),
	}
	// The policy gives no deferred labels, so the Forbid check at decision
	// passes; the result's own labels are refused.
	tool := a.Tools[0].(*deferTool)
	a.Tools[0] = agentloop.Func(tool.Spec(), func(ctx context.Context, in json.RawMessage) (agentloop.Result, error) {
		r, _ := tool.CallWith(ctx, agentloop.Call{Input: in, Deferred: []string{"private"}})
		return r, nil
	})
	a.Forbid = []string{"private"}
	out := run(t, a, s, "go")
	if len(out.Refused) != 1 {
		t.Fatalf("refused = %+v", out.Refused)
	}
	info, _ := s.Call("c1")
	if info.Result.Content != agentloop.RefusedResult {
		t.Errorf("result = %+v", info.Result)
	}
	var res *agentloop.ToolResultEntry
	for _, e := range s.Entries() {
		if e.ToolResult != nil {
			res = e.ToolResult
		}
	}
	if len(res.Usage) != 1 || res.Quarantine != nil {
		t.Errorf("refused entry = %+v", res)
	}
	if out.Usage.InputTokens != 2*fake.DefaultUsage.InputTokens+1000 {
		t.Errorf("usage = %+v", out.Usage)
	}
}

func TestInputCheckerDenies(t *testing.T) {
	s, _ := newSession(t)
	tool := &deferTool{check: "The answer schema is not usable."}
	a := &agentloop.Agent{
		Model: fake.Script(fake.ToolUse("", fake.Call("c1", "reader", "")), fake.Text("done")),
		Tools: []agentloop.Tool{tool},
	}
	run(t, a, s, "go")
	info, _ := s.Call("c1")
	if info.Decision.Verdict != agentloop.Deny || info.Decision.Reason != tool.check || len(tool.got) != 0 {
		t.Errorf("decision = %+v", info.Decision)
	}
}

// For any policy labels, deferred labels and forbidden labels: the
// session never takes a deferred label before the result, and never
// holds a forbidden one.
func TestPropertyDeferredLabels(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		labels := genSomeLabels().Draw(t, "labels")
		deferred := genSomeLabels().Draw(t, "deferred")
		forbid := genSomeLabels().Draw(t, "forbid")
		s, err := agentloop.OpenSession(agentloop.NewMemoryStore())
		if err != nil {
			t.Fatal(err)
		}
		var atCall []string
		started := false
		tool := &deferTool{}
		a := &agentloop.Agent{
			Model: fake.Script(fake.ToolUse("", fake.Call("c1", "reader", "")), fake.Text("done")),
			Tools: []agentloop.Tool{tool},
			Policy: agentloop.PolicyFunc(func(context.Context, agentloop.ToolCall, agentloop.View) agentloop.Decision {
				return agentloop.Decision{Verdict: agentloop.Allow, Labels: labels, Deferred: deferred}
			}),
			Forbid: forbid,
			Observe: func(ev agentloop.Event) {
				if ev.Entry != nil && ev.Entry.Type == agentloop.EntryCallStarted {
					atCall, started = s.Labels(), true
				}
			},
		}
		out, err := a.Run(context.Background(), s, "go")
		if err != nil {
			t.Fatal(err)
		}
		bad := len(agentloop.Forbidden(append(slices.Clone(labels), deferred...), forbid)) > 0
		if bad == started {
			t.Fatalf("forbidden %v, started %v", bad, started)
		}
		if started {
			for _, l := range deferred {
				if !slices.Contains(labels, l) && slices.Contains(atCall, l) {
					t.Fatalf("deferred %s held before the result: %v", l, atCall)
				}
			}
		}
		if f := agentloop.Forbidden(s.Labels(), forbid); len(f) > 0 {
			t.Fatalf("session holds forbidden %v (outcome %+v)", f, out)
		}
	})
}

// A tool's model calls can cross the token limit in the middle of a turn.
// The turn's later calls are then not run, and the run ends at the limit.
func TestTokenLimitBetweenToolCalls(t *testing.T) {
	s, _ := newSession(t)
	tool := &deferTool{}
	a := &agentloop.Agent{
		Model:  fake.Script(fake.ToolUse("", fake.Call("c1", "reader", ""), fake.Call("c2", "reader", "")), fake.Text("never")),
		Tools:  []agentloop.Tool{tool},
		Limits: agentloop.Limits{MaxTokens: 500},
	}
	out := run(t, a, s, "go")
	wantReason(t, out, agentloop.ReasonLimit)
	if out.Limit != agentloop.LimitTokens || len(tool.got) != 1 {
		t.Fatalf("outcome %+v, %d calls ran", out, len(tool.got))
	}
	info, _ := s.Call("c2")
	if info.Result == nil || !info.Result.IsError || info.Started {
		t.Errorf("second call = %+v", info)
	}
}
