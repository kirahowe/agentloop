package agentloop_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"sync/atomic"
	"testing"
	"time"

	"agentloop"
	"agentloop/fake"
)

func TestToolLoop(t *testing.T) {
	c := newCounter()
	m := fake.Script(
		fake.ToolUse("looking", fake.Call("a", "echo", input("one")), fake.Call("b", "echo", input("two"))),
		fake.Text("done"),
	)
	var events []agentloop.Event
	a := &agentloop.Agent{
		Model: m, ModelID: "m1", System: "be brief", Tools: []agentloop.Tool{echoTool("echo", c)},
		Observe: func(e agentloop.Event) { events = append(events, e) },
	}
	s, _ := newSession(t)
	out := run(t, a, s, "go")

	wantReason(t, out, agentloop.ReasonCompleted)
	if out.Text != "done" || out.Turns != 2 || out.Usage.Total() != 30 {
		t.Fatalf("outcome %+v", out)
	}
	want := []agentloop.EntryType{
		agentloop.EntryRunStarted, agentloop.EntryMessage, agentloop.EntryMessage,
		agentloop.EntryDecision, agentloop.EntryCallStarted, agentloop.EntryToolResult,
		agentloop.EntryDecision, agentloop.EntryCallStarted, agentloop.EntryToolResult,
		agentloop.EntryMessage, agentloop.EntryRunEnded,
	}
	if got := types(s.Entries()); !reflect.DeepEqual(got, want) {
		t.Fatalf("entries %v\nwant    %v", got, want)
	}
	for i, e := range s.Entries() {
		if e.Seq != i+1 || e.Time.IsZero() {
			t.Fatalf("entry %d: seq %d time %v", i, e.Seq, e.Time)
		}
	}

	msgs := s.Messages()
	if len(msgs) != 4 {
		t.Fatalf("want 4 messages, got %d: %+v", len(msgs), msgs)
	}
	checkConversation(t, msgs)
	if msgs[2].Parts[0].ToolResult.Content != "echo: one" || msgs[2].Parts[1].ToolResult.Content != "echo: two" {
		t.Fatalf("results %+v", msgs[2])
	}
	if msgs[1].Native == nil || msgs[1].Native.Provider != "fake" {
		t.Fatal("the assistant turn lost its native record")
	}

	reqs := m.Requests()
	if len(reqs) != 2 || reqs[0].Model != "m1" || reqs[0].System != "be brief" {
		t.Fatalf("requests %+v", reqs)
	}
	if len(reqs[0].Messages) != 1 || len(reqs[1].Messages) != 3 {
		t.Fatalf("request sizes %d, %d", len(reqs[0].Messages), len(reqs[1].Messages))
	}
	if len(reqs[0].Tools) != 1 || reqs[0].Tools[0].Name != "echo" {
		t.Fatalf("tools %+v", reqs[0].Tools)
	}

	entries, deltas := 0, 0
	for _, e := range events {
		if e.Entry != nil {
			entries++
		}
		if e.Delta != nil {
			deltas++
		}
	}
	if entries != len(s.Entries()) || deltas != 2 {
		t.Fatalf("observed %d entries and %d deltas", entries, deltas)
	}

	turns := s.Turns()
	if len(turns) != 2 || turns[0].Stop != agentloop.StopToolUse || turns[1].Stop != agentloop.StopEndTurn || turns[0].Message != 1 || turns[1].Message != 3 {
		t.Fatalf("turns %+v", turns)
	}
	if s.Usage().Total() != 30 {
		t.Fatalf("session usage %+v", s.Usage())
	}
}

func TestRunWithNothingToDo(t *testing.T) {
	a := &agentloop.Agent{Model: fake.Script()}
	s, _ := newSession(t)
	out := run(t, a, s, "")
	wantReason(t, out, agentloop.ReasonCompleted)
	if len(a.Model.(*fake.Model).Requests()) != 0 {
		t.Fatal("the model was called")
	}
	// Run again on a finished conversation.
	m := fake.Script(fake.Text("hello"))
	a = &agentloop.Agent{Model: m}
	run(t, a, s, "hi")
	out = run(t, a, s, "")
	wantReason(t, out, agentloop.ReasonCompleted)
	if out.Text != "hello" || len(m.Requests()) != 1 {
		t.Fatalf("outcome %+v, %d requests", out, len(m.Requests()))
	}
}

func TestSetupErrors(t *testing.T) {
	s, _ := newSession(t)
	c := newCounter()
	bad := agentloop.Func(agentloop.ToolSpec{Name: "bad", Parameters: json.RawMessage(`{"type":"nonsense"}`)}, nil)
	remote := agentloop.Func(agentloop.ToolSpec{Name: "remote", Parameters: json.RawMessage(`{"$ref":"https://example.com/schema.json"}`)}, nil)
	cases := map[string]*agentloop.Agent{
		"no model":       {},
		"bad schema":     {Model: fake.Script(), Tools: []agentloop.Tool{bad}},
		"remote ref":     {Model: fake.Script(), Tools: []agentloop.Tool{remote}},
		"duplicate name": {Model: fake.Script(), Tools: []agentloop.Tool{echoTool("x", c), echoTool("x", c)}},
	}
	for name, a := range cases {
		if _, err := a.Run(context.Background(), s, "hi"); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
	if len(s.Entries()) != 0 {
		t.Fatalf("a failed setup wrote %d entries", len(s.Entries()))
	}
}

func TestPolicyDenyAndAllow(t *testing.T) {
	c := newCounter()
	m := fake.Script(
		fake.ToolUse("", fake.Call("a", "echo", input("secret")), fake.Call("b", "echo", input("fine"))),
		fake.Text("ok"),
	)
	var views []agentloop.View
	a := &agentloop.Agent{
		Model: m, Tools: []agentloop.Tool{echoTool("echo", c)},
		Policy: agentloop.PolicyFunc(func(ctx context.Context, call agentloop.ToolCall, v agentloop.View) agentloop.Decision {
			views = append(views, v)
			if string(call.Input) == input("secret") {
				return agentloop.Decision{Verdict: agentloop.Deny, Reason: "not that one"}
			}
			return agentloop.Decision{Verdict: agentloop.Allow}
		}),
	}
	s, _ := newSession(t)
	out := run(t, a, s, "go")
	wantReason(t, out, agentloop.ReasonCompleted)

	if c.count("secret") != 0 || c.count("fine") != 1 {
		t.Fatalf("runs: %v", c.runs)
	}
	res := resultsOf(s.Messages())
	if !res["a"].IsError || res["a"].Content != "not that one" {
		t.Fatalf("denied result %+v", res["a"])
	}
	if res["b"].IsError || res["b"].Content != "echo: fine" {
		t.Fatalf("allowed result %+v", res["b"])
	}
	if got := s.View().ToolCalls; got != 2 {
		t.Fatalf("view ToolCalls = %d", got)
	}
	if len(views) != 2 || views[0].ToolCalls != 0 || views[1].ToolCalls != 1 {
		t.Fatalf("policy saw %+v", views)
	}
	info, _ := s.Call("a")
	if info.Started || !info.Finished || info.Decision.Verdict != agentloop.Deny {
		t.Fatalf("call a: %+v", info)
	}
	checkConversation(t, s.Messages())
}

func TestLabels(t *testing.T) {
	c := newCounter()
	labeller := agentloop.Func(
		agentloop.ToolSpec{Name: "read"},
		func(context.Context, json.RawMessage) (agentloop.Result, error) {
			return agentloop.Result{Content: "contents", Labels: []string{"private"}}, nil
		})
	m := fake.Script(
		fake.ToolUse("", fake.Call("r", "read", ""), fake.Call("n", "echo", input("net"))),
		fake.ToolUse("", fake.Call("n2", "echo", input("net again"))),
		fake.Text("done"),
	)
	var seen [][]string
	a := &agentloop.Agent{
		Model: m, Tools: []agentloop.Tool{labeller, echoTool("echo", c)},
		Policy: agentloop.PolicyFunc(func(ctx context.Context, call agentloop.ToolCall, v agentloop.View) agentloop.Decision {
			seen = append(seen, v.Labels)
			switch {
			case call.Name == "read":
				return agentloop.Decision{Verdict: agentloop.Allow, Labels: []string{"reading"}}
			case v.Has("private"):
				return agentloop.Decision{Verdict: agentloop.Deny, Reason: "no egress after private"}
			}
			return agentloop.Decision{Verdict: agentloop.Allow}
		}),
	}
	s, _ := newSession(t)
	out := run(t, a, s, "go")
	wantReason(t, out, agentloop.ReasonCompleted)

	if got := s.Labels(); !reflect.DeepEqual(got, []string{"private", "reading"}) {
		t.Fatalf("labels %v", got)
	}
	// Calls are settled one at a time, so the second call of the first
	// turn already sees what the first one's decision and result added.
	if !slices.Equal(seen[0], nil) || !slices.Equal(seen[1], []string{"private", "reading"}) {
		t.Fatalf("policy saw %v", seen)
	}
	res := resultsOf(s.Messages())
	if c.all() != 0 || res["n"].Content != "no egress after private" || res["n2"].Content != "no egress after private" {
		t.Fatalf("runs %v results %+v", c.runs, res)
	}
	if res["r"].Content != "contents" {
		t.Fatalf("%+v", res["r"])
	}
}

func TestSchemaRejectionAndUnknownTool(t *testing.T) {
	c := newCounter()
	asked := 0
	m := fake.Script(
		fake.ToolUse("",
			fake.Call("bad", "echo", `{"text":5}`),
			fake.Call("missing", "echo", `{}`),
			fake.Call("junk", "echo", `not json`),
			fake.Call("none", "nothing", `{}`),
		),
		fake.Text("done"),
	)
	a := &agentloop.Agent{
		Model: m, Tools: []agentloop.Tool{echoTool("echo", c)},
		Policy: agentloop.PolicyFunc(func(context.Context, agentloop.ToolCall, agentloop.View) agentloop.Decision {
			asked++
			return agentloop.Decision{Verdict: agentloop.Allow}
		}),
	}
	s, _ := newSession(t)
	out := run(t, a, s, "go")
	wantReason(t, out, agentloop.ReasonCompleted)
	if c.all() != 0 || asked != 0 {
		t.Fatalf("tool ran %d times, policy asked %d times", c.all(), asked)
	}
	res := resultsOf(s.Messages())
	for _, id := range []string{"bad", "missing", "junk", "none"} {
		if !res[id].IsError {
			t.Errorf("%s: %+v", id, res[id])
		}
	}
	contains(t, res["bad"].Content, "schema")
	contains(t, res["none"].Content, "no tool called nothing")
	for _, e := range s.Entries() {
		if e.Type == agentloop.EntryDecision && e.Decision.Verdict != agentloop.Deny {
			t.Fatalf("decision %+v", e.Decision)
		}
	}
	checkConversation(t, s.Messages())
}

func TestToolFailureAndTimeout(t *testing.T) {
	boom := agentloop.Func(agentloop.ToolSpec{Name: "boom"}, func(context.Context, json.RawMessage) (agentloop.Result, error) {
		return agentloop.Result{}, errors.New("disk on fire")
	})
	slow := agentloop.Func(agentloop.ToolSpec{Name: "slow"}, func(ctx context.Context, _ json.RawMessage) (agentloop.Result, error) {
		<-ctx.Done()
		return agentloop.Result{}, ctx.Err()
	})
	m := fake.Script(fake.ToolUse("", fake.Call("b", "boom", ""), fake.Call("s", "slow", "")), fake.Text("done"))
	a := &agentloop.Agent{Model: m, Tools: []agentloop.Tool{boom, slow}, ToolTimeout: 20 * time.Millisecond}
	s, _ := newSession(t)
	out := run(t, a, s, "go")
	wantReason(t, out, agentloop.ReasonCompleted)
	res := resultsOf(s.Messages())
	if !res["b"].IsError {
		t.Fatalf("%+v", res["b"])
	}
	contains(t, res["b"].Content, "disk on fire")
	if !res["s"].IsError {
		t.Fatalf("%+v", res["s"])
	}
	contains(t, res["s"].Content, "timed out")
}

func TestHoldResolveAndResume(t *testing.T) {
	c := newCounter()
	m := fake.New(callsThenDone(fake.Call("a", "echo", input("danger")), fake.Call("b", "echo", input("safe"))))
	a := &agentloop.Agent{
		Model: m, Tools: []agentloop.Tool{echoTool("echo", c)},
		Policy: agentloop.PolicyFunc(func(_ context.Context, call agentloop.ToolCall, _ agentloop.View) agentloop.Decision {
			if string(call.Input) == input("danger") {
				return agentloop.Decision{Verdict: agentloop.Hold, Reason: "needs a person"}
			}
			return agentloop.Decision{Verdict: agentloop.Allow}
		}),
	}
	store := agentloop.NewMemoryStore()
	s := reopen(t, store)
	out := run(t, a, s, "go")
	wantReason(t, out, agentloop.ReasonHeld)
	if len(out.Held) != 1 || out.Held[0].ID != "a" {
		t.Fatalf("held %+v", out.Held)
	}
	// The rest of the turn settled.
	if c.count("safe") != 1 || c.count("danger") != 0 {
		t.Fatalf("runs %v", c.runs)
	}
	if got := s.Held(); len(got) != 1 || got[0].ID != "a" {
		t.Fatalf("Held() %+v", got)
	}
	if got := s.Pending(); len(got) != 1 || got[0].ID != "a" {
		t.Fatalf("Pending() %+v", got)
	}
	// A new prompt does not get past a held call.
	out = run(t, a, s, "more")
	wantReason(t, out, agentloop.ReasonHeld)
	for _, msg := range s.Messages() {
		if msg.Role == agentloop.User && msg.Text() == "more" {
			t.Fatal("the prompt was added while a call was held")
		}
	}

	// Resolve errors.
	if err := s.Resolve("b", agentloop.Decision{Verdict: agentloop.Allow}); err == nil {
		t.Error("resolved a call that is not held")
	}
	if err := s.Resolve("nope", agentloop.Decision{Verdict: agentloop.Allow}); err == nil {
		t.Error("resolved a call that does not exist")
	}
	if err := s.Resolve("a", agentloop.Decision{Verdict: agentloop.Hold}); err == nil {
		t.Error("resolved a call with hold")
	}
	if len(s.Held()) != 1 {
		t.Fatal("a refused Resolve changed the session")
	}

	// Resolve survives a reopen, and a fresh process carries on.
	if err := s.Resolve("a", agentloop.Decision{Verdict: agentloop.Allow, Labels: []string{"approved"}}); err != nil {
		t.Fatal(err)
	}
	if err := s.Resolve("a", agentloop.Decision{Verdict: agentloop.Allow}); err == nil {
		t.Error("resolved a call twice")
	}
	s = reopen(t, store)
	if len(s.Held()) != 0 || !slices.Contains(s.Labels(), "approved") {
		t.Fatalf("held %v labels %v", s.Held(), s.Labels())
	}
	out = run(t, a, s, "")
	wantReason(t, out, agentloop.ReasonCompleted)
	if c.count("danger") != 1 || c.count("safe") != 1 {
		t.Fatalf("runs %v", c.runs)
	}
	checkConversation(t, s.Messages())
	msgs := s.Messages()
	if msgs[2].Parts[0].ToolResult.CallID != "a" || msgs[2].Parts[1].ToolResult.CallID != "b" {
		t.Fatalf("results are not in call order: %+v", msgs[2])
	}
	if got := s.View().ToolCalls; got != 2 {
		t.Fatalf("ToolCalls = %d", got)
	}
}

func TestHoldResolvedWithDeny(t *testing.T) {
	c := newCounter()
	m := fake.New(callsThenDone(fake.Call("a", "echo", input("danger"))))
	a := &agentloop.Agent{
		Model: m, Tools: []agentloop.Tool{echoTool("echo", c)},
		Policy: agentloop.PolicyFunc(func(context.Context, agentloop.ToolCall, agentloop.View) agentloop.Decision {
			return agentloop.Decision{Verdict: agentloop.Hold}
		}),
	}
	s, _ := newSession(t)
	wantReason(t, run(t, a, s, "go"), agentloop.ReasonHeld)
	if err := s.Resolve("a", agentloop.Decision{Verdict: agentloop.Deny, Reason: "no thanks", Labels: []string{"ignored"}}); err != nil {
		t.Fatal(err)
	}
	wantReason(t, run(t, a, s, ""), agentloop.ReasonCompleted)
	if c.all() != 0 {
		t.Fatal("a denied call ran")
	}
	if res := resultsOf(s.Messages())["a"]; !res.IsError || res.Content != "no thanks" {
		t.Fatalf("%+v", res)
	}
	if len(s.Labels()) != 0 {
		t.Fatalf("labels of a denial were added: %v", s.Labels())
	}
}

func TestBadVerdictIsDenied(t *testing.T) {
	c := newCounter()
	m := fake.New(callsThenDone(fake.Call("a", "echo", input("x"))))
	a := &agentloop.Agent{
		Model: m, Tools: []agentloop.Tool{echoTool("echo", c)},
		Policy: agentloop.PolicyFunc(func(context.Context, agentloop.ToolCall, agentloop.View) agentloop.Decision {
			return agentloop.Decision{}
		}),
	}
	s, _ := newSession(t)
	wantReason(t, run(t, a, s, "go"), agentloop.ReasonCompleted)
	if c.all() != 0 {
		t.Fatal("a call with no verdict ran")
	}
}

func TestLimits(t *testing.T) {
	var n atomic.Int64
	loop := func(*agentloop.Request) (*agentloop.Response, error) {
		return fake.ToolUse("", fake.Call(fmt.Sprintf("c%d", n.Add(1)), "echo", input("x"))), nil
	}
	t.Run("turns", func(t *testing.T) {
		c := newCounter()
		a := &agentloop.Agent{Model: fake.New(loop), Tools: []agentloop.Tool{echoTool("echo", c)}, Limits: agentloop.Limits{MaxTurns: 3}}
		s, _ := newSession(t)
		out := run(t, a, s, "go")
		wantReason(t, out, agentloop.ReasonLimit)
		if out.Limit != agentloop.LimitTurns || out.Turns != 3 || c.all() != 3 {
			t.Fatalf("%+v, %d runs", out, c.all())
		}
		checkConversation(t, s.Messages())
		// The limit is per run: another run gets three more turns.
		out = run(t, a, s, "again")
		if out.Turns != 3 {
			t.Fatalf("second run %+v", out)
		}
	})
	t.Run("tokens", func(t *testing.T) {
		c := newCounter()
		a := &agentloop.Agent{Model: fake.New(loop), Tools: []agentloop.Tool{echoTool("echo", c)}, Limits: agentloop.Limits{MaxTokens: 40}}
		s, _ := newSession(t)
		out := run(t, a, s, "go")
		wantReason(t, out, agentloop.ReasonLimit)
		if out.Limit != agentloop.LimitTokens || out.Turns != 3 || out.Usage.Total() != 45 {
			t.Fatalf("%+v", out)
		}
	})
	t.Run("tool calls", func(t *testing.T) {
		c := newCounter()
		m := fake.Script(fake.ToolUse("",
			fake.Call("a", "echo", input("1")), fake.Call("b", "echo", input("2")), fake.Call("c", "echo", input("3"))))
		a := &agentloop.Agent{Model: m, Tools: []agentloop.Tool{echoTool("echo", c)}, Limits: agentloop.Limits{MaxToolCalls: 2}}
		s, _ := newSession(t)
		out := run(t, a, s, "go")
		wantReason(t, out, agentloop.ReasonLimit)
		if out.Limit != agentloop.LimitToolCalls || c.all() != 2 {
			t.Fatalf("%+v, %d runs", out, c.all())
		}
		res := resultsOf(s.Messages())
		if !res["c"].IsError {
			t.Fatalf("c: %+v", res["c"])
		}
		checkConversation(t, s.Messages())
	})
	t.Run("a call exactly at the limit is fine", func(t *testing.T) {
		c := newCounter()
		m := fake.Script(fake.ToolUse("", fake.Call("a", "echo", input("1"))), fake.Text("done"))
		a := &agentloop.Agent{Model: m, Tools: []agentloop.Tool{echoTool("echo", c)}, Limits: agentloop.Limits{MaxToolCalls: 1}}
		s, _ := newSession(t)
		wantReason(t, run(t, a, s, "go"), agentloop.ReasonCompleted)
	})
}

func TestStops(t *testing.T) {
	t.Run("refusal", func(t *testing.T) {
		a := &agentloop.Agent{Model: fake.Script(fake.WithStop(fake.Text("no"), agentloop.StopRefusal))}
		s, _ := newSession(t)
		out := run(t, a, s, "go")
		wantReason(t, out, agentloop.ReasonRefused)
		if out.Text != "no" {
			t.Fatalf("%+v", out)
		}
		// Resuming without a prompt reports the same thing.
		wantReason(t, run(t, a, s, ""), agentloop.ReasonRefused)
	})
	t.Run("max tokens", func(t *testing.T) {
		a := &agentloop.Agent{Model: fake.Script(fake.WithStop(fake.Text("cut"), agentloop.StopMaxTokens))}
		s, _ := newSession(t)
		wantReason(t, run(t, a, s, "go"), agentloop.ReasonMaxTokens)
	})
	t.Run("pause goes on", func(t *testing.T) {
		m := fake.Script(fake.WithStop(fake.Text("partway"), agentloop.StopPause), fake.Text("whole"))
		a := &agentloop.Agent{Model: m}
		s, _ := newSession(t)
		out := run(t, a, s, "go")
		wantReason(t, out, agentloop.ReasonCompleted)
		if out.Text != "whole" || out.Turns != 2 {
			t.Fatalf("%+v", out)
		}
	})
}

func TestCutOffTurnGetsErrorResults(t *testing.T) {
	c := newCounter()
	cut := fake.WithStop(fake.ToolUse("", fake.Call("a", "echo", input("x"))), agentloop.StopMaxTokens)
	m := fake.Script(cut, fake.Text("recovered"))
	a := &agentloop.Agent{Model: m, Tools: []agentloop.Tool{echoTool("echo", c)}}
	s, _ := newSession(t)
	wantReason(t, run(t, a, s, "go"), agentloop.ReasonMaxTokens)
	if len(s.Pending()) != 0 {
		t.Fatal("the cut-off call was left open when the run ended")
	}
	checkConversation(t, s.Messages())
	out := run(t, a, s, "again")
	wantReason(t, out, agentloop.ReasonCompleted)
	if c.all() != 0 {
		t.Fatal("a cut-off call ran")
	}
	res := resultsOf(s.Messages())["a"]
	if !res.IsError {
		t.Fatalf("%+v", res)
	}
	contains(t, res.Content, "not run")
	checkConversation(t, s.Messages())
}

func TestRetries(t *testing.T) {
	overloaded := &agentloop.ModelError{Kind: agentloop.Overloaded, Status: 529, Message: "busy"}
	newAgent := func(m agentloop.Model, p *agentloop.RetryPolicy, sl *sleeper) *agentloop.Agent {
		return &agentloop.Agent{Model: m, Retry: p, Sleep: sl.sleep, Now: func() time.Time { return epoch }}
	}
	policy := &agentloop.RetryPolicy{Retries: 3, Backoff: 2 * time.Second, MaxWait: 60 * time.Second}

	t.Run("backoff doubles and each retry is logged", func(t *testing.T) {
		sl := &sleeper{}
		m := fake.Script(overloaded, overloaded, fake.Text("ok"))
		s, _ := newSession(t)
		out := run(t, newAgent(m, policy, sl), s, "go")
		wantReason(t, out, agentloop.ReasonCompleted)
		if !slices.Equal(sl.waits, []time.Duration{2 * time.Second, 4 * time.Second}) {
			t.Fatalf("waits %v", sl.waits)
		}
		if n := countType(s.Entries(), agentloop.EntryModelError); n != 2 {
			t.Fatalf("%d model_error entries", n)
		}
		for _, e := range s.Entries() {
			if e.Type == agentloop.EntryModelError && (!e.ModelError.Retry || e.ModelError.Error.Kind != agentloop.Overloaded) {
				t.Fatalf("%+v", e.ModelError)
			}
		}
		if out.Turns != 1 {
			t.Fatalf("turns = %d", out.Turns)
		}
	})
	t.Run("exhausted", func(t *testing.T) {
		sl := &sleeper{}
		m := fake.Script(overloaded, overloaded, overloaded)
		s, _ := newSession(t)
		out := run(t, newAgent(m, &agentloop.RetryPolicy{Retries: 2, Backoff: time.Second}, sl), s, "go")
		wantReason(t, out, agentloop.ReasonModelError)
		if out.Error == nil || out.Error.Kind != agentloop.Overloaded || len(sl.waits) != 2 {
			t.Fatalf("%+v waits %v", out, sl.waits)
		}
		entries := s.Entries()
		var last *agentloop.ModelErrorEntry
		for _, e := range entries {
			if e.Type == agentloop.EntryModelError {
				last = e.ModelError
			}
		}
		if last.Retry {
			t.Fatal("the last failure was logged as retried")
		}
		if entries[len(entries)-1].Type != agentloop.EntryRunEnded {
			t.Fatal("no run_ended")
		}
	})
	t.Run("rate limit waits for retry-after", func(t *testing.T) {
		sl := &sleeper{}
		rl := &agentloop.ModelError{Kind: agentloop.RateLimited, Message: "slow down", RetryAfter: epoch.Add(10 * time.Second)}
		s, _ := newSession(t)
		out := run(t, newAgent(fake.Script(rl, fake.Text("ok")), policy, sl), s, "go")
		wantReason(t, out, agentloop.ReasonCompleted)
		if !slices.Equal(sl.waits, []time.Duration{10 * time.Second}) {
			t.Fatalf("waits %v", sl.waits)
		}
	})
	t.Run("a wait beyond the maximum ends the run", func(t *testing.T) {
		sl := &sleeper{}
		rl := &agentloop.ModelError{Kind: agentloop.RateLimited, Message: "slow down", RetryAfter: epoch.Add(5 * time.Minute)}
		s, _ := newSession(t)
		out := run(t, newAgent(fake.Script(rl, fake.Text("ok")), policy, sl), s, "go")
		wantReason(t, out, agentloop.ReasonModelError)
		if len(sl.waits) != 0 || out.Error.Kind != agentloop.RateLimited {
			t.Fatalf("%+v waits %v", out, sl.waits)
		}
	})
	t.Run("retry-after later than the backoff wins", func(t *testing.T) {
		sl := &sleeper{}
		e := &agentloop.ModelError{Kind: agentloop.Unreachable, Message: "down", RetryAfter: epoch.Add(9 * time.Second)}
		s, _ := newSession(t)
		run(t, newAgent(fake.Script(e, fake.Text("ok")), policy, sl), s, "go")
		if !slices.Equal(sl.waits, []time.Duration{9 * time.Second}) {
			t.Fatalf("waits %v", sl.waits)
		}
	})
	t.Run("other kinds are not retried", func(t *testing.T) {
		for _, kind := range []agentloop.ErrorKind{agentloop.Unauthorized, agentloop.Billing, agentloop.NotFound, agentloop.InvalidRequest} {
			sl := &sleeper{}
			s, _ := newSession(t)
			out := run(t, newAgent(fake.Script(&agentloop.ModelError{Kind: kind, Message: "x"}, fake.Text("ok")), policy, sl), s, "go")
			wantReason(t, out, agentloop.ReasonModelError)
			if len(sl.waits) != 0 || out.Error.Kind != kind {
				t.Fatalf("%s: %+v", kind, out)
			}
		}
	})
	t.Run("a plain error counts as unreachable", func(t *testing.T) {
		sl := &sleeper{}
		s, _ := newSession(t)
		out := run(t, newAgent(fake.Script(errors.New("connection reset"), fake.Text("ok")), policy, sl), s, "go")
		wantReason(t, out, agentloop.ReasonCompleted)
		if len(sl.waits) != 1 {
			t.Fatalf("waits %v", sl.waits)
		}
	})
	t.Run("nil policy uses the default", func(t *testing.T) {
		sl := &sleeper{}
		s, _ := newSession(t)
		out := run(t, newAgent(fake.Script(overloaded, fake.Text("ok")), nil, sl), s, "go")
		wantReason(t, out, agentloop.ReasonCompleted)
		if !slices.Equal(sl.waits, []time.Duration{agentloop.DefaultRetry.Backoff}) {
			t.Fatalf("waits %v", sl.waits)
		}
	})
	t.Run("a session can be resumed after a model error", func(t *testing.T) {
		sl := &sleeper{}
		s, _ := newSession(t)
		a := newAgent(fake.Script(&agentloop.ModelError{Kind: agentloop.Billing, Message: "x"}, fake.Text("ok")), policy, sl)
		wantReason(t, run(t, a, s, "go"), agentloop.ReasonModelError)
		out := run(t, a, s, "")
		wantReason(t, out, agentloop.ReasonCompleted)
		checkConversation(t, s.Messages())
	})
}

func TestCancelDuringModelCall(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	m := fake.New(func(*agentloop.Request) (*agentloop.Response, error) {
		cancel()
		// What an adapter returns when its request is cancelled.
		return nil, &agentloop.ModelError{Kind: agentloop.Unreachable, Message: "context canceled"}
	})
	sl := &sleeper{}
	a := &agentloop.Agent{Model: m, Sleep: sl.sleep}
	s, _ := newSession(t)
	out, err := a.Run(ctx, s, "go")
	if err != nil {
		t.Fatal(err)
	}
	wantReason(t, out, agentloop.ReasonCanceled)
	if countType(s.Entries(), agentloop.EntryModelError) != 0 || len(sl.waits) != 0 {
		t.Fatal("a cancelled request was logged or retried as a model error")
	}
	if last := s.Entries()[len(s.Entries())-1]; last.Type != agentloop.EntryRunEnded || last.RunEnded.Reason != agentloop.ReasonCanceled {
		t.Fatalf("last entry %+v", last)
	}
}

func TestCancelDuringRetryWait(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	m := fake.Script(&agentloop.ModelError{Kind: agentloop.Overloaded, Message: "busy"}, fake.Text("ok"))
	a := &agentloop.Agent{Model: m, Sleep: func(ctx context.Context, d time.Duration) error {
		cancel()
		return ctx.Err()
	}}
	s, _ := newSession(t)
	out, err := a.Run(ctx, s, "go")
	if err != nil {
		t.Fatal(err)
	}
	wantReason(t, out, agentloop.ReasonCanceled)
}

func TestCancelAlreadyDone(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	m := fake.Script(fake.Text("never"))
	a := &agentloop.Agent{Model: m}
	s, _ := newSession(t)
	out, err := a.Run(ctx, s, "go")
	if err != nil {
		t.Fatal(err)
	}
	wantReason(t, out, agentloop.ReasonCanceled)
}

func TestCancelDuringTool(t *testing.T) {
	for _, returnsError := range []bool{true, false} {
		ctx, cancel := context.WithCancel(context.Background())
		c := newCounter()
		stuck := agentloop.Func(agentloop.ToolSpec{Name: "stuck"}, func(ctx context.Context, _ json.RawMessage) (agentloop.Result, error) {
			cancel()
			<-ctx.Done()
			if returnsError {
				return agentloop.Result{}, ctx.Err()
			}
			return agentloop.Result{Content: "finished anyway"}, nil
		})
		m := fake.Script(
			fake.ToolUse("", fake.Call("a", "stuck", ""), fake.Call("b", "echo", input("later"))),
			fake.Text("done"),
		)
		a := &agentloop.Agent{Model: m, Tools: []agentloop.Tool{stuck, echoTool("echo", c)}}
		store := agentloop.NewMemoryStore()
		s := reopen(t, store)
		out, err := a.Run(ctx, s, "go")
		if err != nil {
			t.Fatal(err)
		}
		wantReason(t, out, agentloop.ReasonCanceled)
		info, _ := s.Call("a")
		if !info.Finished || info.Result == nil {
			t.Fatalf("the open call has no result: %+v", info)
		}
		if returnsError {
			// A call cut off by cancellation may have taken effect, and
			// the model is told so rather than that it was canceled.
			contains(t, info.Result.Content, "may or may not have taken effect")
			if !info.Result.IsError {
				t.Fatalf("%+v", info.Result)
			}
		} else if info.Result.Content != "finished anyway" {
			t.Fatalf("%+v", info.Result)
		}
		if info, _ := s.Call("b"); info.Started || info.Finished {
			t.Fatalf("a later call was touched: %+v", info)
		}
		last := s.Entries()[len(s.Entries())-1]
		if last.Type != agentloop.EntryRunEnded {
			t.Fatalf("last entry %v", last.Type)
		}

		// Resuming runs the call that never started, and nothing twice.
		s = reopen(t, store)
		out = run(t, a, s, "")
		wantReason(t, out, agentloop.ReasonCompleted)
		if c.count("later") != 1 {
			t.Fatalf("runs %v", c.runs)
		}
		checkConversation(t, s.Messages())
	}
}
