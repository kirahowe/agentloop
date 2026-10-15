package agentloop_test

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/fake"
)

// A turn the provider paused may still carry calls that were cut off. The
// next request must answer them, or the provider refuses the conversation.
func TestPausedTurnCallsAreAnswered(t *testing.T) {
	c := newCounter()
	m := fake.Script(
		fake.WithStop(fake.ToolUse("", fake.Call("p1", "echo", input("x"))), agentloop.StopPause),
		fake.Text("done"),
	)
	a := &agentloop.Agent{Model: m, Tools: []agentloop.Tool{echoTool("echo", c)}}
	s, _ := newSession(t)
	out := run(t, a, s, "go")
	wantReason(t, out, agentloop.ReasonCompleted)
	reqs := m.Requests()
	if len(reqs) != 2 {
		t.Fatalf("%d requests", len(reqs))
	}
	checkConversation(t, reqs[1].Messages)
	if c.all() != 0 {
		t.Fatal("a cut-off call ran")
	}
}

// What a caller is handed must not be the session's own memory.
func TestSessionDoesNotShareWhatItHandsOut(t *testing.T) {
	policy := agentloop.PolicyFunc(func(context.Context, agentloop.ToolCall, agentloop.View) agentloop.Decision {
		return agentloop.Decision{Verdict: agentloop.Allow, Labels: []string{"private"}}
	})
	labels := []string{"untrusted"}
	tool := agentloop.Func(agentloop.ToolSpec{Name: "t"}, func(context.Context, json.RawMessage) (agentloop.Result, error) {
		return agentloop.Result{Content: "ok", Labels: labels}, nil
	})
	a := &agentloop.Agent{Model: fake.Script(fake.ToolUse("", fake.Call("a", "t", "")), fake.Text("done")), Tools: []agentloop.Tool{tool}, Policy: policy}
	s, store := newSession(t)
	out := run(t, a, s, "go")
	out.Text, out.Turns = "MUTATED", -1
	labels[0] = "MUTATED"
	for _, e := range s.Entries() {
		if e.Type == agentloop.EntryRunEnded && (e.RunEnded.Text != "done" || e.RunEnded.Turns != 2) {
			t.Fatalf("the run_ended entry follows the caller's Outcome: %+v", e.RunEnded)
		}
	}
	if got := s.Labels(); !reflect.DeepEqual(got, []string{"private", "untrusted"}) {
		t.Fatalf("labels %v", got)
	}
	if !reflect.DeepEqual(s.Entries(), reopen(t, store).Entries()) {
		t.Fatal("the session differs from the one reopened from its store")
	}
}

// A session that has run is the session that reopens, even when the
// provider's raw JSON has spacing, HTML characters or escapes that JSON
// storage does not keep, and when a tool's output is not valid UTF-8.
func TestReopenedSessionEqualsLive(t *testing.T) {
	native := json.RawMessage("{ \"a\" : \"<b>&\\u2028\",\n \"sig\" : \"AB==\" }")
	resp := fake.ToolUse("hi", fake.Call("a", "t", " {  } "), fake.Call("b", "t", "{bad"))
	resp.Message.Native = &agentloop.Native{Provider: "fake", Model: "m", Data: native}
	tool := agentloop.Func(agentloop.ToolSpec{Name: "t"}, func(context.Context, json.RawMessage) (agentloop.Result, error) {
		return agentloop.Result{Content: "bin\xff\xfe"}, nil
	})
	m := fake.Script(resp, fake.Text("done"))
	a := &agentloop.Agent{Model: m, Tools: []agentloop.Tool{tool}}
	s, store := newSession(t)
	run(t, a, s, "go")
	r := reopen(t, store)
	if !reflect.DeepEqual(s.Messages(), r.Messages()) {
		t.Fatalf("live and reopened conversations differ:\n%+v\n%+v", s.Messages(), r.Messages())
	}
	// What the model is given on a later request is what was recorded.
	last := m.Requests()[1].Messages[1]
	if !reflect.DeepEqual(last, r.Messages()[1]) {
		t.Fatalf("the replayed turn differs from the reopened one:\n%+v\n%+v", last, r.Messages()[1])
	}
	var v any
	if err := json.Unmarshal(last.Native.Data, &v); err != nil || v.(map[string]any)["sig"] != "AB==" {
		t.Fatalf("native record %s: %v", last.Native.Data, err)
	}
}

// A call allowed by one run can be settled by an agent that no longer has
// the tool. Nothing runs and nothing panics.
func TestResumeWithoutTheTool(t *testing.T) {
	c := newCounter()
	a := &agentloop.Agent{Model: fake.New(callsThenDone(fake.Call("a", "echo", input("x")))), Tools: []agentloop.Tool{echoTool("echo", c)}}
	s, store := newSession(t)
	run(t, a, s, "go")
	// Cut the log after the allow decision.
	var kept []agentloop.Entry
	for _, e := range s.Entries() {
		if e.Type == agentloop.EntryCallStarted {
			break
		}
		kept = append(kept, e)
	}
	_ = store
	s2 := reopen(t, agentloop.NewMemoryStore(kept...))
	b := &agentloop.Agent{Model: fake.Script(fake.Text("ok"))}
	out := run(t, b, s2, "")
	wantReason(t, out, agentloop.ReasonCompleted)
	info, _ := s2.Call("a")
	if !info.Finished || !info.Result.IsError || !strings.Contains(info.Result.Content, "no tool called echo") {
		t.Fatalf("%+v", info)
	}
	if c.all() != 1 {
		t.Fatalf("the tool ran %d times", c.all())
	}
}

// Two Runs on one session would interleave their entries.
func TestConcurrentRunIsRefused(t *testing.T) {
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	tool := agentloop.Func(agentloop.ToolSpec{Name: "slow"}, func(context.Context, json.RawMessage) (agentloop.Result, error) {
		once.Do(func() { close(entered) })
		<-release
		return agentloop.Result{Content: "ok"}, nil
	})
	a := &agentloop.Agent{Model: fake.New(callsThenDone(fake.Call("a", "slow", ""))), Tools: []agentloop.Tool{tool}}
	s, store := newSession(t)
	done := make(chan error, 1)
	go func() {
		_, err := a.Run(context.Background(), s, "go")
		done <- err
	}()
	<-entered
	before := len(s.Entries())
	if _, err := a.Run(context.Background(), s, "again"); err == nil || !strings.Contains(err.Error(), "already in progress") {
		t.Fatalf("second Run: %v", err)
	}
	if got := len(s.Entries()); got != before {
		t.Fatalf("the refused Run wrote %d entries", got-before)
	}
	// Reading and resolving stay available meanwhile.
	s.Messages()
	if err := s.Resolve("a", agentloop.Decision{Verdict: agentloop.Allow}); err == nil {
		t.Fatal("resolved a call that is not held")
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	// The session is free again, and its log is sound.
	if _, err := a.Run(context.Background(), s, ""); err != nil {
		t.Fatal(err)
	}
	r := reopen(t, store)
	if !reflect.DeepEqual(s.Entries(), r.Entries()) {
		t.Fatal("log differs from live session")
	}
	checkConversation(t, r.Messages())
}

// A refused or failed Run frees the session too.
func TestRunReleasesSessionOnError(t *testing.T) {
	s, _ := newSession(t)
	bad := &agentloop.Agent{}
	if _, err := bad.Run(context.Background(), s, "x"); err == nil {
		t.Fatal("no model")
	}
	ok := &agentloop.Agent{Model: fake.Script(fake.Text("hi"))}
	if _, err := ok.Run(context.Background(), s, "x"); err != nil {
		t.Fatal(err)
	}
}

// Hammer the session from several goroutines under -race: one Run, while
// others read and refuse.
func TestConcurrentReadersAndRefusedRuns(t *testing.T) {
	c := newCounter()
	a := &agentloop.Agent{Model: fake.New(callsThenDone(fake.Call("a", "echo", input("x")), fake.Call("b", "echo", input("y")))),
		Tools: []agentloop.Tool{echoTool("echo", c)}}
	s, store := newSession(t)
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				touch(s)
				a.Run(context.Background(), s, "") // may be refused
			}
		}()
	}
	if _, err := a.Run(context.Background(), s, "go"); err != nil && !strings.Contains(err.Error(), "already in progress") {
		t.Fatal(err)
	}
	wg.Wait()
	r := reopen(t, store)
	if !reflect.DeepEqual(s.Entries(), r.Entries()) {
		t.Fatal("log differs from live session")
	}
	for _, key := range []string{"x", "y"} {
		if c.count(key) > 1 {
			t.Fatalf("%s ran %d times", key, c.count(key))
		}
	}
}

// A torn last line that Load ignores must also be the one the next append
// removes, even when it is valid JSON that is not an entry, or is followed
// by blank space.
func TestTornTailsLoadAndRepairAlike(t *testing.T) {
	for _, raw := range []string{`""`, "0\n ", "0\n\n", "[1]\n  \n", `{"seq":1,"ty`, "\n\n", " "} {
		path := filepath.Join(t.TempDir(), "s.jsonl")
		if err := os.WriteFile(path, []byte(raw), 0o600); err != nil {
			t.Fatal(err)
		}
		store := agentloop.NewFileStore(path)
		if _, err := store.Load(); err != nil {
			continue // a bad line before the end is an error, by design
		}
		s, err := agentloop.OpenSession(store)
		if err != nil {
			t.Fatal(err)
		}
		a := &agentloop.Agent{Model: fake.Script(fake.Text("hi"))}
		if _, err := a.Run(context.Background(), s, "x"); err != nil {
			t.Fatalf("%q: %v", raw, err)
		}
		if _, err := agentloop.OpenSession(agentloop.NewFileStore(path)); err != nil {
			t.Fatalf("%q: after a run the file does not reopen: %v", raw, err)
		}
	}
}

// A log cannot smuggle tool results into the conversation as a message;
// they would have no call to sit after.
func TestMessageEntryCannotHoldResults(t *testing.T) {
	user := func(parts ...agentloop.Part) agentloop.Entry {
		return agentloop.Entry{Seq: 1, Type: agentloop.EntryMessage, Message: &agentloop.MessageEntry{Message: agentloop.Message{Role: agentloop.User, Parts: parts}}}
	}
	for _, e := range []agentloop.Entry{
		user(agentloop.Part{Type: agentloop.ToolResultPart}),
		user(agentloop.Part{Type: agentloop.ToolResultPart, ToolResult: &agentloop.ToolResult{CallID: "zz"}}),
	} {
		if _, err := agentloop.OpenSession(agentloop.NewMemoryStore(e)); err == nil {
			t.Fatalf("accepted %+v", e)
		}
	}
}

// TestPropertyHolds is property 11. A held call waits; resolving it
// allows or denies exactly that call; the next run carries it out once;
// resolving anything else is an error that writes nothing.
func TestPropertyHolds(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		n := rapid.IntRange(1, 5).Draw(t, "calls")
		verdicts := rapid.SliceOfN(rapid.SampledFrom([]agentloop.Verdict{agentloop.Allow, agentloop.Hold, agentloop.Hold, agentloop.Deny}), n, n).Draw(t, "verdicts")
		var calls []agentloop.ToolCall
		byText := map[string]agentloop.Verdict{}
		for i := range verdicts {
			text := fmt.Sprintf("t%d", i)
			calls = append(calls, fake.Call(fmt.Sprintf("c%d", i), "echo", input(text)))
			byText[text] = verdicts[i]
		}
		c := newCounter()
		asked := map[string]int{}
		a := &agentloop.Agent{
			Model: fake.New(callsThenDone(calls...)), Tools: []agentloop.Tool{echoTool("echo", c)},
			Policy: agentloop.PolicyFunc(func(_ context.Context, call agentloop.ToolCall, _ agentloop.View) agentloop.Decision {
				asked[call.ID]++
				var v struct{ Text string }
				json.Unmarshal(call.Input, &v)
				return agentloop.Decision{Verdict: byText[v.Text], Reason: "policy " + call.ID}
			}),
		}
		store := agentloop.NewMemoryStore()
		s := openRapid(t, store)
		out := runRapid(t, a, s, "go")
		var wantHeld []string
		for i, v := range verdicts {
			if v == agentloop.Hold {
				wantHeld = append(wantHeld, calls[i].ID)
			}
		}
		if len(wantHeld) == 0 {
			if out.Reason != agentloop.ReasonCompleted {
				t.Fatalf("nothing held but %s", out.Reason)
			}
			return
		}
		if out.Reason != agentloop.ReasonHeld || !reflect.DeepEqual(ids(out.Held), wantHeld) {
			t.Fatalf("outcome %+v, want held %v", out, wantHeld)
		}
		// A person decides on some of them. Nothing else may resolve.
		resolved := map[string]agentloop.Verdict{}
		for _, id := range wantHeld {
			choice := rapid.SampledFrom([]string{"allow", "deny", "leave"}).Draw(t, "choice")
			if choice == "leave" {
				continue
			}
			before := len(s.Entries())
			if err := s.Resolve(id, agentloop.Decision{Verdict: agentloop.Hold}); err == nil {
				t.Fatal("resolved with hold")
			}
			if err := s.Resolve(id, agentloop.Decision{Verdict: "maybe"}); err == nil {
				t.Fatal("resolved with a made-up verdict")
			}
			if len(s.Entries()) != before {
				t.Fatal("a refused Resolve wrote an entry")
			}
			v := agentloop.Allow
			if choice == "deny" {
				v = agentloop.Deny
			}
			if err := s.Resolve(id, agentloop.Decision{Verdict: v, Reason: "person " + id}); err != nil {
				t.Fatal(err)
			}
			resolved[id] = v
			if err := s.Resolve(id, agentloop.Decision{Verdict: v}); err == nil {
				t.Fatal("resolved twice")
			}
		}
		for i, v := range verdicts {
			id := calls[i].ID
			if v != agentloop.Hold {
				before := len(s.Entries())
				if err := s.Resolve(id, agentloop.Decision{Verdict: agentloop.Allow}); err == nil {
					t.Fatalf("resolved %s, which is %s, not held", id, v)
				}
				if len(s.Entries()) != before {
					t.Fatal("a refused Resolve wrote an entry")
				}
			}
		}
		if err := s.Resolve("nope", agentloop.Decision{Verdict: agentloop.Allow}); err == nil {
			t.Fatal("resolved an unknown call")
		}

		// Resume from what was stored.
		s = openRapid(t, store)
		out = runRapid(t, a, s, "")
		var stillHeld []string
		for _, id := range wantHeld {
			if _, ok := resolved[id]; !ok {
				stillHeld = append(stillHeld, id)
			}
		}
		if len(stillHeld) > 0 {
			if out.Reason != agentloop.ReasonHeld || !reflect.DeepEqual(ids(out.Held), stillHeld) {
				t.Fatalf("outcome %+v, want held %v", out, stillHeld)
			}
		} else if out.Reason != agentloop.ReasonCompleted {
			t.Fatalf("all resolved but %s", out.Reason)
		}
		for i, v := range verdicts {
			id, text := calls[i].ID, fmt.Sprintf("t%d", i)
			info, _ := s.Call(id)
			ran := c.count(text)
			switch {
			case v == agentloop.Allow:
				if ran != 1 || !info.Finished || info.Result.IsError {
					t.Fatalf("%s: allowed, ran %d, %+v", id, ran, info)
				}
			case v == agentloop.Deny:
				if ran != 0 || info.Result == nil || info.Result.Content != "policy "+id {
					t.Fatalf("%s: denied, ran %d, %+v", id, ran, info)
				}
			case resolved[id] == agentloop.Allow:
				if ran != 1 || !info.Finished || info.Result.Content != "echo: "+text {
					t.Fatalf("%s: resolved allow, ran %d, %+v", id, ran, info)
				}
			case resolved[id] == agentloop.Deny:
				if ran != 0 || info.Result == nil || info.Result.Content != "person "+id || !info.Result.IsError {
					t.Fatalf("%s: resolved deny, ran %d, %+v", id, ran, info)
				}
			default:
				if ran != 0 || info.Finished || !info.Held() {
					t.Fatalf("%s: still held, ran %d, %+v", id, ran, info)
				}
			}
			if asked[id] != 1 {
				t.Fatalf("the policy was asked about %s %d times", id, asked[id])
			}
		}
		// Finally approve the rest; each runs once, and the model finishes.
		for _, id := range stillHeld {
			if err := s.Resolve(id, agentloop.Decision{Verdict: agentloop.Allow}); err != nil {
				t.Fatal(err)
			}
		}
		if len(stillHeld) > 0 {
			out = runRapid(t, a, openRapid(t, store), "")
			if out.Reason != agentloop.ReasonCompleted {
				t.Fatalf("%+v", out)
			}
			for i, v := range verdicts {
				if v == agentloop.Hold && resolved[calls[i].ID] != agentloop.Deny && c.count(fmt.Sprintf("t%d", i)) != 1 {
					t.Fatalf("call %d ran %d times", i, c.count(fmt.Sprintf("t%d", i)))
				}
			}
		}
		checkRequestShape(t, openRapid(t, store).Messages())
	})
}

func ids(calls []agentloop.HeldCall) []string {
	var out []string
	for _, c := range calls {
		out = append(out, c.ID)
	}
	return out
}

func openRapid(t *rapid.T, store agentloop.Store) *agentloop.Session {
	s, err := agentloop.OpenSession(store)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func runRapid(t *rapid.T, a *agentloop.Agent, s *agentloop.Session, prompt string) *agentloop.Outcome {
	out, err := a.Run(context.Background(), s, prompt)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	return out
}

// checkRequestShape is property 3 on a conversation.
func checkRequestShape(t *rapid.T, msgs []agentloop.Message) {
	if len(msgs) == 0 || msgs[0].Role != agentloop.User {
		t.Fatalf("bad start: %+v", msgs)
	}
	for i, m := range msgs {
		calls := m.ToolCalls()
		if m.Role != agentloop.Assistant || len(calls) == 0 {
			continue
		}
		if i+1 >= len(msgs) || len(msgs[i+1].Parts) != len(calls) {
			t.Fatalf("calls at %d are not answered one for one: %+v", i, msgs)
		}
		for j, c := range calls {
			if p := msgs[i+1].Parts[j]; p.ToolResult == nil || p.ToolResult.CallID != c.ID {
				t.Fatalf("result %d does not answer %s", j, c.ID)
			}
		}
	}
}
