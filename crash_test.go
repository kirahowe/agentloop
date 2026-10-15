package agentloop_test

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/fake"
)

// crashScenario builds an agent over a fixed turn of calls and returns it
// with the counter of tool executions, keyed by the text each was given.
func crashScenario(held bool) (*agentloop.Agent, *counter) {
	c := newCounter()
	labeller := agentloop.Func(
		agentloop.ToolSpec{Name: "read", Parameters: json.RawMessage(echoSchema)},
		func(_ context.Context, in json.RawMessage) (agentloop.Result, error) {
			c.hit("read")
			return agentloop.Result{Content: "contents", Labels: []string{"private"}}, nil
		})
	m := fake.New(callsThenDone(
		fake.Call("c1", "echo", input("one")),
		fake.Call("c2", "read", input("file")),
		fake.Call("c3", "nothing", ""),
		fake.Call("c4", "echo", `{"text":7}`),
		fake.Call("c5", "echo", input("danger")),
		fake.Call("c6", "echo", input("six")),
	))
	a := &agentloop.Agent{
		Model: m, Tools: []agentloop.Tool{echoTool("echo", c), labeller},
		Policy: agentloop.PolicyFunc(func(_ context.Context, call agentloop.ToolCall, v agentloop.View) agentloop.Decision {
			if held && string(call.Input) == input("danger") {
				return agentloop.Decision{Verdict: agentloop.Hold}
			}
			if string(call.Input) == input("danger") {
				return agentloop.Decision{Verdict: agentloop.Deny, Reason: "dangerous"}
			}
			return agentloop.Decision{Verdict: agentloop.Allow, Labels: []string{"seen:" + call.ID}}
		}),
	}
	return a, c
}

// finish drives a session to completion, approving any held call.
func finish(t *testing.T, a *agentloop.Agent, s *agentloop.Session, prompt string) *agentloop.Outcome {
	t.Helper()
	for i := 0; i < 5; i++ {
		out := run(t, a, s, prompt)
		prompt = ""
		if out.Reason != agentloop.ReasonHeld {
			return out
		}
		for _, call := range out.Held {
			if err := s.Resolve(call.ID, agentloop.Decision{Verdict: agentloop.Allow}); err != nil {
				t.Fatal(err)
			}
		}
	}
	t.Fatal("the session stayed held")
	return nil
}

// TestCrashAtEveryPoint cuts the log of a finished run at every length,
// reopens it, and resumes. Whatever the cut, the call that started before
// it is never run again, and the conversation ends valid.
func TestCrashAtEveryPoint(t *testing.T) {
	for _, held := range []bool{false, true} {
		t.Run(fmt.Sprintf("held=%v", held), func(t *testing.T) {
			a, _ := crashScenario(held)
			s, _ := newSession(t)
			out := finish(t, a, s, "go")
			wantReason(t, out, agentloop.ReasonCompleted)
			full := s.Entries()
			if len(full) < 20 {
				t.Fatalf("only %d entries", len(full))
			}

			for k := 0; k <= len(full); k++ {
				prefix := full[:k]
				started := map[string]bool{}
				for _, e := range prefix {
					if e.Type == agentloop.EntryCallStarted {
						started[e.CallStarted.CallID] = true
					}
				}

				a2, c2 := crashScenario(held)
				store := agentloop.NewMemoryStore(prefix...)
				s2 := reopen(t, store)
				prompt := ""
				if len(s2.Messages()) == 0 {
					prompt = "go"
				}
				out := finish(t, a2, s2, prompt)
				wantReason(t, out, agentloop.ReasonCompleted)

				// A fresh process reading the log sees the same session.
				s3 := reopen(t, store)
				entries := s3.Entries()
				msgs := s3.Messages()
				checkConversation(t, msgs)
				if last := msgs[len(msgs)-1]; last.Role != agentloop.Assistant || last.Text() != "done" {
					t.Fatalf("k=%d: conversation ends with %+v", k, last)
				}
				res := resultsOf(msgs)
				if len(res) != 6 {
					t.Fatalf("k=%d: %d results", k, len(res))
				}

				startedBy := map[string]int{}
				for _, e := range entries {
					if e.Type == agentloop.EntryCallStarted {
						startedBy[e.CallStarted.CallID]++
					}
				}
				for id, n := range startedBy {
					if n != 1 {
						t.Fatalf("k=%d: call %s started %d times", k, id, n)
					}
				}
				// Executions in the resumed run: nothing that had started
				// runs again, and nothing runs twice.
				ran := map[string]int{"c1": c2.count("one"), "c2": c2.count("read"), "c6": c2.count("six")}
				for id, n := range ran {
					if n > 1 || (started[id] && n != 0) {
						t.Fatalf("k=%d: call %s ran %d times after a crash (started in prefix: %v)", k, id, n, started[id])
					}
				}
				if c2.count("danger") != 0 && held == false {
					t.Fatalf("k=%d: the denied call ran", k)
				}
				// Every call is decided before it starts.
				for id := range startedBy {
					info, _ := s3.Call(id)
					if info.Decision == nil || info.Decision.Verdict != agentloop.Allow {
						t.Fatalf("k=%d: %s started without an allow", k, id)
					}
				}
			}
		})
	}
}

// A call that was started when the run died is reported as such, once,
// and not run again.
func TestCrashAfterCallStarted(t *testing.T) {
	a, _ := crashScenario(false)
	s, _ := newSession(t)
	finish(t, a, s, "go")
	full := s.Entries()
	cut := -1
	for i, e := range full {
		if e.Type == agentloop.EntryCallStarted && e.CallStarted.CallID == "c2" {
			cut = i + 1
		}
	}
	a2, c2 := crashScenario(false)
	s2 := reopen(t, agentloop.NewMemoryStore(full[:cut]...))
	info, _ := s2.Call("c2")
	if !info.Started || info.Finished {
		t.Fatalf("%+v", info)
	}
	wantReason(t, run(t, a2, s2, ""), agentloop.ReasonCompleted)
	if c2.count("read") != 0 {
		t.Fatal("the started call ran again")
	}
	res := resultsOf(s2.Messages())["c2"]
	if !res.IsError {
		t.Fatalf("%+v", res)
	}
	contains(t, res.Content, "may or may not have taken effect")
	// Its labels never arrived, since its result never did.
	for _, l := range s2.Labels() {
		if l == "private" {
			t.Fatal("labels of a result that was never recorded")
		}
	}
	// The calls after it still ran.
	if c2.count("six") != 1 {
		t.Fatalf("runs %v", c2.runs)
	}
}
