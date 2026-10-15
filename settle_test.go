package agentloop_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/fake"
)

// A turn that ends without tool_use and holds calls has them answered
// before the run ends, so the session is a valid request at rest.
func TestCutOffCallsAreSettledWhenTheTurnEnds(t *testing.T) {
	cases := map[string]struct {
		stop   agentloop.Stop
		reason agentloop.Reason
	}{
		"end_turn":   {agentloop.StopEndTurn, agentloop.ReasonCompleted},
		"max_tokens": {agentloop.StopMaxTokens, agentloop.ReasonMaxTokens},
		"refusal":    {agentloop.StopRefusal, agentloop.ReasonRefused},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			c := newCounter()
			m := fake.Script(fake.WithStop(fake.ToolUse("", fake.Call("a", "echo", input("x")), fake.Call("b", "echo", input("y"))), tc.stop))
			a := &agentloop.Agent{Model: m, Tools: []agentloop.Tool{echoTool("echo", c)}}
			s, _ := newSession(t)
			wantReason(t, run(t, a, s, "go"), tc.reason)
			if len(s.Pending()) != 0 || len(s.Held()) != 0 {
				t.Fatalf("pending %v held %v", s.Pending(), s.Held())
			}
			checkConversation(t, s.Messages())
			if c.all() != 0 {
				t.Fatal("a cut-off call ran")
			}
			es := s.Entries()
			if last := es[len(es)-1]; last.Type != agentloop.EntryRunEnded {
				t.Fatalf("last entry %s", last.Type)
			}
			// The results are in the log before run_ended.
			if got := len(resultsOf(s.Messages())); got != 2 {
				t.Fatalf("%d results", got)
			}
			for _, id := range []string{"a", "b"} {
				res := resultsOf(s.Messages())[id]
				if !res.IsError {
					t.Fatalf("%s: %+v", id, res)
				}
				contains(t, res.Content, "not run")
			}
		})
	}
}

// Logs written before calls were settled at the end of the turn, and logs
// cut by a crash between the turn and its results, are still settled by the
// next Run.
func TestUnsettledCutOffTurnIsSettledByTheNextRun(t *testing.T) {
	for _, stop := range []agentloop.Stop{agentloop.StopMaxTokens, agentloop.StopEndTurn, agentloop.StopRefusal} {
		t.Run(string(stop), func(t *testing.T) {
			c := newCounter()
			store := seqStore(
				msgEntry(agentloop.Text(agentloop.User, "hi"), "", agentloop.Usage{}),
				msgEntry(assistantCalls("a"), stop, agentloop.Usage{}),
			)
			s := reopen(t, store)
			if len(s.Pending()) != 1 {
				t.Fatal("setup")
			}
			m := fake.Script(fake.Text("recovered"))
			a := &agentloop.Agent{Model: m, Tools: []agentloop.Tool{echoTool("echo", c)}}
			wantReason(t, run(t, a, s, "again"), agentloop.ReasonCompleted)
			if c.all() != 0 || len(s.Pending()) != 0 {
				t.Fatalf("ran %d, pending %v", c.all(), s.Pending())
			}
			checkConversation(t, m.Requests()[0].Messages)
		})
	}
}

func TestBadToolCallIDsAreAProviderFailure(t *testing.T) {
	cases := map[string]struct {
		script []any
		msg    string
	}{
		"duplicate in a turn": {
			[]any{fake.ToolUse("", fake.Call("a", "echo", input("x")), fake.Call("a", "echo", input("y")))},
			"the model reused a tool call id",
		},
		"empty": {
			[]any{fake.ToolUse("", fake.Call("", "echo", input("x")))},
			"the model returned a tool call with no id",
		},
		"reused from an earlier turn": {
			[]any{fake.ToolUse("", fake.Call("a", "echo", input("x"))), fake.ToolUse("", fake.Call("a", "echo", input("y")))},
			"the model reused a tool call id",
		},
		"bad id on a cut-off turn": {
			[]any{fake.WithStop(fake.ToolUse("", fake.Call("", "echo", input("x"))), agentloop.StopMaxTokens)},
			"the model returned a tool call with no id",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			c := newCounter()
			m := fake.Script(tc.script...)
			a := &agentloop.Agent{Model: m, Tools: []agentloop.Tool{echoTool("echo", c)}}
			s, store := newSession(t)
			out, err := a.Run(t.Context(), s, "go")
			if err != nil {
				t.Fatalf("a bad id is an outcome, got error %v", err)
			}
			wantReason(t, out, agentloop.ReasonModelError)
			if out.Error == nil || out.Error.Kind != agentloop.InvalidRequest || out.Error.Message != tc.msg {
				t.Fatalf("error %+v", out.Error)
			}
			// Not retried.
			if want := len(tc.script); len(m.Requests()) != want {
				t.Fatalf("%d requests, want %d", len(m.Requests()), want)
			}
			es := s.Entries()
			if last := es[len(es)-1]; last.Type != agentloop.EntryRunEnded || last.RunEnded.Error == nil {
				t.Fatalf("run_ended missing: %v", types(es))
			}
			// The bad turn is not in the log, and the session is at rest.
			var turns int
			for _, e := range es {
				if e.Type == agentloop.EntryMessage && e.Message.Message.Role == agentloop.Assistant {
					turns++
				}
			}
			if turns != len(tc.script)-1 {
				t.Fatalf("%d assistant turns recorded", turns)
			}
			if out.Turns != turns {
				t.Fatalf("outcome counts %d turns, log %d", out.Turns, turns)
			}
			checkConversation(t, s.Messages())
			if len(s.Pending()) != 0 {
				t.Fatalf("pending %v", s.Pending())
			}
			// And it reopens.
			reopen(t, store)
		})
	}
}

func TestOpenSessionChecksSequenceNumbers(t *testing.T) {
	user := msgEntry(agentloop.Text(agentloop.User, "hi"), "", agentloop.Usage{})
	at := func(seq int, e agentloop.Entry) agentloop.Entry { e.Seq = seq; return e }
	cases := map[string]struct {
		entries []agentloop.Entry
		bad     string
	}{
		"starts at zero": {[]agentloop.Entry{at(0, user)}, "entry 1 has seq 0"},
		"starts at two":  {[]agentloop.Entry{at(2, user)}, "entry 1 has seq 2"},
		"gap":            {[]agentloop.Entry{at(1, user), at(2, user), at(4, user)}, "entry 3 has seq 4"},
		"repeat":         {[]agentloop.Entry{at(1, user), at(2, user), at(2, user)}, "entry 3 has seq 2"},
		"out of order":   {[]agentloop.Entry{at(1, user), at(3, user), at(2, user)}, "entry 2 has seq 3"},
	}
	for name, tc := range cases {
		_, err := agentloop.OpenSession(agentloop.NewMemoryStore(tc.entries...))
		if err == nil || !strings.Contains(err.Error(), tc.bad) {
			t.Errorf("%s: error %v, want one containing %q", name, err, tc.bad)
		}
	}
	if _, err := agentloop.OpenSession(seqStore(user, user, user)); err != nil {
		t.Fatal(err)
	}
}

// A torn last line is dropped before the sequence is checked, so a log cut
// mid-write still opens and the next append continues the numbering.
func TestFileStoreTornTailKeepsSequenceValid(t *testing.T) {
	path := filepath.Join(t.TempDir(), "log.jsonl")
	a := &agentloop.Agent{Model: fake.Script(fake.Text("one"), fake.Text("two"))}
	s := reopen(t, agentloop.NewFileStore(path))
	run(t, a, s, "go")
	n := len(s.Entries())

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	torn := append(raw, []byte(`{"seq":`+"99"+`,"time":"2026-01-01T00:0`)...)
	if err := os.WriteFile(path, torn, 0o600); err != nil {
		t.Fatal(err)
	}
	s = reopen(t, agentloop.NewFileStore(path))
	if len(s.Entries()) != n {
		t.Fatalf("%d entries, want %d", len(s.Entries()), n)
	}
	run(t, a, s, "again")
	s = reopen(t, agentloop.NewFileStore(path))
	for i, e := range s.Entries() {
		if e.Seq != i+1 {
			t.Fatalf("entry %d has seq %d", i, e.Seq)
		}
	}

	// A complete line with the wrong number is damage, torn tail or not.
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	bad := strings.Replace(lines[1], `"seq":2`, `"seq":5`, 1)
	if bad == lines[1] {
		t.Fatal("setup: no seq to change")
	}
	lines[1] = bad
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := agentloop.OpenSession(agentloop.NewFileStore(path)); err == nil || !strings.Contains(err.Error(), "seq 5") {
		t.Fatalf("error %v", err)
	}
}
