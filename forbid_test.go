package agentloop_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"sort"
	"strings"
	"testing"

	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/fake"
)

var someLabels = []string{"contaminated", "private"}

func genSomeLabels() *rapid.Generator[[]string] {
	return rapid.SliceOfN(rapid.SampledFrom(someLabels), 0, 3)
}

// Forbidden is the sorted intersection, with no repeats, never nil.
func TestForbiddenIsTheIntersection(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		labels := genSomeLabels().Draw(t, "labels")
		forbid := genSomeLabels().Draw(t, "forbid")
		want := []string{}
		for _, l := range someLabels {
			if slices.Contains(labels, l) && slices.Contains(forbid, l) {
				want = append(want, l)
			}
		}
		got := agentloop.Forbidden(labels, forbid)
		if got == nil || !reflect.DeepEqual(got, want) {
			t.Fatalf("Forbidden(%v, %v) = %#v, want %v", labels, forbid, got, want)
		}
	})
}

// labelTool returns, for each call, the content and labels named by the
// call's input.
func labelTool(name string) agentloop.Tool {
	return agentloop.Func(
		agentloop.ToolSpec{Name: name, Parameters: json.RawMessage(`{"type":"object"}`)},
		func(_ context.Context, in json.RawMessage) (agentloop.Result, error) {
			var v struct {
				Text   string   `json:"text"`
				Labels []string `json:"labels"`
			}
			json.Unmarshal(in, &v)
			return agentloop.Result{Content: "got " + v.Text, Labels: v.Labels, Log: "LOG-" + v.Text}, nil
		})
}

func noModel(t *testing.T) agentloop.Model {
	return fake.New(func(*agentloop.Request) (*agentloop.Response, error) {
		t.Fatal("the model was called")
		return nil, nil
	})
}

func TestARunThatWouldStartForbiddenDoesNotStart(t *testing.T) {
	s, store := newSession(t)
	a := &agentloop.Agent{Model: noModel(t), Labels: []string{"private"}, Forbid: []string{"private"}}
	out := run(t, a, s, "go")
	wantReason(t, out, agentloop.ReasonForbidden)
	if !reflect.DeepEqual(out.Forbidden, []string{"private"}) {
		t.Fatalf("forbidden %v", out.Forbidden)
	}
	if n := len(reopen(t, store).Entries()); n != 0 {
		t.Fatalf("a forbidden run wrote %d entries", n)
	}

	// A session that already holds the label, from a run without the
	// rule, is not carried on either.
	b := &agentloop.Agent{Model: fake.Script(fake.Text("hi")), Labels: []string{"private"}}
	s = reopen(t, store)
	wantReason(t, run(t, b, s, "go"), agentloop.ReasonCompleted)
	before := len(s.Entries())
	c := &agentloop.Agent{Model: noModel(t), Forbid: []string{"private"}}
	out = run(t, c, s, "more")
	wantReason(t, out, agentloop.ReasonForbidden)
	if len(s.Entries()) != before {
		t.Fatal("the forbidden run wrote to the session")
	}

	// A forbid that names nothing the session holds changes nothing.
	d := &agentloop.Agent{Model: fake.Script(fake.Text("ok")), Forbid: []string{"contaminated"}}
	wantReason(t, run(t, d, s, "more"), agentloop.ReasonCompleted)
}

func TestAForbiddenResultIsRefused(t *testing.T) {
	m := fake.Script(
		fake.ToolUse("", fake.Call("c1", "t", `{"text":"SECRET-1","labels":["private"]}`),
			fake.Call("c2", "t", `{"text":"web page","labels":["contaminated"]}`)),
		fake.Text("done"))
	a := &agentloop.Agent{Model: m, Tools: []agentloop.Tool{labelTool("t")}, Forbid: []string{"private"}}
	s, _ := newSession(t)
	out := run(t, a, s, "go")
	wantReason(t, out, agentloop.ReasonCompleted)
	if got := s.Labels(); !reflect.DeepEqual(got, []string{"contaminated"}) {
		t.Fatalf("labels %v", got)
	}
	want := []agentloop.Refusal{{CallID: "c1", Tool: "t", Labels: []string{"private"}}}
	if !reflect.DeepEqual(out.Refused, want) {
		t.Fatalf("refused %+v", out.Refused)
	}
	results := resultsOf(s.Messages())
	if r := results["c1"]; r.Content != agentloop.RefusedResult || !r.IsError {
		t.Fatalf("c1's result %+v", r)
	}
	if r := results["c2"]; r.Content != "got web page" {
		t.Fatalf("c2's result %+v", r)
	}
	// Nothing of the refused result is kept anywhere: not for the model,
	// not in the log, not in the requests the model was sent.
	for _, e := range s.Entries() {
		if e.Type == agentloop.EntryToolResult && e.ToolResult.Result.CallID == "c1" {
			b, _ := json.Marshal(e)
			if strings.Contains(string(b), "SECRET") || !strings.Contains(e.ToolResult.Log, "refused") {
				t.Fatalf("result entry %s", b)
			}
		}
	}
	for _, req := range m.Requests()[1:] {
		rb, _ := json.Marshal(req.Messages)
		if strings.Contains(string(rb), "got SECRET-1") || strings.Contains(string(rb), "LOG-SECRET") {
			t.Fatalf("the model saw the refused result: %s", rb)
		}
	}
}

// A decision that would add a forbidden label is a deny, whatever the
// policy says, and a person cannot allow it either.
func TestAnAllowThatWouldAddAForbiddenLabelIsADeny(t *testing.T) {
	c := newCounter()
	m := fake.New(callsThenDone(fake.Call("a", "echo", input("x"))))
	a := &agentloop.Agent{
		Model: m, Tools: []agentloop.Tool{echoTool("echo", c)}, Forbid: []string{"private"},
		Policy: agentloop.PolicyFunc(func(context.Context, agentloop.ToolCall, agentloop.View) agentloop.Decision {
			return agentloop.Decision{Verdict: agentloop.Allow, Labels: []string{"private"}}
		}),
	}
	s, _ := newSession(t)
	wantReason(t, run(t, a, s, "go"), agentloop.ReasonCompleted)
	if c.all() != 0 {
		t.Fatal("the call ran")
	}
	if r := resultsOf(s.Messages())["a"]; r.Content != agentloop.ForbiddenDecision {
		t.Fatalf("result %+v", r)
	}
	if len(s.Labels()) != 0 {
		t.Fatalf("labels %v", s.Labels())
	}

	held := &agentloop.Agent{
		Model: fake.New(callsThenDone(fake.Call("b", "echo", input("y")))), Tools: []agentloop.Tool{echoTool("echo", c)},
		Forbid: []string{"private"},
		Policy: agentloop.PolicyFunc(func(context.Context, agentloop.ToolCall, agentloop.View) agentloop.Decision {
			return agentloop.Decision{Verdict: agentloop.Hold, Labels: []string{"private"}}
		}),
	}
	s, _ = newSession(t)
	out := run(t, held, s, "go")
	wantReason(t, out, agentloop.ReasonHeld)
	err := held.Resolve(context.Background(), s, agentloop.Resolution{CallID: "b", Verdict: agentloop.Allow, InputSHA256: out.Held[0].InputSHA256})
	if !errors.Is(err, agentloop.ErrPolicyDenies) {
		t.Fatalf("resolve: %v", err)
	}
}

// Whatever the tools return and whatever the policy adds, a session never
// holds a forbidden label, every result carrying one is refused and
// reported, and every other result reaches the model as it was.
func TestASessionNeverHoldsAForbiddenLabel(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		forbid := genSomeLabels().Draw(t, "forbid")
		seed := genSomeLabels().Draw(t, "seed")
		policyLabels := genSomeLabels().Draw(t, "policy labels")
		n := rapid.IntRange(0, 6).Draw(t, "calls")
		var calls []agentloop.ToolCall
		want := map[string]bool{}
		for i := range n {
			labels := genSomeLabels().Draw(t, fmt.Sprintf("labels %d", i))
			in, _ := json.Marshal(map[string]any{"text": fmt.Sprintf("text-%d", i), "labels": labels})
			id := fmt.Sprintf("c%d", i)
			calls = append(calls, fake.Call(id, "t", string(in)))
			want[id] = len(agentloop.Forbidden(labels, forbid)) > 0
		}
		m := fake.New(callsThenDone(calls...))
		a := &agentloop.Agent{
			Model: m, Tools: []agentloop.Tool{labelTool("t")}, Labels: seed, Forbid: forbid,
			Policy: agentloop.PolicyFunc(func(context.Context, agentloop.ToolCall, agentloop.View) agentloop.Decision {
				return agentloop.Decision{Verdict: agentloop.Allow, Labels: policyLabels}
			}),
		}
		s, err := agentloop.OpenSession(agentloop.NewMemoryStore())
		if err != nil {
			t.Fatal(err)
		}
		out, err := a.Run(context.Background(), s, "go")
		if err != nil {
			t.Fatal(err)
		}
		if bad := agentloop.Forbidden(s.Labels(), forbid); len(bad) > 0 {
			t.Fatalf("the session holds %v", bad)
		}
		if len(agentloop.Forbidden(seed, forbid)) > 0 {
			if out.Reason != agentloop.ReasonForbidden || len(s.Entries()) != 0 {
				t.Fatalf("a forbidden seed ran: %+v", out)
			}
			return
		}
		results := resultsOf(s.Messages())
		denied := len(agentloop.Forbidden(policyLabels, forbid)) > 0
		var refused []string
		for _, rf := range out.Refused {
			refused = append(refused, rf.CallID)
		}
		sort.Strings(refused)
		var wantRefused []string
		for i := range n {
			id := fmt.Sprintf("c%d", i)
			r := results[id]
			switch {
			case denied:
				if r.Content != agentloop.ForbiddenDecision {
					t.Fatalf("%s: %+v", id, r)
				}
			case want[id]:
				wantRefused = append(wantRefused, id)
				if r.Content != agentloop.RefusedResult {
					t.Fatalf("%s: %+v", id, r)
				}
			default:
				if r.Content != fmt.Sprintf("got text-%d", i) {
					t.Fatalf("%s: %+v", id, r)
				}
			}
		}
		sort.Strings(wantRefused)
		if !reflect.DeepEqual(refused, wantRefused) {
			t.Fatalf("refused %v, want %v", refused, wantRefused)
		}
	})
}
