package agentloop_test

import (
	"context"
	"encoding/json"
	"errors"
	"maps"
	"testing"

	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/fake"
)

// fixedTool is a Narrowed tool that adds the operator's fixed arguments
// to what the model wrote, as an MCP tool with fixed_args does, and
// records what it ran on.
type fixedTool struct {
	fixed map[string]json.RawMessage
	got   []map[string]json.RawMessage
}

func (t *fixedTool) Spec() agentloop.ToolSpec {
	return agentloop.ToolSpec{Name: "send", Parameters: json.RawMessage(`{"type":"object"}`)}
}

func (t *fixedTool) EffectiveInput(in json.RawMessage) (json.RawMessage, map[string]json.RawMessage, bool) {
	var args map[string]json.RawMessage
	if json.Unmarshal(in, &args) != nil || args == nil {
		return nil, nil, false
	}
	for k := range t.fixed {
		if _, ok := args[k]; ok {
			return nil, nil, false
		}
	}
	maps.Copy(args, t.fixed)
	out, _ := json.Marshal(args)
	return out, maps.Clone(t.fixed), true
}

func (t *fixedTool) Call(_ context.Context, in json.RawMessage) (agentloop.Result, error) {
	out, _, ok := t.EffectiveInput(in)
	if !ok {
		return agentloop.Result{Content: "refused", IsError: true}, nil
	}
	var args map[string]json.RawMessage
	json.Unmarshal(out, &args)
	t.got = append(t.got, args)
	return agentloop.Result{Content: "sent"}, nil
}

// holdUnless holds every call unless its input's group is "Kira only",
// as a consent rule on group would, and records the input it was shown.
type holdUnless struct{ seen []json.RawMessage }

func (p *holdUnless) Decide(_ context.Context, c agentloop.ToolCall, _ agentloop.View) agentloop.Decision {
	p.seen = append(p.seen, c.Input)
	var in struct{ Group string }
	json.Unmarshal(c.Input, &in)
	if in.Group == "Kira only" {
		return agentloop.Decision{Verdict: agentloop.Allow}
	}
	return agentloop.Decision{Verdict: agentloop.Hold}
}

func group(g string) map[string]json.RawMessage {
	b, _ := json.Marshal(g)
	return map[string]json.RawMessage{"group": b}
}

// The policy decides on the input the call runs on, so consent on a
// fixed argument applies.
func TestPolicySeesTheEffectiveInput(t *testing.T) {
	s, _ := newSession(t)
	tool := &fixedTool{fixed: group("Kira only")}
	pol := &holdUnless{}
	a := &agentloop.Agent{Model: fake.Script(fake.ToolUse("", fake.Call("c1", "send", `{"subject":"hi"}`)), fake.Text("done")),
		Tools: []agentloop.Tool{tool}, Policy: pol}
	wantReason(t, run(t, a, s, "go"), agentloop.ReasonCompleted)
	if len(tool.got) != 1 || len(pol.seen) != 1 || agentloop.InputDigest(pol.seen[0]) != agentloop.InputDigest(json.RawMessage(`{"group":"Kira only","subject":"hi"}`)) {
		t.Fatalf("policy saw %s, tool ran %d times", pol.seen, len(tool.got))
	}
}

// Whatever the groups, an approval given for the call as held runs it
// only while the fixed group is the one it was held with; otherwise the
// call is refused at resume, never run.
func TestPropertyApprovalIsBoundToFixedArguments(t *testing.T) {
	groups := rapid.SampledFrom([]string{"Kira only", "Public list", "Board", "kira only", "Kira only "})
	rapid.Check(t, func(rt *rapid.T) {
		before, after := groups.Draw(rt, "before"), groups.Draw(rt, "after")
		// A crash between the approval and the run, with the change in
		// between, as well as a change before the approval.
		crash := rapid.Bool().Draw(rt, "resolved-before-the-change")
		s, store := newSession(t)
		tool := &fixedTool{fixed: group(before)}
		a := &agentloop.Agent{Model: fake.Script(fake.ToolUse("", fake.Call("c1", "send", `{"subject":"hi"}`)), fake.Text("done"), fake.Text("done")),
			Tools: []agentloop.Tool{tool}, Policy: agentloop.PolicyFunc(func(context.Context, agentloop.ToolCall, agentloop.View) agentloop.Decision {
				return agentloop.Decision{Verdict: agentloop.Hold}
			})}
		out, err := a.Run(context.Background(), s, "go")
		if err != nil || out.Reason != agentloop.ReasonHeld || len(out.Held) != 1 {
			rt.Fatalf("%v %+v", err, out)
		}
		h := out.Held[0]
		eff, _ := json.Marshal(map[string]any{"group": before, "subject": "hi"})
		if h.InputSHA256 != agentloop.InputDigest(eff) || agentloop.InputDigest(h.EffectiveInput) != h.InputSHA256 || string(h.FixedArgs["group"]) != string(group(before)["group"]) {
			rt.Fatalf("held %+v", h)
		}
		r := agentloop.Resolution{CallID: "c1", Verdict: agentloop.Allow, InputSHA256: h.InputSHA256}
		if crash {
			if err := a.Resolve(context.Background(), s, r); err != nil {
				rt.Fatal(err)
			}
		}
		tool.fixed = group(after)
		s = reopen(t, store)
		err = a.Resolve(context.Background(), s, r)
		same := before == after
		switch {
		case crash:
			// Already resolved: a no-op.
			if err != nil {
				rt.Fatal(err)
			}
		case same && err != nil:
			rt.Fatalf("unchanged settings refused: %v", err)
		case !same && !errors.Is(err, agentloop.ErrSettingsChanged):
			rt.Fatalf("changed settings gave %v", err)
		}
		if _, err := a.Run(context.Background(), s, ""); err != nil {
			rt.Fatal(err)
		}
		info, _ := s.Call("c1")
		if same {
			if len(tool.got) != 1 || info.Result.IsError {
				rt.Fatalf("ran %v, result %+v", tool.got, info.Result)
			}
			return
		}
		if len(tool.got) != 0 {
			rt.Fatalf("ran with changed settings: %v", tool.got)
		}
		if !crash {
			// Never resolved: still held, and the person is asked again.
			if !info.Held() {
				rt.Fatalf("%+v", info)
			}
			return
		}
		if !info.Finished || info.Result.Content != agentloop.SettingsChanged {
			rt.Fatalf("result %+v", info.Result)
		}
	})
}

// Regression (Codex review): a hold shown again after the operator's
// settings changed took a new digest, which no resolution could name, so
// the session was stuck. It keeps its hold's digest: a deny goes
// through, and an allow is refused as a settings change.
func TestAHoldShownAgainKeepsItsDigest(t *testing.T) {
	for _, verdict := range []agentloop.Verdict{agentloop.Allow, agentloop.Deny} {
		s, store := newSession(t)
		tool := &fixedTool{fixed: group("Kira only")}
		a := &agentloop.Agent{Model: fake.Script(fake.ToolUse("", fake.Call("c1", "send", `{"subject":"hi"}`)), fake.Text("done")),
			Tools: []agentloop.Tool{tool}, Policy: agentloop.PolicyFunc(func(context.Context, agentloop.ToolCall, agentloop.View) agentloop.Decision {
				return agentloop.Decision{Verdict: agentloop.Hold}
			})}
		first := run(t, a, s, "go")
		tool.fixed = group("Public list")
		s = reopen(t, store)
		again := run(t, a, s, "")
		if len(again.Held) != 1 || again.Held[0].InputSHA256 != first.Held[0].InputSHA256 {
			t.Fatalf("held again %+v, first %+v", again.Held, first.Held)
		}
		r := agentloop.Resolution{CallID: "c1", Verdict: verdict, InputSHA256: again.Held[0].InputSHA256}
		err := a.Resolve(context.Background(), s, r)
		if verdict == agentloop.Allow {
			if !errors.Is(err, agentloop.ErrSettingsChanged) {
				t.Fatalf("allow: %v", err)
			}
			continue
		}
		if err != nil {
			t.Fatalf("deny: %v", err)
		}
		run(t, a, s, "")
		if len(tool.got) != 0 {
			t.Fatalf("ran %v", tool.got)
		}
	}
}
