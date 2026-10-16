package trifecta

import (
	"context"
	"reflect"
	"slices"
	"testing"

	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop"
)

func genLabels() *rapid.Generator[[]string] {
	return rapid.SliceOfN(rapid.SampledFrom(Labels), 0, 3)
}

func genTool() *rapid.Generator[Tool] {
	return rapid.Custom(func(t *rapid.T) Tool {
		return Tool{
			Class:      rapid.SampledFrom([]Class{Read, Write, Egress, Act, Act, Egress, "", "sned"}).Draw(t, "class"),
			Labels:     genLabels().Draw(t, "labels"),
			AlwaysHold: rapid.IntRange(0, 4).Draw(t, "always") == 0,
		}
	})
}

// The rule, stated once more as the brief states it, so the two can be
// compared on every input.
func TestDecideIsTheRule(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		labels := genLabels().Draw(t, "session")
		tool := genTool().Draw(t, "tool")
		private, contaminated := slices.Contains(labels, Private), slices.Contains(labels, Contaminated)
		want := agentloop.Allow
		switch {
		case tool.AlwaysHold:
			want = agentloop.Hold
		case tool.Class == Egress && private && contaminated:
			want = agentloop.Hold
		case tool.Class == Act && contaminated:
			want = agentloop.Hold
		case tool.Class != Read && tool.Class != Write && tool.Class != Egress && tool.Class != Act:
			want = agentloop.Hold
		}
		if got := Decide(labels, tool); got != want {
			t.Fatalf("Decide(%v, %+v) = %s, want %s", labels, tool, got, want)
		}
	})
}

// No set of labels lets an egress call through once the session holds
// both private and contaminated, or an act call once it holds
// contaminated; and more labels never turn a hold into an allow.
func TestDecideIsMonotone(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		small := genLabels().Draw(t, "small")
		big := Union(small, genLabels().Draw(t, "more"))
		tool := genTool().Draw(t, "tool")
		a, b := Decide(small, tool), Decide(big, tool)
		if a == agentloop.Hold && b != agentloop.Hold {
			t.Fatalf("%v holds but %v allows %+v", small, big, tool)
		}
		if a == agentloop.Deny || b == agentloop.Deny {
			t.Fatal("the rule never denies")
		}
	})
}

func TestDecideTable(t *testing.T) {
	both := []string{Contaminated, Private}
	cases := []struct {
		labels []string
		tool   Tool
		want   agentloop.Verdict
	}{
		{nil, Tool{Class: Egress}, agentloop.Allow},
		{[]string{Private}, Tool{Class: Egress}, agentloop.Allow},
		{[]string{Contaminated}, Tool{Class: Egress}, agentloop.Allow},
		{both, Tool{Class: Egress}, agentloop.Hold},
		{both, Tool{Class: Read}, agentloop.Allow},
		{both, Tool{Class: Write}, agentloop.Allow},
		{[]string{Private}, Tool{Class: Act}, agentloop.Allow},
		{[]string{Contaminated}, Tool{Class: Act}, agentloop.Hold},
		{nil, Tool{Class: Read, AlwaysHold: true}, agentloop.Hold},
		{nil, Tool{Class: "reed"}, agentloop.Hold},
	}
	for _, c := range cases {
		if got := Decide(c.labels, c.tool); got != c.want {
			t.Errorf("Decide(%v, %+v) = %s, want %s", c.labels, c.tool, got, c.want)
		}
	}
}

func TestPolicyDecide(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		tool := genTool().Draw(t, "tool")
		p := Policy{Tools: map[string]Tool{"t": tool}}
		view := agentloop.View{Labels: Union(nil, genLabels().Draw(t, "session"))}
		d := p.Decide(context.Background(), agentloop.ToolCall{ID: "c", Name: "t"}, view)
		if d.Verdict != Decide(view.Labels, tool) {
			t.Fatalf("policy %s, rule %s", d.Verdict, Decide(view.Labels, tool))
		}
		if !reflect.DeepEqual(append([]string{}, d.Labels...), append([]string{}, tool.Labels...)) {
			t.Fatalf("decision labels %v, tool labels %v", d.Labels, tool.Labels)
		}
		if d.Verdict == agentloop.Hold && d.Reason == "" {
			t.Fatal("a hold has no reason")
		}
		// The decision's labels are its own copy.
		if len(d.Labels) > 0 {
			d.Labels[0] = "x"
			if tool.Labels[0] == "x" {
				t.Fatal("the decision shares the policy's labels")
			}
		}
		other := p.Decide(context.Background(), agentloop.ToolCall{ID: "c", Name: "u"}, view)
		if other.Verdict != agentloop.Deny {
			t.Fatalf("a tool with no entry got %s", other.Verdict)
		}
	})
}

func TestCheckLabels(t *testing.T) {
	if err := CheckLabels([]string{Private, Contaminated}); err != nil {
		t.Fatal(err)
	}
	for _, bad := range []string{"Private", "untrusted", "", "private "} {
		if CheckLabels([]string{Private, bad}) == nil {
			t.Errorf("%q passed", bad)
		}
	}
}

// With Forbid, a call to a tool whose static labels include a forbidden
// label is denied, since allowing it would give the session that label;
// every other call is decided exactly as without Forbid.
func TestPolicyDeniesToolsWhoseLabelsAreForbidden(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		tool := genTool().Draw(t, "tool")
		forbid := genLabels().Draw(t, "forbid")
		view := agentloop.View{Labels: Union(nil, genLabels().Draw(t, "session"))}
		call := agentloop.ToolCall{ID: "c", Name: "t"}
		with := Policy{Tools: map[string]Tool{"t": tool}, Forbid: forbid}.Decide(context.Background(), call, view)
		without := Policy{Tools: map[string]Tool{"t": tool}}.Decide(context.Background(), call, view)
		if len(agentloop.Forbidden(tool.Labels, forbid)) > 0 {
			if with.Verdict != agentloop.Deny || with.Reason != ForbiddenReason || len(with.Labels) != 0 {
				t.Fatalf("a tool labelled %v under forbid %v got %+v", tool.Labels, forbid, with)
			}
			return
		}
		if !reflect.DeepEqual(with, without) {
			t.Fatalf("forbid %v changed %+v into %+v", forbid, without, with)
		}
	})
}
