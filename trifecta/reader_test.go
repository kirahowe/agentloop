package trifecta

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop"
)

func readerCall(inner string) agentloop.ToolCall {
	in, _ := json.Marshal(map[string]any{"tool": inner, "input": map[string]any{}, "question": "q", "schema": map[string]any{}})
	return agentloop.ToolCall{ID: "c", Name: "read_quarantined", Input: in}
}

func readerPolicy() Policy {
	return Policy{
		Tools: map[string]Tool{
			"read_quarantined": {Class: Read},
			"mail_read":        {Class: Read, Labels: []string{Contaminated, Private}},
			"web_fetch":        {Class: Read, Labels: []string{Contaminated}},
			"held_read":        {Class: Read, AlwaysHold: true, Labels: []string{Contaminated}},
			"web_post":         {Class: Egress},
			"elsewhere":        {Class: Read},
		},
		Readers: map[string][]string{"read_quarantined": {"mail_read", "web_fetch", "held_read", "web_post", "missing"}},
	}
}

func TestDecideReader(t *testing.T) {
	tests := []struct {
		name     string
		policy   func(*Policy)
		call     agentloop.ToolCall
		session  []string
		verdict  agentloop.Verdict
		labels   []string
		deferred []string
		reason   string
	}{
		{name: "inner static labels are deferred", call: readerCall("mail_read"),
			verdict: agentloop.Allow, labels: []string{}, deferred: []string{Contaminated, Private}},
		{name: "a contaminated, private session may still read", call: readerCall("web_fetch"), session: []string{Contaminated, Private},
			verdict: agentloop.Allow, labels: []string{}, deferred: []string{Contaminated}},
		{name: "the reader's own labels are not deferred", policy: func(p *Policy) {
			p.Tools["read_quarantined"] = Tool{Class: Read, Labels: []string{Private}}
		}, call: readerCall("web_fetch"), verdict: agentloop.Allow, labels: []string{Private}, deferred: []string{Contaminated}},
		{name: "an inner hold holds the call", call: readerCall("held_read"),
			verdict: agentloop.Hold, reason: "Every call to this tool waits for a person."},
		{name: "the reader's own hold holds the call", policy: func(p *Policy) {
			p.Tools["read_quarantined"] = Tool{Class: Read, AlwaysHold: true}
		}, call: readerCall("web_fetch"), verdict: agentloop.Hold, reason: "Every call to this tool waits for a person."},
		{name: "an inner tool not on the reader's list", call: readerCall("elsewhere"),
			verdict: agentloop.Deny, reason: "cannot read through elsewhere"},
		{name: "an inner tool with no entry", call: readerCall("missing"),
			verdict: agentloop.Deny, reason: "has no policy"},
		{name: "an inner tool that is not class read", call: readerCall("web_post"),
			verdict: agentloop.Deny, reason: "of class read"},
		{name: "input naming no inner tool", call: agentloop.ToolCall{Name: "read_quarantined", Input: json.RawMessage(`{"tool":7}`)},
			verdict: agentloop.Deny, reason: "does not name an inner tool"},
		{name: "a forbidden inner label denies", policy: func(p *Policy) { p.Forbid = []string{Private} },
			call: readerCall("mail_read"), verdict: agentloop.Deny, reason: ForbiddenReason},
		{name: "a forbidden reader label denies", policy: func(p *Policy) {
			p.Forbid = []string{Private}
			p.Tools["read_quarantined"] = Tool{Class: Read, Labels: []string{Private}}
		}, call: readerCall("web_fetch"), verdict: agentloop.Deny, reason: ForbiddenReason},
		{name: "a forbidden label the inner tool lacks does not deny", policy: func(p *Policy) { p.Forbid = []string{Private} },
			call: readerCall("web_fetch"), verdict: agentloop.Allow, labels: []string{}, deferred: []string{Contaminated}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := readerPolicy()
			if tc.policy != nil {
				tc.policy(&p)
			}
			d := p.Decide(context.Background(), tc.call, agentloop.View{Labels: tc.session})
			if d.Verdict != tc.verdict || !strings.Contains(d.Reason, tc.reason) {
				t.Fatalf("decision = %+v, want %s %q", d, tc.verdict, tc.reason)
			}
			if tc.verdict == agentloop.Allow {
				if !slices.Equal(append([]string{}, d.Labels...), tc.labels) || !slices.Equal(d.Deferred, tc.deferred) {
					t.Errorf("labels %v deferred %v, want %v %v", d.Labels, d.Deferred, tc.labels, tc.deferred)
				}
			}
		})
	}
}

// For any policy and session: a reader call never adds the inner tool's
// labels at decision time, it is held exactly when the reader's entry or
// the inner one would be held, and a forbidden label in either set
// always denies it.
func TestPropertyDecideReader(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		outer := genTool().Draw(t, "outer")
		inner := genTool().Draw(t, "inner")
		forbid := genLabels().Draw(t, "forbid")
		session := genLabels().Draw(t, "session")
		p := Policy{
			Tools:   map[string]Tool{"r": outer, "i": inner},
			Forbid:  forbid,
			Readers: map[string][]string{"r": {"i"}},
		}
		call := agentloop.ToolCall{Name: "r", Input: json.RawMessage(`{"tool":"i"}`)}
		d := p.Decide(context.Background(), call, agentloop.View{Labels: session})
		forbidden := len(agentloop.Forbidden(append(slices.Clone(outer.Labels), inner.Labels...), forbid)) > 0
		switch {
		case inner.Class != Read || forbidden:
			if d.Verdict != agentloop.Deny {
				t.Fatalf("decision %+v, want deny", d)
			}
		case Decide(session, outer) == agentloop.Hold || Decide(session, inner) == agentloop.Hold:
			if d.Verdict != agentloop.Hold {
				t.Fatalf("decision %+v, want hold", d)
			}
		default:
			if d.Verdict != agentloop.Allow || !slices.Equal(d.Labels, outer.Labels) || !slices.Equal(d.Deferred, inner.Labels) {
				t.Fatalf("decision %+v for outer %+v inner %+v", d, outer, inner)
			}
		}
	})
}
