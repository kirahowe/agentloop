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
			"web_ask":          {Class: Egress, Labels: []string{Contaminated}},
			"notes_write":      {Class: Write},
			"web_post":         {Class: Act},
			"elsewhere":        {Class: Read},
		},
		Readers: map[string][]string{"read_quarantined": {"mail_read", "web_fetch", "held_read", "web_ask", "notes_write", "web_post", "missing"}},
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
		{name: "an act inner tool is denied", call: readerCall("web_post"),
			verdict: agentloop.Deny, reason: "of class read or egress"},
		{name: "a write inner tool is denied", call: readerCall("notes_write"),
			verdict: agentloop.Deny, reason: "of class read or egress"},
		{name: "a clean caller reads through egress", call: readerCall("web_ask"),
			verdict: agentloop.Allow, labels: []string{}, deferred: []string{Contaminated}},
		{name: "a private caller reads through egress", call: readerCall("web_ask"), session: []string{Private},
			verdict: agentloop.Allow, labels: []string{}, deferred: []string{Contaminated}},
		{name: "a contaminated caller reads through egress", call: readerCall("web_ask"), session: []string{Contaminated},
			verdict: agentloop.Allow, labels: []string{}, deferred: []string{Contaminated}},
		{name: "a private, contaminated caller is held on egress", call: readerCall("web_ask"), session: []string{Contaminated, Private},
			verdict: agentloop.Hold, reason: "cannot send data out"},
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
// the inner one would be held under the caller's labels, an inner tool
// that is not read or egress is denied, and a forbidden label in either
// set always denies it.
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
		case (inner.Class != Read && inner.Class != Egress) || forbidden:
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

// An egress inner call is decided exactly as the same call made by the
// caller directly would be: the reader adds no way around the trifecta,
// and no stricter than calling the tool directly. It is held exactly
// when the caller holds both labels or the tool always holds.
func TestPropertyReaderEgressIsTheCallersCall(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		inner := genTool().Draw(t, "inner")
		inner.Class = Egress
		session := genLabels().Draw(t, "session")
		p := Policy{
			Tools:   map[string]Tool{"r": {Class: Read}, "i": inner},
			Readers: map[string][]string{"r": {"i"}},
		}
		viaReader := p.Decide(context.Background(), agentloop.ToolCall{Name: "r", Input: json.RawMessage(`{"tool":"i"}`)}, agentloop.View{Labels: session})
		direct := p.Decide(context.Background(), agentloop.ToolCall{Name: "i", Input: json.RawMessage(`{}`)}, agentloop.View{Labels: session})
		if viaReader.Verdict != direct.Verdict {
			t.Fatalf("via the reader %+v, directly %+v", viaReader, direct)
		}
		held := inner.AlwaysHold || (slices.Contains(session, Private) && slices.Contains(session, Contaminated))
		if held != (viaReader.Verdict == agentloop.Hold) {
			t.Fatalf("session %v: %+v", session, viaReader)
		}
	})
}
