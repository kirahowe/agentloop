// Package trifecta keeps an agent from holding all three legs of the
// lethal trifecta at once: private data, untrusted content, and a way to
// send data where an outsider can see it.
//
// A session's labels say what it has seen: private when it has seen the
// operator's private data, contaminated when it has seen anything an
// outsider could have written. Labels only grow. Each tool has a class
// saying what a call to it can do, and the rule (Decide) holds a call that
// would complete the trifecta, or act in the operator's name on a
// contaminated session's say-so, until a person resolves it.
//
// Everything here is pure: no I/O, no clock, no state.
package trifecta

import (
	"context"
	"fmt"
	"slices"
	"sort"

	"github.com/kirahowe/agentloop"
)

// The labels. There are no others: any other name is an error, so a typo
// cannot drop a label.
const (
	// Private: the session has seen the operator's private data.
	Private = "private"
	// Contaminated: the session has seen content an outsider could have
	// written, or that a contaminated session wrote.
	Contaminated = "contaminated"
)

// Labels lists every label, sorted.
var Labels = []string{Contaminated, Private}

// KnownLabel reports whether l is a label.
func KnownLabel(l string) bool { return slices.Contains(Labels, l) }

// CheckLabels returns an error naming the first label that is not one.
func CheckLabels(labels []string) error {
	for _, l := range labels {
		if !KnownLabel(l) {
			return fmt.Errorf("unknown label %q (want one of %v)", l, Labels)
		}
	}
	return nil
}

// A Class says what a call to a tool can do.
type Class string

const (
	// Read returns content, changes nothing, and sends nothing out that an
	// outsider sees.
	Read Class = "read"
	// Write changes relay's own state only.
	Write Class = "write"
	// Egress can put data the model chose where an outsider can see it.
	Egress Class = "egress"
	// Act does something in the operator's name outside relay.
	Act Class = "act"
)

// Classes lists every class.
var Classes = []Class{Read, Write, Egress, Act}

// KnownClass reports whether c is a class.
func KnownClass(c Class) bool { return slices.Contains(Classes, c) }

// A Tool is one tool's policy entry.
type Tool struct {
	Class Class `json:"class"`
	// Labels are added to the session when a call to the tool is allowed,
	// before it runs: what its results are known to carry.
	Labels []string `json:"labels,omitempty"`
	// AlwaysHold holds every call to the tool for a person.
	AlwaysHold bool `json:"always_hold,omitempty"`
}

// Decide is the rule. Given the session's labels when a call is decided
// and the tool's entry, it says whether the call runs or waits for a
// person:
//
//   - always_hold: hold
//   - egress, and the session holds both private and contaminated: hold
//   - act, and the session holds contaminated: hold
//   - otherwise: allow
//
// It never denies; a person does that. A class it does not know is held,
// so a mistake fails closed.
func Decide(labels []string, t Tool) agentloop.Verdict {
	has := func(l string) bool { return slices.Contains(labels, l) }
	switch {
	case t.AlwaysHold:
		return agentloop.Hold
	case t.Class == Egress && has(Private) && has(Contaminated):
		return agentloop.Hold
	case t.Class == Act && has(Contaminated):
		return agentloop.Hold
	case !KnownClass(t.Class):
		return agentloop.Hold
	}
	return agentloop.Allow
}

// A Policy is the rule over a set of tools. It is an agentloop.Policy.
type Policy struct {
	// Tools is the entry for each tool, by name. A call to a tool with no
	// entry is denied.
	Tools map[string]Tool
}

var _ agentloop.Policy = Policy{}

// Decide decides a call from the session's labels. The decision carries
// the tool's labels, which the session takes on when the call is allowed,
// now or by a person later.
func (p Policy) Decide(_ context.Context, call agentloop.ToolCall, view agentloop.View) agentloop.Decision {
	t, ok := p.Tools[call.Name]
	if !ok {
		return agentloop.Decision{Verdict: agentloop.Deny, Reason: fmt.Sprintf("The tool %s has no policy.", call.Name)}
	}
	d := agentloop.Decision{Verdict: Decide(view.Labels, t), Labels: slices.Clone(t.Labels)}
	if d.Verdict == agentloop.Hold {
		d.Reason = holdReason(t)
	}
	return d
}

// holdReason says why a call waits, for the record and the person.
func holdReason(t Tool) string {
	switch {
	case t.AlwaysHold:
		return "Every call to this tool waits for a person."
	case t.Class == Egress:
		return "The session has seen private data and untrusted content, so it cannot send data out without a person's approval."
	case t.Class == Act:
		return "The session has seen untrusted content, so it cannot act for the operator without a person's approval."
	}
	return fmt.Sprintf("The tool's class %q is unknown.", t.Class)
}

// Union is the labels of a and b, sorted, with no repeats.
func Union(a, b []string) []string {
	set := map[string]bool{}
	for _, l := range a {
		set[l] = true
	}
	for _, l := range b {
		set[l] = true
	}
	out := make([]string, 0, len(set))
	for l := range set {
		out = append(out, l)
	}
	sort.Strings(out)
	return out
}
