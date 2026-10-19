// Package trifecta keeps an agent from holding all three legs of the
// lethal trifecta at once: private data, untrusted content, and a way to
// send data where an outsider can see it.
//
// A session's labels say what it has seen: private when it has seen the
// operator's private data, contaminated when it has seen anything an
// outsider could have written. Labels only grow. Each tool has a class
// saying what a call to it can do, and the rule (Decide) holds a call that
// would complete the trifecta until a person resolves it. A call that acts
// where others can see it, spends money or cannot be undone always waits
// for the operator's consent: given at the time, or in advance as a
// standing rule the operator wrote (Consent).
//
// Everything here is pure: no I/O, no clock, no state.
package trifecta

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"slices"
	"sort"
	"strings"
	"unicode"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/internal/strictjson"
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
	// Edit changes the operator's own things outside relay: nobody else
	// sees the change, and the earlier state survives, so it can be
	// undone (the operator's own reminders, a draft in their mail). A
	// call is held once the session is contaminated, since what it
	// writes leaves relay without its labels: read back later, it would
	// come in clean. A tool that overwrites or deletes with no history
	// kept is Act.
	Edit Class = "edit"
	// Egress can put data the model chose where an outsider can see it.
	Egress Class = "egress"
	// Act does something others see, spends money, or cannot be undone:
	// sending a campaign, a refund, a post, a payment.
	Act Class = "act"
)

// Classes lists every class.
var Classes = []Class{Read, Write, Edit, Egress, Act}

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
	// Consent is the operator's standing consent for an act tool: rules,
	// each a map from input argument names to the exact JSON values a
	// call must have. A call whose input matches every pair of some rule
	// runs without a hold (see Consented). Only act tools have it.
	Consent []map[string]json.RawMessage `json:"consent,omitempty"`
}

// Decide is the rule for a call whose input no consent rule covers.
// Given the session's labels when a call is decided and the tool's
// entry, it says whether the call runs or waits for a person:
//
//   - always_hold: hold
//   - act: hold, whatever the labels
//   - edit, and the session holds contaminated: hold
//   - egress, and the session holds both private and contaminated: hold
//   - otherwise (read, write): allow, whatever the labels
//
// It never denies; a person does that. A class it does not know is held,
// so a mistake fails closed. DecideCall adds the operator's standing
// consent.
func Decide(labels []string, t Tool) agentloop.Verdict {
	has := func(l string) bool { return slices.Contains(labels, l) }
	switch {
	case t.AlwaysHold:
		return agentloop.Hold
	case t.Class == Act:
		return agentloop.Hold
	case t.Class == Edit && has(Contaminated):
		return agentloop.Hold
	case t.Class == Egress && has(Private) && has(Contaminated):
		return agentloop.Hold
	case !KnownClass(t.Class):
		return agentloop.Hold
	}
	return agentloop.Allow
}

// DecideCall is the whole rule for one call: Decide, except that an act
// call (without always_hold) whose input matches one of the tool's
// consent rules is allowed. It returns the verdict and the index of the
// rule that allowed it, or -1 when no rule did.
func DecideCall(labels []string, t Tool, input json.RawMessage) (agentloop.Verdict, int) {
	if t.Class == Act && !t.AlwaysHold {
		if i := Consented(t.Consent, input); i >= 0 {
			return agentloop.Allow, i
		}
	}
	return Decide(labels, t), -1
}

// Consented is the index of the first rule that input matches, or -1. A
// rule matches when the input is a JSON object that has every key the
// rule names, each with a value equal to the rule's: the same JSON value
// whatever its spacing, key order or string escapes. A number matches
// only one written the same way (1 is not 1.0), and a string only the
// same characters, with no Unicode folding. An empty rule matches
// nothing, and so does input that is not one object, or that names a key
// twice anywhere, since a tool could read either value. Two keys that
// differ only in case (group and Group) count as the same key twice:
// many decoders, Go's encoding/json among them, read either into one
// field.
func Consented(rules []map[string]json.RawMessage, input json.RawMessage) int {
	if len(rules) == 0 || strictjson.DuplicateKeys(input) != nil {
		return -1
	}
	v := normal(input)
	obj, ok := v.(map[string]any)
	if !ok || foldCollides(v) {
		return -1
	}
	for i, rule := range rules {
		if len(rule) > 0 && matches(rule, obj) {
			return i
		}
	}
	return -1
}

func matches(rule map[string]json.RawMessage, obj map[string]any) bool {
	for k, want := range rule {
		got, ok := obj[k]
		if !ok || strictjson.DuplicateKeys(want) != nil {
			return false
		}
		w := normal(want)
		if w == errValue || !equal(got, w) {
			return false
		}
	}
	return true
}

// foldCollides reports whether any object in v has two keys that are
// equal under Unicode case folding.
func foldCollides(v any) bool {
	switch x := v.(type) {
	case map[string]any:
		seen := map[string]bool{}
		for k, e := range x {
			f := fold(k)
			if seen[f] || foldCollides(e) {
				return true
			}
			seen[f] = true
		}
	case []any:
		for _, e := range x {
			if foldCollides(e) {
				return true
			}
		}
	}
	return false
}

// fold maps each rune of s to the least rune of its case-folding orbit,
// so strings equal under folding (K, k and the Kelvin sign) fold alike.
func fold(s string) string {
	var b strings.Builder
	for _, r := range s {
		least := r
		for f := unicode.SimpleFold(r); f != r; f = unicode.SimpleFold(f) {
			least = min(least, f)
		}
		b.WriteRune(least)
	}
	return b.String()
}

// errValue stands for JSON that does not parse as one value; it equals
// nothing.
var errValue = new(int)

// normal is the JSON value in b with numbers kept as written, or errValue.
func normal(b []byte) any {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return errValue
	}
	if _, err := dec.Token(); err != io.EOF {
		return errValue
	}
	return v
}

// equal compares two normal values.
func equal(a, b any) bool {
	switch x := a.(type) {
	case map[string]any:
		y, ok := b.(map[string]any)
		if !ok || len(x) != len(y) {
			return false
		}
		for k, v := range x {
			w, ok := y[k]
			if !ok || !equal(v, w) {
				return false
			}
		}
		return true
	case []any:
		y, ok := b.([]any)
		if !ok || len(x) != len(y) {
			return false
		}
		for i := range x {
			if !equal(x[i], y[i]) {
				return false
			}
		}
		return true
	case json.Number:
		y, ok := b.(json.Number)
		return ok && x == y
	case string:
		y, ok := b.(string)
		return ok && x == y
	case bool:
		y, ok := b.(bool)
		return ok && x == y
	case nil:
		return b == nil
	}
	return false
}

// A Policy is the rule over a set of tools. It is an agentloop.Policy.
type Policy struct {
	// Tools is the entry for each tool, by name. A call to a tool with no
	// entry is denied.
	Tools map[string]Tool
	// Forbid is the labels the session may never hold. A call to a tool
	// whose static labels include one is denied: allowing it would give
	// the session that label before the call even ran.
	Forbid []string
	// Readers names the quarantined reader tools, each with the inner
	// tools it may read through. A reader call is decided as its inner
	// call would be (see DecideReader).
	Readers map[string][]string
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
	if inner, ok := p.Readers[call.Name]; ok {
		return p.decideReader(call, t, inner, view)
	}
	if len(agentloop.Forbidden(t.Labels, p.Forbid)) > 0 {
		return agentloop.Decision{Verdict: agentloop.Deny, Reason: ForbiddenReason}
	}
	verdict, rule := DecideCall(view.Labels, t, call.Input)
	d := agentloop.Decision{Verdict: verdict, Labels: slices.Clone(t.Labels)}
	if rule >= 0 {
		d.Consent = &rule
	}
	if d.Verdict == agentloop.Hold {
		d.Reason = holdReason(t)
	}
	return d
}

// decideReader decides a call to a quarantined reader, whose input names
// the inner tool it reads through in "tool".
//
// The call is decided as the inner call would be: held if the reader's
// own entry or the inner tool's holds it, with that entry's reason. The
// decision's labels are the reader's own static labels. The inner tool's
// static labels describe what the reader sees, not what the caller sees,
// so they are deferred to the result, where the reader decides what
// passes through. A forbidden label among either set denies the call.
// An inner tool that is not on the reader's list, has no entry, or is
// not class read or egress is denied.
//
// An egress inner tool is the one exception to reading only through
// read tools, made for a web search that fetches its results: the
// caller writes the inner input (the query), so the inner call is
// decided under the caller's labels, as if the caller made it. A caller
// that is private and contaminated is held. The reader model sees only
// the inner tool's output and never writes an input. Write and act
// tools change things, so the reader never runs them.
func (p Policy) decideReader(call agentloop.ToolCall, outer Tool, allowed []string, view agentloop.View) agentloop.Decision {
	deny := func(reason string) agentloop.Decision {
		return agentloop.Decision{Verdict: agentloop.Deny, Reason: reason}
	}
	var in struct {
		Tool string `json:"tool"`
	}
	if err := json.Unmarshal(call.Input, &in); err != nil || in.Tool == "" {
		return deny(fmt.Sprintf("The input to %s does not name an inner tool.", call.Name))
	}
	if !slices.Contains(allowed, in.Tool) {
		return deny(fmt.Sprintf("%s cannot read through %s.", call.Name, in.Tool))
	}
	inner, ok := p.Tools[in.Tool]
	if !ok {
		return deny(fmt.Sprintf("The tool %s has no policy.", in.Tool))
	}
	if inner.Class != Read && inner.Class != Egress {
		return deny(fmt.Sprintf("%s reads only through tools of class read or egress, and %s is %q.", call.Name, in.Tool, inner.Class))
	}
	if len(agentloop.Forbidden(append(slices.Clone(outer.Labels), inner.Labels...), p.Forbid)) > 0 {
		return deny(ForbiddenReason)
	}
	d := agentloop.Decision{Verdict: agentloop.Allow, Labels: slices.Clone(outer.Labels), Deferred: slices.Clone(inner.Labels)}
	switch {
	case Decide(view.Labels, outer) == agentloop.Hold:
		d.Verdict, d.Reason = agentloop.Hold, holdReason(outer)
	case Decide(view.Labels, inner) == agentloop.Hold:
		d.Verdict, d.Reason = agentloop.Hold, holdReason(inner)
	}
	return d
}

// ForbiddenReason is what the model is told about a call to a tool whose
// results always carry a label the session may not hold.
const ForbiddenReason = "This tool's results carry a label this session may not hold, so it cannot be called here."

// holdReason says why a call waits, for the record and the person.
func holdReason(t Tool) string {
	switch {
	case t.AlwaysHold:
		return "Every call to this tool waits for a person."
	case t.Class == Act:
		return "This tool does something others see, spends money, or cannot be undone, so each call waits for the operator's consent."
	case t.Class == Edit:
		return "The session has seen untrusted content, so it cannot change the operator's things outside relay without a person's approval."
	case t.Class == Egress:
		return "The session has seen private data and untrusted content, so it cannot send data out without a person's approval."
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
