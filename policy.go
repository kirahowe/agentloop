package agentloop

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"
)

// Verdict is a policy's answer about one tool call.
type Verdict string

const (
	// Allow lets the call run.
	Allow Verdict = "allow"
	// Deny refuses the call. The model is told the reason.
	Deny Verdict = "deny"
)

// A Decision is a verdict with the reasons and consequences that go with it.
type Decision struct {
	Verdict Verdict `json:"verdict"`
	// Reason is shown to the model on deny, and recorded always.
	Reason string `json:"reason,omitempty"`
	// Labels are added to the session when the call is allowed, before
	// its result exists.
	Labels []string `json:"labels,omitempty"`
}

// A Policy decides whether a tool call may run. It is asked only about
// calls to tools the agent has, whose input already matches the schema.
type Policy interface {
	Decide(ctx context.Context, call ToolCall, view View) Decision
}

// PolicyFunc makes a Policy from a function.
type PolicyFunc func(ctx context.Context, call ToolCall, view View) Decision

// Decide calls f.
func (f PolicyFunc) Decide(ctx context.Context, call ToolCall, view View) Decision {
	return f(ctx, call, view)
}

// View is the session as a policy sees it.
type View struct {
	// Labels held by the session, sorted.
	Labels []string
}

// Has reports whether the session holds the label.
func (v View) Has(label string) bool {
	for _, l := range v.Labels {
		if l == label {
			return true
		}
	}
	return false
}

// Forbidden is the labels in labels that forbid names, sorted, with no
// repeats: what a session given labels would hold that it may not. It is
// empty, never nil, when there are none.
func Forbidden(labels, forbid []string) []string {
	out := []string{}
	for _, l := range labels {
		if slices.Contains(forbid, l) && !slices.Contains(out, l) {
			out = append(out, l)
		}
	}
	sort.Strings(out)
	return out
}

// MaxLabelBytes is the longest a label may be.
const MaxLabelBytes = 100

// CheckLabel says why l is not a label, or returns nil. Labels are short
// provenance names such as "private" or "contaminated": 1 to MaxLabelBytes
// bytes, starting with a lower-case letter or digit and continuing with
// lower-case letters, digits and the characters _ . : @ / + -. Nothing
// else is allowed, so case, spacing, invisible, bidi and lookalike
// variants of a name cannot exist, and a forbid list cannot miss one by
// a byte.
func CheckLabel(l string) error {
	switch {
	case l == "":
		return errors.New("a label is empty")
	case len(l) > MaxLabelBytes:
		return fmt.Errorf("a label is longer than %d bytes", MaxLabelBytes)
	}
	for i := 0; i < len(l); i++ {
		c := l[i]
		alnum := c >= 'a' && c <= 'z' || c >= '0' && c <= '9'
		if !alnum && (i == 0 || !strings.ContainsRune("_.:@/+-", rune(c))) {
			return errors.New("a label may only use a-z, 0-9 and _ . : @ / + - , starting with a letter or digit")
		}
	}
	return nil
}

// CheckLabels returns the first error CheckLabel gives for labels.
func CheckLabels(labels []string) error {
	for _, l := range labels {
		if err := CheckLabel(l); err != nil {
			return fmt.Errorf("%q: %w", l, err)
		}
	}
	return nil
}
