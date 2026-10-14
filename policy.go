package agentloop

import "context"

// Verdict is a policy's answer about one tool call.
type Verdict string

const (
	// Allow lets the call run.
	Allow Verdict = "allow"
	// Deny refuses the call. The model is told the reason.
	Deny Verdict = "deny"
	// Hold parks the call until a person resolves it.
	Hold Verdict = "hold"
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
	// ToolCalls is the number of calls decided so far in the session. A
	// held call counts once a person has resolved it.
	ToolCalls int
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
