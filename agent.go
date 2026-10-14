package agentloop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

// An Agent is a model, the tools it was given and the rules around them.
// Its zero value runs nothing: Model is required, and the only tools are
// the ones in Tools.
type Agent struct {
	Model   Model
	ModelID string
	System  string
	Tools   []Tool
	// Policy decides each call after the tool and its input check out. Nil
	// allows every call to a tool the agent has.
	Policy          Policy
	Effort          Effort
	MaxOutputTokens int
	Limits          Limits
	// Retry is how failed model calls are retried. Nil means DefaultRetry.
	Retry *RetryPolicy
	// ToolTimeout bounds one tool call. Zero means no bound.
	ToolTimeout time.Duration
	// Observe sees every entry as it is appended and every streamed delta.
	// It runs on the loop's goroutine; keep it quick.
	Observe func(Event)

	// Now and Sleep stand in for the clock, so tests can exercise retries
	// without waiting. Nil means the real ones. Sleep returns early with
	// the context's error if the context ends.
	Now   func() time.Time
	Sleep func(ctx context.Context, d time.Duration) error
}

// Limits stop a run that spends too much. A zero field is no limit.
type Limits struct {
	// MaxTurns is the number of model calls in one run.
	MaxTurns int `json:"max_turns,omitempty"`
	// MaxToolCalls is the number of tool calls run in one run. Calls the
	// model makes beyond it are refused and the run ends.
	MaxToolCalls int `json:"max_tool_calls,omitempty"`
	// MaxTokens is Usage.Total across one run's model calls. It is checked
	// before each call, so the last call can overshoot it.
	MaxTokens int64 `json:"max_tokens,omitempty"`
}

// RetryPolicy says how a failed model call is retried.
type RetryPolicy struct {
	// Retries is the number of further attempts after the first.
	Retries int
	// Backoff is the first wait for overloaded or unreachable errors. It
	// doubles each time.
	Backoff time.Duration
	// MaxWait is the longest single wait, rate limits included. A longer
	// wait ends the run instead. Zero means no limit.
	MaxWait time.Duration
}

// DefaultRetry is used when an Agent sets no Retry.
var DefaultRetry = RetryPolicy{Retries: 3, Backoff: 2 * time.Second, MaxWait: 60 * time.Second}

// Reason is why a run ended.
type Reason string

const (
	// ReasonCompleted: the model finished its turn, or there was nothing to do.
	ReasonCompleted Reason = "completed"
	// ReasonMaxTokens: the last response was cut off by its token cap.
	ReasonMaxTokens Reason = "max_tokens"
	// ReasonRefused: the model declined to answer.
	ReasonRefused Reason = "refused"
	// ReasonHeld: calls wait for a person; see Outcome.Held.
	ReasonHeld Reason = "held"
	// ReasonLimit: a Limits field was reached; see Outcome.Limit.
	ReasonLimit Reason = "limit"
	// ReasonModelError: the model call failed for good; see Outcome.Error.
	ReasonModelError Reason = "model_error"
	// ReasonCanceled: the context ended.
	ReasonCanceled Reason = "canceled"
)

// Names of the limits, as reported in Outcome.Limit.
const (
	LimitTurns     = "max_turns"
	LimitToolCalls = "max_tool_calls"
	LimitTokens    = "max_tokens"
)

// An Outcome is how a run ended. Everything but a setup or store failure
// is one.
type Outcome struct {
	Reason Reason `json:"reason"`
	// Text is the last assistant text.
	Text  string      `json:"text,omitempty"`
	Error *ModelError `json:"error,omitempty"`
	Held  []ToolCall  `json:"held,omitempty"`
	Limit string      `json:"limit,omitempty"`
	// Turns and Usage count this run only.
	Turns int   `json:"turns"`
	Usage Usage `json:"usage"`
}

// An Event is one thing a run reports to Observe: either an entry just
// appended or a streamed delta.
type Event struct {
	Entry *Entry
	Delta *Delta
}

// Run carries the session forward until the model is done, something
// needs a person, or something stops it. It first settles the calls an
// earlier run left open, so it resumes a session from any point. A
// non-empty prompt is added as a user message once those are settled.
//
// Run returns an error only if the agent is set up wrongly or the store
// fails. Anything else is an Outcome.
func (a *Agent) Run(ctx context.Context, s *Session, prompt string) (*Outcome, error) {
	if a.Model == nil {
		return nil, errors.New("agent has no model")
	}
	tb, err := newToolbox(a.Tools)
	if err != nil {
		return nil, err
	}
	r := &runner{a: a, s: s, tb: tb, retry: DefaultRetry}
	if a.Retry != nil {
		r.retry = *a.Retry
	}
	if err := r.record(Entry{Type: EntryRunStarted, RunStarted: &RunStartedEntry{Prompt: prompt}}); err != nil {
		return nil, err
	}
	out, err := r.drive(ctx, prompt)
	if err != nil {
		return nil, err
	}
	if err := r.record(Entry{Type: EntryRunEnded, RunEnded: out}); err != nil {
		return nil, err
	}
	return out, nil
}

type runner struct {
	a     *Agent
	s     *Session
	tb    *toolbox
	retry RetryPolicy

	turns     int
	usage     Usage
	toolCalls int
}

func (r *runner) record(e Entry) error {
	got, err := r.s.record(e)
	if err != nil {
		return err
	}
	if r.a.Observe != nil {
		r.a.Observe(Event{Entry: &got})
	}
	return nil
}

func (r *runner) outcome(reason Reason) *Outcome {
	return &Outcome{Reason: reason, Text: r.s.lastText(), Turns: r.turns, Usage: r.usage}
}

func (r *runner) drive(ctx context.Context, prompt string) (*Outcome, error) {
	if out, err := r.settleAndCheck(ctx); out != nil || err != nil {
		return out, err
	}
	if prompt != "" {
		err := r.record(Entry{Type: EntryMessage, Message: &MessageEntry{Message: Text(User, prompt)}})
		if err != nil {
			return nil, err
		}
	} else if !r.modelIsDue() {
		return r.endOfTurn(), nil
	}
	for {
		if ctx.Err() != nil {
			return r.outcome(ReasonCanceled), nil
		}
		if lim := r.limitReached(); lim != "" {
			out := r.outcome(ReasonLimit)
			out.Limit = lim
			return out, nil
		}
		resp, out, err := r.generate(ctx)
		if out != nil || err != nil {
			return out, err
		}
		r.turns++
		r.usage.add(resp.Usage)
		msg := keepable(resp.Message)
		err = r.record(Entry{Type: EntryMessage, Message: &MessageEntry{
			Message: msg, Stop: resp.Stop, Usage: resp.Usage, Model: resp.Model, Refusal: resp.Refusal,
		}})
		if err != nil {
			return nil, err
		}
		switch resp.Stop {
		case StopToolUse:
			if len(msg.ToolCalls()) == 0 {
				return r.outcome(ReasonCompleted), nil
			}
			if out, err := r.settleAndCheck(ctx); out != nil || err != nil {
				return out, err
			}
		case StopPause:
		case StopMaxTokens:
			return r.outcome(ReasonMaxTokens), nil
		case StopRefusal:
			return r.outcome(ReasonRefused), nil
		default:
			return r.outcome(ReasonCompleted), nil
		}
	}
}

// modelIsDue reports whether the conversation is waiting for the
// model: it ends on a user message, or on a turn the provider paused.
func (r *runner) modelIsDue() bool {
	msgs := r.s.Messages()
	if len(msgs) == 0 {
		return false
	}
	if msgs[len(msgs)-1].Role == User {
		return true
	}
	turn, ok := r.s.lastTurn()
	return ok && turn.Stop == StopPause
}

// endOfTurn is the outcome of a run that finds the model has nothing
// left to say, named for how the last turn ended.
func (r *runner) endOfTurn() *Outcome {
	if turn, ok := r.s.lastTurn(); ok {
		switch turn.Stop {
		case StopMaxTokens:
			return r.outcome(ReasonMaxTokens)
		case StopRefusal:
			return r.outcome(ReasonRefused)
		}
	}
	return r.outcome(ReasonCompleted)
}

func (r *runner) limitReached() string {
	l := r.a.Limits
	switch {
	case l.MaxTurns > 0 && r.turns >= l.MaxTurns:
		return LimitTurns
	case l.MaxTokens > 0 && r.usage.Total() >= l.MaxTokens:
		return LimitTokens
	}
	return ""
}

// generate makes the model call, retrying under the policy. It returns
// the response, or the outcome that ends the run.
func (r *runner) generate(ctx context.Context) (*Response, *Outcome, error) {
	req := &Request{
		Model:           r.a.ModelID,
		System:          r.a.System,
		Messages:        r.s.Messages(),
		Tools:           r.tb.specs(),
		Effort:          r.a.Effort,
		MaxOutputTokens: r.a.MaxOutputTokens,
	}
	var observe func(Delta)
	if r.a.Observe != nil {
		observe = func(d Delta) { r.a.Observe(Event{Delta: &d}) }
	}
	for attempt := 0; ; attempt++ {
		resp, err := r.a.Model.Generate(ctx, req, observe)
		if err == nil && resp == nil {
			err = &ModelError{Kind: InvalidRequest, Message: "the model returned no response"}
		}
		if err == nil {
			return resp, nil, nil
		}
		// A cancelled request fails however the adapter words it; that is
		// the context ending, not the provider failing.
		if ctx.Err() != nil {
			return nil, r.outcome(ReasonCanceled), nil
		}
		me := AsModelError(err)
		wait, retry := r.retryWait(me, attempt)
		e := Entry{Type: EntryModelError, ModelError: &ModelErrorEntry{Error: *me, Retry: retry}}
		if retry {
			e.ModelError.WaitMillis = wait.Milliseconds()
		}
		if err := r.record(e); err != nil {
			return nil, nil, err
		}
		if !retry {
			out := r.outcome(ReasonModelError)
			out.Error = me
			return nil, out, nil
		}
		if err := r.sleep(ctx, wait); err != nil {
			return nil, r.outcome(ReasonCanceled), nil
		}
	}
}

// retryWait says whether to try again after the error, and how long to
// wait first.
func (r *runner) retryWait(me *ModelError, attempt int) (time.Duration, bool) {
	p := r.retry
	if !me.Transient() || attempt >= p.Retries {
		return 0, false
	}
	backoff := p.Backoff
	for i := 0; i < attempt && backoff < time.Hour; i++ {
		backoff *= 2
	}
	until := time.Duration(0)
	if !me.RetryAfter.IsZero() {
		until = me.RetryAfter.Sub(r.now())
	}
	wait := backoff
	if me.Kind == RateLimited && !me.RetryAfter.IsZero() {
		wait = until
	} else if until > wait {
		wait = until
	}
	if wait < 0 {
		wait = 0
	}
	if p.MaxWait > 0 && wait > p.MaxWait {
		return 0, false
	}
	return wait, true
}

func (r *runner) now() time.Time {
	if r.a.Now != nil {
		return r.a.Now()
	}
	return time.Now()
}

func (r *runner) sleep(ctx context.Context, d time.Duration) error {
	if r.a.Sleep != nil {
		return r.a.Sleep(ctx, d)
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// settleAndCheck settles the last turn's open calls and returns the
// outcome if that ends the run.
func (r *runner) settleAndCheck(ctx context.Context) (*Outcome, error) {
	res, err := r.settle(ctx)
	if err != nil {
		return nil, err
	}
	switch {
	case res.canceled:
		return r.outcome(ReasonCanceled), nil
	case len(res.held) > 0:
		out := r.outcome(ReasonHeld)
		out.Held = res.held
		return out, nil
	case res.limit:
		out := r.outcome(ReasonLimit)
		out.Limit = LimitToolCalls
		return out, nil
	}
	return nil, nil
}

type settled struct {
	canceled bool
	held     []ToolCall
	limit    bool
}

// settle gives every open call of the last assistant turn a result, or
// leaves it held. See DESIGN.md for the rules.
func (r *runner) settle(ctx context.Context) (settled, error) {
	var res settled
	turn, ok := r.s.lastTurn()
	if !ok {
		return res, nil
	}
	cutOff := turn.Stop != StopToolUse
	for _, id := range turn.Calls {
		info, _ := r.s.Call(id)
		if info.Finished {
			continue
		}
		switch {
		case cutOff:
			msg := fmt.Sprintf("This call was not run: the turn ended (%s) before the model finished it.", turn.Stop)
			if err := r.finish(id, ToolResult{Content: msg, IsError: true}, nil); err != nil {
				return res, err
			}
			continue
		case info.Started:
			msg := "The run stopped while this call was running, so it may or may not have taken effect. It was not run again."
			if err := r.finish(id, ToolResult{Content: msg, IsError: true}, nil); err != nil {
				return res, err
			}
			continue
		}
		if ctx.Err() != nil {
			res.canceled = true
			return res, nil
		}
		if info.Decision == nil {
			d := r.decide(ctx, info.Call)
			if err := r.recordDecision(info.Call, d); err != nil {
				return res, err
			}
			info, _ = r.s.Call(id)
		}
		switch info.Decision.Verdict {
		case Hold:
			res.held = append(res.held, info.Call)
		case Allow:
			if max := r.a.Limits.MaxToolCalls; max > 0 && r.toolCalls >= max {
				msg := "This call was not run: the limit on tool calls was reached."
				if err := r.finish(id, ToolResult{Content: msg, IsError: true}, nil); err != nil {
					return res, err
				}
				res.limit = true
				continue
			}
			canceled, err := r.execute(ctx, info.Call)
			if err != nil {
				return res, err
			}
			if canceled {
				res.canceled = true
				return res, nil
			}
		default:
			msg := info.Decision.Reason
			if msg == "" {
				msg = "The call was denied."
			}
			if err := r.finish(id, ToolResult{Content: msg, IsError: true}, nil); err != nil {
				return res, err
			}
		}
	}
	return res, nil
}

// decide runs the three checks: the tool exists, the input matches its
// schema, the policy allows it. Anything but a clear verdict is a deny.
func (r *runner) decide(ctx context.Context, call ToolCall) Decision {
	if why := r.tb.check(call.Name, call.Input); why != "" {
		return Decision{Verdict: Deny, Reason: why}
	}
	if r.a.Policy == nil {
		return Decision{Verdict: Allow}
	}
	d := r.a.Policy.Decide(ctx, call, r.s.View())
	switch d.Verdict {
	case Allow, Deny, Hold:
		return d
	}
	return Decision{Verdict: Deny, Reason: fmt.Sprintf("The policy gave no usable verdict (%q).", d.Verdict)}
}

func (r *runner) recordDecision(call ToolCall, d Decision) error {
	return r.record(Entry{Type: EntryDecision, Decision: &DecisionEntry{
		CallID: call.ID, Tool: call.Name, Verdict: d.Verdict, Reason: d.Reason, Labels: d.Labels,
	}})
}

func (r *runner) finish(id string, res ToolResult, labels []string) error {
	res.CallID = id
	return r.record(Entry{Type: EntryToolResult, ToolResult: &ToolResultEntry{Result: res, Labels: labels}})
}

// execute runs an allowed call and records its result. It reports
// whether the context ended meanwhile.
func (r *runner) execute(ctx context.Context, call ToolCall) (canceled bool, err error) {
	if err := r.record(Entry{Type: EntryCallStarted, CallStarted: &CallStartedEntry{CallID: call.ID}}); err != nil {
		return false, err
	}
	r.toolCalls++
	callCtx, cancel := ctx, context.CancelFunc(func() {})
	if r.a.ToolTimeout > 0 {
		callCtx, cancel = context.WithTimeout(ctx, r.a.ToolTimeout)
	}
	out, callErr := r.tb.tools[call.Name].Call(callCtx, call.Input)
	timedOut := callCtx.Err() == context.DeadlineExceeded && ctx.Err() == nil
	cancel()

	res := ToolResult{Content: out.Content, IsError: out.IsError}
	if callErr != nil {
		res.IsError = true
		switch {
		case ctx.Err() != nil:
			res.Content = fmt.Sprintf("The call was canceled before it finished: %v", callErr)
		case timedOut:
			res.Content = fmt.Sprintf("The tool timed out after %s.", r.a.ToolTimeout)
		default:
			res.Content = fmt.Sprintf("The tool failed: %v", callErr)
		}
	}
	if err := r.finish(call.ID, res, out.Labels); err != nil {
		return false, err
	}
	return ctx.Err() != nil, nil
}

// keepable returns the assistant message in a form the log can hold. A
// call with no input gets {}. A call whose input is not JSON could not be
// written as an entry, so its input becomes the raw text as a JSON string,
// which no tool's object schema accepts. The model is then told the input
// was wrong, as with any other bad input.
func keepable(m Message) Message {
	m.Role = Assistant
	fixed := false
	parts := make([]Part, len(m.Parts))
	copy(parts, m.Parts)
	for i, p := range parts {
		if p.Type != ToolCallPart || p.ToolCall == nil {
			continue
		}
		call := *p.ToolCall
		switch {
		case len(call.Input) == 0:
			call.Input = json.RawMessage(`{}`)
		case !json.Valid(call.Input):
			call.Input, _ = json.Marshal(string(p.ToolCall.Input))
		default:
			continue
		}
		parts[i].ToolCall = &call
		fixed = true
	}
	if fixed {
		m.Parts = parts
	}
	return m
}
