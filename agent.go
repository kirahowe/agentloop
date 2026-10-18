package agentloop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/kirahowe/agentloop/internal/strictjson"
)

// An Agent is a model, the tools it was given and the rules around them.
// Its zero value runs nothing: Model is required, and the only tools are
// the ones in Tools.
type Agent struct {
	Model   Model
	ModelID string
	System  string
	Tools   []Tool
	// Labels are added to the session whenever a run starts, before
	// anything else happens in it.
	Labels []string
	// Forbid is the labels the session may never hold. A run on a session
	// that holds one, or would from Labels, does not start (ReasonForbidden).
	// A tool result carrying one is refused: the model is told only that
	// (RefusedResult) and the session does not take the label. An allow
	// whose decision would add one is a deny.
	Forbid []string
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
	// Pins are digests of tool definitions, by tool name, that the session
	// must keep. A run records the ones the session lacks, and a run on a
	// session that pinned a different digest for a tool does not start:
	// Run returns a *PinError and records nothing. The digests are opaque
	// here; the mcp package makes them.
	Pins map[string]string
	// DisplayRefPrefix, when set, names where a result's display is kept:
	// its ref is the prefix and DisplayID of the call, such as
	// "shown:2026-10-18/" and "toolu_01abc". The model is told the ref.
	DisplayRefPrefix string
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
	// wait ends the run instead. Zero means no cap: any wait is accepted.
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
	// ReasonForbidden: the session holds, or would start with, a label
	// Agent.Forbid names, so nothing ran; see Outcome.Forbidden.
	ReasonForbidden Reason = "forbidden"
)

// RefusedResult is what the model is told in place of a tool result that
// carried a label the session may not hold.
const RefusedResult = "This result was refused: it carries a label this session may not hold."

// ForbiddenDecision is the reason recorded for a call whose decision
// would have given the session a label it may not hold.
const ForbiddenDecision = "This call would give the session a label it may not hold, so it was not run."

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
	Held  []HeldCall  `json:"held,omitempty"`
	// LabelSources, for held, says which entry first gave the session
	// each of its labels, so whoever decides can see why it holds them.
	LabelSources map[string]LabelSource `json:"label_sources,omitempty"`
	Limit        string                 `json:"limit,omitempty"`
	// Forbidden, for forbidden, is the labels the session held or would
	// have started with that Agent.Forbid names.
	Forbidden []string `json:"forbidden,omitempty"`
	// Refused lists the tool results of this run that were refused for
	// carrying a forbidden label, whatever the reason the run ended.
	Refused []Refusal `json:"refused,omitempty"`
	// Turns and Usage count this run only.
	Turns int   `json:"turns"`
	Usage Usage `json:"usage"`
}

// A HeldCall is a call waiting for a person, with the digest of its input
// that a resolution must name (see Agent.Resolve).
type HeldCall struct {
	ToolCall
	InputSHA256 string `json:"input_sha256"`
}

// A Refusal is one tool result the session was not given, because it
// carried labels the session may not hold.
type Refusal struct {
	CallID string   `json:"call_id"`
	Tool   string   `json:"tool"`
	Labels []string `json:"labels"`
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
// Run returns an error only if the agent is set up wrongly, the store
// fails, or another Run is already in progress on the session (at most one
// runs at a time). Anything else is an Outcome.
func (a *Agent) Run(ctx context.Context, s *Session, prompt string) (*Outcome, error) {
	if a.Model == nil {
		return nil, errors.New("agent has no model")
	}
	tb, err := newToolbox(a.Tools)
	if err != nil {
		return nil, err
	}
	if err := s.begin(); err != nil {
		return nil, err
	}
	defer s.end()
	if err := s.CheckPins(a.Pins); err != nil {
		return nil, err
	}
	// A session that holds a forbidden label, or would from the agent's
	// own, is not run at all: nothing is recorded and the model is not
	// called.
	if bad := Forbidden(append(s.Labels(), a.Labels...), a.Forbid); len(bad) > 0 {
		return &Outcome{Reason: ReasonForbidden, Forbidden: bad}, nil
	}
	r := &runner{a: a, s: s, tb: tb, retry: DefaultRetry}
	if a.Retry != nil {
		r.retry = *a.Retry
	}
	if err := r.record(Entry{Type: EntryRunStarted, RunStarted: &RunStartedEntry{Prompt: prompt}}); err != nil {
		return nil, err
	}
	if e, err := s.AddLabels(a.Labels, LabelSeed); err != nil {
		return nil, err
	} else if e != nil && a.Observe != nil {
		a.Observe(Event{Entry: e})
	}
	if e, err := s.pin(a.Pins); err != nil {
		return nil, err
	} else if e != nil && a.Observe != nil {
		a.Observe(Event{Entry: e})
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
	refused   []Refusal
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
	return &Outcome{Reason: reason, Text: r.s.lastText(), Turns: r.turns, Usage: r.usage, Refused: r.refused}
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
		msg := keepable(resp.Message)
		if bad := firstBad(badCallID(r.s, msg), unbounded(msg)); bad != "" {
			// The provider broke its own contract. Nothing of the turn is
			// kept, and asking again would not change what it says.
			out := r.outcome(ReasonModelError)
			out.Error = &ModelError{Kind: InvalidRequest, Message: bad}
			return out, nil
		}
		r.turns++
		r.usage.add(resp.Usage)
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
			// Calls in a paused turn were cut off like any others, and
			// the model must see that they were not run.
			if out, err := r.settleAndCheck(ctx); out != nil || err != nil {
				return out, err
			}
		case StopMaxTokens:
			if err := r.settleCutOff(ctx); err != nil {
				return nil, err
			}
			return r.outcome(ReasonMaxTokens), nil
		case StopRefusal:
			if err := r.settleCutOff(ctx); err != nil {
				return nil, err
			}
			return r.outcome(ReasonRefused), nil
		default:
			if err := r.settleCutOff(ctx); err != nil {
				return nil, err
			}
			return r.outcome(ReasonCompleted), nil
		}
	}
}

// settleCutOff answers the calls of a turn that ended without tool_use
// before the run ends, so the session is a conversation a model can take
// whenever no call is held. Nothing runs: settle gives each open call of
// such a turn its cut-off result without consulting the policy or the tools.
func (r *runner) settleCutOff(ctx context.Context) error {
	_, err := r.settle(ctx)
	return err
}

func firstBad(problems ...string) string {
	for _, p := range problems {
		if p != "" {
			return p
		}
	}
	return ""
}

// unbounded says why a response cannot be written to the log, if its tool
// calls' input or the provider's record is nested deeper, or has a longer
// number, than strictjson allows: every line agentloop writes must stay
// readable by strict parsers.
func unbounded(m Message) string {
	for _, c := range m.ToolCalls() {
		if err := strictjson.Bounded(c.Input); err != nil {
			return "the model returned a tool call whose input has " + err.Error()
		}
	}
	if m.Native != nil && len(m.Native.Data) > 0 {
		if err := strictjson.Bounded(m.Native.Data); err != nil {
			return "the provider's record of the turn has " + err.Error()
		}
	}
	return ""
}

// badCallID says what is wrong with the tool call ids of a message, or
// returns "" if nothing is. Ids must be present and not used before in the
// session, or results could not be matched to calls.
func badCallID(s *Session, m Message) string {
	seen := map[string]bool{}
	for _, c := range m.ToolCalls() {
		if c.ID == "" {
			return "the model returned a tool call with no id"
		}
		if _, dup := s.Call(c.ID); dup || seen[c.ID] {
			return "the model reused a tool call id"
		}
		seen[c.ID] = true
	}
	return ""
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
	return r.retry.Wait(me, attempt, r.now())
}

// Wait says whether to try a model call again after it failed with me on
// the given attempt (0 for the first), and how long to wait first, as of
// now.
func (p RetryPolicy) Wait(me *ModelError, attempt int, now time.Time) (time.Duration, bool) {
	if !me.Transient() || attempt >= p.Retries {
		return 0, false
	}
	backoff := p.Backoff
	for i := 0; i < attempt && backoff < time.Hour; i++ {
		backoff *= 2
	}
	until := time.Duration(0)
	if !me.RetryAfter.IsZero() {
		until = me.RetryAfter.Sub(now)
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
		for _, c := range res.held {
			out.Held = append(out.Held, HeldCall{ToolCall: c, InputSHA256: InputDigest(c.Input)})
		}
		out.LabelSources = r.s.LabelSources()
		return out, nil
	case res.limit != "":
		out := r.outcome(ReasonLimit)
		out.Limit = res.limit
		return out, nil
	}
	return nil, nil
}

type settled struct {
	canceled bool
	held     []ToolCall
	// limit names the limit that refused a call, if one did.
	limit string
}

// stoppedWhileRunning is the result of a call that was started and never
// seen to finish, whether the process died or the run was canceled: either
// way the call may or may not have taken effect.
const stoppedWhileRunning = "The run stopped while this call was running, so it may or may not have taken effect. It was not run again."

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
			if err := r.finish(id, ToolResult{Content: msg, IsError: true}, nil, ""); err != nil {
				return res, err
			}
			continue
		case info.Started:
			msg := stoppedWhileRunning
			if err := r.finish(id, ToolResult{Content: msg, IsError: true}, nil, ""); err != nil {
				return res, err
			}
			continue
		}
		if ctx.Err() != nil {
			res.canceled = true
			return res, nil
		}
		// An allow a policy gave in an earlier run, to a call that never
		// started, is stale: the session may have learned more since. It
		// is decided again. A person's allow stands.
		stale := info.Decision != nil && info.Decision.Verdict == Allow && !info.Resolved
		if info.Decision == nil || stale {
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
				if err := r.finish(id, ToolResult{Content: msg, IsError: true}, nil, ""); err != nil {
					return res, err
				}
				res.limit = LimitToolCalls
				continue
			}
			// A tool that calls a model (a quarantined reader) spends
			// tokens, so the token limit is checked between calls too.
			if max := r.a.Limits.MaxTokens; max > 0 && r.usage.Total() >= max {
				msg := "This call was not run: the limit on tokens was reached."
				if err := r.finish(id, ToolResult{Content: msg, IsError: true}, nil, ""); err != nil {
					return res, err
				}
				res.limit = LimitTokens
				continue
			}
			canceled, err := r.execute(ctx, info.Call, info.Decision.Deferred)
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
			if err := r.finish(id, ToolResult{Content: msg, IsError: true}, nil, ""); err != nil {
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
	case Allow:
		// Whatever the policy says, an allow may not give the session a
		// label it may not hold, now or with the result.
		if len(Forbidden(append(slices.Clone(d.Labels), d.Deferred...), r.a.Forbid)) > 0 {
			return Decision{Verdict: Deny, Reason: ForbiddenDecision}
		}
		return d
	case Deny, Hold:
		return d
	}
	return Decision{Verdict: Deny, Reason: fmt.Sprintf("The policy gave no usable verdict (%q).", d.Verdict)}
}

func (r *runner) recordDecision(call ToolCall, d Decision) error {
	e := &DecisionEntry{CallID: call.ID, Tool: call.Name, Verdict: d.Verdict, Reason: d.Reason, Labels: d.Labels, Deferred: d.Deferred, Consent: d.Consent}
	if d.Verdict != Allow {
		e.Consent = nil
		// Deferred labels go with an allow only.
		e.Deferred = nil
	}
	if d.Verdict == Hold {
		// A hold adds nothing: its labels are asked for again when a
		// person allows the call.
		e.Labels = nil
		e.InputSHA256 = InputDigest(call.Input)
	}
	return r.record(Entry{Type: EntryDecision, Decision: e})
}

func (r *runner) finish(id string, res ToolResult, labels []string, log string) error {
	return r.finishEntry(id, ToolResultEntry{Result: res, Labels: labels, Log: log})
}

func (r *runner) finishEntry(id string, e ToolResultEntry) error {
	e.Result.CallID = id
	return r.record(Entry{Type: EntryToolResult, ToolResult: &e})
}

// execute runs an allowed call and records its result. It reports
// whether the context ended meanwhile.
func (r *runner) execute(ctx context.Context, call ToolCall, deferred []string) (canceled bool, err error) {
	tool, ok := r.tb.tools[call.Name]
	if !ok {
		// Allowed in an earlier run, whose agent had the tool; this one
		// does not. Nothing runs.
		msg := fmt.Sprintf("This call was not run: there is no tool called %s.", call.Name)
		return false, r.finish(call.ID, ToolResult{Content: msg, IsError: true}, nil, "")
	}
	if err := r.record(Entry{Type: EntryCallStarted, CallStarted: &CallStartedEntry{CallID: call.ID}}); err != nil {
		return false, err
	}
	r.toolCalls++
	callCtx, cancel := ctx, context.CancelFunc(func() {})
	if r.a.ToolTimeout > 0 {
		callCtx, cancel = context.WithTimeout(ctx, r.a.ToolTimeout)
	}
	var out Result
	var callErr error
	if ct, ok := tool.(CallTool); ok {
		out, callErr = ct.CallWith(callCtx, Call{ID: call.ID, Input: call.Input, Labels: r.s.Labels(), Deferred: slices.Clone(deferred)})
	} else {
		out, callErr = tool.Call(callCtx, call.Input)
	}
	timedOut := callCtx.Err() == context.DeadlineExceeded && ctx.Err() == nil
	cancel()

	res := ToolResult{Content: out.Content, IsError: out.IsError}
	if callErr != nil {
		res.IsError = true
		switch {
		case ctx.Err() != nil:
			// The tool may have done its work before it noticed.
			res.Content = stoppedWhileRunning
		case timedOut:
			res.Content = fmt.Sprintf("The tool timed out after %s.", r.a.ToolTimeout)
		default:
			res.Content = fmt.Sprintf("The tool failed: %v", callErr)
		}
	}
	labels, log, report := out.Labels, out.Log, out.Quarantine
	// What the tool's model calls cost is spent whatever became of the
	// result.
	for _, u := range out.Usage {
		r.usage.add(u.Usage)
	}
	shown := out.Display
	if bad := Forbidden(out.Labels, r.a.Forbid); len(bad) > 0 {
		shown = nil
		// The model never sees the result and the session never takes its
		// labels. Nothing of what the tool said is kept, not even in the
		// log: it is what the session may not hold.
		res = ToolResult{Content: RefusedResult, IsError: true}
		labels, report = nil, nil
		log = fmt.Sprintf("agentloop refused this result: it carried %v, which this session may not hold.", bad)
		r.refused = append(r.refused, Refusal{CallID: call.ID, Tool: call.Name, Labels: bad})
	}
	var display *Display
	if shown != nil && callErr == nil {
		display, res.Content, log = r.display(call, *shown, out.Labels, deferred, res.Content, log)
	}
	e := ToolResultEntry{Result: res, Labels: labels, Log: log, Usage: out.Usage, Quarantine: report, Display: display}
	if err := r.finishEntry(call.ID, e); err != nil {
		return false, err
	}
	return ctx.Err() != nil, nil
}

// display is what is kept of a result's display, the content the model
// is given with it, and the operator's log. The display is labelled with
// the result's labels, the labels deferred to it and the session's, as
// anything the call made would be. One that would carry a label the
// session may not hold is dropped, and only the log says so. The content
// gains a note naming the display's ref, and nothing of the display.
func (r *runner) display(call ToolCall, d Display, result, deferred []string, content, log string) (*Display, string, string) {
	d.Labels = unionLabels(d.Labels, result, deferred, r.s.Labels())
	d.Sources = append([]Source{}, d.Sources...)
	d.Ref = ""
	if bad := Forbidden(d.Labels, r.a.Forbid); len(bad) > 0 {
		why := fmt.Sprintf("agentloop dropped this result's display: it carried %v, which this session may not hold.", bad)
		if log != "" {
			why = log + "\n" + why
		}
		return nil, content, why
	}
	if r.a.DisplayRefPrefix != "" {
		d.Ref = r.a.DisplayRefPrefix + DisplayID(call.ID)
	}
	return &d, withDisplayNote(content, d.Ref), log
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
