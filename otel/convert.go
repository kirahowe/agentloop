package otel

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sort"
	"strconv"
	"time"

	"github.com/kirahowe/agentloop"
)

// Options say what the log does not.
type Options struct {
	// ConversationID is gen_ai.conversation.id. The log carries none, so
	// the caller names the session, for example by its file name.
	ConversationID string
	// AgentName is gen_ai.agent.name and is part of the run span's name.
	AgentName string
	// Provider is gen_ai.provider.name, such as "anthropic" or "openai".
	// When empty, the provider recorded in the assistant messages is used,
	// and the attribute is left out if there are none.
	Provider string
	// Model is gen_ai.request.model. The log records only the model that
	// served a turn, so when this is empty that model stands in for it.
	Model string
	// ServiceName is the resource's service.name. Default "agentloop".
	ServiceName string
	// TraceSeed makes trace and span ids. Default: ConversationID.
	TraceSeed string
	// ParentTraceID and ParentSpanID (hex, 32 and 16 characters) hang
	// every run under a span made elsewhere, such as an eval's
	// invoke_workflow span. All runs then share that trace id. Both are
	// needed; if either is empty they are ignored.
	ParentTraceID string
	ParentSpanID  string
	// IncludeContent adds the opt-in content attributes. See the package
	// documentation. Off by default.
	IncludeContent bool
}

const (
	scopeName = "github.com/kirahowe/agentloop/otel"
	schemaURL = "https://opentelemetry.io/schemas/gen-ai-dev/1.42.0-dev"
)

// Marshal is Traces as JSON.
func Marshal(entries []agentloop.Entry, opts Options) ([]byte, error) {
	return json.Marshal(Traces(entries, opts))
}

// Traces converts a session log to OTLP traces, one trace per run.
func Traces(entries []agentloop.Entry, opts Options) TracesData {
	c := &converter{opts: opts, calls: map[string]*call{}, labels: map[string]bool{}}
	if c.opts.TraceSeed == "" {
		c.opts.TraceSeed = opts.ConversationID
	}
	for i := range entries {
		c.entry(i, entries)
	}
	if c.cur != nil {
		c.closeRun(c.cur, c.last, nil)
	}
	for _, r := range c.runs {
		c.runSpan(r)
	}
	c.callSpans()

	sort.SliceStable(c.spans, func(i, j int) bool { return c.spans[i].order < c.spans[j].order })
	spans := make([]Span, len(c.spans))
	for i, s := range c.spans {
		spans[i] = s.Span
	}
	service := opts.ServiceName
	if service == "" {
		service = "agentloop"
	}
	return TracesData{ResourceSpans: []ResourceSpans{{
		Resource: Resource{Attributes: []KeyValue{str("service.name", service)}},
		ScopeSpans: []ScopeSpans{{
			Scope:     Scope{Name: scopeName},
			Spans:     spans,
			SchemaURL: schemaURL,
		}},
	}}}
}

type orderedSpan struct {
	Span
	order int
}

// A run is one run_started and what follows it until run_ended or a crash.
type run struct {
	index    int
	traceID  string
	spanID   string
	startSeq int
	start    time.Time
	end      time.Time
	out      *agentloop.Outcome // nil for a run that crashed
	labels   []string           // the session's labels when the run closed
	usage    agentloop.Usage    // summed from its messages
	provider string
	events   []Event
}

// A call is everything the log says about one tool call.
type call struct {
	id, name  string
	input     json.RawMessage
	touched   bool
	firstSeq  int
	first     time.Time
	last      time.Time
	started   bool
	verdict   agentloop.Verdict
	events    []Event
	result    *agentloop.ToolResultEntry
	resultSeq int
	owner     *run
}

type converter struct {
	opts   Options
	spans  []orderedSpan
	runs   []*run
	cur    *run // the open run
	prev   *run // the latest run, open or closed
	last   time.Time
	calls  map[string]*call
	order  []*call
	labels map[string]bool

	model    string
	provider string
	history  []historyMessage
}

// anchor is the run that entries belong to now: the open one, or when
// none is open the latest.
func (c *converter) anchor() *run {
	if c.cur != nil {
		return c.cur
	}
	return c.prev
}

func (c *converter) entry(i int, entries []agentloop.Entry) {
	e := entries[i]
	c.last = e.Time
	switch e.Type {
	case agentloop.EntryRunStarted:
		if c.cur != nil {
			c.closeRun(c.cur, prevTime(i, entries), nil)
		}
		r := &run{index: len(c.runs), startSeq: e.Seq, start: e.Time}
		r.traceID = c.traceID(r.index)
		r.spanID = c.spanID(r.traceID, "run", e.Seq)
		c.runs = append(c.runs, r)
		c.cur, c.prev = r, r
	case agentloop.EntryRunEnded:
		if c.cur != nil && e.RunEnded != nil {
			c.closeRun(c.cur, e.Time, e.RunEnded)
		}
	case agentloop.EntryLabels:
		if e.Labels == nil {
			return
		}
		for _, l := range e.Labels.Labels {
			c.labels[l] = true
		}
		if r := c.anchor(); r != nil {
			attrs := []KeyValue{strArray("agentloop.labels", e.Labels.Labels)}
			if e.Labels.Source != "" {
				attrs = append(attrs, str("agentloop.labels.source", e.Labels.Source))
			}
			r.events = append(r.events, Event{nanos(e.Time), "agentloop.labels", attrs})
		}
	case agentloop.EntryPins:
		if e.Pins != nil {
			if r := c.anchor(); r != nil {
				names := make([]string, 0, len(e.Pins.Pins))
				for n := range e.Pins.Pins {
					names = append(names, n)
				}
				sort.Strings(names)
				r.events = append(r.events, Event{nanos(e.Time), "agentloop.pins", []KeyValue{strArray("agentloop.pins.tools", names)}})
			}
		}
	case agentloop.EntryMessage:
		if e.Message != nil {
			c.message(i, entries)
		}
	case agentloop.EntryDecision:
		if e.Decision != nil {
			c.decision(e)
		}
	case agentloop.EntryCallStarted:
		if e.CallStarted != nil {
			k := c.call(e.CallStarted.CallID, "", e)
			k.started = true
			k.events = append(k.events, Event{TimeUnixNano: nanos(e.Time), Name: "agentloop.call.started"})
		}
	case agentloop.EntryToolResult:
		if e.ToolResult != nil {
			k := c.call(e.ToolResult.Result.CallID, "", e)
			k.result, k.resultSeq = e.ToolResult, e.Seq
			if r := c.anchor(); r != nil {
				for _, u := range e.ToolResult.Usage {
					r.usage.InputTokens += u.Usage.InputTokens
					r.usage.OutputTokens += u.Usage.OutputTokens
					r.usage.CacheReadTokens += u.Usage.CacheReadTokens
					r.usage.CacheWriteTokens += u.Usage.CacheWriteTokens
				}
			}
			for _, l := range e.ToolResult.Labels {
				c.labels[l] = true
			}
			c.addToolResult(e.ToolResult.Result)
		}
	case agentloop.EntryModelError:
		if e.ModelError != nil {
			c.modelError(i, entries)
		}
	}
}

func prevTime(i int, entries []agentloop.Entry) time.Time {
	if i == 0 {
		return entries[i].Time
	}
	return entries[i-1].Time
}

// call finds or makes the state of a call and notes that e touched it.
func (c *converter) call(id, name string, e agentloop.Entry) *call {
	k := c.calls[id]
	if k == nil {
		k = &call{id: id}
		c.calls[id] = k
		c.order = append(c.order, k)
	}
	if k.name == "" {
		k.name = name
	}
	if !k.touched {
		k.touched = true
		k.firstSeq = e.Seq
		k.first = e.Time
	}
	k.last = e.Time
	k.owner = c.anchor()
	return k
}

func (c *converter) decision(e agentloop.Entry) {
	d := e.Decision
	k := c.call(d.CallID, d.Tool, e)
	k.verdict = d.Verdict
	if d.Verdict == agentloop.Allow {
		for _, l := range d.Labels {
			c.labels[l] = true
		}
	}
	attrs := []KeyValue{str("agentloop.policy.verdict", string(d.Verdict))}
	if len(d.Labels) > 0 {
		attrs = append(attrs, strArray("agentloop.labels", d.Labels))
	}
	if c.opts.IncludeContent {
		if d.Reason != "" {
			attrs = append(attrs, str("agentloop.policy.reason", d.Reason))
		}
	}
	k.events = append(k.events, Event{nanos(e.Time), "agentloop.policy.decision", attrs})
}

func (c *converter) message(i int, entries []agentloop.Entry) {
	e := entries[i]
	m := e.Message
	if m.Message.Role != agentloop.Assistant {
		c.history = append(c.history, historyMessage{Role: "user", Parts: convertParts(m.Message)})
		return
	}
	if m.Message.Native != nil && m.Message.Native.Provider != "" {
		c.provider = m.Message.Native.Provider
	}
	if m.Model != "" {
		c.model = m.Model
	}
	for _, tc := range m.Message.ToolCalls() {
		k := c.calls[tc.ID]
		if k == nil {
			k = &call{id: tc.ID}
			c.calls[tc.ID] = k
			c.order = append(c.order, k)
		}
		k.name, k.input = tc.Name, tc.Input
	}
	r := c.anchor()
	if r != nil {
		r.usage.InputTokens += m.Usage.InputTokens
		r.usage.OutputTokens += m.Usage.OutputTokens
		r.usage.CacheReadTokens += m.Usage.CacheReadTokens
		r.usage.CacheWriteTokens += m.Usage.CacheWriteTokens
		if c.provider != "" {
			r.provider = c.provider
		}
	}

	reqModel := c.requestModel(m.Model)
	attrs := c.inferenceAttrs(reqModel)
	if m.Model != "" {
		attrs = append(attrs, str("gen_ai.response.model", m.Model))
	}
	if m.Stop != "" {
		attrs = append(attrs,
			strArray("gen_ai.response.finish_reasons", []string{finishReason(m.Stop)}),
			str("agentloop.stop", string(m.Stop)))
	}
	attrs = append(attrs, usageAttrs(m.Usage)...)
	out := historyMessage{Role: "assistant", Parts: convertParts(m.Message)}
	if c.opts.IncludeContent {
		if v, ok := jsonString(c.history); ok {
			attrs = append(attrs, str("gen_ai.input.messages", v))
		}
		if v, ok := jsonString([]historyMessage{out}); ok {
			attrs = append(attrs, str("gen_ai.output.messages", v))
		}
		if m.Refusal != "" {
			attrs = append(attrs, str("agentloop.refusal", m.Refusal))
		}
	}
	c.history = append(c.history, out)

	c.addSpan(r, "chat", e.Seq, Span{
		Name:       name("chat", reqModel),
		Kind:       KindClient,
		Attributes: attrs,
	}, prevTime(i, entries), e.Time)
}

func (c *converter) modelError(i int, entries []agentloop.Entry) {
	e := entries[i]
	me := e.ModelError
	kind := string(me.Error.Kind)
	if kind == "" {
		kind = "_OTHER"
	}
	reqModel := c.requestModel("")
	attrs := append(c.inferenceAttrs(reqModel),
		str("error.type", kind),
		boolean("agentloop.retry", me.Retry))
	if me.WaitMillis > 0 {
		attrs = append(attrs, integer("agentloop.retry.wait_ms", me.WaitMillis))
	}
	exc := []KeyValue{str("exception.type", kind)}
	if c.opts.IncludeContent && me.Error.Message != "" {
		exc = append(exc, str("exception.message", me.Error.Message))
	}
	c.addSpan(c.anchor(), "chat", e.Seq, Span{
		Name:       name("chat", reqModel),
		Kind:       KindClient,
		Attributes: attrs,
		Events:     []Event{{nanos(e.Time), "gen_ai.client.operation.exception", exc}},
		Status:     &Status{Code: StatusError, Message: kind},
	}, prevTime(i, entries), e.Time)
}

func (c *converter) requestModel(served string) string {
	switch {
	case c.opts.Model != "":
		return c.opts.Model
	case served != "":
		return served
	}
	return c.model
}

func (c *converter) providerName() string {
	if c.opts.Provider != "" {
		return c.opts.Provider
	}
	return c.provider
}

func (c *converter) inferenceAttrs(model string) []KeyValue {
	attrs := []KeyValue{str("gen_ai.operation.name", "chat")}
	if p := c.providerName(); p != "" {
		attrs = append(attrs, str("gen_ai.provider.name", p))
	}
	if model != "" {
		attrs = append(attrs, str("gen_ai.request.model", model))
	}
	if c.opts.ConversationID != "" {
		attrs = append(attrs, str("gen_ai.conversation.id", c.opts.ConversationID))
	}
	return attrs
}

// usageAttrs: gen_ai.usage.input_tokens includes cached tokens, which
// agentloop's InputTokens does not.
func usageAttrs(u agentloop.Usage) []KeyValue {
	attrs := []KeyValue{
		integer("gen_ai.usage.input_tokens", u.InputTokens+u.CacheReadTokens+u.CacheWriteTokens),
		integer("gen_ai.usage.output_tokens", u.OutputTokens),
	}
	if u.CacheReadTokens != 0 {
		attrs = append(attrs, integer("gen_ai.usage.cache_read.input_tokens", u.CacheReadTokens))
	}
	if u.CacheWriteTokens != 0 {
		attrs = append(attrs, integer("gen_ai.usage.cache_write.input_tokens", u.CacheWriteTokens))
	}
	return attrs
}

func finishReason(s agentloop.Stop) string {
	switch s {
	case agentloop.StopEndTurn:
		return "stop"
	case agentloop.StopToolUse:
		return "tool_call"
	case agentloop.StopMaxTokens:
		return "length"
	case agentloop.StopRefusal:
		return "content_filter"
	}
	return string(s)
}

func name(op, subject string) string {
	if subject == "" {
		return op
	}
	return op + " " + subject
}

// addSpan puts a child span in run r's trace, or in a trace of its own
// when no run has started yet. The end is never before the start.
func (c *converter) addSpan(r *run, kind string, seq int, s Span, start, end time.Time) string {
	if r != nil {
		s.TraceID, s.ParentSpanID = r.traceID, r.spanID
	} else {
		s.TraceID = c.traceID(-1)
	}
	s.SpanID = c.spanID(s.TraceID, kind, seq)
	if end.Before(start) {
		start = end
	}
	s.StartTimeUnixNano, s.EndTimeUnixNano = nanos(start), nanos(end)
	c.spans = append(c.spans, orderedSpan{s, seq})
	return s.SpanID
}

func maxTime(a, b time.Time) time.Time {
	if b.Before(a) {
		return a
	}
	return b
}

func (c *converter) closeRun(r *run, end time.Time, out *agentloop.Outcome) {
	r.end, r.out = end, out
	r.labels = sortedKeys(c.labels)
	c.cur = nil
}

func (c *converter) runSpan(r *run) {
	attrs := []KeyValue{str("gen_ai.operation.name", "invoke_agent")}
	provider := c.opts.Provider
	if provider == "" {
		provider = r.provider
	}
	if provider != "" {
		attrs = append(attrs, str("gen_ai.provider.name", provider))
	}
	if c.opts.AgentName != "" {
		attrs = append(attrs, str("gen_ai.agent.name", c.opts.AgentName))
	}
	if c.opts.Model != "" {
		attrs = append(attrs, str("gen_ai.request.model", c.opts.Model))
	}
	if c.opts.ConversationID != "" {
		attrs = append(attrs, str("gen_ai.conversation.id", c.opts.ConversationID))
	}
	usage := r.usage
	var status *Status
	if out := r.out; out != nil {
		usage = out.Usage
		attrs = append(attrs,
			str("agentloop.run.reason", string(out.Reason)),
			integer("agentloop.run.turns", int64(out.Turns)))
		if out.Limit != "" {
			attrs = append(attrs, str("agentloop.run.limit", out.Limit))
		}
		if out.Reason == agentloop.ReasonModelError {
			kind := "_OTHER"
			if out.Error != nil && out.Error.Kind != "" {
				kind = string(out.Error.Kind)
			}
			attrs = append(attrs, str("error.type", kind))
			status = &Status{Code: StatusError, Message: kind}
		}
		if c.opts.IncludeContent && out.Text != "" {
			msg := historyMessage{Role: "assistant", Parts: []historyPart{{Type: "text", Content: out.Text}}}
			if v, ok := jsonString([]historyMessage{msg}); ok {
				attrs = append(attrs, str("gen_ai.output.messages", v))
			}
		}
	} else {
		attrs = append(attrs, str("agentloop.run.reason", "crashed"), str("error.type", "crashed"))
		status = &Status{Code: StatusError, Message: "crashed"}
	}
	attrs = append(attrs, usageAttrs(usage)...)
	if len(r.labels) > 0 {
		attrs = append(attrs, strArray("agentloop.labels", r.labels))
	}
	end := r.end
	if end.Before(r.start) {
		end = r.start
	}
	c.spans = append(c.spans, orderedSpan{Span{
		TraceID:           r.traceID,
		SpanID:            r.spanID,
		ParentSpanID:      c.parentSpan(),
		Name:              name("invoke_agent", c.opts.AgentName),
		Kind:              KindInternal,
		StartTimeUnixNano: nanos(r.start),
		EndTimeUnixNano:   nanos(end),
		Attributes:        attrs,
		Events:            r.events,
		Status:            status,
	}, r.startSeq})
}

// callSpans turns every call the log shows any sign of into a span in the
// trace of the run that last touched it.
func (c *converter) callSpans() {
	for _, k := range c.order {
		if !k.touched {
			continue
		}
		attrs := []KeyValue{
			str("gen_ai.operation.name", "execute_tool"),
			str("gen_ai.tool.name", k.name),
			str("gen_ai.tool.call.id", k.id),
			str("gen_ai.tool.type", "function"),
		}
		if c.opts.AgentName != "" {
			attrs = append(attrs, str("gen_ai.agent.name", c.opts.AgentName))
		}
		if c.opts.ConversationID != "" {
			attrs = append(attrs, str("gen_ai.conversation.id", c.opts.ConversationID))
		}

		state, errType := "decided", ""
		switch {
		case k.result != nil:
			state = "finished"
			switch {
			case k.verdict == agentloop.Deny:
				errType = "policy_denied"
			case !k.started && k.result.Result.IsError:
				errType = "not_executed"
			case k.result.Result.IsError:
				errType = "tool_error"
			}
		case k.started:
			state, errType = "unfinished", "unfinished"
		}
		attrs = append(attrs, str("agentloop.call.state", state))
		if k.verdict != "" {
			attrs = append(attrs, str("agentloop.policy.verdict", string(k.verdict)))
		}
		var status *Status
		if errType != "" {
			attrs = append(attrs, str("error.type", errType))
			status = &Status{Code: StatusError, Message: errType}
		}
		if k.result != nil && len(k.result.Labels) > 0 {
			attrs = append(attrs, strArray("agentloop.labels", k.result.Labels))
		}
		if c.opts.IncludeContent {
			if len(k.input) > 0 {
				attrs = append(attrs, str("gen_ai.tool.call.arguments", string(k.input)))
			}
			if k.result != nil {
				if k.result.Result.IsError {
					attrs = append(attrs, str("agentloop.tool.error", k.result.Result.Content))
				} else {
					attrs = append(attrs, str("gen_ai.tool.call.result", k.result.Result.Content))
				}
				if k.result.Log != "" {
					attrs = append(attrs, str("agentloop.tool.log", k.result.Log))
				}
			}
		}

		// A call decided in one run and finished in another belongs to the
		// later run, and cannot start before it.
		start := k.first
		if k.owner != nil && start.Before(k.owner.start) {
			start = k.owner.start
		}
		callSpan := c.addSpan(k.owner, "call", k.firstSeq, Span{
			Name:       name("execute_tool", k.name),
			Kind:       KindInternal,
			Attributes: attrs,
			Events:     k.events,
			Status:     status,
		}, start, k.last)
		// The model calls a tool made are chat spans under the tool span. The log has no times for them, so each spans the
		// tool call.
		if k.result != nil {
			for j, u := range k.result.Usage {
				reqModel := u.Model
				if reqModel == "" {
					reqModel = c.requestModel("")
				}
				ia := c.inferenceAttrs(reqModel)
				if u.Model != "" {
					ia = append(ia, str("gen_ai.response.model", u.Model))
				}
				ia = append(ia, usageAttrs(u.Usage)...)
				rs := Span{Name: name("chat", reqModel), Kind: KindClient, Attributes: ia}
				if k.owner != nil {
					rs.TraceID = k.owner.traceID
				} else {
					rs.TraceID = c.traceID(-1)
				}
				rs.ParentSpanID = callSpan
				rs.SpanID = c.spanID(rs.TraceID, "toolmodel"+strconv.Itoa(j), k.resultSeq)
				rs.StartTimeUnixNano, rs.EndTimeUnixNano = nanos(start), nanos(maxTime(start, k.last))
				c.spans = append(c.spans, orderedSpan{rs, k.firstSeq})
			}
		}
	}
}

func (c *converter) traceID(index int) string {
	if c.opts.ParentTraceID != "" && c.opts.ParentSpanID != "" {
		return c.opts.ParentTraceID
	}
	return hexID(16, c.opts.TraceSeed, "trace", strconv.Itoa(index))
}

func (c *converter) parentSpan() string {
	if c.opts.ParentTraceID != "" {
		return c.opts.ParentSpanID
	}
	return ""
}

func (c *converter) spanID(trace, kind string, seq int) string {
	return hexID(8, trace, kind, strconv.Itoa(seq))
}

// hexID hashes its parts into n bytes of hex that are not all zero, which
// OTLP does not allow for an id. Each part is quoted so that no two
// different lists of parts hash the same input.
func hexID(n int, parts ...string) string {
	h := sha256.New()
	for _, p := range parts {
		h.Write([]byte(strconv.Quote(p)))
		h.Write([]byte{0})
	}
	b := h.Sum(nil)[:n]
	zero := true
	for _, x := range b {
		if x != 0 {
			zero = false
			break
		}
	}
	if zero {
		b[n-1] = 1
	}
	return hex.EncodeToString(b)
}

func nanos(t time.Time) string {
	if t.IsZero() {
		return "0"
	}
	n := t.UnixNano()
	if n < 0 {
		n = 0
	}
	return strconv.FormatInt(n, 10)
}

func sortedKeys(m map[string]bool) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func jsonString(v any) (string, bool) {
	b, err := json.Marshal(v)
	if err != nil {
		return "", false
	}
	return string(b), true
}

// The history types are the convention's input and output message shape
// (model/gen-ai/gen-ai-input-messages.json and gen-ai-output-messages.json).
type historyMessage struct {
	Role  string        `json:"role"`
	Parts []historyPart `json:"parts"`
}

type historyPart struct {
	Type      string          `json:"type"`
	Content   string          `json:"content,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
	Response  *string         `json:"response,omitempty"`
}

func convertParts(m agentloop.Message) []historyPart {
	parts := []historyPart{}
	for _, p := range m.Parts {
		switch p.Type {
		case agentloop.TextPart:
			parts = append(parts, historyPart{Type: "text", Content: p.Text})
		case agentloop.ReasoningPart:
			parts = append(parts, historyPart{Type: "reasoning", Content: p.Text})
		case agentloop.ToolCallPart:
			if p.ToolCall != nil {
				hp := historyPart{Type: "tool_call", ID: p.ToolCall.ID, Name: p.ToolCall.Name}
				if json.Valid(p.ToolCall.Input) {
					hp.Arguments = p.ToolCall.Input
				}
				parts = append(parts, hp)
			}
		}
	}
	return parts
}

// addToolResult adds a result to the history. Consecutive results share
// one message, as the session gathers them.
func (c *converter) addToolResult(r agentloop.ToolResult) {
	resp := r.Content
	part := historyPart{Type: "tool_call_response", ID: r.CallID, Response: &resp}
	if n := len(c.history); n > 0 && c.history[n-1].Role == "tool" {
		c.history[n-1].Parts = append(c.history[n-1].Parts, part)
		return
	}
	c.history = append(c.history, historyMessage{Role: "tool", Parts: []historyPart{part}})
}
