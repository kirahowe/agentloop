// Package acpagent serves agentloop sessions over the Agent Client
// Protocol. Everything but the shell is pure: the translation of a run's
// events into session/update notifications, the stop reason of a turn,
// the prompt text, and the turn machine take values and return values.
package acpagent

import (
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"time"

	acp "github.com/caelis-labs/acp-go-sdk"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/trifecta"
)

// Update is one session/update the agent sends.
type Update = acp.SessionUpdate

// Options are what the translation needs to know beyond the events.
type Options struct {
	// Kinds is the ACP kind of each tool, by name (the request's
	// tool_kinds). A tool not listed is "other".
	Kinds map[string]acp.ToolKind
	// ContextWindow is the model's context window in tokens, as its
	// provider describes it. Zero means unknown, and then a usage_update
	// is sent only for a priced model, with a size of 0.
	ContextWindow int64
	// Prices is what the model costs, when its provider's adapter knows.
	// With it, each usage_update carries the session's cumulative cost in
	// US dollars; without it, none does.
	Prices *agentloop.Prices
	// PriceOf is the price of any model of the provider, nil for one it
	// does not know. It prices the model calls a tool made for the run (a
	// quarantined reader's), each at its own model's price. A nil PriceOf
	// prices none, so a session with such calls reports no cost.
	PriceOf func(model string) *agentloop.Prices
	// Notices says the client advertised session notices, which carry
	// retry status. Without them nothing is sent for a model error.
	Notices bool
}

// A Translator turns a session's events into updates. It is a value:
// Step returns the next one and never changes the one it is called on.
//
// A message's id is its entry's sequence number: "a<seq>" for an assistant
// message and "u<seq>" for a user message, so replay gives the ids a live
// session gave. Deltas stream before their entry exists, so they carry the
// id of the entry that comes next. If that entry is a model error or the
// end of the run instead of a message, the attempt's text keeps an id no
// message has, and a retry streams under a new one rather than appending
// to what the failed attempt said.
type Translator struct {
	opts Options
	// seq is the sequence number of the last entry seen.
	seq int
	// text and thought are what deltas streamed since that entry.
	text, thought string
	// attempt counts failed model calls since the last message.
	attempt int
	// cost is what the session's model calls have cost so far, in this
	// run and before it: each assistant message at Options.Prices, and
	// each call a tool made at its own model's price. unpriced says some
	// call could not be priced, and then no cost is sent, never a partial
	// one.
	cost     float64
	unpriced bool
	// used is the context the last assistant message took, which an
	// update after a tool's model calls repeats.
	used int64
	// labels are the session's labels as of the last entry, sorted. Every
	// tool call, tool call update and usage update carries them.
	labels []string
}

// NewTranslator makes the translator for a session's live events.
func NewTranslator(opts Options) Translator { return Translator{opts: opts} }

// WithHistory is the translator for a session that already has these
// entries, as a reopened one has, so the cost it reports includes
// earlier turns, priced as they were live.
func (t Translator) WithHistory(entries []agentloop.Entry) Translator {
	for _, e := range entries {
		t = t.spend(e)
	}
	return t
}

// spend adds what an entry's model calls cost.
func (t Translator) spend(e agentloop.Entry) Translator {
	switch {
	case e.Type == agentloop.EntryMessage && e.Message != nil && e.Message.Message.Role == agentloop.Assistant:
		u := e.Message.Usage
		t.used = u.InputTokens + u.CacheReadTokens + u.CacheWriteTokens + u.OutputTokens
		if p := t.opts.Prices; p != nil {
			t.cost += p.Cost(u)
		}
	case e.Type == agentloop.EntryToolResult && e.ToolResult != nil:
		for _, u := range e.ToolResult.Usage {
			var p *agentloop.Prices
			if t.opts.PriceOf != nil {
				p = t.opts.PriceOf(u.Model)
			}
			if p == nil {
				t.unpriced = true
				continue
			}
			t.cost += p.Cost(u.Usage)
		}
	}
	return t
}

// WithLabels is the translator for a session that already holds labels.
func (t Translator) WithLabels(labels []string) Translator {
	t.labels = trifecta.Union(t.labels, labels)
	return t
}

// Labels are the session's labels as the translator has seen them.
func (t Translator) Labels() []string { return slices.Clone(t.labels) }

// entryLabels are the labels an entry adds to the session, as the session
// itself counts them.
func entryLabels(e agentloop.Entry) []string {
	switch {
	case e.Type == agentloop.EntryLabels && e.Labels != nil:
		return e.Labels.Labels
	case e.Type == agentloop.EntryDecision && e.Decision != nil && e.Decision.Verdict == agentloop.Allow:
		return e.Decision.Labels
	case e.Type == agentloop.EntryToolResult && e.ToolResult != nil:
		return e.ToolResult.Labels
	}
	return nil
}

// labelsMeta is the _meta that shows the session's labels to a client.
func labelsMeta(labels []string) map[string]json.RawMessage {
	if labels == nil {
		labels = []string{}
	}
	b, _ := json.Marshal(labels)
	return map[string]json.RawMessage{"agentloop/labels": b}
}

// Step takes one event and returns the next translator and the updates to
// send for it.
func (t Translator) Step(ev agentloop.Event) (Translator, []Update) {
	switch {
	case ev.Delta != nil:
		return t.delta(*ev.Delta)
	case ev.Entry != nil:
		return t.entry(*ev.Entry)
	}
	return t, nil
}

func (t Translator) delta(d agentloop.Delta) (Translator, []Update) {
	// Text goes to the client through JSON, as it goes to the log, so it
	// is kept in the form both will have: invalid UTF-8 replaced.
	text := wireText(d.Text)
	if text == "" {
		return t, nil
	}
	id := assistantID(t.seq + 1)
	switch d.Type {
	case agentloop.TextPart:
		t.text += text
		return t, []Update{messageChunk(id, text)}
	case agentloop.ReasoningPart:
		t.thought += text
		return t, []Update{thoughtChunk(id, text)}
	}
	return t, nil
}

// entry translates one entry. When the entry changes the session's labels
// and none of its updates carries them, a session_info_update with
// _meta["agentloop/labels"] is added, so a client always knows the labels
// as they are.
func (t Translator) entry(e agentloop.Entry) (Translator, []Update) {
	before := t.labels
	t, ups := t.entryUpdates(e)
	if slices.Equal(before, t.labels) {
		return t, ups
	}
	for _, u := range ups {
		if u.ToolCall != nil || u.ToolCallUpdate != nil || u.UsageUpdate != nil {
			return t, ups
		}
	}
	return t, append(ups, LabelsUpdate(t.labels))
}

// LabelsUpdate is a session_info_update that only shows the session's
// labels, under _meta["agentloop/labels"].
func LabelsUpdate(labels []string) Update {
	return acp.SessionUpdate{SessionInfoUpdate: &acp.SessionSessionInfoUpdate{Meta: labelsMeta(labels)}}
}

func (t Translator) entryUpdates(e agentloop.Entry) (Translator, []Update) {
	streamedText, streamedThought := t.text, t.thought
	t.seq, t.text, t.thought = e.Seq, "", ""
	t = t.spend(e)
	if add := entryLabels(e); len(add) > 0 {
		t.labels = trifecta.Union(t.labels, add)
	}
	switch e.Type {
	case agentloop.EntryRunStarted:
		t.attempt = 0
	case agentloop.EntryMessage:
		t.attempt = 0
		m := e.Message
		if m == nil || m.Message.Role != agentloop.Assistant {
			// The client showed the prompt as it sent it.
			return t, nil
		}
		ups := assistantChunks(e.Seq, *m, streamedText, streamedThought)
		for _, c := range m.Message.ToolCalls() {
			ups = append(ups, toolCall(c, t.opts.Kinds, acp.ToolCallStatusPending, nil, t.labels))
		}
		if u, ok := t.usage(t.labels); ok {
			ups = append(ups, u)
		}
		return t, ups
	case agentloop.EntryCallStarted:
		if e.CallStarted == nil {
			return t, nil
		}
		u := acp.UpdateToolCall(acp.ToolCallId(e.CallStarted.CallID), acp.WithUpdateStatus(acp.ToolCallStatusInProgress))
		u.ToolCallUpdate.Meta = labelsMeta(t.labels)
		return t, []Update{u}
	case agentloop.EntryToolResult:
		if e.ToolResult == nil {
			return t, nil
		}
		ups := []Update{toolResult(*e.ToolResult, t.labels)}
		// A tool's own model calls change the cost now, not at the next
		// turn, which may never come.
		if len(e.ToolResult.Usage) > 0 {
			if u, ok := t.usage(t.labels); ok {
				ups = append(ups, u)
			}
		}
		return t, ups
	case agentloop.EntryModelError:
		if e.ModelError == nil {
			return t, nil
		}
		t.attempt++
		if !t.opts.Notices {
			return t, nil
		}
		return t, []Update{notice(e, t.attempt)}
	}
	return t, nil
}

// Replay is what a client is sent when it loads a session: each message
// as chunks, and each call as one tool_call with its final status. A held
// call is pending, and a call that started and never finished is
// in_progress. Notices are never replayed. If the options give a context
// window, or prices, the last assistant turn's usage follows, with the
// cost of every turn.
func Replay(opts Options, entries []agentloop.Entry) []Update {
	results := map[string]agentloop.ToolResultEntry{}
	started := map[string]bool{}
	// A call shows the session's labels as of its last entry (its result,
	// its start, or the message that made it), as it did live; the usage
	// shows them as they are now.
	var labels []string
	labelsAt := map[int][]string{}
	callAt := map[string]int{}
	for _, e := range entries {
		labels = trifecta.Union(labels, entryLabels(e))
		labelsAt[e.Seq] = labels
		switch {
		case e.Type == agentloop.EntryToolResult && e.ToolResult != nil:
			callAt[e.ToolResult.Result.CallID] = e.Seq
		case e.Type == agentloop.EntryCallStarted && e.CallStarted != nil:
			callAt[e.CallStarted.CallID] = e.Seq
		}
		switch {
		case e.Type == agentloop.EntryToolResult && e.ToolResult != nil:
			results[e.ToolResult.Result.CallID] = *e.ToolResult
		case e.Type == agentloop.EntryCallStarted && e.CallStarted != nil:
			started[e.CallStarted.CallID] = true
		}
	}
	var ups []Update
	var last *agentloop.MessageEntry
	for _, e := range entries {
		if e.Type != agentloop.EntryMessage || e.Message == nil {
			continue
		}
		m := e.Message
		if m.Message.Role != agentloop.Assistant {
			for _, p := range m.Message.Parts {
				if p.Type == agentloop.TextPart && p.Text != "" {
					ups = append(ups, acp.SessionUpdate{UserMessageChunk: &acp.SessionUpdateUserMessageChunk{
						Content: acp.TextBlock(p.Text), MessageId: acp.Ptr(userID(e.Seq)),
					}})
				}
			}
			continue
		}
		last = m
		ups = append(ups, assistantChunks(e.Seq, *m, "", "")...)
		for _, c := range m.Message.ToolCalls() {
			status := acp.ToolCallStatusPending
			if started[c.ID] {
				status = acp.ToolCallStatusInProgress
			}
			var res *agentloop.ToolResultEntry
			if r, ok := results[c.ID]; ok {
				res = &r
			}
			at, ok := callAt[c.ID]
			if !ok {
				at = e.Seq
			}
			ups = append(ups, toolCall(c, opts.Kinds, status, res, labelsAt[at]))
		}
	}
	if last != nil {
		if u, ok := NewTranslator(opts).WithHistory(entries).usage(labels); ok {
			ups = append(ups, u)
		}
	}
	if len(labels) > 0 {
		ups = append(ups, LabelsUpdate(labels))
	}
	return ups
}

func assistantID(seq int) acp.MessageId { return acp.MessageId("a" + strconv.Itoa(seq)) }
func userID(seq int) acp.MessageId      { return acp.MessageId("u" + strconv.Itoa(seq)) }

func messageChunk(id acp.MessageId, text string) Update {
	return acp.SessionUpdate{AgentMessageChunk: &acp.SessionUpdateAgentMessageChunk{
		Content: acp.TextBlock(text), MessageId: acp.Ptr(id),
	}}
}

func thoughtChunk(id acp.MessageId, text string) Update {
	return acp.SessionUpdate{AgentThoughtChunk: &acp.SessionUpdateAgentThoughtChunk{
		Content: acp.TextBlock(text), MessageId: acp.Ptr(id),
	}}
}

// assistantChunks sends what deltas did not: for each text or reasoning
// part, the part of it beyond what was streamed, so a model that streams
// nothing loses nothing and one that streams everything sends nothing
// twice. If what was streamed is not the start of the message's text, it
// cannot be taken back, and nothing more of that kind is sent.
//
// A refusal the provider explained outside the text is sent after it, as
// the client would otherwise show nothing.
func assistantChunks(seq int, m agentloop.MessageEntry, streamedText, streamedThought string) []Update {
	id := assistantID(seq)
	var ups []Update
	full := map[agentloop.PartType]string{}
	sent := map[agentloop.PartType]int{}
	streamed := map[agentloop.PartType]string{agentloop.TextPart: streamedText, agentloop.ReasoningPart: streamedThought}
	for _, p := range m.Message.Parts {
		if p.Type == agentloop.TextPart || p.Type == agentloop.ReasoningPart {
			full[p.Type] += p.Text
		}
	}
	diverged := map[agentloop.PartType]bool{}
	for pt, s := range streamed {
		diverged[pt] = !strings.HasPrefix(full[pt], s)
		sent[pt] = len(s)
	}
	at := map[agentloop.PartType]int{}
	for _, p := range m.Message.Parts {
		if (p.Type != agentloop.TextPart && p.Type != agentloop.ReasoningPart) || p.Text == "" {
			continue
		}
		start, end := at[p.Type], at[p.Type]+len(p.Text)
		at[p.Type] = end
		if diverged[p.Type] || end <= sent[p.Type] {
			continue
		}
		rest := full[p.Type][max(start, sent[p.Type]):end]
		if p.Type == agentloop.TextPart {
			ups = append(ups, messageChunk(id, rest))
		} else {
			ups = append(ups, thoughtChunk(id, rest))
		}
	}
	if extra := refusalText(m); extra != "" {
		ups = append(ups, messageChunk(id, extra))
	}
	return ups
}

// refusalText is the chunk a refusal adds after the message's text: the
// provider's explanation when the text does not already hold it.
func refusalText(m agentloop.MessageEntry) string {
	if m.Stop != agentloop.StopRefusal || m.Refusal == "" {
		return ""
	}
	var text strings.Builder
	for _, p := range m.Message.Parts {
		if p.Type == agentloop.TextPart {
			text.WriteString(p.Text)
		}
	}
	if strings.Contains(text.String(), m.Refusal) {
		return ""
	}
	if text.Len() > 0 {
		return "\n\n" + m.Refusal
	}
	return m.Refusal
}

// toolCall is a call as a client first sees it, with the session's labels
// under _meta["agentloop/labels"].
func toolCall(c agentloop.ToolCall, kinds map[string]acp.ToolKind, status acp.ToolCallStatus, res *agentloop.ToolResultEntry, labels []string) Update {
	tc := &acp.SessionUpdateToolCall{
		ToolCallId: acp.ToolCallId(c.ID),
		Title:      Title(c),
		Name:       acp.Ptr(c.Name),
		Kind:       Kind(c.Name, kinds),
		Status:     status,
		RawInput:   rawJSON(c.Input),
		Meta:       labelsMeta(labels),
	}
	if res != nil {
		tc.Status, tc.Content, tc.RawOutput = resultFields(*res)
		addQuarantineMeta(tc.Meta, *res)
	}
	return acp.SessionUpdate{ToolCall: tc}
}

func toolResult(r agentloop.ToolResultEntry, labels []string) Update {
	u := &acp.SessionToolCallUpdate{ToolCallId: acp.ToolCallId(r.Result.CallID), Meta: labelsMeta(labels)}
	var status acp.ToolCallStatus
	status, u.Content, u.RawOutput = resultFields(r)
	u.Status = &status
	addQuarantineMeta(u.Meta, r)
	return acp.SessionUpdate{ToolCallUpdate: u}
}

// addQuarantineMeta puts a quarantined reader's report in
// _meta["agentloop/quarantine"]: the inner tool, whether the answer's
// schema admits free text, and the labels the result left out, so a
// client can show that a reader answered.
func addQuarantineMeta(meta map[string]json.RawMessage, r agentloop.ToolResultEntry) {
	q := r.Quarantine
	if q == nil {
		return
	}
	dropped := q.Dropped
	if dropped == nil {
		dropped = []string{}
	}
	b, _ := json.Marshal(agentloop.QuarantineReport{Tool: q.Tool, FreeText: q.FreeText, Dropped: dropped})
	meta["agentloop/quarantine"] = b
}

// resultFields is how a result shows on a call: failed when it is an
// error, the content as text, then the display if there is one (see
// DisplayContent), and the raw result. The operator's log
// (a tool's stderr) is not shown.
func resultFields(r agentloop.ToolResultEntry) (acp.ToolCallStatus, []acp.ToolCallContent, any) {
	status := acp.ToolCallStatusCompleted
	if r.Result.IsError {
		status = acp.ToolCallStatusFailed
	}
	content := []acp.ToolCallContent{acp.ToolContent(acp.TextBlock(r.Result.Content))}
	if r.Display != nil {
		content = append(content, DisplayContent(*r.Display))
	}
	raw := map[string]any{"content": r.Result.Content, "is_error": r.Result.IsError}
	return status, content, raw
}

// DisplayMeta is the _meta key of a display's content block.
const DisplayMeta = "relay/display"

// DisplayContent is a result's display as a client is sent it: one more
// text block in the call's content, the title and body for a client
// that knows nothing of displays, and the whole display under
// _meta["relay/display"] for one that shows them apart. Labels and
// sources are always arrays.
//
// The display can hold an outsider's words, and a client that knows
// nothing of displays renders the block as markdown, so the block's text
// is inert (Inert): a remote image in it is never fetched, and no link
// in it is live. The display itself, under _meta, is unchanged.
func DisplayContent(d agentloop.Display) acp.ToolCallContent {
	if d.Labels == nil {
		d.Labels = []string{}
	}
	if d.Sources == nil {
		d.Sources = []agentloop.Source{}
	}
	b, _ := json.Marshal(d)
	block := acp.TextBlock(Inert(d.Title + "\n\n" + d.Body))
	block.Text.Meta = map[string]json.RawMessage{DisplayMeta: b}
	return acp.ToolContent(block)
}

// usage is the usage_update as of the last assistant message, with the
// session's cost so far when every call in it is priced. Used is that
// message's own count, which is what is in context. A priced model with
// no known window still reports its cost, with a size of 0: the protocol
// requires the size, and zero is the one value that claims nothing about
// the window.
func (t Translator) usage(labels []string) (Update, bool) {
	if t.opts.ContextWindow <= 0 && t.opts.Prices == nil {
		return Update{}, false
	}
	up := &acp.SessionUsageUpdate{Used: uint64(max(t.used, 0)), Size: uint64(max(t.opts.ContextWindow, 0)), Meta: labelsMeta(labels)}
	if t.opts.Prices != nil && !t.unpriced {
		up.Cost = &acp.Cost{Amount: t.cost, Currency: "USD"}
	}
	return acp.SessionUpdate{UsageUpdate: up}, true
}

// notice reports a failed model call: a warning with the wait when the
// loop will retry, an error when it will not. The wait and the time of the
// retry go in _meta["agentloop/retry"], so a client can count down.
func notice(e agentloop.Entry, attempt int) Update {
	me := e.ModelError
	desc := me.Error.Message
	n := &acp.SessionUpdateNotice{Severity: acp.NoticeSeverityError, Title: failedTitle(me.Error.Kind)}
	if desc != "" {
		n.Description = &desc
	}
	if me.Retry {
		wait := time.Duration(me.WaitMillis) * time.Millisecond
		secs := (me.WaitMillis + 999) / 1000
		n.Severity = acp.NoticeSeverityWarning
		n.Title = fmt.Sprintf("%s, retrying in %d s", problem(me.Error.Kind), secs)
		retry := map[string]any{
			"kind":     string(me.Error.Kind),
			"wait_ms":  me.WaitMillis,
			"retry_at": e.Time.Add(wait).UTC().Format(time.RFC3339Nano),
			"attempt":  attempt,
		}
		if me.Error.Status != 0 {
			retry["status"] = me.Error.Status
		}
		b, _ := json.Marshal(retry)
		n.Meta = map[string]json.RawMessage{"agentloop/retry": b}
	}
	return acp.SessionUpdate{Notice: n}
}

func problem(k agentloop.ErrorKind) string {
	switch k {
	case agentloop.RateLimited:
		return "Rate limited"
	case agentloop.Overloaded:
		return "The provider is overloaded"
	case agentloop.Unreachable:
		return "The provider is unreachable"
	}
	return "The model call failed"
}

func failedTitle(k agentloop.ErrorKind) string {
	if k == "" {
		return "The model call failed"
	}
	return "The model call failed: " + string(k)
}

// rawJSON passes a call's input through as JSON. The log only holds valid
// JSON, but an input that is not is sent as a string rather than breaking
// the notification.
func rawJSON(b json.RawMessage) any {
	if len(b) == 0 {
		return nil
	}
	if !json.Valid(b) {
		return string(b)
	}
	return b
}

// wireText is s as it comes back from JSON.
func wireText(s string) string {
	b, _ := json.Marshal(s)
	var out string
	json.Unmarshal(b, &out)
	return out
}
