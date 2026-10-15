// Package acpagent serves agentloop sessions over the Agent Client
// Protocol. Everything but the shell is pure: the translation of a run's
// events into session/update notifications, the stop reason of a turn,
// the prompt text, and the turn machine take values and return values.
package acpagent

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	acp "github.com/caelis-labs/acp-go-sdk"

	"github.com/kirahowe/agentloop"
)

// Update is one session/update the agent sends.
type Update = acp.SessionUpdate

// Options are what the translation needs to know beyond the events.
type Options struct {
	// Kinds is the ACP kind of each tool, by name (the request's
	// tool_kinds). A tool not listed is "other".
	Kinds map[string]acp.ToolKind
	// ContextWindow is the model's context window in tokens. Zero means
	// unknown, and then no usage_update is sent: its size is required.
	ContextWindow int64
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
}

// NewTranslator makes the translator for a session's live events.
func NewTranslator(opts Options) Translator { return Translator{opts: opts} }

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

func (t Translator) entry(e agentloop.Entry) (Translator, []Update) {
	streamedText, streamedThought := t.text, t.thought
	t.seq, t.text, t.thought = e.Seq, "", ""
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
			ups = append(ups, toolCall(c, t.opts.Kinds, acp.ToolCallStatusPending, nil))
		}
		if u, ok := t.usage(m.Usage); ok {
			ups = append(ups, u)
		}
		return t, ups
	case agentloop.EntryCallStarted:
		if e.CallStarted == nil {
			return t, nil
		}
		return t, []Update{acp.UpdateToolCall(acp.ToolCallId(e.CallStarted.CallID),
			acp.WithUpdateStatus(acp.ToolCallStatusInProgress))}
	case agentloop.EntryToolResult:
		if e.ToolResult == nil {
			return t, nil
		}
		return t, []Update{toolResult(*e.ToolResult)}
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
// window, the last assistant turn's usage follows.
func Replay(opts Options, entries []agentloop.Entry) []Update {
	results := map[string]agentloop.ToolResultEntry{}
	started := map[string]bool{}
	for _, e := range entries {
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
			ups = append(ups, toolCall(c, opts.Kinds, status, res))
		}
	}
	if last != nil {
		if u, ok := (Translator{opts: opts}).usage(last.Usage); ok {
			ups = append(ups, u)
		}
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

func toolCall(c agentloop.ToolCall, kinds map[string]acp.ToolKind, status acp.ToolCallStatus, res *agentloop.ToolResultEntry) Update {
	tc := &acp.SessionUpdateToolCall{
		ToolCallId: acp.ToolCallId(c.ID),
		Title:      Title(c),
		Name:       acp.Ptr(c.Name),
		Kind:       Kind(c.Name, kinds),
		Status:     status,
		RawInput:   rawJSON(c.Input),
	}
	if res != nil {
		tc.Status, tc.Content, tc.RawOutput, tc.Meta = resultFields(*res)
	}
	return acp.SessionUpdate{ToolCall: tc}
}

func toolResult(r agentloop.ToolResultEntry) Update {
	u := &acp.SessionToolCallUpdate{ToolCallId: acp.ToolCallId(r.Result.CallID)}
	var status acp.ToolCallStatus
	status, u.Content, u.RawOutput, u.Meta = resultFields(r)
	u.Status = &status
	return acp.SessionUpdate{ToolCallUpdate: u}
}

// resultFields is how a result shows on a call: failed when it is an
// error, the content as text, the raw result, and its labels under
// _meta["agentloop/labels"].
func resultFields(r agentloop.ToolResultEntry) (acp.ToolCallStatus, []acp.ToolCallContent, any, map[string]json.RawMessage) {
	status := acp.ToolCallStatusCompleted
	if r.Result.IsError {
		status = acp.ToolCallStatusFailed
	}
	content := []acp.ToolCallContent{acp.ToolContent(acp.TextBlock(r.Result.Content))}
	raw := map[string]any{"content": r.Result.Content, "is_error": r.Result.IsError}
	var meta map[string]json.RawMessage
	if len(r.Labels) > 0 {
		b, _ := json.Marshal(r.Labels)
		meta = map[string]json.RawMessage{"agentloop/labels": b}
	}
	return status, content, raw, meta
}

func (t Translator) usage(u agentloop.Usage) (Update, bool) {
	if t.opts.ContextWindow <= 0 {
		return Update{}, false
	}
	used := u.InputTokens + u.CacheReadTokens + u.CacheWriteTokens + u.OutputTokens
	return acp.SessionUpdate{UsageUpdate: &acp.SessionUsageUpdate{
		Used: uint64(max(used, 0)), Size: uint64(t.opts.ContextWindow),
	}}, true
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
