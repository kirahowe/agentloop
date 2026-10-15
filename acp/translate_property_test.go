package acpagent

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"

	acp "github.com/caelis-labs/acp-go-sdk"
	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/internal/scenario"
	"github.com/kirahowe/agentloop/trifecta"
)

// run is one generated scenario: the events every run reported, in order,
// with deltas re-split, and the log it left.
type run struct {
	opts    Options
	events  []agentloop.Event
	entries []agentloop.Entry
}

func genOptions(t *rapid.T) Options {
	kinds := map[string]acp.ToolKind{}
	for _, name := range []string{"echo", "any", "lab", "ghost"} {
		if rapid.Bool().Draw(t, "haskind") {
			kinds[name] = rapid.SampledFrom([]acp.ToolKind{acp.ToolKindRead, acp.ToolKindEdit, acp.ToolKindExecute,
				acp.ToolKindFetch, acp.ToolKindOther, "bogus"}).Draw(t, "kind")
		}
	}
	return Options{
		Kinds:         kinds,
		ContextWindow: rapid.SampledFrom([]int64{0, 0, 200000}).Draw(t, "window"),
		Notices:       rapid.Bool().Draw(t, "notices"),
	}
}

// genRun drives a scenario from internal/scenario and records it. Crashes
// are left out: an entry made durable by a store that then fails is in
// the log but was never reported, which is the store failing, not the
// translation.
func genRun(t *rapid.T) run {
	cfg := scenario.GenConfig(t)
	sc := scenario.New(t, cfg, scenario.Faults{
		Cancel: rapid.Bool().Draw(t, "cancel"),
		Wild:   rapid.Bool().Draw(t, "wild"),
	})
	var events []agentloop.Event
	sc.Observe = func(ev agentloop.Event) { events = append(events, ev) }
	sc.Execute()
	entries, err := sc.Store.Load()
	if err != nil {
		t.Fatal(err)
	}
	return run{opts: genOptions(t), events: resplit(t, events), entries: entries}
}

// resplit gives each streamed response a random shape. Before a message,
// its deltas are split at random rune boundaries, cut to a random prefix,
// or dropped. Before anything else (a failed or cancelled model call), a
// partial stream is sometimes made up, as a call that fails mid-stream
// leaves.
func resplit(t *rapid.T, events []agentloop.Event) []agentloop.Event {
	var out, group []agentloop.Event
	for _, ev := range events {
		if ev.Delta != nil {
			group = append(group, ev)
			continue
		}
		e := ev.Entry
		if e.Type == agentloop.EntryMessage && e.Message.Message.Role == agentloop.Assistant {
			out = append(out, reshape(t, group)...)
		} else {
			out = append(out, group...)
			if rapid.IntRange(0, 4).Draw(t, "partial?") == 0 {
				out = append(out, splitDelta(t, agentloop.Delta{
					Type: rapid.SampledFrom([]agentloop.PartType{agentloop.TextPart, agentloop.ReasoningPart}).Draw(t, "ptype"),
					Text: rapid.SampledFrom([]string{"par", "half a sentence", "café \U0001F600"}).Draw(t, "ptext"),
				})...)
			}
		}
		group = nil
		out = append(out, ev)
	}
	return append(out, group...)
}

func reshape(t *rapid.T, group []agentloop.Event) []agentloop.Event {
	total := 0
	for _, ev := range group {
		total += utf8.RuneCountInString(ev.Delta.Text)
	}
	keep := total
	switch rapid.IntRange(0, 3).Draw(t, "stream") {
	case 0:
		keep = 0
	case 1:
		keep = rapid.IntRange(0, total).Draw(t, "keep")
	}
	var out []agentloop.Event
	for _, ev := range group {
		r := []rune(ev.Delta.Text)
		n := min(len(r), keep)
		keep -= n
		if n == 0 {
			continue
		}
		out = append(out, splitDelta(t, agentloop.Delta{Type: ev.Delta.Type, Text: string(r[:n])})...)
	}
	return out
}

func splitDelta(t *rapid.T, d agentloop.Delta) []agentloop.Event {
	r := []rune(d.Text)
	var out []agentloop.Event
	for len(r) > 0 {
		n := rapid.IntRange(1, len(r)).Draw(t, "piece")
		piece := agentloop.Delta{Type: d.Type, Text: string(r[:n])}
		out = append(out, agentloop.Event{Delta: &piece})
		r = r[n:]
	}
	return out
}

func translateAll(opts Options, events []agentloop.Event) []Update {
	tr := NewTranslator(opts)
	var all []Update
	for _, ev := range events {
		var ups []Update
		tr, ups = tr.Step(ev)
		all = append(all, ups...)
	}
	return all
}

// stream is what a client makes of a list of updates.
type stream struct {
	text, thought map[acp.MessageId]string
	ids           []acp.MessageId
	calls         map[string]*callView
	order         []string
	notices       int
	usage         int
	info          int
}

type callView struct {
	statuses []acp.ToolCallStatus
	starts   int
	final    acp.SessionUpdateToolCall
}

func read(t *rapid.T, ups []Update) stream {
	s := stream{text: map[acp.MessageId]string{}, thought: map[acp.MessageId]string{}, calls: map[string]*callView{}}
	seen := map[acp.MessageId]bool{}
	note := func(id *acp.MessageId) acp.MessageId {
		if id == nil {
			t.Fatal("a chunk has no messageId")
		}
		if !seen[*id] {
			seen[*id] = true
			s.ids = append(s.ids, *id)
		}
		return *id
	}
	for _, u := range ups {
		switch {
		case u.AgentMessageChunk != nil:
			s.text[note(u.AgentMessageChunk.MessageId)] += u.AgentMessageChunk.Content.Text.Text
		case u.AgentThoughtChunk != nil:
			s.thought[note(u.AgentThoughtChunk.MessageId)] += u.AgentThoughtChunk.Content.Text.Text
		case u.UserMessageChunk != nil:
			s.text[note(u.UserMessageChunk.MessageId)] += u.UserMessageChunk.Content.Text.Text
		case u.ToolCall != nil:
			id := string(u.ToolCall.ToolCallId)
			c := s.calls[id]
			if c == nil {
				c = &callView{}
				s.calls[id] = c
				s.order = append(s.order, id)
			}
			c.starts++
			c.statuses = append(c.statuses, u.ToolCall.Status)
			c.final = *u.ToolCall
		case u.ToolCallUpdate != nil:
			id := string(u.ToolCallUpdate.ToolCallId)
			c := s.calls[id]
			if c == nil {
				t.Fatalf("tool_call_update for %s before its tool_call", id)
			}
			up := u.ToolCallUpdate
			if up.Status == nil {
				t.Fatalf("tool_call_update for %s has no status", id)
			}
			c.statuses = append(c.statuses, *up.Status)
			c.final.Status = *up.Status
			if up.Content != nil {
				c.final.Content = up.Content
			}
			if up.RawOutput != nil {
				c.final.RawOutput = up.RawOutput
			}
			if up.Meta != nil {
				c.final.Meta = up.Meta
			}
		case u.Notice != nil:
			s.notices++
		case u.UsageUpdate != nil:
			s.usage++
		case u.SessionInfoUpdate != nil:
			s.info++
		default:
			t.Fatalf("an unexpected update: %+v", u)
		}
	}
	return s
}

func terminal(s acp.ToolCallStatus) bool {
	return s == acp.ToolCallStatusCompleted || s == acp.ToolCallStatusFailed
}

// TestPropertyTranslate holds P1.1 to P1.4 over generated runs.
func TestPropertyTranslate(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		r := genRun(t)
		live := translateAll(r.opts, r.events)
		replay := Replay(r.opts, r.entries)

		// P1.1 Valid wire data.
		for _, u := range append(append([]Update{}, live...), replay...) {
			if err := validateUpdate(u); err != nil {
				t.Fatal(err)
			}
		}
		ls, rs := read(t, live), read(t, replay)
		checkLifecycle(t, ls, r.entries)
		checkLabels(t, live, r.entries)
		checkLiveIsReplay(t, ls, rs, r.entries)
		checkTextConserved(t, ls, r.entries)
		if !r.opts.Notices && ls.notices > 0 {
			t.Fatalf("%d notices sent to a client that did not ask for them", ls.notices)
		}
		if rs.notices > 0 {
			t.Fatal("replay sent a notice")
		}
		if r.opts.ContextWindow == 0 && ls.usage+rs.usage > 0 {
			t.Fatal("usage_update sent with no context window")
		}
	})
}

// P1.2 Tool call lifecycle: for each call, one tool_call, pending, first;
// at most one in_progress; at most one terminal status and nothing after
// it. The calls with a terminal status are exactly those with a
// tool_result entry, so held calls have none.
func checkLifecycle(t *rapid.T, s stream, entries []agentloop.Entry) {
	results := map[string]bool{}
	for _, e := range entries {
		if e.Type == agentloop.EntryToolResult {
			results[e.ToolResult.Result.CallID] = true
		}
	}
	for _, id := range s.order {
		c := s.calls[id]
		if c.starts != 1 || c.statuses[0] != acp.ToolCallStatusPending {
			t.Fatalf("call %s: %d tool_call updates, statuses %v", id, c.starts, c.statuses)
		}
		inProgress, ended := 0, false
		for _, st := range c.statuses[1:] {
			if ended {
				t.Fatalf("call %s: an update after its terminal status: %v", id, c.statuses)
			}
			switch {
			case st == acp.ToolCallStatusInProgress:
				inProgress++
			case terminal(st):
				ended = true
			default:
				t.Fatalf("call %s: status %q after the start", id, st)
			}
		}
		if inProgress > 1 {
			t.Fatalf("call %s went in_progress %d times", id, inProgress)
		}
		if ended != results[id] {
			t.Fatalf("call %s: terminal=%v, tool_result in the log=%v", id, ended, results[id])
		}
	}
	for id := range results {
		if s.calls[id] == nil {
			t.Fatalf("call %s has a result but was never announced", id)
		}
	}
}

// P1.3 Live equals replay: per messageId, the same chunk text, and per
// call the same final status, content, input and output. A live id that
// replay lacks belongs to a model call that produced no message.
func checkLiveIsReplay(t *rapid.T, live, replay stream, entries []agentloop.Entry) {
	messages := map[acp.MessageId]bool{}
	for _, e := range entries {
		if e.Type == agentloop.EntryMessage && e.Message.Message.Role == agentloop.Assistant {
			messages[assistantID(e.Seq)] = true
		}
	}
	for _, id := range replay.ids {
		if strings.HasPrefix(string(id), "u") {
			continue
		}
		if live.text[id] != replay.text[id] || live.thought[id] != replay.thought[id] {
			t.Fatalf("message %s: live %q/%q, replay %q/%q", id, live.text[id], live.thought[id], replay.text[id], replay.thought[id])
		}
	}
	for _, id := range live.ids {
		if messages[id] {
			if live.text[id] != replay.text[id] || live.thought[id] != replay.thought[id] {
				t.Fatalf("message %s: live %q/%q, replay %q/%q", id, live.text[id], live.thought[id], replay.text[id], replay.thought[id])
			}
		}
	}
	if fmt.Sprint(live.order) != fmt.Sprint(replay.order) {
		t.Fatalf("calls live %v, replay %v", live.order, replay.order)
	}
	for _, id := range live.order {
		a, b := live.calls[id].final, replay.calls[id].final
		ja, _ := json.Marshal(acp.SessionUpdate{ToolCall: &a})
		jb, _ := json.Marshal(acp.SessionUpdate{ToolCall: &b})
		if string(ja) != string(jb) {
			t.Fatalf("call %s ends differently:\nlive   %s\nreplay %s", id, ja, jb)
		}
	}
}

// P1.4 Text is conserved: for each assistant message, the chunk text
// under its id is its text parts joined, plus a refusal's explanation
// when the text does not hold it, however the deltas were split. The
// same holds for reasoning.
func checkTextConserved(t *rapid.T, s stream, entries []agentloop.Entry) {
	for _, e := range entries {
		if e.Type != agentloop.EntryMessage || e.Message.Message.Role != agentloop.Assistant {
			continue
		}
		var text, thought strings.Builder
		for _, p := range e.Message.Message.Parts {
			switch p.Type {
			case agentloop.TextPart:
				text.WriteString(p.Text)
			case agentloop.ReasoningPart:
				thought.WriteString(p.Text)
			}
		}
		want := text.String()
		if m := e.Message; m.Stop == agentloop.StopRefusal && m.Refusal != "" && !strings.Contains(want, m.Refusal) {
			if want != "" {
				want += "\n\n"
			}
			want += m.Refusal
		}
		id := assistantID(e.Seq)
		if s.text[id] != want || s.thought[id] != thought.String() {
			t.Fatalf("message %s: sent %q/%q, the message holds %q/%q", id, s.text[id], s.thought[id], want, thought.String())
		}
	}
}

// Labels: every tool_call, tool_call_update, usage_update and
// session_info_update carries the session's labels under
// _meta["agentloop/labels"]; they only grow, they are never more than the
// session holds by the end of the run, and the last ones shown are the
// session's final labels, so a change is never left unshown.
func checkLabels(t *rapid.T, ups []Update, entries []agentloop.Entry) {
	var final []string
	for _, e := range entries {
		final = trifecta.Union(final, entryLabels(e))
	}
	var prev []string
	for _, u := range ups {
		var meta map[string]json.RawMessage
		switch {
		case u.ToolCall != nil:
			meta = u.ToolCall.Meta
		case u.ToolCallUpdate != nil:
			meta = u.ToolCallUpdate.Meta
		case u.UsageUpdate != nil:
			meta = u.UsageUpdate.Meta
		case u.SessionInfoUpdate != nil:
			meta = u.SessionInfoUpdate.Meta
		default:
			continue
		}
		var got []string
		if err := json.Unmarshal(meta["agentloop/labels"], &got); err != nil || got == nil {
			t.Fatalf("an update has no labels: %v", meta)
		}
		for _, l := range prev {
			if !slices.Contains(got, l) {
				t.Fatalf("label %s disappeared: %v after %v", l, got, prev)
			}
		}
		for _, l := range got {
			if !slices.Contains(final, l) {
				t.Fatalf("label %s shown that the session never held", l)
			}
		}
		prev = got
	}
	if len(final) > 0 && !slices.Equal(prev, final) {
		t.Fatalf("the client last saw labels %v, the session holds %v", prev, final)
	}
}
