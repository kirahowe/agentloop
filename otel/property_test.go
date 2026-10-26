package otel_test

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/otel"
	"pgregory.net/rapid"
)

// sentinel starts every piece of content the generator makes up. The
// default export must contain none of it.
const sentinel = "SECRETCONTENT"

type gen struct {
	t       *rapid.T
	entries []agentloop.Entry
	now     time.Time
	nextID  int
	// open calls: ids of the last assistant turn that have no result
	open     []string
	toolName map[string]string
	input    map[string]string
}

func (g *gen) text(label string) string {
	return sentinel + rapid.StringMatching(`[a-z0-9 ]{0,12}`).Draw(g.t, label)
}

func (g *gen) add(typ agentloop.EntryType, f func(*agentloop.Entry)) {
	g.now = g.now.Add(time.Duration(rapid.IntRange(0, 3000).Draw(g.t, "gap_ms")) * time.Millisecond)
	e := agentloop.Entry{Seq: len(g.entries) + 1, Time: g.now, Type: typ}
	f(&e)
	g.entries = append(g.entries, e)
}

var labelNames = []string{"private", "contaminated"}

func (g *gen) labels() []string {
	var out []string
	for _, l := range labelNames {
		if rapid.Bool().Draw(g.t, "label") {
			out = append(out, l)
		}
	}
	return out
}

func (g *gen) usage() agentloop.Usage {
	n := func(l string) int64 { return rapid.Int64Range(0, 5000).Draw(g.t, l) }
	return agentloop.Usage{InputTokens: n("in"), OutputTokens: n("out"), CacheReadTokens: n("cr"), CacheWriteTokens: n("cw")}
}

func (g *gen) modelErr() *agentloop.ModelError {
	kinds := []agentloop.ErrorKind{agentloop.RateLimited, agentloop.Overloaded, agentloop.Unreachable, agentloop.Unauthorized, agentloop.InvalidRequest}
	return &agentloop.ModelError{Kind: rapid.SampledFrom(kinds).Draw(g.t, "kind"), Message: g.text("err")}
}

func (g *gen) result(id string, isErr bool, labels []string) {
	g.add(agentloop.EntryToolResult, func(e *agentloop.Entry) {
		e.ToolResult = &agentloop.ToolResultEntry{
			Result: agentloop.ToolResult{CallID: id, Content: g.text("content"), IsError: isErr},
			Labels: labels,
			Log:    g.text("log"),
		}
		if rapid.Bool().Draw(g.t, "toolmodel") {
			for i, n := 0, rapid.IntRange(1, 2).Draw(g.t, "ntoolmodel"); i < n; i++ {
				e.ToolResult.Usage = append(e.ToolResult.Usage, agentloop.ModelUsage{
					Model: rapid.SampledFrom([]string{"", "tool-model-a"}).Draw(g.t, "tmodel"), Usage: g.usage()})
			}
		}
	})
}

func (g *gen) decide(id string, v agentloop.Verdict) {
	g.add(agentloop.EntryDecision, func(e *agentloop.Entry) {
		d := &agentloop.DecisionEntry{CallID: id, Tool: g.toolName[id], Verdict: v, Labels: g.labels()}
		if rapid.Bool().Draw(g.t, "reason") {
			d.Reason = g.text("reason")
		}
		e.Decision = d
	})
}

func (g *gen) start(id string) {
	g.add(agentloop.EntryCallStarted, func(e *agentloop.Entry) { e.CallStarted = &agentloop.CallStartedEntry{CallID: id} })
}

// settleOne gives one open call its fate.
func (g *gen) settleOne(id string) {
	switch rapid.IntRange(0, 3).Draw(g.t, "fate") {
	case 0: // allowed, ran
		g.decide(id, agentloop.Allow)
		g.start(id)
		g.result(id, rapid.Bool().Draw(g.t, "failed"), g.labels())
	case 1: // denied
		g.decide(id, agentloop.Deny)
		g.result(id, true, nil)
	case 2: // cut off
		g.result(id, true, nil)
	case 3: // started, never finished
		g.decide(id, agentloop.Allow)
		g.start(id)
	}
}

func (g *gen) assistant(withCalls bool) {
	var parts []agentloop.Part
	if rapid.Bool().Draw(g.t, "has_text") {
		parts = append(parts, agentloop.Part{Type: agentloop.TextPart, Text: g.text("text")})
	}
	if rapid.Bool().Draw(g.t, "has_reasoning") {
		parts = append(parts, agentloop.Part{Type: agentloop.ReasoningPart, Text: g.text("reasoning")})
	}
	var ids []string
	if withCalls {
		for i, n := 0, rapid.IntRange(1, 3).Draw(g.t, "ncalls"); i < n; i++ {
			g.nextID++
			id := fmt.Sprintf("call_%d", g.nextID)
			name := rapid.SampledFrom([]string{"read", "write", "send"}).Draw(g.t, "tool")
			in := fmt.Sprintf(`{"q":%q}`, g.text("input"))
			g.toolName[id], g.input[id] = name, in
			ids = append(ids, id)
			parts = append(parts, agentloop.Part{Type: agentloop.ToolCallPart, ToolCall: &agentloop.ToolCall{ID: id, Name: name, Input: json.RawMessage(in)}})
		}
	}
	stop := agentloop.StopEndTurn
	if withCalls {
		stop = agentloop.StopToolUse
	} else {
		stop = rapid.SampledFrom([]agentloop.Stop{agentloop.StopEndTurn, agentloop.StopMaxTokens, agentloop.StopRefusal, agentloop.StopPause}).Draw(g.t, "stop")
	}
	g.add(agentloop.EntryMessage, func(e *agentloop.Entry) {
		m := &agentloop.MessageEntry{
			Message: agentloop.Message{Role: agentloop.Assistant, Parts: parts,
				Native: &agentloop.Native{Provider: "anthropic", Model: "m", Data: json.RawMessage(fmt.Sprintf(`{"x":%q}`, g.text("native")))}},
			Stop: stop, Usage: g.usage(), Model: rapid.SampledFrom([]string{"", "claude-a", "claude-b"}).Draw(g.t, "model"),
		}
		if stop == agentloop.StopRefusal {
			m.Refusal = g.text("refusal")
		}
		e.Message = m
	})
	g.open = ids
}

func (g *gen) run() {
	g.add(agentloop.EntryRunStarted, func(e *agentloop.Entry) {
		e.RunStarted = &agentloop.RunStartedEntry{Prompt: g.text("prompt")}
	})
	if rapid.Bool().Draw(g.t, "pins") {
		g.nextID++
		g.add(agentloop.EntryPins, func(e *agentloop.Entry) {
			e.Pins = &agentloop.PinsEntry{Pins: map[string]string{fmt.Sprintf("tool_%d", g.nextID): g.text("pin")}}
		})
	}
	if l := g.labels(); len(l) > 0 && rapid.Bool().Draw(g.t, "seed") {
		g.add(agentloop.EntryLabels, func(e *agentloop.Entry) {
			e.Labels = &agentloop.LabelsEntry{Labels: l, Source: agentloop.LabelSeed}
		})
	}
	if rapid.Bool().Draw(g.t, "user") {
		g.add(agentloop.EntryMessage, func(e *agentloop.Entry) {
			e.Message = &agentloop.MessageEntry{Message: agentloop.Text(agentloop.User, g.text("user"))}
		})
	}
	crashed := false
	for turn, n := 0, rapid.IntRange(0, 3).Draw(g.t, "turns"); turn < n && !crashed; turn++ {
		for i, k := 0, rapid.IntRange(0, 2).Draw(g.t, "errors"); i < k; i++ {
			g.add(agentloop.EntryModelError, func(e *agentloop.Entry) {
				e.ModelError = &agentloop.ModelErrorEntry{Error: *g.modelErr(), Retry: rapid.Bool().Draw(g.t, "retry"), WaitMillis: rapid.Int64Range(0, 2000).Draw(g.t, "wait")}
			})
		}
		g.assistant(rapid.Bool().Draw(g.t, "calls"))
		for _, id := range g.open {
			g.settleOne(id)
		}
		g.open = nil
		crashed = rapid.IntRange(0, 9).Draw(g.t, "crash") == 0
	}
	if crashed {
		return // no run_ended
	}
	reason := rapid.SampledFrom([]agentloop.Reason{agentloop.ReasonCompleted, agentloop.ReasonMaxTokens, agentloop.ReasonRefused, agentloop.ReasonLimit, agentloop.ReasonCanceled, agentloop.ReasonModelError}).Draw(g.t, "reason")
	out := &agentloop.Outcome{Reason: reason, Text: g.text("outcome"), Turns: rapid.IntRange(0, 5).Draw(g.t, "nturns"), Usage: g.usage()}
	switch reason {
	case agentloop.ReasonModelError:
		out.Error = g.modelErr()
	case agentloop.ReasonLimit:
		out.Limit = agentloop.LimitTurns
	}
	g.add(agentloop.EntryRunEnded, func(e *agentloop.Entry) { e.RunEnded = out })
}

// between: labels are added by hand.
func (g *gen) between() {
	if l := g.labels(); len(l) > 0 && rapid.Bool().Draw(g.t, "addlabels") {
		g.add(agentloop.EntryLabels, func(e *agentloop.Entry) {
			e.Labels = &agentloop.LabelsEntry{Labels: l, Source: agentloop.LabelSeed}
		})
	}
}

func genLog(t *rapid.T) []agentloop.Entry {
	g := &gen{t: t, now: time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC), toolName: map[string]string{}, input: map[string]string{}}
	for i, n := 0, rapid.IntRange(0, 4).Draw(t, "runs"); i < n; i++ {
		g.run()
		g.between()
	}
	// The generator must only make logs a session accepts.
	if _, err := agentloop.OpenSession(agentloop.NewMemoryStore(g.entries...)); err != nil {
		t.Fatalf("generator made an invalid log: %v", err)
	}
	return g.entries
}

func genOptions(t *rapid.T) otel.Options {
	s := func(l string) string { return rapid.StringMatching(`[a-z0-9-]{0,8}`).Draw(t, l) }
	return otel.Options{
		ConversationID: s("conv"), AgentName: s("agent"), Provider: rapid.SampledFrom([]string{"", "anthropic", "openai"}).Draw(t, "provider"),
		Model: s("model"), ServiceName: s("service"), TraceSeed: s("seed"),
	}
}

func spansOf(t interface{ Fatal(...any) }, raw []byte) []map[string]any {
	var root struct {
		ResourceSpans []struct {
			ScopeSpans []struct{ Spans []map[string]any }
		}
	}
	if err := json.Unmarshal(raw, &root); err != nil {
		t.Fatal(err)
	}
	var out []map[string]any
	for _, rs := range root.ResourceSpans {
		for _, ss := range rs.ScopeSpans {
			out = append(out, ss.Spans...)
		}
	}
	return out
}

func TestPropertyWellFormed(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		entries, opts := genLog(t), genOptions(t)
		for _, content := range []bool{false, true} {
			opts.IncludeContent = content
			raw, err := otel.Marshal(entries, opts)
			if err != nil {
				t.Fatal(err)
			}
			if err := validateOTLP(raw); err != nil {
				t.Fatalf("content=%v: %v", content, err)
			}
		}
	})
}

func TestPropertyShapeOfTraces(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		entries, opts := genLog(t), genOptions(t)
		raw, _ := otel.Marshal(entries, opts)
		spans := spansOf(t, raw)

		var runs, assistant, errs int
		calls := map[string]bool{}
		for _, e := range entries {
			switch e.Type {
			case agentloop.EntryRunStarted:
				runs++
			case agentloop.EntryMessage:
				if e.Message.Message.Role == agentloop.Assistant {
					assistant++
				}
			case agentloop.EntryModelError:
				errs++
			case agentloop.EntryDecision:
				calls[e.Decision.CallID] = true
			case agentloop.EntryCallStarted:
				calls[e.CallStarted.CallID] = true
			case agentloop.EntryToolResult:
				calls[e.ToolResult.Result.CallID] = true
				assistant += len(e.ToolResult.Usage) // a tool's model calls are chat spans too
			}
		}
		var roots, chats, tools int
		traces := map[string]int{}
		for _, s := range spans {
			name := s["name"].(string)
			_, hasParent := s["parentSpanId"]
			switch {
			case strings.HasPrefix(name, "invoke_agent"):
				roots++
				traces[s["traceId"].(string)]++
				if hasParent {
					t.Fatalf("a run span has a parent")
				}
			case strings.HasPrefix(name, "chat"):
				chats++
			case strings.HasPrefix(name, "execute_tool"):
				tools++
			default:
				t.Fatalf("unexpected span %q", name)
			}
		}
		if roots != runs || len(traces) != runs {
			t.Fatalf("%d runs gave %d run spans in %d traces", runs, roots, len(traces))
		}
		if chats != assistant+errs {
			t.Fatalf("%d chat spans for %d assistant messages and %d model errors", chats, assistant, errs)
		}
		if tools != len(calls) {
			t.Fatalf("%d tool spans for %d calls", tools, len(calls))
		}
	})
}

func TestPropertyNoContentByDefault(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		entries, opts := genLog(t), genOptions(t)
		opts.IncludeContent = false
		raw, _ := otel.Marshal(entries, opts)
		if i := strings.Index(string(raw), sentinel); i >= 0 {
			t.Fatalf("content leaked into the default export: ...%s...", raw[max(0, i-80):min(len(raw), i+80)])
		}
		// And the option does turn it on, or the check above proves nothing.
		opts.IncludeContent = true
		with, _ := otel.Marshal(entries, opts)
		raw2 := string(with)
		for _, e := range entries {
			if e.Type == agentloop.EntryToolResult && !strings.Contains(raw2, e.ToolResult.Log) {
				t.Fatalf("content mode lost a tool log")
			}
		}
	})
}

func TestPropertyDeterministic(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		entries, opts := genLog(t), genOptions(t)
		opts.IncludeContent = rapid.Bool().Draw(t, "content")
		a, _ := otel.Marshal(entries, opts)
		b, _ := otel.Marshal(entries, opts)
		// The same log read back from a store, as another process would.
		loaded, err := agentloop.NewMemoryStore(entries...).Load()
		if err != nil {
			t.Fatal(err)
		}
		c, _ := otel.Marshal(loaded, opts)
		if string(a) != string(b) || string(a) != string(c) {
			t.Fatal("the same log gave different output")
		}
		// The seed in use: an empty TraceSeed means the conversation id,
		// and "" + "x" would be the same seed as a conversation named x.
		seed := opts.TraceSeed
		if seed == "" {
			seed = opts.ConversationID
		}
		other := opts
		other.TraceSeed = seed + "x"
		d, _ := otel.Marshal(entries, other)
		ids := map[string]bool{}
		for _, s := range spansOf(t, a) {
			ids[s["traceId"].(string)] = true
		}
		for _, s := range spansOf(t, d) {
			if ids[s["traceId"].(string)] {
				t.Fatal("two seeds share a trace id")
			}
		}
	})
}

// TestPropertyUsage: the input tokens on a chat span count the cached ones.
func TestPropertyUsage(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		entries, opts := genLog(t), genOptions(t)
		raw, _ := otel.Marshal(entries, opts)
		var want []int64
		for _, e := range entries {
			if e.Type == agentloop.EntryMessage && e.Message.Message.Role == agentloop.Assistant {
				u := e.Message.Usage
				want = append(want, u.InputTokens+u.CacheReadTokens+u.CacheWriteTokens)
			}
		}
		var got []int64
		spans := spansOf(t, raw)
		runIDs := map[string]bool{}
		for _, s := range spans {
			if strings.HasPrefix(s["name"].(string), "invoke_agent") {
				runIDs[s["spanId"].(string)] = true
			}
		}
		for _, s := range spans {
			// Reader calls under a tool span are checked elsewhere.
			if !strings.HasPrefix(s["name"].(string), "chat") || (s["parentSpanId"] != nil && !runIDs[s["parentSpanId"].(string)]) {
				continue
			}
			for _, a := range s["attributes"].([]any) {
				kv := a.(map[string]any)
				if kv["key"] == "gen_ai.usage.input_tokens" {
					var n int64
					fmt.Sscan(kv["value"].(map[string]any)["intValue"].(string), &n)
					got = append(got, n)
				}
			}
		}
		if fmt.Sprint(want) != fmt.Sprint(got) {
			t.Fatalf("input tokens: want %v, got %v", want, got)
		}
	})
}
