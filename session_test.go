package agentloop_test

import (
	"encoding/json"
	"reflect"
	"testing"

	"agentloop"
)

func msgEntry(m agentloop.Message, stop agentloop.Stop, u agentloop.Usage) agentloop.Entry {
	return agentloop.Entry{Type: agentloop.EntryMessage, Message: &agentloop.MessageEntry{Message: m, Stop: stop, Usage: u, Model: "m"}}
}

func assistantCalls(ids ...string) agentloop.Message {
	m := agentloop.Message{Role: agentloop.Assistant}
	for _, id := range ids {
		m.Parts = append(m.Parts, agentloop.Part{Type: agentloop.ToolCallPart, ToolCall: &agentloop.ToolCall{ID: id, Name: "t", Input: json.RawMessage(`{}`)}})
	}
	return m
}

func decisionEntry(id string, v agentloop.Verdict, labels ...string) agentloop.Entry {
	return agentloop.Entry{Type: agentloop.EntryDecision, Decision: &agentloop.DecisionEntry{CallID: id, Tool: "t", Verdict: v, Labels: labels}}
}

func startedEntry(id string) agentloop.Entry {
	return agentloop.Entry{Type: agentloop.EntryCallStarted, CallStarted: &agentloop.CallStartedEntry{CallID: id}}
}

func resultEntry(id, content string, labels ...string) agentloop.Entry {
	return agentloop.Entry{Type: agentloop.EntryToolResult, ToolResult: &agentloop.ToolResultEntry{
		Result: agentloop.ToolResult{CallID: id, Content: content}, Labels: labels}}
}

func TestSessionDerivesStateFromEntries(t *testing.T) {
	u := agentloop.Usage{InputTokens: 3, OutputTokens: 4}
	s := reopen(t, seqStore(
		msgEntry(agentloop.Text(agentloop.User, "hi"), "", agentloop.Usage{}),
		msgEntry(assistantCalls("a", "b", "c"), agentloop.StopToolUse, u),
		decisionEntry("a", agentloop.Allow, "z-label", "a-label"),
		startedEntry("a"),
		decisionEntry("b", agentloop.Hold, "never"),
		decisionEntry("c", agentloop.Deny, "also-never"),
		resultEntry("c", "denied"),
		resultEntry("a", "ok", "from-result"),
	))

	if got := s.Labels(); !reflect.DeepEqual(got, []string{"a-label", "from-result", "z-label"}) {
		t.Fatalf("labels %v", got)
	}
	if s.Usage() != u {
		t.Fatalf("usage %+v", s.Usage())
	}
	if got := s.View(); got.ToolCalls != 2 || !got.Has("z-label") || got.Has("never") {
		t.Fatalf("view %+v", got)
	}
	a, _ := s.Call("a")
	if !a.Started || !a.Finished || a.Decision.Verdict != agentloop.Allow || a.Result.Content != "ok" {
		t.Fatalf("a: %+v", a)
	}
	b, _ := s.Call("b")
	if !b.Held() || b.Started || b.Finished {
		t.Fatalf("b: %+v", b)
	}
	if _, ok := s.Call("zz"); ok {
		t.Fatal("found a call that does not exist")
	}
	if held := s.Held(); len(held) != 1 || held[0].ID != "b" {
		t.Fatalf("held %+v", held)
	}
	if p := s.Pending(); len(p) != 1 || p[0].ID != "b" {
		t.Fatalf("pending %+v", p)
	}

	// Results arrived in the order c, a; the message keeps call order.
	msgs := s.Messages()
	if len(msgs) != 3 {
		t.Fatalf("%d messages", len(msgs))
	}
	parts := msgs[2].Parts
	if len(parts) != 2 || parts[0].ToolResult.CallID != "a" || parts[1].ToolResult.CallID != "c" {
		t.Fatalf("results %+v", parts)
	}
	if turns := s.Turns(); len(turns) != 1 || turns[0].Stop != agentloop.StopToolUse || !reflect.DeepEqual(turns[0].Calls, []string{"a", "b", "c"}) {
		t.Fatalf("turns %+v", turns)
	}
}

func TestSessionLateResultJoinsItsMessage(t *testing.T) {
	store := seqStore(
		msgEntry(agentloop.Text(agentloop.User, "hi"), "", agentloop.Usage{}),
		msgEntry(assistantCalls("a", "b"), agentloop.StopToolUse, agentloop.Usage{}),
		decisionEntry("b", agentloop.Allow),
		resultEntry("b", "second"),
		decisionEntry("a", agentloop.Hold),
	)
	s := reopen(t, store)
	before := s.Messages()
	if err := s.Resolve("a", agentloop.Decision{Verdict: agentloop.Deny, Reason: "no"}); err != nil {
		t.Fatal(err)
	}
	if len(before[2].Parts) != 1 {
		t.Fatal("a message handed out earlier changed")
	}
	// a was denied; a run would now record its result.
	entries := append(s.Entries(), resultEntry("a", "no"))
	s = reopen(t, seqStore(entries...))
	msgs := s.Messages()
	if len(msgs) != 3 || len(msgs[2].Parts) != 2 || msgs[2].Parts[0].ToolResult.CallID != "a" {
		t.Fatalf("%+v", msgs)
	}
	checkConversation(t, msgs)
}

func TestSessionRefusesEntriesThatDoNotFit(t *testing.T) {
	user := msgEntry(agentloop.Text(agentloop.User, "hi"), "", agentloop.Usage{})
	turn := msgEntry(assistantCalls("a"), agentloop.StopToolUse, agentloop.Usage{})
	cases := map[string][]agentloop.Entry{
		"result for unknown call":    {user, resultEntry("x", "?")},
		"decision for unknown call":  {user, decisionEntry("x", agentloop.Allow)},
		"start for unknown call":     {user, startedEntry("x")},
		"start without a decision":   {user, turn, startedEntry("a")},
		"start after a deny":         {user, turn, decisionEntry("a", agentloop.Deny), startedEntry("a")},
		"start twice":                {user, turn, decisionEntry("a", agentloop.Allow), startedEntry("a"), startedEntry("a")},
		"result twice":               {user, turn, resultEntry("a", "1"), resultEntry("a", "2")},
		"decision after result":      {user, turn, resultEntry("a", "1"), decisionEntry("a", agentloop.Allow)},
		"reused call id":             {user, turn, msgEntry(assistantCalls("a"), agentloop.StopToolUse, agentloop.Usage{})},
		"call with no id":            {user, msgEntry(assistantCalls(""), agentloop.StopToolUse, agentloop.Usage{})},
		"two calls with the same id": {user, msgEntry(assistantCalls("a", "a"), agentloop.StopToolUse, agentloop.Usage{})},
		"unknown type":               {{Type: "nonsense"}},
		"missing payload":            {{Type: agentloop.EntryDecision}},
	}
	for name, entries := range cases {
		if _, err := agentloop.OpenSession(seqStore(entries...)); err == nil {
			t.Errorf("%s: opened", name)
		}
	}
}

func TestResolveWritesDecisionEntry(t *testing.T) {
	s := reopen(t, seqStore(
		msgEntry(agentloop.Text(agentloop.User, "hi"), "", agentloop.Usage{}),
		msgEntry(assistantCalls("a"), agentloop.StopToolUse, agentloop.Usage{}),
		decisionEntry("a", agentloop.Hold),
	))
	if err := s.Resolve("a", agentloop.Decision{Verdict: agentloop.Allow, Reason: "ok by me", Labels: []string{"approved"}}); err != nil {
		t.Fatal(err)
	}
	e := s.Entries()[len(s.Entries())-1]
	if e.Type != agentloop.EntryDecision || !e.Decision.Resolved || e.Decision.Reason != "ok by me" || e.Seq != 4 {
		t.Fatalf("%+v", e)
	}
	info, _ := s.Call("a")
	if info.Held() || info.Decision.Verdict != agentloop.Allow {
		t.Fatalf("%+v", info)
	}
	if got := s.View().ToolCalls; got != 1 {
		t.Fatalf("ToolCalls = %d", got)
	}
}

func TestSessionRebuildsFromAgentRun(t *testing.T) {
	a, _ := crashScenario(false)
	s, store := newSession(t)
	finish(t, a, s, "go")
	s2 := reopen(t, store)
	if !reflect.DeepEqual(s.Messages(), s2.Messages()) || !reflect.DeepEqual(s.Labels(), s2.Labels()) ||
		s.Usage() != s2.Usage() || !reflect.DeepEqual(s.Turns(), s2.Turns()) || s.View().ToolCalls != s2.View().ToolCalls {
		t.Fatal("a reopened session differs from the one that wrote the log")
	}
}
