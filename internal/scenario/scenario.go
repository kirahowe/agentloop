// Package scenario is the harness behind the property tests, for reuse by
// any package that wants to drive the same generated runs. A scenario is a
// random model script, random tools, a random policy, random limits and retry policy,
// run against a store that can "crash". Every check that must hold at
// every point lives here, so each property test only chooses which faults
// to inject.
package scenario

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"sort"
	"strings"
	"sync"
	"time"

	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/fake"
)

var ErrCrash = errors.New("simulated crash")

// CrashStore keeps JSON lines in memory and can die on the Nth append,
// either before or after the line becomes durable.
type CrashStore struct {
	mu      sync.Mutex
	lines   [][]byte
	appends int
	crashAt int
	after   bool
	dead    bool
}

func (c *CrashStore) Load() ([]agentloop.Entry, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]agentloop.Entry, 0, len(c.lines))
	for _, b := range c.lines {
		var e agentloop.Entry
		if err := json.Unmarshal(b, &e); err != nil {
			return nil, err
		}
		out = append(out, e)
	}
	return out, nil
}

func (c *CrashStore) Append(e agentloop.Entry) error {
	b, err := json.Marshal(e)
	if err != nil {
		return err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.dead {
		return ErrCrash
	}
	c.appends++
	if c.crashAt > 0 && c.appends == c.crashAt {
		c.dead = true
		if c.after {
			c.lines = append(c.lines, b)
		}
		return ErrCrash
	}
	c.lines = append(c.lines, b)
	return nil
}

// arm revives the store, as a restarted process would find it, and makes
// the nth append from now on fail. Zero means never.
func (c *CrashStore) arm(n int, after bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.dead, c.appends, c.crashAt, c.after = false, 0, n, after
}

func (c *CrashStore) Snapshot() [][]byte {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([][]byte, len(c.lines))
	for i, l := range c.lines {
		out[i] = bytes.Clone(l)
	}
	return out
}

// ---- generated pieces ----

type CallSpec struct {
	Name  string
	Input int
}

const (
	StText = iota
	StToolUse
	StErr
	StHang
	StNil
)

type Step struct {
	Kind       int
	Stop       agentloop.Stop
	Calls      []CallSpec
	ErrKind    agentloop.ErrorKind
	Plain      bool
	RetryAfter int
	In, Out    int64
	Text       string
	Reasoning  bool
	// BadID makes the model break the id contract: 1 repeats an id within
	// the turn, 2 leaves one empty, 3 reuses the id of an earlier step's call.
	BadID int
}

var callNames = []string{"echo", "echo", "any", "lab", "ghost"}

func genStep() *rapid.Generator[Step] {
	return rapid.Custom(func(t *rapid.T) Step {
		var s Step
		s.Kind = rapid.SampledFrom([]int{StText, StToolUse, StToolUse, StToolUse, StToolUse, StToolUse, StErr, StErr, StHang, StNil}).Draw(t, "kind")
		s.Text = rapid.SampledFrom([]string{"", "hello", "a <b> & c", "x\ny", "caf\u00e9"}).Draw(t, "text")
		s.In = rapid.Int64Range(0, 40).Draw(t, "in")
		s.Out = rapid.Int64Range(0, 40).Draw(t, "out")
		s.Reasoning = rapid.Bool().Draw(t, "reasoning")
		switch s.Kind {
		case StText:
			s.Stop = rapid.SampledFrom([]agentloop.Stop{agentloop.StopEndTurn, agentloop.StopEndTurn, agentloop.StopEndTurn,
				agentloop.StopMaxTokens, agentloop.StopRefusal, agentloop.StopPause, agentloop.StopToolUse}).Draw(t, "stop")
		case StToolUse:
			s.Calls = rapid.SliceOfN(rapid.Custom(func(t *rapid.T) CallSpec {
				return CallSpec{
					Name:  rapid.SampledFrom(callNames).Draw(t, "name"),
					Input: rapid.SampledFrom([]int{0, 0, 0, 0, 1, 2, 3, 4, 5, 6}).Draw(t, "input"),
				}
			}), 0, 4).Draw(t, "calls")
			s.BadID = rapid.SampledFrom([]int{0, 0, 0, 0, 0, 0, 0, 1, 2, 3}).Draw(t, "badid")
			s.Stop = rapid.SampledFrom([]agentloop.Stop{agentloop.StopToolUse, agentloop.StopToolUse, agentloop.StopToolUse,
				agentloop.StopToolUse, agentloop.StopToolUse, agentloop.StopEndTurn, agentloop.StopMaxTokens,
				agentloop.StopPause, agentloop.StopRefusal}).Draw(t, "stop")
		case StErr:
			s.ErrKind = rapid.SampledFrom([]agentloop.ErrorKind{agentloop.RateLimited, agentloop.Overloaded, agentloop.Unreachable,
				agentloop.Unauthorized, agentloop.Billing, agentloop.NotFound, agentloop.InvalidRequest}).Draw(t, "errkind")
			s.Plain = rapid.IntRange(0, 5).Draw(t, "plain") == 0
			s.RetryAfter = rapid.IntRange(-10, 120).Draw(t, "retryafter")
		}
		return s
	})
}

type Behavior struct {
	Kind   int
	Labels []string
}

const (
	BOK = iota
	BIsError
	BGoError
	BCancelThenOK
	BBlockUntilCancel
	BBlockUntilTimeout
)

type Rule struct {
	Base       agentloop.Verdict
	Labels     []string
	IfLabel    string
	Then       agentloop.Verdict
	ThenLabels []string
}

var labelNames = []string{"private", "untrusted", "egress", "l4"}

func genLabels() *rapid.Generator[[]string] {
	return rapid.SliceOfN(rapid.SampledFrom(labelNames), 0, 3)
}

func genRule() *rapid.Generator[Rule] {
	return rapid.Custom(func(t *rapid.T) Rule {
		vs := []agentloop.Verdict{agentloop.Allow, agentloop.Allow, agentloop.Deny, agentloop.Hold, agentloop.Hold, "maybe"}
		return Rule{
			Base:       rapid.SampledFrom(vs).Draw(t, "base"),
			Labels:     genLabels().Draw(t, "labels"),
			IfLabel:    rapid.SampledFrom(append([]string{""}, labelNames...)).Draw(t, "iflabel"),
			Then:       rapid.SampledFrom(vs).Draw(t, "then"),
			ThenLabels: genLabels().Draw(t, "thenlabels"),
		}
	})
}

// config is everything random about one scenario.
type Config struct {
	Steps       []Step
	Behaviors   map[string]Behavior
	NilPolicy   bool
	Rules       map[string]Rule
	Limits      agentloop.Limits
	Retry       *agentloop.RetryPolicy
	ToolTimeout time.Duration
	Iterations  int
	Wild        bool
}

func GenConfig(t *rapid.T) Config {
	var c Config
	c.Steps = rapid.SliceOfN(genStep(), 0, 9).Draw(t, "steps")
	c.Behaviors = map[string]Behavior{}
	c.Rules = map[string]Rule{}
	for _, n := range []string{"echo", "any", "lab"} {
		c.Behaviors[n] = Behavior{
			Kind:   rapid.SampledFrom([]int{BOK, BOK, BOK, BIsError, BGoError, BCancelThenOK, BBlockUntilCancel, BBlockUntilTimeout}).Draw(t, "behavior"),
			Labels: genLabels().Draw(t, "toollabels"),
		}
		c.Rules[n] = genRule().Draw(t, "rule")
	}
	c.NilPolicy = rapid.IntRange(0, 3).Draw(t, "nilpolicy") == 0
	if rapid.Bool().Draw(t, "limited") {
		c.Limits.MaxTurns = rapid.IntRange(0, 3).Draw(t, "maxturns")
		c.Limits.MaxToolCalls = rapid.IntRange(0, 2).Draw(t, "maxtools")
		c.Limits.MaxTokens = rapid.Int64Range(0, 90).Draw(t, "maxtokens")
	}
	if rapid.Bool().Draw(t, "customretry") {
		c.Retry = &agentloop.RetryPolicy{
			Retries: rapid.IntRange(0, 4).Draw(t, "retries"),
			Backoff: time.Duration(rapid.IntRange(0, 5).Draw(t, "backoff")) * time.Second,
			MaxWait: time.Duration(rapid.IntRange(0, 30).Draw(t, "maxwait")) * time.Second,
		}
	}
	if rapid.IntRange(0, 4).Draw(t, "timeout") == 0 {
		c.ToolTimeout = time.Millisecond
	}
	c.Iterations = rapid.IntRange(1, 8).Draw(t, "iterations")
	return c
}

// ---- the scenario ----

type Faults struct {
	Crash  bool
	Cancel bool
	Wild   bool
}

type Scenario struct {
	t      *rapid.T
	cfg    Config
	faults Faults
	Store  *CrashStore
	// file, once set by HandOver, is where the log is kept instead.
	file string
	// Observe, if set, sees every event of every run, after the
	// scenario's own checks.
	Observe func(agentloop.Event)
	model   *fake.Model
	clock   time.Time

	mu          sync.Mutex
	modelIdx    int
	returned    []agentloop.Message // as the model made them, in order
	toolRuns    map[string]int
	cancel      context.CancelFunc // armed for this run, or nil
	cancelFired bool
	preCanceled bool
	waits       []time.Duration
	handed      [][]string // label slices given to the core
	asked       map[string]agentloop.Verdict
	prevLines   [][]byte
	prevSeen    []string // labels seen at the last check
	resolveAll  bool     // the drain phase approves held calls
	draining    bool
}

func New(t *rapid.T, cfg Config, f Faults) *Scenario {
	sc := &Scenario{t: t, cfg: cfg, faults: f, Store: &CrashStore{}, clock: time.Date(2026, 3, 1, 9, 0, 0, 0, time.UTC),
		toolRuns: map[string]int{}, asked: map[string]agentloop.Verdict{}}
	sc.model = fake.New(sc.respond)
	return sc
}

// log is where the scenario's log is: the crash store, or the file it was
// handed over to.
func (sc *Scenario) log() interface {
	agentloop.Store
	Snapshot() [][]byte
} {
	if sc.file != "" {
		return fileLog{agentloop.NewFileStore(sc.file), sc.file}
	}
	return sc.Store
}

// fileLog is a session file read as the scenario reads its crash store.
type fileLog struct {
	*agentloop.FileStore
	path string
}

func (f fileLog) Snapshot() [][]byte {
	raw, err := os.ReadFile(f.path)
	if err != nil {
		return nil
	}
	var out [][]byte
	for _, l := range bytes.Split(raw, []byte("\n")) {
		if len(bytes.TrimSpace(l)) > 0 {
			out = append(out, l)
		}
	}
	return out
}

// HandOver writes the log to a session file at path and keeps it there
// from now on: every check reads the file, so whatever runs the session
// next (an ACP server, say) is held to the same invariants. The scenario
// does not run the session itself after this.
func (sc *Scenario) HandOver(path string) {
	var b []byte
	for _, l := range sc.Store.Snapshot() {
		b = append(append(b, l...), '\n')
	}
	if err := os.WriteFile(path, b, 0o600); err != nil {
		sc.t.Fatal(err)
	}
	sc.file = path
}

func (sc *Scenario) entries() []agentloop.Entry {
	es, err := sc.log().Load()
	if err != nil {
		sc.t.Fatalf("the persisted log does not load: %v", err)
	}
	return es
}

func callID(stepIdx, i int) string { return fmt.Sprintf("c%d_%d", stepIdx, i) }

func inputFor(kind int, id string) json.RawMessage {
	switch kind {
	case 0:
		return json.RawMessage(fmt.Sprintf(`{"text":%q}`, id))
	case 1:
		return json.RawMessage(fmt.Sprintf(`{"text":%q,"x":1}`, id))
	case 2:
		return json.RawMessage(fmt.Sprintf(`{"text":7,"id":%q}`, id))
	case 3:
		return json.RawMessage(fmt.Sprintf(`[%q]`, id))
	case 4:
		return json.RawMessage(fmt.Sprintf(`{"text":%q`, id)) // not JSON
	case 5:
		return nil
	}
	return json.RawMessage(`null`)
}

// respond is the model. It checks the request, then answers from the script.
func (sc *Scenario) respond(req *agentloop.Request) (*agentloop.Response, error) {
	t := sc.t
	sc.checkRequest(req)

	sc.mu.Lock()
	idx := sc.modelIdx
	sc.modelIdx++
	sc.mu.Unlock()

	if idx >= len(sc.cfg.Steps) {
		return sc.remember(sc.native(fake.Text("done"), idx)), nil
	}
	st := sc.cfg.Steps[idx]
	switch st.Kind {
	case StHang:
		if sc.cancel != nil {
			sc.fire()
			return nil, context.Canceled
		}
		return sc.remember(sc.native(fake.Text("done"), idx)), nil
	case StNil:
		return nil, nil
	case StErr:
		if st.Plain {
			return nil, fmt.Errorf("plain failure %d", idx)
		}
		me := &agentloop.ModelError{Kind: st.ErrKind, Message: fmt.Sprintf("step %d", idx)}
		if st.RetryAfter != 0 {
			me.RetryAfter = sc.clock.Add(time.Duration(st.RetryAfter) * time.Second)
		}
		return nil, me
	}
	var resp *agentloop.Response
	if st.Kind == StText {
		resp = fake.Text(st.Text)
		resp.Stop = st.Stop
	} else {
		calls := make([]agentloop.ToolCall, len(st.Calls))
		for i, c := range st.Calls {
			id := callID(idx, i)
			calls[i] = agentloop.ToolCall{ID: id, Name: c.Name, Input: inputFor(c.Input, id)}
		}
		switch {
		case st.BadID == 1 && len(calls) > 1:
			calls[1].ID = calls[0].ID
		case st.BadID == 2 && len(calls) > 0:
			calls[len(calls)-1].ID = ""
		case st.BadID == 3 && len(calls) > 0 && idx > 0:
			// An id an earlier step used: a repeat if that call was
			// recorded, otherwise just an unusual but valid id.
			calls[0].ID = callID(idx-1, 0)
		}
		resp = fake.ToolUse(st.Text, calls...)
		resp.Stop = st.Stop
	}
	if st.Reasoning {
		resp.Message.Parts = append([]agentloop.Part{{Type: agentloop.ReasoningPart, Text: "hmm"}}, resp.Message.Parts...)
	}
	resp.Usage = agentloop.Usage{InputTokens: st.In, OutputTokens: st.Out}
	if resp.Stop == agentloop.StopRefusal {
		resp.Refusal = "no"
	}
	_ = t
	return sc.remember(sc.native(resp, idx)), nil
}

// native sets the provider record of a turn: compact JSON, or, for the
// wild runs, JSON with whitespace, HTML characters and escapes.
func (sc *Scenario) native(r *agentloop.Response, idx int) *agentloop.Response {
	if sc.faults.Wild {
		data := fmt.Sprintf("{ \"i\" : %d ,\n \"s\" : \"a<b>&c \\u00e9 \\u2028\",  \"sig\":\"AB==\" }", idx)
		r.Message.Native = &agentloop.Native{Provider: "fake", Model: "fake-model", Data: json.RawMessage(data)}
	} else {
		r.Message.Native = &agentloop.Native{Provider: "fake", Model: "fake-model", Data: json.RawMessage(fmt.Sprintf(`{"i":%d,"sig":"AB=="}`, idx))}
	}
	return r
}

func (sc *Scenario) remember(r *agentloop.Response) *agentloop.Response {
	sc.mu.Lock()
	sc.returned = append(sc.returned, deepCopyMessage(sc.t, r.Message))
	sc.mu.Unlock()
	return r
}

func (sc *Scenario) fire() {
	sc.cancelFired = true
	sc.cancel()
}

func deepCopyMessage(t *rapid.T, m agentloop.Message) agentloop.Message {
	out := m
	if m.Parts != nil {
		out.Parts = make([]agentloop.Part, len(m.Parts))
	}
	for i, p := range m.Parts {
		out.Parts[i] = p
		if p.ToolCall != nil {
			c := *p.ToolCall
			c.Input = bytes.Clone(c.Input)
			out.Parts[i].ToolCall = &c
		}
	}
	if m.Native != nil {
		n := *m.Native
		n.Data = bytes.Clone(n.Data)
		out.Native = &n
	}
	return out
}

// canon is what the log holds for a message the model made: tool inputs
// made storable, and everything passed through JSON once.
func canon(t *rapid.T, m agentloop.Message) agentloop.Message {
	m = deepCopyMessage(t, m)
	for i, p := range m.Parts {
		if p.ToolCall == nil {
			continue
		}
		switch {
		case len(p.ToolCall.Input) == 0:
			m.Parts[i].ToolCall.Input = json.RawMessage(`{}`)
		case !json.Valid(p.ToolCall.Input):
			m.Parts[i].ToolCall.Input, _ = json.Marshal(string(p.ToolCall.Input))
		}
	}
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("a model message cannot be marshaled: %v", err)
	}
	var out agentloop.Message
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func assistants(msgs []agentloop.Message) []agentloop.Message {
	var out []agentloop.Message
	for _, m := range msgs {
		if m.Role == agentloop.Assistant {
			out = append(out, m)
		}
	}
	return out
}

// checkRequest holds properties 3 and 4: what the model is sent is a
// valid conversation, and the assistant turns in it are the ones the model
// made, unchanged.
func (sc *Scenario) checkRequest(req *agentloop.Request) {
	t := sc.t
	msgs := req.Messages
	if len(msgs) == 0 || msgs[0].Role != agentloop.User {
		t.Fatalf("the first message is not from the user: %+v", msgs)
	}
	known := map[string]bool{}
	for i, m := range msgs {
		if m.Role == agentloop.Assistant {
			calls := m.ToolCalls()
			for _, c := range calls {
				if known[c.ID] {
					t.Fatalf("call id %s appears twice", c.ID)
				}
				known[c.ID] = true
			}
			if len(calls) == 0 {
				continue
			}
			if i+1 >= len(msgs) {
				t.Fatalf("message %d has calls with no results after it", i)
			}
			next := msgs[i+1]
			if next.Role != agentloop.User || len(next.Parts) != len(calls) {
				t.Fatalf("calls %+v are not answered by exactly their results: %+v", calls, next)
			}
			for j, c := range calls {
				p := next.Parts[j]
				if p.Type != agentloop.ToolResultPart || p.ToolResult == nil || p.ToolResult.CallID != c.ID {
					t.Fatalf("result %d should answer %s: %+v", j, c.ID, p)
				}
			}
			continue
		}
		for _, p := range m.Parts {
			if p.Type == agentloop.ToolResultPart {
				if i == 0 || len(msgs[i-1].ToolCalls()) == 0 {
					t.Fatalf("a result at message %d follows no calls", i)
				}
				found := false
				for _, c := range msgs[i-1].ToolCalls() {
					found = found || c.ID == p.ToolResult.CallID
				}
				if !found {
					t.Fatalf("a result references the unknown call %s", p.ToolResult.CallID)
				}
			}
		}
	}

	// Replay fidelity.
	got := assistants(msgs)
	var persisted []agentloop.Message
	for _, e := range sc.entries() {
		if e.Type == agentloop.EntryMessage && e.Message.Message.Role == agentloop.Assistant {
			persisted = append(persisted, e.Message.Message)
		}
	}
	if len(got) != len(persisted) {
		t.Fatalf("the request has %d assistant turns, the log %d", len(got), len(persisted))
	}
	sc.mu.Lock()
	returned := sc.returned
	sc.mu.Unlock()
	j := 0
	for i, g := range got {
		if !reflect.DeepEqual(g, persisted[i]) {
			t.Fatalf("assistant turn %d differs from the log:\nrequest %+v\nlog     %+v", i, g, persisted[i])
		}
		for j < len(returned) && !reflect.DeepEqual(g, canon(t, returned[j])) {
			j++
		}
		if j == len(returned) {
			t.Fatalf("assistant turn %d is not a turn the model made, unchanged: %+v", i, g)
		}
		if g.Native == nil || !bytes.Equal(g.Native.Data, canon(t, returned[j]).Native.Data) {
			t.Fatalf("native bytes of turn %d changed", i)
		}
		j++
	}
}

// ---- tools ----

func (sc *Scenario) validFor(name string, raw json.RawMessage) bool {
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		return false
	}
	obj, ok := v.(map[string]any)
	switch name {
	case "echo":
		if !ok || len(obj) != 1 {
			return false
		}
		_, isStr := obj["text"].(string)
		return isStr
	case "lab":
		if !ok {
			return false
		}
		_, isStr := obj["text"].(string)
		return isStr
	case "any":
		return ok
	}
	return false
}

func (sc *Scenario) tools() []agentloop.Tool {
	var out []agentloop.Tool
	for _, name := range []string{"echo", "any", "lab"} {
		name := name
		spec := agentloop.ToolSpec{Name: name, Description: name}
		switch name {
		case "echo":
			spec.Parameters = json.RawMessage(`{"type":"object","properties":{"text":{"type":"string"}},"required":["text"],"additionalProperties":false}`)
		case "lab":
			spec.Parameters = json.RawMessage(`{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}`)
		}
		out = append(out, agentloop.Func(spec, func(ctx context.Context, in json.RawMessage) (agentloop.Result, error) {
			return sc.callTool(ctx, name, in)
		}))
	}
	return out
}

// callTool is where properties 1 and 2 are checked, from the persisted log
// alone: the call must have been allowed, and started, durably, just now.
func (sc *Scenario) callTool(ctx context.Context, name string, in json.RawMessage) (agentloop.Result, error) {
	t := sc.t
	es := sc.entries()
	last := es[len(es)-1]
	if last.Type != agentloop.EntryCallStarted {
		t.Fatalf("a tool ran without call_started as the last durable entry (last: %s)", last.Type)
	}
	id := last.CallStarted.CallID
	var verdict agentloop.Verdict
	decisions := 0
	for _, e := range es {
		if e.Type == agentloop.EntryDecision && e.Decision.CallID == id {
			verdict = e.Decision.Verdict
			decisions++
		}
		if e.Type == agentloop.EntryCallStarted && e.CallStarted.CallID == id && e.Seq != last.Seq {
			t.Fatalf("call %s started twice", id)
		}
	}
	if verdict != agentloop.Allow {
		t.Fatalf("call %s reached its tool with decision %q (%d decisions)", id, verdict, decisions)
	}
	sc.mu.Lock()
	sc.toolRuns[id]++
	n := sc.toolRuns[id]
	sc.mu.Unlock()
	if n > 1 {
		t.Fatalf("call %s ran %d times", id, n)
	}
	var call *agentloop.ToolCall
	for _, e := range es {
		if e.Type == agentloop.EntryMessage && e.Message.Message.Role == agentloop.Assistant {
			for _, c := range e.Message.Message.ToolCalls() {
				if c.ID == id {
					c := c
					call = &c
				}
			}
		}
	}
	if call == nil || call.Name != name || !bytes.Equal(call.Input, in) {
		t.Fatalf("call %s ran as %s with %q, but the log has %+v", id, name, in, call)
	}
	if !sc.validFor(name, in) {
		t.Fatalf("call %s reached %s with input that does not match its schema: %q", id, name, in)
	}

	b := sc.cfg.Behaviors[name]
	labels := append([]string(nil), b.Labels...)
	sc.handed = append(sc.handed, labels)
	switch b.Kind {
	case BIsError:
		return agentloop.Result{Content: "bad:" + id, IsError: true, Labels: labels}, nil
	case BGoError:
		return agentloop.Result{Content: "ignored", Labels: labels}, errors.New("boom:" + id)
	case BCancelThenOK:
		if sc.cancel != nil {
			sc.fire()
		}
	case BBlockUntilCancel:
		if sc.cancel != nil {
			sc.fire()
			<-ctx.Done()
			return agentloop.Result{}, ctx.Err()
		}
	case BBlockUntilTimeout:
		if sc.cfg.ToolTimeout > 0 && !sc.draining {
			<-ctx.Done()
			return agentloop.Result{}, ctx.Err()
		}
	}
	return agentloop.Result{Content: "ok:" + id, Labels: labels}, nil
}

// ---- policy ----

func (sc *Scenario) policy() agentloop.Policy {
	if sc.cfg.NilPolicy {
		return nil
	}
	return agentloop.PolicyFunc(func(ctx context.Context, call agentloop.ToolCall, v agentloop.View) agentloop.Decision {
		t := sc.t
		es := sc.entries()
		want := oracleLabels(es)
		if !reflect.DeepEqual(append([]string{}, v.Labels...), want) {
			t.Fatalf("the policy saw labels %v, the log holds %v", v.Labels, want)
		}
		if !sort.StringsAreSorted(v.Labels) {
			t.Fatalf("labels are not sorted: %v", v.Labels)
		}
		if n := oracleDecided(es); v.ToolCalls != n {
			t.Fatalf("the policy saw %d decided calls, the log holds %d", v.ToolCalls, n)
		}
		if sc.validFor(call.Name, call.Input) == false {
			t.Fatalf("the policy was asked about call %s that fails its checks: %+v", call.ID, call)
		}
		r := sc.cfg.Rules[call.Name]
		d := agentloop.Decision{Verdict: r.Base, Reason: "because " + call.ID, Labels: append([]string(nil), r.Labels...)}
		if r.IfLabel != "" && v.Has(r.IfLabel) {
			d.Verdict = r.Then
			d.Labels = append([]string(nil), r.ThenLabels...)
		}
		sc.handed = append(sc.handed, d.Labels)
		sc.mu.Lock()
		sc.asked[call.ID] = d.Verdict
		sc.mu.Unlock()
		return d
	})
}

func oracleLabels(es []agentloop.Entry) []string {
	set := map[string]bool{}
	for _, e := range es {
		switch e.Type {
		case agentloop.EntryDecision:
			if e.Decision.Verdict == agentloop.Allow {
				for _, l := range e.Decision.Labels {
					set[l] = true
				}
			}
		case agentloop.EntryToolResult:
			for _, l := range e.ToolResult.Labels {
				set[l] = true
			}
		case agentloop.EntryLabels:
			for _, l := range e.Labels.Labels {
				set[l] = true
			}
		}
	}
	out := []string{}
	for l := range set {
		out = append(out, l)
	}
	sort.Strings(out)
	return out
}

func oracleDecided(es []agentloop.Entry) int {
	final := map[string]bool{}
	for _, e := range es {
		if e.Type == agentloop.EntryDecision && e.Decision.Verdict != agentloop.Hold {
			final[e.Decision.CallID] = true
		}
	}
	return len(final)
}

// ---- derived state = replayed state, append-only ----

func (sc *Scenario) checkSession(s *agentloop.Session) {
	t := sc.t
	lines := sc.log().Snapshot()
	for i, l := range sc.prevLines {
		if i >= len(lines) || !bytes.Equal(l, lines[i]) {
			t.Fatalf("persisted entry %d changed", i+1)
		}
	}
	sc.prevLines = lines

	live := s.Entries()
	if len(live) != len(lines) {
		t.Fatalf("the session holds %d entries, the store %d", len(live), len(lines))
	}
	for i, e := range live {
		if e.Seq != i+1 {
			t.Fatalf("entry %d has seq %d", i, e.Seq)
		}
		b, _ := json.Marshal(e)
		if !bytes.Equal(b, lines[i]) {
			t.Fatalf("live entry %d differs from the persisted one:\nlive %s\nlog  %s", i+1, b, lines[i])
		}
	}

	r, err := agentloop.OpenSession(sc.log())
	if err != nil {
		t.Fatalf("the persisted log does not replay: %v", err)
	}
	eq := func(what string, a, b any) {
		if !reflect.DeepEqual(a, b) {
			t.Fatalf("%s differ after %d entries:\nlive     %+v\nreplayed %+v", what, len(live), a, b)
		}
	}
	eq("entries", s.Entries(), r.Entries())
	eq("messages", s.Messages(), r.Messages())
	eq("labels", s.Labels(), r.Labels())
	eq("usage", s.Usage(), r.Usage())
	eq("held", s.Held(), r.Held())
	eq("pending", s.Pending(), r.Pending())
	eq("turns", s.Turns(), r.Turns())
	eq("view", s.View(), r.View())
	for _, turn := range s.Turns() {
		for _, id := range turn.Calls {
			a, _ := s.Call(id)
			b, ok := r.Call(id)
			if !ok {
				t.Fatalf("replay lost call %s", id)
			}
			eq("call "+id, a, b)
		}
	}

	// Labels only grow, and match the log.
	labels := s.Labels()
	for _, l := range sc.prevSeen {
		if !s.View().Has(l) {
			t.Fatalf("label %s was removed", l)
		}
	}
	sc.prevSeen = labels
	eq("labels vs oracle", append([]string{}, labels...), oracleLabels(sc.entries()))
}

// ---- driving ----

func (sc *Scenario) agent() *agentloop.Agent {
	a := &agentloop.Agent{
		Model:       sc.model,
		ModelID:     "fake-model",
		System:      "sys",
		Tools:       sc.tools(),
		Policy:      sc.policy(),
		Limits:      sc.cfg.Limits,
		Retry:       sc.cfg.Retry,
		ToolTimeout: sc.cfg.ToolTimeout,
		Now:         func() time.Time { return sc.clock },
		Sleep: func(ctx context.Context, d time.Duration) error {
			sc.waits = append(sc.waits, d)
			if sc.cancel != nil && !sc.cancelFired && len(sc.waits)%2 == 0 {
				sc.fire()
			}
			return ctx.Err()
		},
	}
	return a
}

// Agent is the scenario's agent, for running its session somewhere else
// after HandOver.
func (sc *Scenario) Agent() *agentloop.Agent { return sc.agent() }

func (sc *Scenario) open() *agentloop.Session {
	s, err := agentloop.OpenSession(sc.log())
	if err != nil {
		sc.t.Fatalf("a persisted log does not reopen: %v", err)
	}
	sc.checkSession(s)
	return s
}

// iterate runs once: fresh session from what was persisted, fresh agent.
// It returns the outcome, or nil if the run crashed.
func (sc *Scenario) iterate(prompt string, crashAt int, after bool, cancelMode int) *agentloop.Outcome {
	t := sc.t
	s := sc.open()
	a := sc.agent()
	a.Observe = func(ev agentloop.Event) {
		if ev.Entry != nil {
			sc.checkSession(s)
		}
		if sc.Observe != nil {
			sc.Observe(ev)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sc.cancelFired, sc.preCanceled, sc.cancel, sc.waits = false, false, nil, nil
	switch cancelMode {
	case 1:
		cancel()
		sc.preCanceled = true
	case 2:
		sc.cancel = cancel
	}
	sc.Store.arm(crashAt, after)
	out, err := a.Run(ctx, s, prompt)
	sc.Store.arm(0, false)
	// The armed cancel belongs to this run. Left set, it would make the
	// tools and the model of a later run outside iterate (an agent from
	// Agent, run by a server) "cancel" a context that is not theirs and
	// then wait for their own forever.
	sc.cancel = nil
	if err != nil {
		if !errors.Is(err, ErrCrash) {
			t.Fatalf("Run failed for a reason other than the crash: %v", err)
		}
		return nil
	}
	sc.checkRun(s, a, out, prompt)
	// A caller may do what it likes with what it was handed; the session
	// must not notice.
	out.Text, out.Turns, out.Held, out.Error = "MUTATED", -1, nil, nil
	for _, l := range sc.handed {
		for i := range l {
			l[i] = "MUT"
		}
	}
	sc.handed = nil
	sc.checkSession(s)
	return out
}

func (sc *Scenario) retryPolicy() agentloop.RetryPolicy {
	if sc.cfg.Retry != nil {
		return *sc.cfg.Retry
	}
	return agentloop.DefaultRetry
}

// checkRun holds the limit, outcome and retry properties for one finished run.
func (sc *Scenario) checkRun(s *agentloop.Session, a *agentloop.Agent, out *agentloop.Outcome, prompt string) {
	t := sc.t
	es := sc.entries()
	start := -1
	for i, e := range es {
		if e.Type == agentloop.EntryRunStarted {
			start = i
		}
	}
	if start < 0 {
		t.Fatal("no run_started")
	}
	run := es[start:]
	if run[0].RunStarted.Prompt != prompt {
		t.Fatalf("run_started has prompt %q, want %q", run[0].RunStarted.Prompt, prompt)
	}
	endE := run[len(run)-1]
	if endE.Type != agentloop.EntryRunEnded {
		t.Fatalf("the last entry is %s, not run_ended", endE.Type)
	}
	for _, e := range run[1 : len(run)-1] {
		if e.Type == agentloop.EntryRunEnded || e.Type == agentloop.EntryRunStarted {
			t.Fatalf("run markers inside a run")
		}
	}
	ob, _ := json.Marshal(out)
	eb, _ := json.Marshal(endE.RunEnded)
	if !bytes.Equal(ob, eb) {
		t.Fatalf("run_ended %s differs from the outcome %s", eb, ob)
	}

	turns, started := 0, 0
	var usage agentloop.Usage
	var before agentloop.Usage // usage before the last response
	limitMsg := false
	for _, e := range run {
		switch e.Type {
		case agentloop.EntryMessage:
			if e.Message.Message.Role == agentloop.Assistant {
				turns++
				before = usage
				usage.InputTokens += e.Message.Usage.InputTokens
				usage.OutputTokens += e.Message.Usage.OutputTokens
				if lim := a.Limits.MaxTokens; lim > 0 && before.Total() >= lim {
					t.Fatalf("the model was called with %d tokens used, limit %d", before.Total(), lim)
				}
			}
		case agentloop.EntryCallStarted:
			started++
		case agentloop.EntryToolResult:
			limitMsg = limitMsg || strings.Contains(e.ToolResult.Result.Content, "limit on tool calls")
		}
	}
	if turns != out.Turns || usage != out.Usage {
		t.Fatalf("outcome counts %d turns %+v, the log %d %+v", out.Turns, out.Usage, turns, usage)
	}
	l := a.Limits
	if l.MaxTurns > 0 && turns > l.MaxTurns {
		t.Fatalf("%d model responses, limit %d", turns, l.MaxTurns)
	}
	if l.MaxToolCalls > 0 && started > l.MaxToolCalls {
		t.Fatalf("%d calls started, limit %d", started, l.MaxToolCalls)
	}

	turnsAll := s.Turns()
	var lastStop agentloop.Stop
	if len(turnsAll) > 0 {
		lastStop = turnsAll[len(turnsAll)-1].Stop
	}
	switch out.Reason {
	case agentloop.ReasonLimit:
		switch out.Limit {
		case agentloop.LimitTurns:
			if l.MaxTurns == 0 || turns != l.MaxTurns {
				t.Fatalf("limit max_turns with %d turns of %d", turns, l.MaxTurns)
			}
		case agentloop.LimitToolCalls:
			if l.MaxToolCalls == 0 || started != l.MaxToolCalls || !limitMsg {
				t.Fatalf("limit max_tool_calls with %d started of %d, refused=%v", started, l.MaxToolCalls, limitMsg)
			}
		case agentloop.LimitTokens:
			if l.MaxTokens == 0 || usage.Total() < l.MaxTokens {
				t.Fatalf("limit max_tokens with %d of %d", usage.Total(), l.MaxTokens)
			}
		default:
			t.Fatalf("limit %q", out.Limit)
		}
	case agentloop.ReasonMaxTokens:
		if lastStop != agentloop.StopMaxTokens {
			t.Fatalf("max_tokens outcome after a %s turn", lastStop)
		}
	case agentloop.ReasonRefused:
		if lastStop != agentloop.StopRefusal {
			t.Fatalf("refused outcome after a %s turn", lastStop)
		}
	case agentloop.ReasonHeld:
		held := s.Held()
		var outHeld []agentloop.ToolCall
		for _, h := range out.Held {
			outHeld = append(outHeld, h.ToolCall)
			if h.InputSHA256 != agentloop.InputDigest(h.Input) {
				t.Fatalf("held call %s has digest %q", h.ID, h.InputSHA256)
			}
		}
		if len(held) == 0 || !reflect.DeepEqual(held, outHeld) {
			t.Fatalf("held outcome %+v, session holds %+v", out.Held, held)
		}
		for _, c := range held {
			info, _ := s.Call(c.ID)
			if !info.Held() || info.Started {
				t.Fatalf("%+v", info)
			}
		}
	case agentloop.ReasonModelError:
		prev := run[len(run)-2]
		if out.Error != nil && out.Error.Kind == agentloop.InvalidRequest &&
			(out.Error.Message == "the model reused a tool call id" || out.Error.Message == "the model returned a tool call with no id") {
			// A bad call id ends the run with nothing recorded for the turn.
			if prev.Type == agentloop.EntryMessage && prev.Message.Message.Role == agentloop.Assistant {
				for _, c := range prev.Message.Message.ToolCalls() {
					if c.ID == "" {
						t.Fatalf("a tool call with no id was recorded")
					}
				}
			}
			break
		}
		if out.Error == nil || prev.Type != agentloop.EntryModelError || prev.ModelError.Retry || prev.ModelError.Error.Kind != out.Error.Kind {
			t.Fatalf("model_error outcome %+v after %s", out.Error, prev.Type)
		}
	case agentloop.ReasonCompleted:
		if len(s.Held()) != 0 {
			t.Fatal("completed with held calls")
		}
		if lastStop == agentloop.StopMaxTokens || lastStop == agentloop.StopRefusal {
			if turns > 0 {
				t.Fatalf("completed after a %s turn", lastStop)
			}
		}
	case agentloop.ReasonCanceled:
		if !sc.cancelFired && !sc.preCanceled {
			t.Fatal("canceled though nothing canceled the run")
		}
	default:
		t.Fatalf("reason %q", out.Reason)
	}
	if (sc.cancelFired || sc.preCanceled) && out.Reason != agentloop.ReasonCanceled {
		// A pre-canceled run with nothing to do may still complete.
		if !(sc.preCanceled && !sc.cancelFired && (out.Reason == agentloop.ReasonCompleted || out.Reason == agentloop.ReasonHeld ||
			out.Reason == agentloop.ReasonMaxTokens || out.Reason == agentloop.ReasonRefused)) {
			t.Fatalf("the run was canceled but ended %s", out.Reason)
		}
	}
	msgs := s.Messages()
	want := ""
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == agentloop.Assistant {
			want = msgs[i].Text()
			break
		}
	}
	if out.Text != want {
		t.Fatalf("outcome text %q, last assistant text %q", out.Text, want)
	}

	sc.checkAtRest(s, out)
	sc.checkRetries(run, a)
}

// checkAtRest holds when a run ends: unless calls are held, the session is
// a request a provider accepts, with every call answered by its result. A
// canceled run may stop part way through a tool_use turn, but never leaves
// a cut-off turn's calls open.
func (sc *Scenario) checkAtRest(s *agentloop.Session, out *agentloop.Outcome) {
	t := sc.t
	if out.Reason == agentloop.ReasonHeld {
		return
	}
	if out.Reason == agentloop.ReasonCanceled {
		turns := s.Turns()
		if len(turns) == 0 || turns[len(turns)-1].Stop == agentloop.StopToolUse {
			return
		}
	}
	if p := s.Pending(); len(p) != 0 {
		t.Fatalf("the run ended %s with unanswered calls %+v", out.Reason, p)
	}
	msgs := s.Messages()
	for i, m := range msgs {
		calls := m.ToolCalls()
		if m.Role != agentloop.Assistant || len(calls) == 0 {
			continue
		}
		if i+1 >= len(msgs) || len(msgs[i+1].Parts) != len(calls) {
			t.Fatalf("the run ended %s, but the calls of message %d are not all answered", out.Reason, i)
		}
		for j, c := range calls {
			if p := msgs[i+1].Parts[j]; msgs[i+1].Role != agentloop.User || p.ToolResult == nil || p.ToolResult.CallID != c.ID {
				t.Fatalf("call %s of message %d is not answered in place", c.ID, i)
			}
		}
	}
}

// checkRetries holds the retry rules of DESIGN.md against the log.
func (sc *Scenario) checkRetries(run []agentloop.Entry, a *agentloop.Agent) {
	t := sc.t
	p := sc.retryPolicy()
	attempt := 0
	var waits []time.Duration
	for _, e := range run {
		switch e.Type {
		case agentloop.EntryMessage, agentloop.EntryRunStarted:
			attempt = 0
		case agentloop.EntryModelError:
			me := e.ModelError.Error
			retry := me.Transient() && attempt < p.Retries
			var wait time.Duration
			if retry {
				backoff := p.Backoff << attempt
				var until time.Duration
				if !me.RetryAfter.IsZero() {
					until = me.RetryAfter.Sub(sc.clock)
				}
				wait = backoff
				if me.Kind == agentloop.RateLimited && !me.RetryAfter.IsZero() {
					wait = until
				} else if until > wait {
					wait = until
				}
				wait = max(wait, 0)
				if p.MaxWait > 0 && wait > p.MaxWait {
					retry = false
				}
			}
			if e.ModelError.Retry != retry {
				t.Fatalf("retry=%v for %s after %d attempts under %+v (wait %v)", e.ModelError.Retry, me.Kind, attempt, p, wait)
			}
			if retry {
				if got := time.Duration(e.ModelError.WaitMillis) * time.Millisecond; got != wait {
					t.Fatalf("recorded wait %v, want %v", got, wait)
				}
				if p.MaxWait > 0 && wait > p.MaxWait {
					t.Fatalf("wait %v beyond the cap %v", wait, p.MaxWait)
				}
				waits = append(waits, wait)
			}
			attempt++
		}
	}
	if !reflect.DeepEqual(waits, sc.waits) && !(len(waits) == 0 && len(sc.waits) == 0) {
		t.Fatalf("slept %v, the log says %v", sc.waits, waits)
	}
}

// resolveHeld settles the session's held calls as a person might.
func (sc *Scenario) resolveHeld(s *agentloop.Session, held []agentloop.ToolCall, crashAt int) {
	t := sc.t
	for _, c := range held {
		choice := 0
		if !sc.resolveAll {
			choice = rapid.IntRange(0, 3).Draw(t, "resolve")
		}
		if choice == 3 {
			continue // leave it held
		}
		d := agentloop.Decision{Verdict: agentloop.Allow, Reason: "ok by person"}
		if choice == 1 {
			d = agentloop.Decision{Verdict: agentloop.Deny, Reason: "no by person"}
		}
		d.Labels = append([]string(nil), genLabels().Draw(t, "resolvelabels")...)
		sc.handed = append(sc.handed, d.Labels)
		if err := s.Resolve(c.ID, d); err != nil {
			t.Fatalf("Resolve: %v", err)
		}
		sc.checkSession(s)
		if err := s.Resolve(c.ID, d); err == nil {
			t.Fatalf("resolving %s twice succeeded", c.ID)
		}
	}
}

// finalChecks holds for a log at rest, whatever happened to produce it.
func (sc *Scenario) finalChecks(s *agentloop.Session, complete bool) {
	t := sc.t
	es := sc.entries()
	decisions := map[string][]agentloop.DecisionEntry{}
	results := map[string][]agentloop.ToolResultEntry{}
	started := map[string]bool{}
	for i, e := range es {
		switch e.Type {
		case agentloop.EntryDecision:
			decisions[e.Decision.CallID] = append(decisions[e.Decision.CallID], *e.Decision)
		case agentloop.EntryCallStarted:
			id := e.CallStarted.CallID
			ds := decisions[id]
			if len(ds) == 0 || ds[len(ds)-1].Verdict != agentloop.Allow {
				t.Fatalf("call %s started without an allow decision before it", id)
			}
			started[id] = true
		case agentloop.EntryToolResult:
			id := e.ToolResult.Result.CallID
			results[id] = append(results[id], *e.ToolResult)
			if len(results[id]) > 1 {
				t.Fatalf("call %s has two results", id)
			}
			_ = i
		}
	}
	for id, ds := range decisions {
		// A policy's allow can be followed by another decision only when
		// the call never started (a stale allow, decided again); a hold
		// only by a person's allow or deny; nothing follows a deny or a
		// person's decision.
		for i, d := range ds {
			if d.Resolved != (i > 0 && ds[i-1].Verdict == agentloop.Hold) {
				t.Fatalf("call %s decisions %+v", id, ds)
			}
			if i+1 < len(ds) && (d.Verdict == agentloop.Deny || d.Resolved) {
				t.Fatalf("call %s decisions %+v", id, ds)
			}
			if d.Resolved && d.Verdict == agentloop.Hold {
				t.Fatalf("call %s resolved with hold", id)
			}
			if d.Verdict == agentloop.Hold && (d.InputSHA256 == "" || len(d.Labels) > 0) {
				t.Fatalf("call %s hold %+v", id, d)
			}
		}
		last := ds[len(ds)-1]
		if r := results[id]; len(r) == 1 && last.Verdict == agentloop.Deny {
			want := last.Reason
			if want == "" {
				want = "The call was denied."
			}
			if r[0].Result.Content != want || !r[0].Result.IsError {
				t.Fatalf("denied call %s got result %+v, want %q", id, r[0].Result, want)
			}
		}
		if len(results[id]) == 1 && last.Verdict == agentloop.Hold {
			t.Fatalf("held call %s has a result", id)
		}
	}
	// Calls of a turn that did not end in tool_use were cut off: nothing
	// about them is decided or run, whatever happens on resume.
	for _, turn := range s.Turns() {
		if turn.Stop == agentloop.StopToolUse {
			continue
		}
		for _, id := range turn.Calls {
			if len(decisions[id]) > 0 || started[id] || sc.toolRuns[id] > 0 {
				t.Fatalf("call %s of a %s turn was decided, started or run", id, turn.Stop)
			}
			if r := results[id]; len(r) == 1 && (!r[0].Result.IsError || !strings.Contains(r[0].Result.Content, "was not run")) {
				t.Fatalf("cut-off call %s got result %+v", id, r[0].Result)
			}
		}
	}
	// A call that ran got its own result or the honest account of a stop.
	for id, n := range sc.toolRuns {
		if n != 1 || !started[id] {
			t.Fatalf("call %s ran %d times, started=%v", id, n, started[id])
		}
		if r := results[id]; len(r) == 1 {
			c := r[0].Result.Content
			if !strings.Contains(c, id) && !strings.Contains(c, "may or may not have taken effect") &&
				!strings.Contains(c, "timed out") {
				t.Fatalf("call %s ran but its result is %q", id, c)
			}
		}
	}
	for id := range started {
		if sc.toolRuns[id] == 0 {
			// Started and never run: only a crash between the two, and
			// then the call must be settled as unknown, never rerun.
			if r := results[id]; len(r) == 1 && !strings.Contains(r[0].Result.Content, "may or may not have taken effect") {
				t.Fatalf("call %s never ran but has result %+v", id, r[0].Result)
			}
		}
	}
	if complete {
		turns := s.Turns()
		if len(s.Held()) != 0 || len(s.Pending()) != 0 {
			t.Fatalf("pending %v held %v at the end", s.Pending(), s.Held())
		}
		for ti, turn := range turns {
			_ = ti
			for _, id := range turn.Calls {
				if len(results[id]) != 1 {
					t.Fatalf("call %s has %d results at the end", id, len(results[id]))
				}
			}
		}
	}
}

// Execute drives the scenario: the faulty iterations, then a clean drain
// that must reach `completed`.
func (sc *Scenario) Execute() {
	t := sc.t
	sc.Faulty()
	sc.PrepareDrain()
	var final *agentloop.Outcome
	for i := 0; i < sc.DrainRuns(); i++ {
		out := sc.iterate("continue", 0, false, 0)
		if out.Reason == agentloop.ReasonHeld {
			s := sc.open()
			sc.resolveHeld(s, s.Held(), 0)
			continue
		}
		if out.Reason == agentloop.ReasonCompleted {
			final = out
			break
		}
	}
	if final == nil {
		t.Fatalf("the conversation never completed; entries:\n%s", dump(sc.entries()))
	}
	sc.CheckAtRest(true)
}

// Faulty runs the scenario's iterations with the faults it was made with,
// each from what the last one left in the log.
func (sc *Scenario) Faulty() {
	t := sc.t
	prompt := "go"
	for i := 0; i < sc.cfg.Iterations; i++ {
		crashAt, cancelMode := 0, 0
		after := false
		if sc.faults.Crash && rapid.IntRange(0, 2).Draw(t, "crash?") > 0 {
			crashAt = rapid.IntRange(1, 25).Draw(t, "crashAt")
			after = rapid.Bool().Draw(t, "crashAfter")
		}
		if sc.faults.Cancel {
			cancelMode = rapid.SampledFrom([]int{0, 0, 1, 2, 2}).Draw(t, "cancelMode")
		}
		out := sc.iterate(prompt, crashAt, after, cancelMode)
		prompt = rapid.SampledFrom([]string{"", "", "more"}).Draw(t, "prompt")
		if out != nil && out.Reason == agentloop.ReasonHeld {
			s := sc.open()
			sc.resolveHeld(s, s.Held(), 0)
		}
	}
}

// PrepareDrain turns the faults off for the runs that follow: no limits,
// enough retries for every scripted error, no tool timeout, and holds
// approved when the scenario resolves them. The script is finite, so the
// conversation completes within DrainRuns runs unless a call stays held.
// Agents made after this have the drain's settings.
func (sc *Scenario) PrepareDrain() {
	sc.draining, sc.resolveAll = true, true
	sc.cfg.Limits = agentloop.Limits{}
	sc.cfg.Retry = &agentloop.RetryPolicy{Retries: len(sc.cfg.Steps) + 1}
	sc.cfg.ToolTimeout = 0
}

// DrainRuns bounds the runs a drain needs to complete.
func (sc *Scenario) DrainRuns() int { return 3*len(sc.cfg.Steps) + 10 }

// CheckAtRest holds the invariants of a log at rest: it reloads to the
// same state, nothing persisted earlier changed, no call ran twice or
// without being allowed and started, and every result is honest. With
// complete, nothing is pending or held and every call has its result.
func (sc *Scenario) CheckAtRest(complete bool) {
	s := sc.open()
	sc.finalChecks(s, complete)
}

func dump(es []agentloop.Entry) string {
	var b strings.Builder
	if len(es) > 40 {
		es = es[len(es)-40:]
	}
	for _, e := range es {
		j, _ := json.Marshal(e)
		b.Write(j)
		b.WriteByte('\n')
	}
	return b.String()
}
