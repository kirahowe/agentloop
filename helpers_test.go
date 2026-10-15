package agentloop_test

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/fake"
)

// counter counts tool executions by what the tool was given.
type counter struct {
	mu    sync.Mutex
	runs  map[string]int
	total int
}

func newCounter() *counter { return &counter{runs: map[string]int{}} }

func (c *counter) hit(key string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.runs[key]++
	c.total++
}

func (c *counter) count(key string) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.runs[key]
}

func (c *counter) all() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.total
}

const echoSchema = `{"type":"object","properties":{"text":{"type":"string"}},"required":["text"],"additionalProperties":false}`

// echoTool answers "echo: <text>" and counts runs by text.
func echoTool(name string, c *counter) agentloop.Tool {
	return agentloop.Func(
		agentloop.ToolSpec{Name: name, Description: "echo", Parameters: json.RawMessage(echoSchema)},
		func(ctx context.Context, in json.RawMessage) (agentloop.Result, error) {
			var v struct{ Text string }
			json.Unmarshal(in, &v)
			c.hit(v.Text)
			return agentloop.Result{Content: "echo: " + v.Text}, nil
		})
}

func input(text string) string { return fmt.Sprintf(`{"text":%q}`, text) }

func newSession(t *testing.T) (*agentloop.Session, *agentloop.MemoryStore) {
	t.Helper()
	store := agentloop.NewMemoryStore()
	s, err := agentloop.OpenSession(store)
	if err != nil {
		t.Fatal(err)
	}
	return s, store
}

func reopen(t *testing.T, store agentloop.Store) *agentloop.Session {
	t.Helper()
	s, err := agentloop.OpenSession(store)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func run(t *testing.T, a *agentloop.Agent, s *agentloop.Session, prompt string) *agentloop.Outcome {
	t.Helper()
	out, err := a.Run(context.Background(), s, prompt)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	return out
}

func wantReason(t *testing.T, out *agentloop.Outcome, reason agentloop.Reason) {
	t.Helper()
	if out.Reason != reason {
		t.Fatalf("reason = %q, want %q (outcome %+v)", out.Reason, reason, out)
	}
}

func types(entries []agentloop.Entry) []agentloop.EntryType {
	var out []agentloop.EntryType
	for _, e := range entries {
		out = append(out, e.Type)
	}
	return out
}

func countType(entries []agentloop.Entry, typ agentloop.EntryType) int {
	n := 0
	for _, e := range entries {
		if e.Type == typ {
			n++
		}
	}
	return n
}

// resultsOf maps call ids to results in a conversation.
func resultsOf(msgs []agentloop.Message) map[string]agentloop.ToolResult {
	out := map[string]agentloop.ToolResult{}
	for _, m := range msgs {
		for _, p := range m.Parts {
			if p.Type == agentloop.ToolResultPart {
				out[p.ToolResult.CallID] = *p.ToolResult
			}
		}
	}
	return out
}

// checkConversation fails unless every assistant tool call is answered by
// exactly one result, in the user message right after it, in call order.
func checkConversation(t *testing.T, msgs []agentloop.Message) {
	t.Helper()
	for i, m := range msgs {
		calls := m.ToolCalls()
		if m.Role != agentloop.Assistant || len(calls) == 0 {
			if m.Role == agentloop.User {
				for _, p := range m.Parts {
					if p.Type == agentloop.ToolResultPart && (i == 0 || len(msgs[i-1].ToolCalls()) == 0) {
						t.Fatalf("message %d holds a result with no call before it", i)
					}
				}
			}
			continue
		}
		if i+1 >= len(msgs) {
			t.Fatalf("calls in message %d have no results", i)
		}
		next := msgs[i+1]
		if next.Role != agentloop.User || len(next.Parts) != len(calls) {
			t.Fatalf("message %d: want %d results after the calls, got %+v", i+1, len(calls), next)
		}
		for j, c := range calls {
			p := next.Parts[j]
			if p.Type != agentloop.ToolResultPart || p.ToolResult.CallID != c.ID {
				t.Fatalf("message %d part %d: want result for %s, got %+v", i+1, j, c.ID, p)
			}
		}
	}
}

// sleeper records waits instead of sleeping.
type sleeper struct {
	mu    sync.Mutex
	waits []time.Duration
}

func (s *sleeper) sleep(ctx context.Context, d time.Duration) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.waits = append(s.waits, d)
	return ctx.Err()
}

var epoch = time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)

// lastResultThenDone answers a tool-use turn with calls until results
// are in, then ends. It keeps working across resumes since it reads only
// the request.
func callsThenDone(calls ...agentloop.ToolCall) func(*agentloop.Request) (*agentloop.Response, error) {
	return func(req *agentloop.Request) (*agentloop.Response, error) {
		last := req.Messages[len(req.Messages)-1]
		if last.Role == agentloop.User && len(last.Parts) > 0 && last.Parts[0].Type == agentloop.ToolResultPart {
			return fake.Text("done"), nil
		}
		return fake.ToolUse("working", calls...), nil
	}
}

func contains(t *testing.T, got, want string) {
	t.Helper()
	if !strings.Contains(got, want) {
		t.Fatalf("%q does not contain %q", got, want)
	}
}

func jsonMarshal(v any) ([]byte, error) { return json.Marshal(v) }

// seqStore makes a store of hand-built entries, numbering them 1, 2, 3, ...
// as a session would have.
func seqStore(entries ...agentloop.Entry) *agentloop.MemoryStore {
	out := make([]agentloop.Entry, len(entries))
	for i, e := range entries {
		e.Seq = i + 1
		out[i] = e
	}
	return agentloop.NewMemoryStore(out...)
}
