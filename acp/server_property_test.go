package acpagent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	acp "github.com/caelis-labs/acp-go-sdk"
	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/fake"
)

// cancelWorld is a generated turn: a model script of text and tool steps,
// a policy that denies some calls, and a tool that fails some. Every tool
// execution is counted by call id.
type cancelWorld struct {
	steps []*agentloop.Response

	mu   sync.Mutex
	runs map[string]int
}

func genCancelWorld(t *rapid.T) *cancelWorld {
	w := &cancelWorld{runs: map[string]int{}}
	n := rapid.IntRange(1, 4).Draw(t, "steps")
	for i := 0; i < n; i++ {
		calls := rapid.IntRange(0, 3).Draw(t, fmt.Sprintf("calls%d", i))
		if calls == 0 || i == n-1 {
			w.steps = append(w.steps, fake.Text(rapid.SampledFrom([]string{"Done.", "", "All\nfinished."}).Draw(t, "text")))
			break
		}
		var cs []agentloop.ToolCall
		for j := 0; j < calls; j++ {
			what := rapid.SampledFrom([]string{"ok", "fail", "deny"}).Draw(t, "what")
			id := fmt.Sprintf("c%d_%d", i, j)
			cs = append(cs, fake.Call(id, "echo", fmt.Sprintf(`{"text":%q}`, what+" "+id)))
		}
		w.steps = append(w.steps, fake.ToolUse(rapid.SampledFrom([]string{"", "Calling."}).Draw(t, "lead"), cs...))
	}
	return w
}

// agent builds a fresh agent for the world. Each agent has its own place
// in the script; a model call past the end says "done".
func (w *cancelWorld) agent(observe func(agentloop.Event)) *agentloop.Agent {
	var mu sync.Mutex
	next := 0
	model := fake.New(func(*agentloop.Request) (*agentloop.Response, error) {
		mu.Lock()
		defer mu.Unlock()
		if next >= len(w.steps) {
			return fake.Text("done"), nil
		}
		next++
		return w.steps[next-1], nil
	})
	tool := agentloop.Func(
		agentloop.ToolSpec{Name: "echo", Description: "echo", Parameters: json.RawMessage(echoSchema)},
		func(ctx context.Context, in json.RawMessage) (agentloop.Result, error) {
			if err := ctx.Err(); err != nil {
				return agentloop.Result{}, err
			}
			var v struct{ Text string }
			json.Unmarshal(in, &v)
			_, id, _ := strings.Cut(v.Text, " ")
			w.mu.Lock()
			w.runs[id]++
			w.mu.Unlock()
			if strings.Contains(string(in), "fail") {
				return agentloop.Result{Content: "it broke", IsError: true}, nil
			}
			return agentloop.Result{Content: "ok"}, nil
		})
	return &agentloop.Agent{
		Model: model,
		Tools: []agentloop.Tool{tool},
		Policy: agentloop.PolicyFunc(func(_ context.Context, c agentloop.ToolCall, _ agentloop.View) agentloop.Decision {
			if strings.Contains(string(c.Input), "deny") {
				return agentloop.Decision{Verdict: agentloop.Deny, Reason: "no"}
			}
			return agentloop.Decision{Verdict: agentloop.Allow}
		}),
		Observe: observe,
	}
}

// events counts the events of an uncancelled run of the world.
func (w *cancelWorld) events(t *rapid.T) int {
	n := 0
	sess, err := agentloop.OpenSession(agentloop.NewMemoryStore())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.agent(func(agentloop.Event) { n++ }).Run(context.Background(), sess, "go"); err != nil {
		t.Fatal(err)
	}
	w.runs = map[string]int{}
	return n
}

// TestPropertyCancelAnyMoment is P3.1. rapid picks the event of the run
// (counted as the server observes them) at which the client sends
// session/cancel; the run is held at that event until the server has
// taken the cancel in, so the cancel always lands while the turn is in
// flight. Then:
//   - the response is cancelled;
//   - every call that started has a terminal update before the response;
//   - a following prompt settles what the cancelled run left, no call runs
//     twice, and the log reloads with exactly one result per started call.
func TestPropertyCancelAnyMoment(t *testing.T) {
	root := t.TempDir()
	rapid.Check(t, func(t *rapid.T) {
		w := genCancelWorld(t)
		at := rapid.IntRange(1, w.events(t)).Draw(t, "cancelAt")

		dir, err := os.MkdirTemp(root, "s")
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { os.RemoveAll(dir) })

		var h *harness
		var id acp.SessionId
		seen := 0
		hold := func(agentloop.Event) {
			seen++
			if seen != at {
				return
			}
			h.client.Cancel(context.Background(), acp.CancelNotification{SessionId: id})
			deadline := time.Now().Add(5 * time.Second)
			for {
				if st, _ := h.srv.State(id); st == Cancelling {
					return
				}
				if time.Now().After(deadline) {
					panic("the server never took the cancel in")
				}
				time.Sleep(100 * time.Microsecond)
			}
		}
		h = serveIn(t, dir, w.agent(hold), Options{}, false)
		id = h.newSession()

		resp, err := h.prompt(id, "go")
		if err != nil || resp.StopReason != acp.StopReasonCancelled {
			t.Fatalf("cancelled at event %d: got %+v, %v", at, resp, err)
		}
		during, _, _ := h.turnOnWire(id, 0)
		status := map[string]string{}
		for _, u := range during {
			var v struct {
				ToolCallID string `json:"toolCallId"`
				Status     string `json:"status"`
			}
			json.Unmarshal(u, &v)
			if v.ToolCallID != "" && v.Status != "" {
				status[v.ToolCallID] = v.Status
			}
		}
		for call, st := range status {
			if st == "in_progress" {
				t.Fatalf("call %s started but has no terminal update before the response", call)
			}
		}

		if _, err := h.prompt(id, "carry on"); err != nil {
			t.Fatalf("the next prompt: %v", err)
		}
		w.mu.Lock()
		for call, n := range w.runs {
			if n > 1 {
				t.Fatalf("call %s ran %d times", call, n)
			}
		}
		w.mu.Unlock()

		entries, err := agentloop.NewFileStore(SessionPath(dir, id)).Load()
		if err != nil {
			t.Fatal(err)
		}
		if _, err := agentloop.OpenSession(agentloop.NewMemoryStore(entries...)); err != nil {
			t.Fatalf("the log does not reopen: %v", err)
		}
		started, results := map[string]int{}, map[string]int{}
		for _, e := range entries {
			switch {
			case e.CallStarted != nil:
				started[e.CallStarted.CallID]++
			case e.ToolResult != nil:
				results[e.ToolResult.Result.CallID]++
			}
		}
		for call, n := range started {
			if n != 1 || results[call] != 1 {
				t.Fatalf("call %s: started %d times, %d results", call, n, results[call])
			}
		}
	})
}
