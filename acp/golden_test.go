package acpagent

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"testing"
	"time"

	acp "github.com/caelis-labs/acp-go-sdk"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/fake"
)

var update = flag.Bool("update", false, "rewrite testdata/golden from the current output")

var goldenStart = time.Date(2026, 10, 15, 9, 0, 0, 0, time.UTC)

type goldenCase struct {
	name    string
	model   *fake.Model
	policy  agentloop.Policy
	tool    func(ctx context.Context, cancel context.CancelFunc, in json.RawMessage) (agentloop.Result, error)
	limits  agentloop.Limits
	notices bool
}

const echoSchema = `{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}`

func echoOK(ctx context.Context, cancel context.CancelFunc, in json.RawMessage) (agentloop.Result, error) {
	var v struct{ Text string }
	json.Unmarshal(in, &v)
	return agentloop.Result{Content: "echo: " + v.Text, Labels: []string{"read"}}, nil
}

func withReasoning(r *agentloop.Response, thought string) *agentloop.Response {
	r.Message.Parts = append([]agentloop.Part{{Type: agentloop.ReasoningPart, Text: thought}}, r.Message.Parts...)
	return r
}

func goldenCases() []goldenCase {
	call := fake.Call("call_1", "echo", `{"text":"hi"}`)
	refusal := fake.WithStop(fake.Text("I can't help with that."), agentloop.StopRefusal)
	refusal.Refusal = "The request was declined by the safety system."
	return []goldenCase{
		{name: "simple_reply", model: fake.Script(withReasoning(fake.Text("Hello there."), "The user said hi."))},
		{name: "tool_loop", model: fake.Script(fake.ToolUse("Let me echo that.", call), fake.Text("It said hi."))},
		{
			name:  "denied_call",
			model: fake.Script(fake.ToolUse("", call), fake.Text("I was not allowed.")),
			policy: agentloop.PolicyFunc(func(context.Context, agentloop.ToolCall, agentloop.View) agentloop.Decision {
				return agentloop.Decision{Verdict: agentloop.Deny, Reason: "Echo is switched off."}
			}),
		},
		{
			name:  "cancelled_call",
			model: fake.Script(fake.ToolUse("Running it.", call)),
			tool: func(ctx context.Context, cancel context.CancelFunc, in json.RawMessage) (agentloop.Result, error) {
				cancel()
				<-ctx.Done()
				return agentloop.Result{}, ctx.Err()
			},
		},
		{
			name: "retry_then_success",
			model: fake.Script(
				&agentloop.ModelError{Kind: agentloop.RateLimited, Status: 429, Message: "rate limit exceeded",
					RetryAfter: goldenStart.Add(12 * time.Second)},
				fake.Text("Sorry for the wait."),
			),
			notices: true,
		},
		{
			name:    "model_error",
			model:   fake.Script(&agentloop.ModelError{Kind: agentloop.Unauthorized, Status: 401, Message: "invalid x-api-key"}),
			notices: true,
		},
		{name: "refusal", model: fake.Script(refusal)},
		{name: "turn_limit", model: fake.Script(fake.ToolUse("", call)), limits: agentloop.Limits{MaxTurns: 1}},
	}
}

// record runs the case and returns its events and outcome, with every
// entry's time set from its sequence number so the output is the same on
// every run.
func (c goldenCase) record(t *testing.T) ([]agentloop.Event, *agentloop.Outcome) {
	t.Helper()
	sess, err := agentloop.OpenSession(agentloop.NewMemoryStore())
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	tool := c.tool
	if tool == nil {
		tool = echoOK
	}
	var events []agentloop.Event
	a := &agentloop.Agent{
		Model:   c.model,
		ModelID: fake.ModelID,
		Tools: []agentloop.Tool{agentloop.Func(
			agentloop.ToolSpec{Name: "echo", Description: "echo", Parameters: json.RawMessage(echoSchema)},
			func(ctx context.Context, in json.RawMessage) (agentloop.Result, error) { return tool(ctx, cancel, in) },
		)},
		Policy: c.policy,
		Limits: c.limits,
		Now:    func() time.Time { return goldenStart },
		Sleep:  func(ctx context.Context, d time.Duration) error { return ctx.Err() },
		Observe: func(ev agentloop.Event) {
			if ev.Entry != nil {
				e := *ev.Entry
				e.Time = goldenStart.Add(time.Duration(e.Seq) * time.Second)
				ev.Entry = &e
			}
			events = append(events, ev)
		},
	}
	out, err := a.Run(ctx, sess, "Say hi.")
	if err != nil {
		t.Fatal(err)
	}
	return events, out
}

func TestGolden(t *testing.T) {
	for _, c := range goldenCases() {
		t.Run(c.name, func(t *testing.T) {
			events, out := c.record(t)
			opts := Options{Kinds: map[string]acp.ToolKind{"echo": acp.ToolKindRead}, ContextWindow: 200000, Notices: c.notices}

			var live bytes.Buffer
			for _, u := range translateAll(opts, events) {
				writeLine(t, &live, u)
			}
			stop, rpcErr := Respond(out)
			if rpcErr != nil {
				writeLine(t, &live, map[string]any{"error": rpcErr})
			} else {
				writeLine(t, &live, map[string]any{"stopReason": stop})
			}
			var entries []agentloop.Entry
			for _, ev := range events {
				if ev.Entry != nil {
					entries = append(entries, *ev.Entry)
				}
			}
			var replay bytes.Buffer
			for _, u := range Replay(opts, entries) {
				writeLine(t, &replay, u)
			}
			compareGolden(t, filepath.Join("testdata", "golden", c.name+".jsonl"), live.Bytes())
			compareGolden(t, filepath.Join("testdata", "golden", c.name+".replay.jsonl"), replay.Bytes())
		})
	}
}

func writeLine(t *testing.T, w *bytes.Buffer, v any) {
	t.Helper()
	if u, ok := v.(Update); ok {
		if err := validateUpdate(u); err != nil {
			t.Fatal(err)
		}
	}
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	w.Write(b)
	w.WriteByte('\n')
}

func compareGolden(t *testing.T, path string, got []byte) {
	t.Helper()
	if *update {
		if err := os.WriteFile(path, got, 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("%v (run go test ./acp -update to create it)", err)
	}
	if !bytes.Equal(got, want) {
		t.Errorf("%s differs (go test ./acp -update rewrites it):\ngot\n%s\nwant\n%s", path, got, want)
	}
}
