package otel_test

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"os"
	"testing"
	"time"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/fake"
	"github.com/kirahowe/agentloop/otel"
)

var update = flag.Bool("update", false, "rewrite the recorded session and the golden files")

var goldenOpts = otel.Options{
	ConversationID: "session-0001",
	AgentName:      "mailer",
	ServiceName:    "agentloop-test",
	TraceSeed:      "golden",
}

func TestGolden(t *testing.T) {
	if *update {
		recordSession(t)
	}
	entries, err := agentloop.NewFileStore("testdata/session.jsonl").Load()
	if err != nil || len(entries) == 0 {
		t.Fatalf("load fixture: %v (%d entries)", err, len(entries))
	}
	for _, c := range []struct {
		file    string
		content bool
	}{
		{"testdata/session.golden.json", false},
		{"testdata/session.content.golden.json", true},
	} {
		opts := goldenOpts
		opts.IncludeContent = c.content
		raw, err := otel.Marshal(entries, opts)
		if err != nil {
			t.Fatal(err)
		}
		var pretty bytes.Buffer
		if err := json.Indent(&pretty, raw, "", "  "); err != nil {
			t.Fatal(err)
		}
		pretty.WriteByte('\n')
		if *update {
			if err := os.WriteFile(c.file, pretty.Bytes(), 0o644); err != nil {
				t.Fatal(err)
			}
			continue
		}
		want, err := os.ReadFile(c.file)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(want, pretty.Bytes()) {
			t.Errorf("%s differs from the converter's output; rerun with -update and review the diff", c.file)
		}
		checkOTLP(t, raw)
	}
}

// stampedStore gives entries fixed times, so a recorded session does not
// change from one recording to the next.
type stampedStore struct{ *agentloop.MemoryStore }

func (s stampedStore) Append(e agentloop.Entry) error {
	e.Time = time.Date(2026, 10, 17, 9, 0, 0, 0, time.UTC).Add(time.Duration(e.Seq) * 1250 * time.Millisecond)
	return s.MemoryStore.Append(e)
}

// recordSession runs a scripted agent: a retried model error, one read
// that adds a label, one call held and then allowed by a person, and one
// call the policy denies.
func recordSession(t *testing.T) {
	read := agentloop.Func(agentloop.ToolSpec{Name: "read_note", Description: "Read a note.", Parameters: json.RawMessage(`{"type":"object"}`)},
		func(context.Context, json.RawMessage) (agentloop.Result, error) {
			return agentloop.Result{Content: "Note: the quarterly numbers are in.", Labels: []string{"private"}, Log: "read 1 file"}, nil
		})
	send := agentloop.Func(agentloop.ToolSpec{Name: "send_mail", Description: "Send mail.", Parameters: json.RawMessage(`{"type":"object"}`)},
		func(context.Context, json.RawMessage) (agentloop.Result, error) {
			return agentloop.Result{Content: "sent"}, nil
		})
	del := agentloop.Func(agentloop.ToolSpec{Name: "delete_all", Description: "Delete everything.", Parameters: json.RawMessage(`{"type":"object"}`)},
		func(context.Context, json.RawMessage) (agentloop.Result, error) {
			return agentloop.Result{Content: "gone"}, nil
		})
	policy := agentloop.PolicyFunc(func(_ context.Context, c agentloop.ToolCall, _ agentloop.View) agentloop.Decision {
		switch c.Name {
		case "send_mail":
			return agentloop.Decision{Verdict: agentloop.Hold, Reason: "Mail leaves the building.", Labels: []string{"private"}}
		case "delete_all":
			return agentloop.Decision{Verdict: agentloop.Deny, Reason: "Never."}
		}
		return agentloop.Decision{Verdict: agentloop.Allow}
	})
	model := fake.Script(
		&agentloop.ModelError{Kind: agentloop.Overloaded, Message: "overloaded, try later"},
		fake.WithUsage(fake.ToolUse("Looking at your notes.",
			fake.Call("c1", "read_note", `{"title":"q3"}`),
			fake.Call("c2", "send_mail", `{"to":"boss@example.com","body":"numbers"}`),
			fake.Call("c3", "delete_all", `{}`)),
			agentloop.Usage{InputTokens: 20, OutputTokens: 40, CacheReadTokens: 300, CacheWriteTokens: 100}),
		fake.WithUsage(fake.Text("Sent. I did not delete anything."),
			agentloop.Usage{InputTokens: 15, OutputTokens: 12, CacheReadTokens: 420}),
	)
	agent := &agentloop.Agent{
		Model: model, ModelID: "fake-1", Tools: []agentloop.Tool{read, send, del}, Policy: policy,
		Labels: []string{"private"},
		Retry:  &agentloop.RetryPolicy{Retries: 2, Backoff: time.Millisecond, MaxWait: time.Millisecond},
	}
	store := stampedStore{agentloop.NewMemoryStore()}
	s, err := agentloop.OpenSession(store)
	if err != nil {
		t.Fatal(err)
	}
	out, err := agent.Run(context.Background(), s, "Mail the boss the Q3 numbers.")
	if err != nil || out.Reason != agentloop.ReasonHeld {
		t.Fatalf("first run: %v %+v", err, out)
	}
	h := out.Held[0]
	if err := agent.Resolve(context.Background(), s, agentloop.Resolution{CallID: h.ID, Verdict: agentloop.Allow, InputSHA256: h.InputSHA256}); err != nil {
		t.Fatal(err)
	}
	if out, err = agent.Run(context.Background(), s, ""); err != nil || out.Reason != agentloop.ReasonCompleted {
		t.Fatalf("second run: %v %+v", err, out)
	}
	entries, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	for _, e := range entries {
		b, _ := json.Marshal(e)
		buf.Write(b)
		buf.WriteByte('\n')
	}
	if err := os.WriteFile("testdata/session.jsonl", buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}
