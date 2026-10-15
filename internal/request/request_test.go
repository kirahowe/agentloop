package request

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/internal/usage"
)

func FuzzLoadRequest(f *testing.F) {
	good, _ := json.Marshal(map[string]any{
		"provider": "anthropic", "model": "m", "session": "/tmp/x/s.jsonl", "prompt": "p",
		"api_key_command": []string{"printf", "k"},
		"tools":           []map[string]any{{"name": "t", "description": "d", "parameters": map[string]any{"type": "object", "properties": map[string]any{"a": map[string]any{"type": "string"}}}}},
		"tool_command":    []string{"x"}, "tool_env": map[string]string{"A": "b"},
		"limits": map[string]any{"max_turns": 3}, "retry": map[string]any{"retries": 1},
		"timeout_seconds": 5, "base_url": "http://127.0.0.1:9", "effort": "high",
	})
	f.Add(good)
	f.Add([]byte(`{}`))
	f.Add([]byte(`null`))
	f.Add([]byte(`{"provider":"openai","model":"m","session":"s","tools":[{"name":"a","parameters":{"$ref":"#"}}],"tool_command":["x"]}`))
	f.Add([]byte(`{"provider":"openai","model":"m","session":"s","tools":[{"name":"a","parameters":{"type":"object","properties":{"a":{"$ref":"#/$defs/a"}},"$defs":{"a":{"$ref":"#/$defs/a"}}}}],"tool_command":["x"]}`))
	f.Add([]byte(`{"provider":"openai","model":"m","session":"s","timeout_seconds":1e308,"tool_timeout_seconds":4.9e-324}`))
	dir := f.TempDir()
	path := filepath.Join(dir, "r.json")
	f.Fuzz(func(t *testing.T, data []byte) {
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatal(err)
		}
		req, err := Load(path)
		if err != nil {
			if _, ok := err.(*usage.Error); !ok {
				t.Fatalf("loadRequest returned %T, not a usage error: %v", err, err)
			}
			if req != nil {
				t.Fatal("request returned with an error")
			}
			return
		}
		// An accepted request has what the run needs.
		if req.Provider == "" || req.Model == "" || req.Session == "" {
			t.Fatalf("accepted an incomplete request: %+v", req)
		}
		if len(req.Tools) > 0 && len(req.ToolCommand) == 0 {
			t.Fatal("tools without a command")
		}
		if Seconds(req.TimeoutSeconds) < 0 || Seconds(req.ToolTimeoutSeconds) < 0 {
			t.Fatal("negative duration")
		}
		if req.TimeoutSeconds > 0 && Seconds(req.TimeoutSeconds) <= 0 {
			t.Fatal("a positive timeout became none")
		}
		a := req.Agent("k")
		sess, _ := agentloop.OpenSession(agentloop.NewMemoryStore())
		// The agent the request describes must start: an accepted request
		// never fails at tool setup.
		if _, err := a.Run(t.Context(), sess, ""); err != nil {
			t.Fatalf("an accepted request does not set up: %v", err)
		}
	})
}

// TestFetchKeyTimeout runs the command in process with a short timeout, so
// the 30 second limit need not be waited for. A child that keeps the pipes
// open must not hold the wait past the deadline.
func TestFetchKeyTimeout(t *testing.T) {
	start := time.Now()
	_, err := FetchKey(context.Background(), []string{"/bin/sh", "-c", "sleep 30 & sleep 30"}, 300*time.Millisecond)
	if err == nil || !strings.Contains(err.Error(), "did not finish within") {
		t.Fatalf("err = %v", err)
	}
	if d := time.Since(start); d > 5*time.Second {
		t.Errorf("took %v", d)
	}
	if KeyTimeout != 30*time.Second {
		t.Errorf("KeyTimeout = %v", KeyTimeout)
	}
}

func TestFetchKeyTrimsAndFails(t *testing.T) {
	key, err := FetchKey(context.Background(), []string{"printf", "  k-1 \n"}, time.Minute)
	if err != nil || key != "k-1" {
		t.Errorf("key %q, err %v", key, err)
	}
	long := strings.Repeat("e", 1000)
	_, err = FetchKey(context.Background(), []string{"/bin/sh", "-c", "echo " + long + " >&2; exit 4"}, time.Minute)
	if err == nil || !strings.HasPrefix(err.Error(), "The api_key_command exited 4: ...") || len(err.Error()) > 300 {
		t.Errorf("err = %v", err)
	}
}
