package request

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
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
		for _, mode := range []Mode{Run, ACP} {
			req, err := Load(path, mode)
			if err != nil {
				if _, ok := err.(*usage.Error); !ok {
					t.Fatalf("loadRequest returned %T, not a usage error: %v", err, err)
				}
				if req != nil {
					t.Fatal("request returned with an error")
				}
				continue
			}
			checkAccepted(t, req, mode)
		}
	})
}

// checkAccepted holds for every request Load accepts in a mode.
func checkAccepted(t *testing.T, req *Request, mode Mode) {
	t.Helper()
	// An accepted request has what the command needs.
	if req.Provider == "" || req.Model == "" {
		t.Fatalf("accepted an incomplete request: %+v", req)
	}
	switch mode {
	case Run:
		if req.Session == "" || req.SessionDir != "" || req.ToolKinds != nil || req.ContextWindow != 0 {
			t.Fatalf("run accepted %+v", req)
		}
	case ACP:
		if req.SessionDir == "" || req.Session != "" || req.Prompt != "" || req.TimeoutSeconds != 0 || req.ContextWindow < 0 {
			t.Fatalf("acp accepted %+v", req)
		}
		for name, kind := range req.ToolKinds {
			if !slices.ContainsFunc(req.Tools, func(s agentloop.ToolSpec) bool { return s.Name == name }) || !slices.Contains(ToolKinds, kind) {
				t.Fatalf("acp accepted tool_kinds %v", req.ToolKinds)
			}
		}
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

// TestModes checks the fields that differ between run and acp (design
// section 2.4): acp needs session_dir and refuses session, prompt and
// timeout_seconds; run refuses the acp-only fields.
func TestModes(t *testing.T) {
	base := func() map[string]any {
		return map[string]any{
			"provider": "anthropic", "model": "m", "api_key_command": []string{"printf", "k"},
			"tools":        []map[string]any{{"name": "look", "parameters": map[string]any{"type": "object"}}},
			"tool_command": []string{"x"},
		}
	}
	acp := func(edit func(m map[string]any)) map[string]any {
		m := base()
		m["session_dir"] = "/tmp/sessions"
		edit(m)
		return m
	}
	run := func(edit func(m map[string]any)) map[string]any {
		m := base()
		m["session"], m["prompt"] = "/tmp/s.jsonl", "hi"
		edit(m)
		return m
	}
	same := func(map[string]any) {}
	cases := []struct {
		name string
		mode Mode
		req  map[string]any
		want string // "" means accepted
	}{
		{"acp minimal", ACP, acp(same), ""},
		{"acp with kinds and window", ACP, acp(func(m map[string]any) {
			m["tool_kinds"], m["context_window"] = map[string]string{"look": "read"}, 200000
		}), ""},
		{"acp needs session_dir", ACP, acp(func(m map[string]any) { delete(m, "session_dir") }), "no session_dir"},
		{"acp refuses session", ACP, acp(func(m map[string]any) { m["session"] = "/tmp/s.jsonl" }), "session is not used"},
		{"acp refuses prompt", ACP, acp(func(m map[string]any) { m["prompt"] = "hi" }), "prompt is not used"},
		{"acp refuses a deadline", ACP, acp(func(m map[string]any) { m["timeout_seconds"] = 5 }), "timeout_seconds is not used"},
		{"acp refuses an unknown kind", ACP, acp(func(m map[string]any) { m["tool_kinds"] = map[string]string{"look": "peek"} }), `kind "peek"`},
		{"acp refuses a kind for no tool", ACP, acp(func(m map[string]any) { m["tool_kinds"] = map[string]string{"ghost": "read"} }), `"ghost"`},
		{"acp refuses a negative window", ACP, acp(func(m map[string]any) { m["context_window"] = -1 }), "context_window"},
		{"acp refuses NUL in session_dir", ACP, acp(func(m map[string]any) { m["session_dir"] = "/tmp/a\x00b" }), "NUL"},
		{"run minimal", Run, run(same), ""},
		{"run refuses session_dir", Run, run(func(m map[string]any) { m["session_dir"] = "/tmp" }), "session_dir is only for agentloop acp"},
		{"run refuses tool_kinds", Run, run(func(m map[string]any) { m["tool_kinds"] = map[string]string{} }), "tool_kinds is only"},
		{"run refuses context_window", Run, run(func(m map[string]any) { m["context_window"] = 1 }), "context_window is only"},
	}
	dir := t.TempDir()
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			b, _ := json.Marshal(c.req)
			path := filepath.Join(dir, "r.json")
			if err := os.WriteFile(path, b, 0o644); err != nil {
				t.Fatal(err)
			}
			req, err := Load(path, c.mode)
			if c.want == "" {
				if err != nil {
					t.Fatalf("refused: %v", err)
				}
				checkAccepted(t, req, c.mode)
				return
			}
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("err = %v, want one containing %q", err, c.want)
			}
		})
	}
}
