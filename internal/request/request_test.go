package request

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/command"
	"github.com/kirahowe/agentloop/internal/usage"
)

func FuzzLoadRequest(f *testing.F) {
	good, _ := json.Marshal(map[string]any{
		"provider": "anthropic", "model": "m", "session": "/tmp/x/s.jsonl", "prompt": "p",
		"api_key_command": []string{"printf", "k"},
		"policy":          map[string]any{"labels": []string{"private"}},
		"tools":           []map[string]any{{"name": "t", "description": "d", "parameters": map[string]any{"type": "object", "properties": map[string]any{"a": map[string]any{"type": "string"}}}}},
		"tool_command":    []string{"x"}, "tool_env": map[string]string{"A": "b"},
		"limits": map[string]any{"max_turns": 3}, "retry": map[string]any{"retries": 1},
		"timeout_seconds": 5, "base_url": "http://127.0.0.1:9", "effort": "high",
	})
	f.Add(good)
	f.Add([]byte(`{}`))
	f.Add([]byte(`{"provider":"anthropic","model":"m","session":"s","prompt":"p","api_key_command":["k"],"tools":[{"name":"mail"}],"tool_command":["x"],"tool_protocol":"envelope","policy":{"labels":["private"],"tools":{"mail":{"labels":["contaminated"]}},"forbid_labels":["x"]}}`))
	f.Add([]byte(`{"provider":"anthropic","model":"m","session":"s","prompt":"p","api_key_command":["k"],"tools":[{"name":"mail"}],"tool_command":["x"],"policy":{"tools":{"mail":{"class":"read"}}},"quarantine":{"model":"small"},"resolve":[]}`))
	f.Add([]byte(`{"provider":"anthropic","model":"m","session":"s","prompt":"p","api_key_command":["k"],"mcp_servers":{"m":{"command":["/bin/m","--c={credentials_file}"],"credentials_command":["k"],"send_labels":true}},"tools":[{"name":"look","protocol":"mcp-stdio","server":"m","tool":"lookup","allow_content":["resource"]}],"policy":{"tools":{"look":{"labels":["private"]}}}}`))
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
		if req.Session == "" || req.SessionDir != "" || req.ToolKinds != nil || req.Title != "" {
			t.Fatalf("run accepted %+v", req)
		}
	case ACP:
		if req.SessionDir == "" || req.Session != "" || req.Prompt != "" || req.TimeoutSeconds != 0 {
			t.Fatalf("acp accepted %+v", req)
		}
		for name, kind := range req.ToolKinds {
			if !slices.Contains(req.toolNames(), name) || !slices.Contains(ToolKinds, kind) {
				t.Fatalf("acp accepted tool_kinds %v", req.ToolKinds)
			}
		}
	}
	if len(req.commandTools()) > 0 && len(req.ToolCommand) == 0 {
		t.Fatal("tools without a command")
	}
	if Seconds(req.TimeoutSeconds) < 0 || Seconds(req.ToolTimeoutSeconds) < 0 {
		t.Fatal("negative duration")
	}
	if req.TimeoutSeconds > 0 && Seconds(req.TimeoutSeconds) <= 0 {
		t.Fatal("a positive timeout became none")
	}
	a := req.Agent("k", req.stubMCP())
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

// A policy becomes the agent's labels, its forbid list and a policy that
// gives each tool's labels, and command tools speak the envelope; with no
// policy the agent has none of those and every tool is allowed.
func TestPolicyBuildsTheAgent(t *testing.T) {
	dir := t.TempDir()
	load := func(m map[string]any) *Request {
		t.Helper()
		b, _ := json.Marshal(m)
		path := filepath.Join(dir, "r.json")
		os.WriteFile(path, b, 0o644)
		req, err := Load(path, Run)
		if err != nil {
			t.Fatal(err)
		}
		return req
	}
	base := map[string]any{
		"provider": "anthropic", "model": "m", "api_key_command": []string{"printf", "k"},
		"session": "/tmp/s.jsonl", "prompt": "p",
		"tools":        []map[string]any{{"name": "web_fetch"}, {"name": "memo_send"}},
		"tool_command": []string{"x"},
		"policy": map[string]any{
			"labels":        []string{"private"},
			"forbid_labels": []string{"secret"},
			"tools": map[string]any{
				"web_fetch": map[string]any{"labels": []string{"contaminated", "web"}},
			},
		},
		"tool_protocol": "envelope",
	}
	a := load(base).Agent("", nil)
	if !reflect.DeepEqual(a.Labels, []string{"private"}) || !reflect.DeepEqual(a.Forbid, []string{"secret"}) {
		t.Fatalf("labels %v forbid %v", a.Labels, a.Forbid)
	}
	call := func(name string) agentloop.Decision {
		return a.Policy.Decide(t.Context(), agentloop.ToolCall{Name: name, Input: json.RawMessage(`{}`)}, agentloop.View{})
	}
	if d := call("web_fetch"); d.Verdict != agentloop.Allow || !reflect.DeepEqual(d.Labels, []string{"contaminated", "web"}) {
		t.Fatalf("web_fetch: %+v", d)
	}
	if d := call("memo_send"); d.Verdict != agentloop.Allow || len(d.Labels) != 0 {
		t.Fatalf("memo_send: %+v", d)
	}
	for _, tool := range a.Tools {
		if ct := tool.(*command.Tool); !ct.Envelope {
			t.Fatalf("tool %+v", ct)
		}
	}
	delete(base, "policy")
	delete(base, "tool_protocol")
	a = load(base).Agent("", nil)
	if a.Policy != nil || a.Labels != nil || a.Forbid != nil || a.Tools[0].(*command.Tool).Envelope {
		t.Fatalf("a request with no policy got %+v", a)
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
		{"acp with kinds", ACP, acp(func(m map[string]any) { m["tool_kinds"] = map[string]string{"look": "read"} }), ""},
		{"acp with a title", ACP, acp(func(m map[string]any) { m["title"] = "relay · hub" }), ""},
		{"acp needs session_dir", ACP, acp(func(m map[string]any) { delete(m, "session_dir") }), "no session_dir"},
		{"acp refuses session", ACP, acp(func(m map[string]any) { m["session"] = "/tmp/s.jsonl" }), "session is not used"},
		{"acp refuses prompt", ACP, acp(func(m map[string]any) { m["prompt"] = "hi" }), "prompt is not used"},
		{"acp refuses a deadline", ACP, acp(func(m map[string]any) { m["timeout_seconds"] = 5 }), "timeout_seconds is not used"},
		{"acp refuses an unknown kind", ACP, acp(func(m map[string]any) { m["tool_kinds"] = map[string]string{"look": "peek"} }), `kind "peek"`},
		{"acp refuses a kind for no tool", ACP, acp(func(m map[string]any) { m["tool_kinds"] = map[string]string{"ghost": "read"} }), `"ghost"`},
		// The window comes from the provider, so a request cannot set one.
		{"acp refuses context_window", ACP, acp(func(m map[string]any) { m["context_window"] = 200000 }), `unknown field "context_window"`},
		{"acp refuses NUL in session_dir", ACP, acp(func(m map[string]any) { m["session_dir"] = "/tmp/a\x00b" }), "NUL"},
		{"run minimal", Run, run(same), ""},
		{"run with a policy", Run, run(func(m map[string]any) {
			m["policy"] = map[string]any{"labels": []string{"private"}, "tools": map[string]any{"look": map[string]any{"labels": []string{"contaminated"}}}}
			m["tool_protocol"] = "envelope"
		}), ""},
		{"a policy with command tools needs the envelope", Run, run(func(m map[string]any) {
			m["policy"] = map[string]any{"labels": []string{"private"}}
		}), "needs \"tool_protocol\""},
		{"a policy with a bad label", Run, run(func(m map[string]any) {
			m["policy"] = map[string]any{"forbid_labels": []string{""}}
			m["tool_protocol"] = "envelope"
		}), "policy.forbid_labels"},
		{"a policy that cannot start", Run, run(func(m map[string]any) {
			m["policy"] = map[string]any{"labels": []string{"private"}, "forbid_labels": []string{"private"}}
			m["tool_protocol"] = "envelope"
		}), "cannot start"},
		{"a policy for no tool", Run, run(func(m map[string]any) {
			m["policy"] = map[string]any{"tools": map[string]any{"ghost": map[string]any{}}}
			m["tool_protocol"] = "envelope"
		}), `"ghost", which is not one of the tools`},
		{"a tool label the policy forbids", Run, run(func(m map[string]any) {
			m["policy"] = map[string]any{"forbid_labels": []string{"private"}, "tools": map[string]any{"look": map[string]any{"labels": []string{"private"}}}}
			m["tool_protocol"] = "envelope"
		}), "forbids"},
		{"run refuses session_dir", Run, run(func(m map[string]any) { m["session_dir"] = "/tmp" }), "session_dir is only for agentloop acp"},
		{"run refuses tool_kinds", Run, run(func(m map[string]any) { m["tool_kinds"] = map[string]string{} }), "tool_kinds is only"},
		{"run refuses title", Run, run(func(m map[string]any) { m["title"] = "x" }), "title is only"},
		{"run refuses new_sessions_only", Run, run(func(m map[string]any) { m["new_sessions_only"] = true }), "new_sessions_only is only"},
		{"acp with new_sessions_only", ACP, acp(func(m map[string]any) { m["new_sessions_only"] = true }), ""},
		{"acp refuses new_sessions_only false", ACP, acp(func(m map[string]any) { m["new_sessions_only"] = false }), "is refused"},
		{"run refuses context_window", Run, run(func(m map[string]any) { m["context_window"] = 1 }), `unknown field "context_window"`},
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

// A request written for the agentloop that held calls is refused with a
// message that says the field is gone, for every removed field wherever
// the document puts it.
func TestRemovedFieldsAreRefused(t *testing.T) {
	removed := map[string]func(m map[string]any){
		"resolve":      func(m map[string]any) { m["resolve"] = []map[string]any{} },
		"quarantine":   func(m map[string]any) { m["quarantine"] = map[string]any{"model": "small"} },
		"workbench":    func(m map[string]any) { m["workbench"] = map[string]any{} },
		"unrestricted": func(m map[string]any) { m["policy"] = map[string]any{"unrestricted": true} },
		"class": func(m map[string]any) {
			m["policy"] = map[string]any{"tools": map[string]any{"look": map[string]any{"class": "read"}}}
		},
		"always_hold": func(m map[string]any) {
			m["policy"] = map[string]any{"tools": map[string]any{"look": map[string]any{"always_hold": true}}}
		},
		"consent": func(m map[string]any) {
			m["policy"] = map[string]any{"tools": map[string]any{"look": map[string]any{"consent": []any{}}}}
		},
	}
	dir := t.TempDir()
	for name, edit := range removed {
		for _, mode := range []Mode{Run, ACP} {
			m := map[string]any{
				"provider": "anthropic", "model": "m", "api_key_command": []string{"printf", "k"},
				"tools": []map[string]any{{"name": "look"}}, "tool_command": []string{"x"}, "tool_protocol": "envelope",
			}
			if mode == Run {
				m["session"], m["prompt"] = "/tmp/s.jsonl", "hi"
			} else {
				m["session_dir"] = "/tmp/sessions"
			}
			edit(m)
			b, _ := json.Marshal(m)
			path := filepath.Join(dir, "r.json")
			os.WriteFile(path, b, 0o644)
			_, err := Load(path, mode)
			if err == nil || !strings.Contains(err.Error(), "was removed") || !strings.Contains(err.Error(), name) {
				t.Errorf("%s, mode %d: err = %v", name, mode, err)
			}
			if _, ok := err.(*usage.Error); !ok {
				t.Errorf("%s: %T, not a usage error", name, err)
			}
		}
	}
}

// A field no version of agentloop had is an unknown field, not a removed
// one.
func TestUnknownFieldIsNotRemoved(t *testing.T) {
	for _, msg := range []string{`unknown field "foo"`, `unknown field "policy.tools.x.labelz"`, `unknown field "policy.class"`, "bad", ""} {
		if removedField(msg) != "" {
			t.Errorf("%q reported as removed", msg)
		}
	}
}

func TestDisplayRefPrefix(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "r.json")
	load := func(prefix string) (*Request, error) {
		b, _ := json.Marshal(map[string]any{"provider": "anthropic", "model": "m", "session": "s", "prompt": "p",
			"api_key_command": []string{"k"}, "display_ref_prefix": prefix})
		os.WriteFile(path, b, 0o644)
		return Load(path, Run)
	}
	for _, good := range []string{"shown:2026-10-18/", "shown:"} {
		r, err := load(good)
		if err != nil {
			t.Errorf("%q: %v", good, err)
			continue
		}
		if a := r.Agent("key", nil); a.DisplayRefPrefix != good {
			t.Errorf("%q: agent %v", good, a)
		}
	}
	for _, bad := range []string{"shown", "Shown:x/", "shown:../x]]", "shown:a b", "shown:" + strings.Repeat("a", 81)} {
		if _, err := load(bad); err == nil {
			t.Errorf("accepted %q", bad)
		}
	}
}
