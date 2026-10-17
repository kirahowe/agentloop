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
	"github.com/kirahowe/agentloop/quarantine"
	"github.com/kirahowe/agentloop/trifecta"
)

func FuzzLoadRequest(f *testing.F) {
	good, _ := json.Marshal(map[string]any{
		"provider": "anthropic", "model": "m", "session": "/tmp/x/s.jsonl", "prompt": "p",
		"api_key_command": []string{"printf", "k"},
		"policy":          map[string]any{"unrestricted": true},
		"tools":           []map[string]any{{"name": "t", "description": "d", "parameters": map[string]any{"type": "object", "properties": map[string]any{"a": map[string]any{"type": "string"}}}}},
		"tool_command":    []string{"x"}, "tool_env": map[string]string{"A": "b"},
		"limits": map[string]any{"max_turns": 3}, "retry": map[string]any{"retries": 1},
		"timeout_seconds": 5, "base_url": "http://127.0.0.1:9", "effort": "high",
	})
	f.Add(good)
	f.Add([]byte(`{}`))
	f.Add([]byte(`{"provider":"anthropic","model":"m","session":"s","prompt":"p","api_key_command":["k"],"tools":[{"name":"mail"}],"tool_command":["x"],"tool_protocol":"envelope","policy":{"tools":{"mail":{"class":"read","labels":["contaminated"]},"read_quarantined":{"class":"read"}}},"quarantine":{"model":"small","tools":["mail"],"retries":2}}`))
	f.Add([]byte(`{"provider":"anthropic","model":"m","session":"s","prompt":"p","api_key_command":["k"],"mcp_servers":{"m":{"command":["/bin/m","--c={credentials_file}"],"credentials_command":["k"],"send_labels":true}},"tools":[{"name":"look","protocol":"mcp-stdio","server":"m","tool":"lookup","allow_content":["resource"]}],"policy":{"tools":{"look":{"class":"read"}}}}`))
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
	a, err := req.Agent("k", req.stubMCP())
	if err != nil {
		t.Fatalf("an accepted request does not build: %v", err)
	}
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

// A real policy becomes the agent's labels and a trifecta policy, and
// envelope tools; an unrestricted one adds neither.
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
			"labels": []string{"private"},
			"tools": map[string]any{
				"web_fetch": map[string]any{"class": "egress", "labels": []string{"contaminated"}},
				"memo_send": map[string]any{"class": "write", "always_hold": true},
			},
		},
		"tool_protocol": "envelope",
	}
	a, _ := load(base).Agent("", nil)
	if !reflect.DeepEqual(a.Labels, []string{"private"}) {
		t.Fatalf("labels %v", a.Labels)
	}
	p, ok := a.Policy.(trifecta.Policy)
	if !ok || p.Tools["web_fetch"].Class != trifecta.Egress || !p.Tools["memo_send"].AlwaysHold ||
		!reflect.DeepEqual(p.Tools["web_fetch"].Labels, []string{"contaminated"}) {
		t.Fatalf("policy %+v", a.Policy)
	}
	for _, tool := range a.Tools {
		if ct := tool.(*command.Tool); !ct.Envelope || !ct.KnownLabel("private") || ct.KnownLabel("x") {
			t.Fatalf("tool %+v", ct)
		}
	}
	base["policy"] = map[string]any{"unrestricted": true}
	delete(base, "tool_protocol")
	a, _ = load(base).Agent("", nil)
	if a.Policy != nil || a.Labels != nil || a.Tools[0].(*command.Tool).Envelope {
		t.Fatalf("an unrestricted request got %+v", a)
	}
}

// TestModes checks the fields that differ between run and acp (design
// section 2.4): acp needs session_dir and refuses session, prompt and
// timeout_seconds; run refuses the acp-only fields.
func TestModes(t *testing.T) {
	base := func() map[string]any {
		return map[string]any{
			"provider": "anthropic", "model": "m", "api_key_command": []string{"printf", "k"},
			"policy":       map[string]any{"unrestricted": true},
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
		{"run refuses resolve without a policy", Run, run(func(m map[string]any) {
			m["resolve"] = []map[string]any{{"call_id": "a", "verdict": "deny", "input_sha256": "x"}}
		}), "resolve needs a policy"},
		{"run with resolve", Run, run(func(m map[string]any) {
			m["policy"] = map[string]any{"tools": map[string]any{"look": map[string]any{"class": "read"}}}
			m["tool_protocol"] = "envelope"
			m["resolve"] = []map[string]any{{"call_id": "a", "verdict": "allow", "input_sha256": "x"}, {"call_id": "b", "verdict": "deny", "reason": "no", "input_sha256": "y"}}
		}), ""},
		{"acp refuses resolve", ACP, acp(func(m map[string]any) {
			m["resolve"] = []map[string]any{{"call_id": "a", "verdict": "allow"}}
		}), "resolve is not used"},
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

func quarantineBase() map[string]any {
	return map[string]any{
		"provider": "anthropic", "model": "big", "api_key_command": []string{"printf", "k"},
		"session": "/tmp/s.jsonl", "prompt": "p",
		"tools": []map[string]any{
			{"name": "mail_read", "parameters": map[string]any{"type": "object"}},
			{"name": "web_post", "parameters": map[string]any{"type": "object"}},
		},
		"tool_command":  []string{"x"},
		"tool_protocol": "envelope",
		"policy": map[string]any{
			"tools": map[string]any{
				"mail_read":        map[string]any{"class": "read", "labels": []string{"contaminated", "private"}},
				"web_post":         map[string]any{"class": "egress"},
				"read_quarantined": map[string]any{"class": "read"},
			},
		},
		"quarantine": map[string]any{"model": "small", "effort": "low", "max_output_tokens": 512, "retries": 2, "tools": []string{"mail_read"}},
	}
}

func TestQuarantineRequests(t *testing.T) {
	q := func(m map[string]any) map[string]any { return m["quarantine"].(map[string]any) }
	pol := func(m map[string]any) map[string]any { return m["policy"].(map[string]any)["tools"].(map[string]any) }
	cases := []struct {
		name string
		mode Mode
		edit func(m map[string]any)
		want string
	}{
		{"accepted", Run, func(map[string]any) {}, ""},
		{"retries default", Run, func(m map[string]any) { delete(q(m), "retries") }, ""},
		{"acp with a kind for the reader", ACP, func(m map[string]any) {
			delete(m, "session")
			delete(m, "prompt")
			m["session_dir"] = "/tmp/sessions"
			m["tool_kinds"] = map[string]string{"read_quarantined": "read"}
		}, ""},
		{"needs a policy", Run, func(m map[string]any) {
			m["policy"] = map[string]any{"unrestricted": true}
			delete(m, "tool_protocol")
		}, "quarantine needs a policy"},
		{"needs a model", Run, func(m map[string]any) { delete(q(m), "model") }, "quarantine has no model"},
		{"bad effort", Run, func(m map[string]any) { q(m)["effort"] = "huge" }, "quarantine.effort"},
		{"negative max_output_tokens", Run, func(m map[string]any) { q(m)["max_output_tokens"] = -1 }, "max_output_tokens"},
		{"negative retries", Run, func(m map[string]any) { q(m)["retries"] = -1 }, "quarantine.retries"},
		{"too many retries", Run, func(m map[string]any) { q(m)["retries"] = MaxQuarantineRetries + 1 }, "quarantine.retries"},
		{"no inner tools", Run, func(m map[string]any) { q(m)["tools"] = []string{} }, "names no tool"},
		{"inner tool not among the tools", Run, func(m map[string]any) { q(m)["tools"] = []string{"ghost"} }, `"ghost", which is not one of the tools`},
		{"inner tool not read", Run, func(m map[string]any) { q(m)["tools"] = []string{"web_post"} }, `whose class is "egress"`},
		{"inner tool twice", Run, func(m map[string]any) { q(m)["tools"] = []string{"mail_read", "mail_read"} }, "twice"},
		{"the reader itself", Run, func(m map[string]any) { q(m)["tools"] = []string{"read_quarantined"} }, "cannot name read_quarantined itself"},
		{"no policy entry for the reader", Run, func(m map[string]any) { delete(pol(m), "read_quarantined") }, `no entry for the tool "read_quarantined"`},
		{"the reader not class read", Run, func(m map[string]any) { pol(m)["read_quarantined"] = map[string]any{"class": "act"} }, "give read_quarantined the class read"},
		{"a policy entry for the reader without quarantine", Run, func(m map[string]any) { delete(m, "quarantine") }, `"read_quarantined", which is not one of the tools`},
		{"a tool named like the reader", Run, func(m map[string]any) {
			m["tools"] = append(m["tools"].([]map[string]any), map[string]any{"name": "read_quarantined"})
		}, "is built in"},
		{"a forbidden inner label", Run, func(m map[string]any) {
			m["policy"].(map[string]any)["forbid_labels"] = []string{"private"}
		}, "forbid_labels forbids"},
		{"unknown quarantine field", Run, func(m map[string]any) { q(m)["max_answer_bytes"] = 10 }, "unknown field"},
	}
	dir := t.TempDir()
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m := quarantineBase()
			c.edit(m)
			b, _ := json.Marshal(m)
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

func TestQuarantineBuildsTheReader(t *testing.T) {
	b, _ := json.Marshal(quarantineBase())
	path := filepath.Join(t.TempDir(), "r.json")
	os.WriteFile(path, b, 0o644)
	req, err := Load(path, Run)
	if err != nil {
		t.Fatal(err)
	}
	a, err := req.Agent("k", req.stubMCP())
	if err != nil {
		t.Fatal(err)
	}
	var reader *quarantine.Reader
	for _, tool := range a.Tools {
		if r, ok := tool.(*quarantine.Reader); ok {
			reader = r
		}
	}
	if reader == nil {
		t.Fatal("no reader tool")
	}
	if reader.ModelID != "small" || reader.Effort != agentloop.EffortLow || reader.MaxOutputTokens != 512 || reader.Retries != 2 ||
		len(reader.Inner) != 1 || reader.Inner[0].Spec().Name != "mail_read" || reader.Retry != agentloop.DefaultRetry ||
		!reflect.DeepEqual(reader.Labels, map[string][]string{"mail_read": {"contaminated", "private"}}) {
		t.Errorf("reader = %+v", reader)
	}
	p := a.Policy.(trifecta.Policy)
	if !reflect.DeepEqual(p.Readers, map[string][]string{"read_quarantined": {"mail_read"}}) {
		t.Errorf("readers = %v", p.Readers)
	}
}

// A reader that cannot be built is an error from Agent, never a request
// that quietly runs without it.
func TestAgentReportsAReaderError(t *testing.T) {
	b, _ := json.Marshal(quarantineBase())
	path := filepath.Join(t.TempDir(), "r.json")
	os.WriteFile(path, b, 0o644)
	req, err := Load(path, Run)
	if err != nil {
		t.Fatal(err)
	}
	// Past Load, so only Agent can notice.
	bad := -1
	req.Quarantine.Retries = &bad
	if a, err := req.Agent("k", req.stubMCP()); err == nil || a != nil {
		t.Fatalf("agent %v, err %v", a, err)
	}
}
