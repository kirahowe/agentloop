package main

import (
	"encoding/json"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/kirahowe/agentloop"
)

// valFixture is a request on disk with a fake API that always answers "done".
type valFixture struct {
	t   *testing.T
	dir string
	api *chaosAPI
	req map[string]any
}

func newValFixture(t *testing.T) *valFixture {
	dir := t.TempDir()
	api := newChaosAPI(t, newTestRand(), nil)
	return &valFixture{t: t, dir: dir, api: api, req: map[string]any{
		"provider":        "anthropic",
		"model":           "claude-test",
		"base_url":        api.URL,
		"session":         filepath.Join(dir, "sess", "session.jsonl"),
		"api_key_command": []string{"printf", "%s", chaosKey},
		"prompt":          "hello",
		"tools": []map[string]any{{
			"name": "t", "description": "d", "parameters": map[string]any{"type": "object"},
		}},
		"tool_command": []string{"/bin/echo"},
		"tool_dir":     dir,
		"retry":        map[string]any{"retries": 0},
	}}
}

func (f *valFixture) session() string { return f.req["session"].(string) }

// exec writes the request (raw if given) and runs `agentloop run` in dir.
func (f *valFixture) exec(raw []byte) chaosRun {
	f.t.Helper()
	if raw == nil {
		raw, _ = json.Marshal(f.req)
	}
	p := filepath.Join(f.dir, "request.json")
	if err := os.WriteFile(p, raw, 0o644); err != nil {
		f.t.Fatal(err)
	}
	return chaosExec(f.t, []string{"run", p}, chaosEnv(), nil)
}

func (f *valFixture) expectUnusable(r chaosRun) {
	f.t.Helper()
	if r.code != 2 || strings.TrimSpace(r.stderr) == "" || r.stdout != "" {
		f.t.Errorf("want exit 2 with a message and empty stdout; got exit %d signal %v, stderr %q, stdout %q", r.code, r.signal, r.stderr, r.stdout)
	}
	if _, err := os.Stat(f.session()); err == nil {
		f.t.Errorf("a session file was written for an unusable request")
	}
	if n := f.api.count(); n != 0 {
		f.t.Errorf("%d API requests for an unusable request", n)
	}
	if strings.Contains(r.stderr, "panic") || strings.Contains(r.stderr, "goroutine ") {
		f.t.Errorf("stderr looks like a crash: %s", r.stderr)
	}
}

func TestUnusableRequestMatrix(t *testing.T) {
	tool := func(m map[string]any) map[string]any { return m["tools"].([]map[string]any)[0] }
	cases := []struct {
		name   string
		mutate func(m map[string]any)
	}{
		{"unknown field", func(m map[string]any) { m["Bogus"] = 1 }},
		{"unknown nested field", func(m map[string]any) { m["limits"] = map[string]any{"max_turn": 3} }},
		{"unknown tool field", func(m map[string]any) { tool(m)["extra"] = true }},
		{"model is a number", func(m map[string]any) { m["model"] = 5 }},
		{"tools is a string", func(m map[string]any) { m["tools"] = "nope" }},
		{"tools is an object", func(m map[string]any) { m["tools"] = map[string]any{} }},
		{"tool_command is a string", func(m map[string]any) { m["tool_command"] = "echo" }},
		{"tool_env value is a number", func(m map[string]any) { m["tool_env"] = map[string]any{"A": 1} }},
		{"timeout is a string", func(m map[string]any) { m["timeout_seconds"] = "5" }},
		{"limits is a list", func(m map[string]any) { m["limits"] = []int{1} }},
		{"max_turns is fractional", func(m map[string]any) { m["limits"] = map[string]any{"max_turns": 1.5} }},
		{"retry is a number", func(m map[string]any) { m["retry"] = 3 }},
		{"refusal_fallback is a string", func(m map[string]any) { m["refusal_fallback"] = "yes" }},
		{"no provider", func(m map[string]any) { delete(m, "provider") }},
		{"provider with capitals", func(m map[string]any) { m["provider"] = "Anthropic" }},
		{"no model", func(m map[string]any) { delete(m, "model") }},
		{"empty session", func(m map[string]any) { m["session"] = "" }},
		{"no api_key_command", func(m map[string]any) { delete(m, "api_key_command") }},
		{"empty api_key_command", func(m map[string]any) { m["api_key_command"] = []string{} }},
		{"api_key_command is a string", func(m map[string]any) { m["api_key_command"] = "printf x" }},
		{"api_key_command empty program", func(m map[string]any) { m["api_key_command"] = []string{"", "x"} }},
		{"api_key_command NUL in program", func(m map[string]any) { m["api_key_command"] = []string{"pri\u0000ntf"} }},
		{"api_key_command NUL in argument", func(m map[string]any) { m["api_key_command"] = []string{"printf", "a\u0000b"} }},
		{"api_key_env is gone", func(m map[string]any) { m["api_key_env"] = "AGENTLOOP_API_KEY" }},
		{"tools without tool_command", func(m map[string]any) { delete(m, "tool_command") }},
		{"empty tool_command with tools", func(m map[string]any) { m["tool_command"] = []string{} }},
		{"empty program name", func(m map[string]any) { m["tool_command"] = []string{""} }},
		{"duplicate tool names", func(m map[string]any) {
			m["tools"] = []map[string]any{tool(m), tool(m)}
		}},
		{"tool without name", func(m map[string]any) { tool(m)["name"] = "" }},
		{"tool name with a space", func(m map[string]any) { tool(m)["name"] = "do it" }},
		{"tool name with a slash", func(m map[string]any) { tool(m)["name"] = "../../bin/sh" }},
		{"tool name too long", func(m map[string]any) { tool(m)["name"] = strings.Repeat("a", 65) }},
		{"tool name unicode", func(m map[string]any) { tool(m)["name"] = "tøøl" }},
		{"parameters is a number", func(m map[string]any) { tool(m)["parameters"] = 5 }},
		{"parameters is a string", func(m map[string]any) { tool(m)["parameters"] = "object" }},
		{"parameters is a list", func(m map[string]any) { tool(m)["parameters"] = []int{} }},
		{"parameters is null", func(m map[string]any) { tool(m)["parameters"] = nil }},
		{"schema with unknown type", func(m map[string]any) { tool(m)["parameters"] = map[string]any{"type": "banana"} }},
		{"schema with remote ref", func(m map[string]any) {
			tool(m)["parameters"] = map[string]any{"type": "object", "properties": map[string]any{"a": map[string]any{"$ref": "http://127.0.0.1:1/x.json"}}}
		}},
		{"schema with file ref", func(m map[string]any) {
			tool(m)["parameters"] = map[string]any{"type": "object", "properties": map[string]any{"a": map[string]any{"$ref": "file:///etc/passwd"}}}
		}},
		{"schema with dangling ref", func(m map[string]any) {
			tool(m)["parameters"] = map[string]any{"type": "object", "properties": map[string]any{"a": map[string]any{"$ref": "#/$defs/nope"}}}
		}},
		{"schema with bad required", func(m map[string]any) {
			tool(m)["parameters"] = map[string]any{"type": "object", "required": "a"}
		}},
		{"schema with bad pattern", func(m map[string]any) {
			tool(m)["parameters"] = map[string]any{"type": "object", "properties": map[string]any{"a": map[string]any{"type": "string", "pattern": "("}}}
		}},
		{"negative timeout", func(m map[string]any) { m["timeout_seconds"] = -1 }},
		{"negative tool timeout", func(m map[string]any) { m["tool_timeout_seconds"] = -0.5 }},
		{"negative limit", func(m map[string]any) { m["limits"] = map[string]any{"max_tokens": -1} }},
		{"negative retries", func(m map[string]any) { m["retry"] = map[string]any{"retries": -1} }},
		{"negative backoff", func(m map[string]any) { m["retry"] = map[string]any{"backoff_seconds": -2} }},
		{"unknown effort", func(m map[string]any) { m["effort"] = "HIGH" }},
		{"base_url not a url", func(m map[string]any) { m["base_url"] = "not a url" }},
		{"base_url wrong scheme", func(m map[string]any) { m["base_url"] = "ftp://example.com" }},
		{"base_url without host", func(m map[string]any) { m["base_url"] = "http://" }},
		{"tool_env key with equals", func(m map[string]any) { m["tool_env"] = map[string]string{"A=B": "c"} }},
		{"tool_env empty key", func(m map[string]any) { m["tool_env"] = map[string]string{"": "c"} }},
		{"tool_env NUL value", func(m map[string]any) { m["tool_env"] = map[string]string{"A": "b\x00c"} }},
		{"session with NUL", func(m map[string]any) { m["session"] = "a\x00b" }},
		{"no prompt on a new session", func(m map[string]any) { delete(m, "prompt") }},
		{"session directory under a file", func(m map[string]any) {
			f := filepath.Join(m["tool_dir"].(string), "afile")
			os.WriteFile(f, []byte("x"), 0o644)
			m["session"] = filepath.Join(f, "sub", "s.jsonl")
		}},
		{"session is a directory", func(m map[string]any) {
			d := filepath.Join(m["tool_dir"].(string), "adir")
			os.MkdirAll(d, 0o755)
			m["session"] = d
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			f := newValFixture(t)
			c.mutate(f.req)
			var raw []byte
			if c.name == "session with NUL" || c.name == "tool_env NUL value" {
				raw, _ = json.Marshal(f.req) // json escapes the NUL as \u0000
			}
			r := f.exec(raw)
			// The session-is-a-directory case has the "session file" there already.
			if c.name == "session is a directory" {
				if r.code != 2 || r.stdout != "" || r.stderr == "" {
					t.Errorf("exit %d stderr %q stdout %q", r.code, r.stderr, r.stdout)
				}
				return
			}
			f.expectUnusable(r)
		})
	}
}

func TestUnusableRawInputs(t *testing.T) {
	huge := strings.Repeat("[", 200000)
	cases := map[string][]byte{
		"empty file":        {},
		"whitespace":        []byte("  \n"),
		"null":              []byte("null"),
		"number":            []byte("7"),
		"string":            []byte(`"x"`),
		"array":             []byte(`[{}]`),
		"empty object":      []byte(`{}`),
		"trailing object":   []byte(`{"provider":"anthropic"}{}`),
		"trailing garbage":  []byte(`{"provider":"anthropic","model":"m","session":"/x"} nope`),
		"truncated":         []byte(`{"provider":"anthropic","model":`),
		"duplicate key":     []byte(`{"provider":"anthropic","provider":"gemini","model":"m","session":"/tmp/x"}`),
		"deeply nested":     []byte(huge),
		"binary":            {0xff, 0xfe, 0x00, 0x01},
		"bom":               []byte("\xef\xbb\xbf{}"),
		"number overflow":   []byte(`{"provider":"anthropic","model":"m","session":"/tmp/x","timeout_seconds":1e999}`),
		"int overflow":      []byte(`{"provider":"anthropic","model":"m","session":"/tmp/x","limits":{"max_turns":99999999999999999999}}`),
		"lone surrogate":    []byte(`{"provider":"anthropic","model":"\ud800","session":""}`),
		"comment":           []byte(`{"provider":"anthropic"} // hi`),
		"unicode field key": []byte(`{"provider":"anthropic","bogus":1,"model":"m","session":"/tmp/x"}`),
	}
	for name, raw := range cases {
		t.Run(name, func(t *testing.T) {
			f := newValFixture(t)
			r := f.exec(raw)
			if r.code != 2 || r.stdout != "" || strings.TrimSpace(r.stderr) == "" {
				t.Errorf("exit %d signal %v stderr %q stdout %q", r.code, r.signal, r.stderr, r.stdout)
			}
		})
	}
}

func TestUsageErrorsOnCommandLine(t *testing.T) {
	for _, args := range [][]string{
		nil, {"run"}, {"run", "a", "b"}, {"chat"}, {"bogus"}, {"run", "/nonexistent/request.json"}, {"run", t.TempDir()}, {"RUN", "x"}, {""},
	} {
		r := chaosExec(t, args, chaosEnv(), nil)
		if r.code != 2 || r.stdout != "" || r.stderr == "" {
			t.Errorf("%q: exit %d stderr %q stdout %q", args, r.code, r.stderr, r.stdout)
		}
	}
}

// TestUnusualButValidRequests: requests that look odd and are fine. Each
// runs to a completed outcome.
func TestUnusualButValidRequests(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(f *valFixture)
		prompt string // the prompt the API should have received, if checked
	}{
		{"empty tools list", func(f *valFixture) { f.req["tools"] = []any{}; delete(f.req, "tool_command") }, ""},
		{"no tools key", func(f *valFixture) { delete(f.req, "tools"); delete(f.req, "tool_command") }, ""},
		{"parameters is empty object", func(f *valFixture) {
			f.req["tools"] = []map[string]any{{"name": "t", "description": "", "parameters": map[string]any{}}}
		}, ""},
		{"parameters absent", func(f *valFixture) {
			f.req["tools"] = []map[string]any{{"name": "t", "description": "d"}}
		}, ""},
		{"unicode prompt", func(f *valFixture) { f.req["prompt"] = "héllo 世界 \U0001F600 ‮ rtl \x01 \u0000 nul" }, ""},
		{"huge prompt", func(f *valFixture) { f.req["prompt"] = strings.Repeat("lorem ipsum ", 500000) }, ""},
		{"huge system", func(f *valFixture) { f.req["system"] = strings.Repeat("x", 2<<20) }, ""},
		{"prompt only whitespace", func(f *valFixture) { f.req["prompt"] = "  \n " }, ""},
		{"relative session", func(f *valFixture) { f.req["session"] = "rel/session.jsonl" }, ""},
		{"session in cwd", func(f *valFixture) { f.req["session"] = "session.jsonl" }, ""},
		{"huge timeout", func(f *valFixture) { f.req["timeout_seconds"] = 1e30 }, ""},
		{"huge tool timeout", func(f *valFixture) { f.req["tool_timeout_seconds"] = 1e300 }, ""},
		{"tiny timeout is still a timeout", nil, ""},
		{"unicode session path", func(f *valFixture) { f.req["session"] = filepath.Join(f.dir, "é世界 \U0001F600", "s .jsonl") }, ""},
		{"tool_env empty value", func(f *valFixture) { f.req["tool_env"] = map[string]string{"A": ""} }, ""},
		{"api_key_command through sh", func(f *valFixture) {
			f.req["api_key_command"] = []string{"/bin/sh", "-c", `printf '%s\n' "$0"`, chaosKey}
		}, ""},
		{"openai-looking effort", func(f *valFixture) { f.req["effort"] = "xhigh" }, ""},
		{"limits all zero", func(f *valFixture) { f.req["limits"] = map[string]any{"max_turns": 0} }, ""},
		{"unusual-looking but legal tool names", func(f *valFixture) {
			f.req["tools"] = []map[string]any{{"name": "A-b_9", "description": "d"}, {"name": strings.Repeat("z", 64), "description": "d"}}
		}, ""},
	}
	for _, c := range cases {
		if c.mutate == nil {
			continue
		}
		t.Run(c.name, func(t *testing.T) {
			f := newValFixture(t)
			c.mutate(f)
			cwd := f.dir
			raw, _ := json.Marshal(f.req)
			p := filepath.Join(f.dir, "request.json")
			os.WriteFile(p, raw, 0o644)
			r := chaosExecIn(t, cwd, []string{"run", p}, chaosEnv())
			if r.code != 0 {
				t.Fatalf("exit %d: %s", r.code, r.stderr)
			}
			lines := r.outputLines()
			var o map[string]any
			json.Unmarshal(lines[len(lines)-1], &o)
			if o["reason"] != "completed" || o["text"] != "done" {
				t.Fatalf("outcome %v", o)
			}
			// The session file is where the request said, relative to cwd.
			s := f.session()
			if !filepath.IsAbs(s) {
				s = filepath.Join(cwd, s)
			}
			if _, err := os.Stat(s); err != nil {
				t.Error(err)
			}
			if p, _ := f.req["prompt"].(string); strings.HasPrefix(p, "héllo") {
				// The entry log keeps the prompt exactly.
				sess := readSession(t, s)
				if got := sess.Messages()[0].Text(); got != p {
					t.Errorf("prompt altered: %q", got)
				}
			}
		})
	}
	t.Run("tiny timeout is still a timeout", func(t *testing.T) {
		f := newValFixture(t)
		f.req["timeout_seconds"] = 1e-9
		r := f.exec(nil)
		lines := r.outputLines()
		var o map[string]any
		json.Unmarshal(lines[len(lines)-1], &o)
		if r.code != 0 || o["reason"] != "canceled" {
			t.Errorf("exit %d outcome %v", r.code, o)
		}
	})
}

// A request of a session path that exists as a file but whose tool_dir is
// missing is still a run: the tool call fails, the run does not.
func TestMissingToolDirIsAToolFailure(t *testing.T) {
	f := newValFixture(t)
	f.req["tool_dir"] = filepath.Join(f.dir, "nope")
	r := f.exec(nil)
	if r.code != 0 {
		t.Fatalf("exit %d: %s", r.code, r.stderr)
	}
}

// --- fuzzing the request parser ---

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
		req, err := loadRequest(path)
		if err != nil {
			if _, ok := err.(*usageError); !ok {
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
		if seconds(req.TimeoutSeconds) < 0 || seconds(req.ToolTimeoutSeconds) < 0 {
			t.Fatal("negative duration")
		}
		if req.TimeoutSeconds > 0 && seconds(req.TimeoutSeconds) <= 0 {
			t.Fatal("a positive timeout became none")
		}
		a := req.agent("k")
		sess, _ := agentloop.OpenSession(agentloop.NewMemoryStore())
		// The agent the request describes must start: an accepted request
		// never fails at tool setup.
		if _, err := a.Run(t.Context(), sess, ""); err != nil {
			t.Fatalf("an accepted request does not set up: %v", err)
		}
	})
}

func newTestRand() *rand.Rand { return rand.New(rand.NewSource(1)) }

func readSession(t *testing.T, path string) *agentloop.Session {
	t.Helper()
	s, err := agentloop.OpenSession(agentloop.NewFileStore(path))
	if err != nil {
		t.Fatal(err)
	}
	return s
}
