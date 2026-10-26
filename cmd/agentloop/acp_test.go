package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
	"unicode/utf8"

	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop"
)

// syncBuf is a buffer that a process's output and the test can share.
type syncBuf struct {
	mu  chan struct{}
	buf bytes.Buffer
}

func newSyncBuf() *syncBuf { s := &syncBuf{mu: make(chan struct{}, 1)}; return s }
func (s *syncBuf) Write(p []byte) (int, error) {
	s.mu <- struct{}{}
	defer func() { <-s.mu }()
	return s.buf.Write(p)
}
func (s *syncBuf) String() string {
	s.mu <- struct{}{}
	defer func() { <-s.mu }()
	return s.buf.String()
}

// --- an ACP client for the real binary ---

// acpProc is `agentloop acp` with pipes on stdin and stdout. Every stdout
// line is kept, in order, raw.
type acpProc struct {
	t      tb
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stderr *syncBuf
	exited chan struct{}
	code   int

	mu     sync.Mutex
	lines  [][]byte
	more   chan struct{} // closed and replaced when a line arrives
	nextID int
}

type acpReq struct {
	dir, sessionDir string
	req             map[string]any
}

// newACPReq is a request for `agentloop acp`: the fake API, a probe tool
// running script, and the key from the keycmd test program.
func newACPReq(t tb, dir, apiURL, script, keycmd string) *acpReq {
	scriptPath := filepath.Join(dir, "tool.sh")
	if err := os.WriteFile(scriptPath, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	r := &acpReq{dir: dir, sessionDir: filepath.Join(dir, "sessions")}
	r.req = map[string]any{
		"provider":        "anthropic",
		"model":           "claude-test",
		"base_url":        apiURL,
		"api_key_command": []string{keycmd, secretKey},
		"session_dir":     r.sessionDir,
		"tools": []map[string]any{{
			"name": "probe", "description": "probe",
			"parameters": map[string]any{"type": "object"},
		}},
		"tool_kinds":   map[string]string{"probe": "execute"},
		"tool_command": []string{"/bin/sh", scriptPath},
		"tool_env":     map[string]string{"PATH": "/usr/bin:/bin", "OUT": filepath.Join(dir, "out.bin")},
		"tool_dir":     dir,
		"retry":        map[string]any{"retries": 0},
	}
	return r
}

func (r *acpReq) write(t tb) string {
	b, _ := json.Marshal(r.req)
	p := filepath.Join(r.dir, "request.json")
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

var keycmdOnce struct {
	sync.Once
	path string
	err  error
}

// keycmdBinary builds testdata/keycmd once for every ACP test.
func keycmdBinary(t tb) string {
	keycmdOnce.Do(func() {
		keycmdOnce.path = filepath.Join(filepath.Dir(binary), "keycmd")
		out, err := exec.Command("go", "build", "-o", keycmdOnce.path, "./testdata/keycmd").CombinedOutput()
		if err != nil {
			keycmdOnce.err = fmt.Errorf("%v: %s", err, out)
		}
	})
	if keycmdOnce.err != nil {
		t.Fatal(keycmdOnce.err)
	}
	return keycmdOnce.path
}

func startACP(t tb, reqPath string) *acpProc {
	t.Helper()
	p := &acpProc{t: t, stderr: newSyncBuf(), exited: make(chan struct{}), more: make(chan struct{})}
	p.cmd = exec.Command(binary, "acp", reqPath)
	p.cmd.Env = []string{"PATH=/usr/bin:/bin", "PARENT_VAR=leaky"}
	p.cmd.Stderr = p.stderr
	var err error
	if p.stdin, err = p.cmd.StdinPipe(); err != nil {
		t.Fatal(err)
	}
	stdout, err := p.cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := p.cmd.Start(); err != nil {
		t.Fatal(err)
	}
	go func() {
		r := bufio.NewReader(stdout)
		for {
			line, err := r.ReadBytes('\n')
			if len(line) > 0 {
				p.mu.Lock()
				p.lines = append(p.lines, line)
				close(p.more)
				p.more = make(chan struct{})
				p.mu.Unlock()
			}
			if err != nil {
				break
			}
		}
		p.cmd.Wait()
		p.code = p.cmd.ProcessState.ExitCode()
		close(p.exited)
	}()
	t.Cleanup(func() {
		p.stdin.Close()
		select {
		case <-p.exited:
		case <-time.After(10 * time.Second):
			p.cmd.Process.Kill()
			<-p.exited
		}
	})
	return p
}

func (p *acpProc) send(v any) {
	b, _ := json.Marshal(v)
	p.stdin.Write(append(b, '\n'))
}

// request sends a request and returns its id.
func (p *acpProc) request(method string, params any) int {
	p.mu.Lock()
	p.nextID++
	id := p.nextID
	p.mu.Unlock()
	p.send(map[string]any{"jsonrpc": "2.0", "id": id, "method": method, "params": params})
	return id
}

type rpcLine struct {
	ID     *int            `json:"id"`
	Method string          `json:"method"`
	Params json.RawMessage `json:"params"`
	Result json.RawMessage `json:"result"`
	Error  *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

// waitFor waits until a line satisfies ok and returns it.
func (p *acpProc) waitFor(what string, ok func(rpcLine) bool) rpcLine {
	p.t.Helper()
	deadline := time.After(15 * time.Second)
	seen := 0
	for {
		p.mu.Lock()
		lines, more := p.lines, p.more
		p.mu.Unlock()
		for ; seen < len(lines); seen++ {
			var l rpcLine
			if json.Unmarshal(lines[seen], &l) == nil && ok(l) {
				return l
			}
		}
		select {
		case <-more:
		case <-p.exited:
			p.t.Fatalf("agentloop exited %d waiting for %s; stderr: %s", p.code, what, p.stderr.String())
		case <-deadline:
			p.t.Fatalf("no %s; stderr: %s", what, p.stderr.String())
		}
	}
}

func (p *acpProc) response(id int) rpcLine {
	p.t.Helper()
	return p.waitFor(fmt.Sprintf("response %d", id), func(l rpcLine) bool { return l.Method == "" && l.ID != nil && *l.ID == id })
}

func (p *acpProc) call(method string, params any) rpcLine {
	p.t.Helper()
	l := p.response(p.request(method, params))
	if l.Error != nil {
		p.t.Fatalf("%s: %+v", method, *l.Error)
	}
	return l
}

func (p *acpProc) initialize() {
	p.t.Helper()
	p.call("initialize", map[string]any{"protocolVersion": 1})
}

func (p *acpProc) newSession() string {
	p.t.Helper()
	l := p.call("session/new", map[string]any{"cwd": "/", "mcpServers": []any{}})
	var r struct {
		SessionID string `json:"sessionId"`
	}
	json.Unmarshal(l.Result, &r)
	return r.SessionID
}

func (p *acpProc) prompt(session, text string) int {
	return p.request("session/prompt", map[string]any{
		"sessionId": session, "prompt": []map[string]any{{"type": "text", "text": text}},
	})
}

// updates are the session/update params seen so far, as maps.
func (p *acpProc) updates() []map[string]any {
	p.mu.Lock()
	defer p.mu.Unlock()
	var out []map[string]any
	for _, line := range p.lines {
		var l struct {
			Method string `json:"method"`
			Params struct {
				Update map[string]any `json:"update"`
			} `json:"params"`
		}
		if json.Unmarshal(line, &l) == nil && l.Method == "session/update" {
			out = append(out, l.Params.Update)
		}
	}
	return out
}

// wait waits for the process to exit and returns its code.
func (p *acpProc) wait(within time.Duration) (int, bool) {
	select {
	case <-p.exited:
		return p.code, true
	case <-time.After(within):
		return 0, false
	}
}

// checkFraming holds P3.2 for everything the process wrote: every stdout
// line is valid UTF-8 and exactly one JSON-RPC 2.0 message, and nothing
// follows the last newline.
func (p *acpProc) checkFraming(t tb) {
	t.Helper()
	p.mu.Lock()
	defer p.mu.Unlock()
	for i, line := range p.lines {
		if !bytes.HasSuffix(line, []byte("\n")) {
			t.Fatalf("stdout ends without a newline: %q", line)
		}
		body := bytes.TrimSuffix(line, []byte("\n"))
		if !utf8.Valid(body) {
			t.Fatalf("line %d is not UTF-8", i)
		}
		dec := json.NewDecoder(bytes.NewReader(body))
		var m map[string]json.RawMessage
		if err := dec.Decode(&m); err != nil {
			t.Fatalf("line %d is not a JSON object: %v: %q", i, err, body)
		}
		if dec.More() {
			t.Fatalf("line %d holds more than one message: %q", i, body)
		}
		if string(m["jsonrpc"]) != `"2.0"` {
			t.Fatalf("line %d is not JSON-RPC 2.0: %q", i, body)
		}
		_, hasID := m["id"]
		_, hasMethod := m["method"]
		_, hasResult := m["result"]
		_, hasError := m["error"]
		if !hasMethod && (!hasID || hasResult == hasError) {
			t.Fatalf("line %d is neither a request, a notification nor a response: %q", i, body)
		}
	}
}

func lockFree(t tb, sessionPath string) bool {
	t.Helper()
	f, err := os.OpenFile(sessionPath+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	return syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB) == nil
}

// --- tests ---

// TestACPEndToEnd runs a tool loop through the real binary: the fake
// Anthropic API, the key from a key command, and a tool script.
func TestACPEndToEnd(t *testing.T) {
	api := newFakeAPI(t, toolReply("toolu_1", "probe", `{"q":"hello"}`), textReply("all done"))
	dir := t.TempDir()
	r := newACPReq(t, dir, api.URL, probeScript, keycmdBinary(t))
	p := startACP(t, r.write(t))
	p.initialize()
	created := p.call("session/new", map[string]any{"cwd": "/", "mcpServers": []any{}})
	var opened struct {
		SessionID     string `json:"sessionId"`
		ConfigOptions []struct {
			ID           string `json:"id"`
			Category     string `json:"category"`
			CurrentValue string `json:"currentValue"`
			Options      []struct {
				Value, Name string
			} `json:"options"`
		} `json:"configOptions"`
	}
	json.Unmarshal(created.Result, &opened)
	id := opened.SessionID
	// The model, as the Models API described it.
	if o := opened.ConfigOptions; len(o) != 1 || o[0].ID != "model" || o[0].Category != "model" || o[0].CurrentValue != "claude-test" ||
		len(o[0].Options) != 1 || o[0].Options[0].Value != "claude-test" || o[0].Options[0].Name != fakeDisplayName {
		t.Fatalf("configOptions: %s", created.Result)
	}
	resp := p.response(p.prompt(id, "go"))
	if resp.Error != nil || !strings.Contains(string(resp.Result), `"stopReason":"end_turn"`) {
		t.Fatalf("prompt: %s %+v", resp.Result, resp.Error)
	}

	var seen []string
	for _, u := range p.updates() {
		kind := u["sessionUpdate"].(string)
		switch kind {
		case "tool_call":
			if u["kind"] != "execute" || u["toolCallId"] != "toolu_1" || u["status"] != "pending" {
				t.Errorf("tool_call: %v", u)
			}
			kind += ":" + u["status"].(string)
		case "tool_call_update":
			kind += ":" + u["status"].(string)
			if u["status"] == "completed" {
				out, _ := json.Marshal(u["content"])
				if !strings.Contains(string(out), "tool=probe") {
					t.Errorf("tool output: %s", out)
				}
				if strings.Contains(string(out), secretKey) || strings.Contains(string(out), "PARENT_VAR") {
					t.Errorf("the tool saw the key or the parent environment: %s", out)
				}
			}
		case "agent_message_chunk":
			kind += ":" + u["content"].(map[string]any)["text"].(string)
		case "usage_update":
			// The window the Models API gave, not one from the request.
			if u["size"] != float64(fakeWindow) {
				t.Errorf("usage_update size %v, want %d", u["size"], fakeWindow)
			}
		}
		seen = append(seen, kind)
	}
	want := "tool_call:pending usage_update tool_call_update:in_progress tool_call_update:completed agent_message_chunk:all done usage_update"
	if got := strings.Join(seen, " "); got != want {
		t.Errorf("updates:\n%s\nwant\n%s", got, want)
	}

	p.stdin.Close()
	if code, ok := p.wait(5 * time.Second); !ok || code != 0 {
		t.Fatalf("exit %d (exited %v); stderr: %s", code, ok, p.stderr.String())
	}
	p.checkFraming(t)
	session := filepath.Join(r.sessionDir, id+".jsonl")
	log := readFile(t, session)
	if strings.Contains(log, secretKey) || strings.Contains(p.stderr.String(), secretKey) {
		t.Error("the key reached the session or stderr")
	}
	if len(api.requests()) != 2 || !lockFree(t, session) {
		t.Errorf("requests %d, lock free %v", len(api.requests()), lockFree(t, session))
	}
}

// A request agentloop acp cannot use exits 2 with a message, before any
// protocol traffic: nothing reaches stdout.
func TestACPUnusableRequests(t *testing.T) {
	api := newFakeAPI(t)
	cases := map[string]func(m map[string]any){
		"no session_dir":   func(m map[string]any) { delete(m, "session_dir") },
		"a session":        func(m map[string]any) { m["session"] = "/tmp/s.jsonl" },
		"a prompt":         func(m map[string]any) { m["prompt"] = "hi" },
		"a deadline":       func(m map[string]any) { m["timeout_seconds"] = 3 },
		"a bad tool kind":  func(m map[string]any) { m["tool_kinds"] = map[string]string{"probe": "poke"} },
		"no key":           func(m map[string]any) { m["api_key_command"] = []string{"/bin/sh", "-c", "exit 3"} },
		"an empty key":     func(m map[string]any) { m["api_key_command"] = []string{"printf", ""} },
		"a missing keycmd": func(m map[string]any) { m["api_key_command"] = []string{"/nonexistent/keycmd"} },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			r := newACPReq(t, t.TempDir(), api.URL, probeScript, keycmdBinary(t))
			mutate(r.req)
			cmd := exec.Command(binary, "acp", r.write(t))
			cmd.Stdin = strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1}}` + "\n")
			var out, errb bytes.Buffer
			cmd.Stdout, cmd.Stderr = &out, &errb
			err := cmd.Run()
			ee, ok := err.(*exec.ExitError)
			if !ok || ee.ExitCode() != 2 || out.Len() != 0 || strings.TrimSpace(errb.String()) == "" {
				t.Fatalf("err %v, stdout %q, stderr %q", err, out.String(), errb.String())
			}
		})
	}
}

// reflexAPI is a fake Anthropic API with no script: it answers a request
// that holds a tool result with text, and any other with a call to probe.
// A prompt holding "hang" streams a little text and then waits for the
// client to go away.
func reflexAPI(t tb) *httptest.Server {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if serveModel(w, r) {
			return
		}
		b, _ := io.ReadAll(r.Body)
		switch {
		case bytes.Contains(b, []byte("tool_result")):
			textReply("done")(w)
		case bytes.Contains(b, []byte("hang")):
			w.Header().Set("Content-Type", "text/event-stream")
			for _, e := range [][2]string{startEv(),
				ev("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
				ev("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Thinking"}}`),
			} {
				fmt.Fprintf(w, "event: %s\ndata: %s\n\n", e[0], e[1])
			}
			w.(http.Flusher).Flush()
			<-r.Context().Done()
		default:
			toolReply("toolu_1", "probe", `{}`)(w)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

// TestPropertyACPFraming is P3.2: whatever bytes a tool prints (newlines,
// NULs, invalid UTF-8, long lines, line and paragraph separators), every
// line the binary writes to stdout is one JSON-RPC 2.0 message in UTF-8,
// and nothing else is written there.
func TestPropertyACPFraming(t *testing.T) {
	api := reflexAPI(t)
	keycmd := keycmdBinary(t)
	root := t.TempDir()
	pieces := []string{"\n", "\r\n", "\x00", "\xff\xfe", "\xc3", " ", " ", "}\n{", `"`, "\\", "\t", "é", "\U0001F600", "{\"jsonrpc\":\"2.0\"}\n"}
	rapid.Check(t, func(rt *rapid.T) {
		var out bytes.Buffer
		for _, i := range rapid.SliceOfN(rapid.IntRange(0, len(pieces)+1), 0, 40).Draw(rt, "pieces") {
			switch {
			case i < len(pieces):
				out.WriteString(pieces[i])
			case i == len(pieces):
				out.WriteString(strings.Repeat("x", rapid.IntRange(1, 200000).Draw(rt, "long")))
			default:
				out.Write(rapid.SliceOfN(rapid.Byte(), 0, 64).Draw(rt, "bytes"))
			}
		}
		dir, err := os.MkdirTemp(root, "f")
		if err != nil {
			rt.Fatal(err)
		}
		defer os.RemoveAll(dir)
		r := newACPReq(rt, dir, api.URL, "#!/bin/sh\ncat \"$OUT\"\n", keycmd)
		if err := os.WriteFile(filepath.Join(dir, "out.bin"), out.Bytes(), 0o644); err != nil {
			rt.Fatal(err)
		}
		p := startACP(rt, r.write(rt))
		p.initialize()
		id := p.newSession()
		if resp := p.response(p.prompt(id, "go")); resp.Error != nil {
			rt.Fatalf("prompt: %+v", *resp.Error)
		}
		p.stdin.Close()
		if code, ok := p.wait(5 * time.Second); !ok || code != 0 {
			rt.Fatalf("exit %d (exited %v)", code, ok)
		}
		p.checkFraming(rt)
		// The tool's output reached the client as the log holds it.
		var got string
		for _, u := range p.updates() {
			if u["status"] == "completed" {
				got = u["content"].([]any)[0].(map[string]any)["content"].(map[string]any)["text"].(string)
			}
		}
		entries, err := agentloop.NewFileStore(filepath.Join(r.sessionDir, id+".jsonl")).Load()
		if err != nil {
			rt.Fatal(err)
		}
		var logged string
		for _, e := range entries {
			if e.ToolResult != nil {
				logged = e.ToolResult.Result.Content
			}
		}
		if got != logged {
			rt.Fatalf("the client got %q, the log holds %q", got, logged)
		}
	})
}

// tb is what the helpers need of a test: a *testing.T or a *rapid.T.
type tb interface {
	Helper()
	Fatal(args ...any)
	Fatalf(format string, args ...any)
	Cleanup(func())
}

// TestPropertyACPClientLoss is P3.3: the client closes stdin at a point
// rapid picks (before initialize, after it, with a session open, while
// the model streams, while a tool runs, after a finished turn), and the
// process exits 0 within 5 seconds with the session's lock released.
func TestPropertyACPClientLoss(t *testing.T) {
	api := reflexAPI(t)
	keycmd := keycmdBinary(t)
	root := t.TempDir()
	rapid.Check(t, func(rt *rapid.T) {
		stage := rapid.SampledFrom([]string{"start", "initialized", "session", "streaming", "tool", "answered"}).Draw(rt, "stage")
		dir, err := os.MkdirTemp(root, "l")
		if err != nil {
			rt.Fatal(err)
		}
		defer os.RemoveAll(dir)
		r := newACPReq(rt, dir, api.URL, "#!/bin/sh\necho started\nsleep 30\n", keycmd)
		if stage == "answered" {
			r = newACPReq(rt, dir, api.URL, "#!/bin/sh\necho quick\n", keycmd)
		}
		p := startACP(rt, r.write(rt))
		var id string
		switch stage {
		case "start":
		case "initialized":
			p.initialize()
		default:
			p.initialize()
			id = p.newSession()
		}
		switch stage {
		case "streaming":
			p.prompt(id, "hang")
			p.waitFor("a streamed chunk", func(l rpcLine) bool { return strings.Contains(string(l.Params), "Thinking") })
		case "tool":
			p.prompt(id, "go")
			p.waitFor("the tool starting", func(l rpcLine) bool { return strings.Contains(string(l.Params), "in_progress") })
		case "answered":
			if resp := p.response(p.prompt(id, "go")); resp.Error != nil {
				rt.Fatalf("prompt: %+v", *resp.Error)
			}
		}
		start := time.Now()
		p.stdin.Close()
		code, ok := p.wait(5 * time.Second)
		if !ok || code != 0 {
			rt.Fatalf("at %s: exit %d, exited %v after %v; stderr: %s", stage, code, ok, time.Since(start), p.stderr.String())
		}
		p.checkFraming(rt)
		if id != "" {
			session := filepath.Join(r.sessionDir, id+".jsonl")
			if !lockFree(rt, session) {
				rt.Fatalf("at %s: the session is still locked", stage)
			}
			if _, err := agentloop.NewFileStore(session).Load(); err != nil {
				rt.Fatalf("at %s: the log does not load: %v", stage, err)
			}
		}
	})
}

// SIGTERM during a turn cancels it: the client gets the cancelled
// response, the tool is stopped, and the process exits 0 with the lock
// released.
func TestACPSIGTERMCancelsTheTurn(t *testing.T) {
	api := reflexAPI(t)
	dir := t.TempDir()
	r := newACPReq(t, dir, api.URL, "#!/bin/sh\necho started\nsleep 30\n", keycmdBinary(t))
	p := startACP(t, r.write(t))
	p.initialize()
	id := p.newSession()
	prompt := p.prompt(id, "go")
	p.waitFor("the tool starting", func(l rpcLine) bool { return strings.Contains(string(l.Params), "in_progress") })
	p.cmd.Process.Signal(syscall.SIGTERM)
	resp := p.response(prompt)
	if resp.Error != nil || !strings.Contains(string(resp.Result), `"stopReason":"cancelled"`) {
		t.Fatalf("response: %s %+v", resp.Result, resp.Error)
	}
	if code, ok := p.wait(5 * time.Second); !ok || code != 0 {
		t.Fatalf("exit %d, exited %v; stderr: %s", code, ok, p.stderr.String())
	}
	var statuses []string
	for _, u := range p.updates() {
		if s, ok := u["status"].(string); ok {
			statuses = append(statuses, s)
		}
	}
	if strings.Join(statuses, " ") != "pending in_progress failed" {
		t.Errorf("call statuses: %v", statuses)
	}
	p.checkFraming(t)
	if !lockFree(t, filepath.Join(r.sessionDir, id+".jsonl")) {
		t.Error("the session is still locked")
	}
}

// Regression: when stdin ended with a prompt in flight, the SDK shut the
// whole connection, writes included, and the process exited without the
// cancelled response that the turn is owed. A client that only closed its
// writing half (relay does, on SIGTERM) never got an answer. Closing stdin
// mid-turn must still deliver the cancelled response before the exit.
func TestACPStdinEOFAnswersThePromptCancelled(t *testing.T) {
	for _, stage := range []string{"tool", "streaming"} {
		t.Run(stage, func(t *testing.T) {
			api := reflexAPI(t)
			dir := t.TempDir()
			r := newACPReq(t, dir, api.URL, "#!/bin/sh\necho started\nsleep 30\n", keycmdBinary(t))
			p := startACP(t, r.write(t))
			p.initialize()
			id := p.newSession()
			var prompt int
			if stage == "tool" {
				prompt = p.prompt(id, "go")
				p.waitFor("the tool starting", func(l rpcLine) bool { return strings.Contains(string(l.Params), "in_progress") })
			} else {
				prompt = p.prompt(id, "hang")
				p.waitFor("a streamed chunk", func(l rpcLine) bool { return strings.Contains(string(l.Params), "Thinking") })
			}
			p.stdin.Close()
			code, ok := p.wait(5 * time.Second)
			if !ok || code != 0 {
				t.Fatalf("exit %d, exited %v; stderr: %s", code, ok, p.stderr.String())
			}
			resp := p.response(prompt)
			if resp.Error != nil || !strings.Contains(string(resp.Result), `"stopReason":"cancelled"`) {
				t.Fatalf("response: %s %+v", resp.Result, resp.Error)
			}
			p.checkFraming(t)
			if !lockFree(t, filepath.Join(r.sessionDir, id+".jsonl")) {
				t.Error("the session is still locked")
			}
		})
	}
}

// sessionFiles is every file in a session directory with its contents,
// but the lock files, which a lock attempt may create and never writes.
func sessionFiles(t tb, dir string) map[string]string {
	t.Helper()
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	out := map[string]string{}
	for _, e := range ents {
		if strings.HasSuffix(e.Name(), ".lock") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		out[e.Name()] = string(b)
	}
	return out
}

// TestPropertyACPExclusiveAcrossProcesses is P4.3 across processes: two
// `agentloop acp` processes share a session directory. A session open in
// A, idle or with a tool running, cannot be loaded or resumed by B, which
// gets "session in use", and the attempts change nothing on disk. Once A
// closes it, B loads it.
func TestPropertyACPExclusiveAcrossProcesses(t *testing.T) {
	api := reflexAPI(t)
	dir := t.TempDir()
	r := newACPReq(t, dir, api.URL, "#!/bin/sh\necho started\nsleep 30\n", keycmdBinary(t))
	req := r.write(t)
	a, b := startACP(t, req), startACP(t, req)
	a.initialize()
	b.initialize()
	rapid.Check(t, func(rt *rapid.T) {
		a.t, b.t = rt, rt
		id := a.newSession()
		busy := rapid.Bool().Draw(rt, "busy")
		var prompt int
		if busy {
			prompt = a.prompt(id, "go")
			a.waitFor("the tool starting", func(l rpcLine) bool {
				return l.Method == "session/update" && strings.Contains(string(l.Params), id) &&
					strings.Contains(string(l.Params), "in_progress")
			})
		}
		disk := sessionFiles(rt, r.sessionDir)
		for i, n := 0, rapid.IntRange(1, 4).Draw(rt, "attempts"); i < n; i++ {
			method := rapid.SampledFrom([]string{"session/load", "session/resume"}).Draw(rt, "method")
			l := b.response(b.request(method, map[string]any{"sessionId": id, "cwd": "/", "mcpServers": []any{}}))
			if l.Error == nil || l.Error.Code != -32603 || l.Error.Message != "session in use" {
				rt.Fatalf("%s from B of a session open in A (busy %v): %s %+v", method, busy, l.Result, l.Error)
			}
			now := sessionFiles(rt, r.sessionDir)
			if fmt.Sprint(now) != fmt.Sprint(disk) {
				rt.Fatalf("%s from B changed the session directory", method)
			}
		}
		if busy {
			a.send(map[string]any{"jsonrpc": "2.0", "method": "session/cancel", "params": map[string]any{"sessionId": id}})
			if l := a.response(prompt); l.Error != nil || !strings.Contains(string(l.Result), `"cancelled"`) {
				rt.Fatalf("the cancelled turn: %s %+v", l.Result, l.Error)
			}
		}
		a.call("session/close", map[string]any{"sessionId": id})
		b.call("session/load", map[string]any{"sessionId": id, "cwd": "/", "mcpServers": []any{}})
		b.call("session/close", map[string]any{"sessionId": id})
	})
	a.t, b.t = t, t
}
