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
)

var binary string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "agentloop-bin")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	binary = filepath.Join(dir, "agentloop")
	if out, err := exec.Command("go", "build", "-o", binary, ".").CombinedOutput(); err != nil {
		fmt.Fprintf(os.Stderr, "build: %v\n%s", err, out)
		os.RemoveAll(dir)
		os.Exit(1)
	}
	code := m.Run()
	os.RemoveAll(dir)
	os.Exit(code)
}

// --- a fake Anthropic server ---

type fakeAPI struct {
	*httptest.Server
	mu      sync.Mutex
	replies []func(http.ResponseWriter)
	bodies  []string
}

func newFakeAPI(t *testing.T, replies ...func(http.ResponseWriter)) *fakeAPI {
	f := &fakeAPI{replies: replies}
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if serveModel(w, r) {
			return
		}
		b, _ := io.ReadAll(r.Body)
		f.mu.Lock()
		n := len(f.bodies)
		f.bodies = append(f.bodies, string(b))
		f.mu.Unlock()
		if n >= len(f.replies) {
			http.Error(w, `{"type":"error","error":{"type":"invalid_request_error","message":"unexpected request"}}`, 400)
			return
		}
		f.replies[n](w)
	}))
	t.Cleanup(f.Close)
	return f
}

// The fake Models API describes every model with this window, a number
// no test request names, so a usage size that matches it came from here.
const (
	fakeWindow      = 987654
	fakeDisplayName = "Claude Test"
)

// serveModel answers GET /v1/models/{id} as the Models API does, and
// reports whether the request was one.
func serveModel(w http.ResponseWriter, r *http.Request) bool {
	id, ok := strings.CutPrefix(r.URL.Path, "/v1/models/")
	if r.Method != http.MethodGet || !ok {
		return false
	}
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{"type":"model","id":%q,"display_name":%q,"created_at":"2026-01-01T00:00:00Z","max_input_tokens":%d,"max_tokens":64000,"capabilities":{}}`,
		id, fakeDisplayName, fakeWindow)
	return true
}

// requests are the bodies of the Messages API requests, in order.
func (f *fakeAPI) requests() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string{}, f.bodies...)
}

func sse(w http.ResponseWriter, evs ...[2]string) {
	w.Header().Set("Content-Type", "text/event-stream")
	for _, e := range evs {
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", e[0], e[1])
	}
}

func ev(name, data string) [2]string { return [2]string{name, data} }

func startEv() [2]string {
	return ev("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}`)
}

func endEvs(reason string) [][2]string {
	return [][2]string{
		ev("message_delta", `{"type":"message_delta","delta":{"stop_reason":"`+reason+`","stop_sequence":null,"stop_details":null},"usage":{"output_tokens":7}}`),
		ev("message_stop", `{"type":"message_stop"}`),
	}
}

func textReply(text string) func(http.ResponseWriter) {
	return func(w http.ResponseWriter) {
		b, _ := json.Marshal(text)
		evs := [][2]string{
			startEv(),
			ev("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
			ev("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":`+string(b)+`}}`),
			ev("content_block_stop", `{"type":"content_block_stop","index":0}`),
		}
		sse(w, append(evs, endEvs("end_turn")...)...)
	}
}

func toolReply(id, name, input string) func(http.ResponseWriter) {
	return func(w http.ResponseWriter) {
		b, _ := json.Marshal(input)
		evs := [][2]string{
			startEv(),
			ev("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"`+id+`","name":"`+name+`","input":{}}}`),
			ev("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":`+string(b)+`}}`),
			ev("content_block_stop", `{"type":"content_block_stop","index":0}`),
		}
		sse(w, append(evs, endEvs("tool_use")...)...)
	}
}

// --- running the binary ---

const secretKey = "sk-secret-key-123"

type fixture struct {
	t       *testing.T
	dir     string
	session string
	req     map[string]any
}

func newFixture(t *testing.T, api *fakeAPI, script string) *fixture {
	dir := t.TempDir()
	f := &fixture{t: t, dir: dir, session: filepath.Join(dir, "sub", "session.jsonl")}
	scriptPath := filepath.Join(dir, "tool.sh")
	if err := os.WriteFile(scriptPath, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	f.req = map[string]any{
		"provider":        "anthropic",
		"model":           "claude-test",
		"base_url":        api.URL,
		"system":          "be brief",
		"prompt":          "go",
		"session":         f.session,
		"api_key_command": []string{"printf", "%s", secretKey},
		"policy":          map[string]any{"unrestricted": true},
		"tools": []map[string]any{{
			"name": "probe", "description": "probe",
			"parameters": map[string]any{"type": "object"},
		}},
		"tool_command": []string{"/bin/sh", scriptPath},
		"tool_env":     map[string]string{"TOOL_X": "1", "PATH": "/usr/bin:/bin"},
		"tool_dir":     dir,
		"retry":        map[string]any{"retries": 0},
	}
	return f
}

func (f *fixture) writeReq() string {
	b, _ := json.Marshal(f.req)
	p := filepath.Join(f.dir, "request.json")
	if err := os.WriteFile(p, b, 0o644); err != nil {
		f.t.Fatal(err)
	}
	return p
}

func (f *fixture) command(args ...string) *exec.Cmd {
	cmd := exec.Command(binary, args...)
	cmd.Env = []string{"PARENT_VAR=leaky", "PATH=/usr/bin:/bin"}
	return cmd
}

type result struct {
	lines  []map[string]any
	stdout string
	stderr string
	code   int
}

func (r result) outcome(t *testing.T) map[string]any {
	t.Helper()
	if len(r.lines) == 0 {
		t.Fatalf("no output; stderr: %s", r.stderr)
	}
	o := r.lines[len(r.lines)-1]
	if o["type"] != "outcome" {
		t.Fatalf("last line is not an outcome: %s", r.stdout)
	}
	return o
}

func parse(t *testing.T, stdout string) []map[string]any {
	var lines []map[string]any
	sc := bufio.NewScanner(strings.NewReader(stdout))
	sc.Buffer(nil, 1<<24)
	for sc.Scan() {
		var m map[string]any
		if err := json.Unmarshal(sc.Bytes(), &m); err != nil {
			t.Fatalf("stdout line is not JSON: %q", sc.Text())
		}
		lines = append(lines, m)
	}
	return lines
}

func finish(t *testing.T, cmd *exec.Cmd, out, errb *bytes.Buffer) result {
	err := cmd.Wait()
	code := 0
	if ee, ok := err.(*exec.ExitError); ok {
		code = ee.ExitCode()
	} else if err != nil {
		t.Fatal(err)
	}
	return result{parse(t, out.String()), out.String(), errb.String(), code}
}

func (f *fixture) run() result {
	f.t.Helper()
	cmd := f.command("run", f.writeReq())
	var out, errb bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errb
	if err := cmd.Start(); err != nil {
		f.t.Fatal(err)
	}
	return finish(f.t, cmd, &out, &errb)
}

// --- tests ---

const probeScript = `#!/bin/sh
echo "tool=$1"
echo "stdin=$(cat)"
env
`

func TestToolLoop(t *testing.T) {
	api := newFakeAPI(t, toolReply("toolu_1", "probe", `{"q":"hello"}`), textReply("all done"))
	f := newFixture(t, api, probeScript)
	res := f.run()
	if res.code != 0 {
		t.Fatalf("exit %d: %s", res.code, res.stderr)
	}
	out := res.outcome(t)
	if out["reason"] != "completed" || out["text"] != "all done" || out["session"] != f.session {
		t.Errorf("outcome = %v", out)
	}
	if out["turns"].(float64) != 2 {
		t.Errorf("turns = %v", out["turns"])
	}
	var types []string
	for _, l := range res.lines {
		types = append(types, l["type"].(string))
	}
	want := "run_started message message decision call_started tool_result message run_ended outcome"
	if got := strings.Join(types, " "); got != want {
		t.Errorf("types = %s\nwant    %s", got, want)
	}
	var result string
	for _, l := range res.lines {
		if tr, ok := l["tool_result"].(map[string]any); ok {
			result = tr["result"].(map[string]any)["content"].(string)
		}
	}
	if !strings.Contains(result, "tool=probe") || !strings.Contains(result, `stdin={"q":"hello"}`) {
		t.Errorf("tool saw: %s", result)
	}
	if !strings.Contains(result, "TOOL_X=1") {
		t.Errorf("tool_env missing: %s", result)
	}
	for _, leak := range []string{secretKey, "PARENT_VAR"} {
		if strings.Contains(result, leak) {
			t.Errorf("tool saw %q: %s", leak, result)
		}
	}
	// The key went to the API, and the session was written under a new directory.
	if len(api.requests()) != 2 {
		t.Errorf("requests = %d", len(api.requests()))
	}
	if _, err := os.Stat(f.session); err != nil {
		t.Error(err)
	}
	if strings.Contains(readFile(t, f.session), secretKey) {
		t.Error("key in the session")
	}
}

func readFile(t *testing.T, p string) string {
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestUnusableRequests(t *testing.T) {
	api := newFakeAPI(t)
	cases := map[string]func(m map[string]any){
		"unknown field":    func(m map[string]any) { m["bogus"] = 1 },
		"unknown provider": func(m map[string]any) { m["provider"] = "gemini" },
		"no tool_command":  func(m map[string]any) { delete(m, "tool_command") },
		"no session":       func(m map[string]any) { delete(m, "session") },
		"no prompt":        func(m map[string]any) { delete(m, "prompt") },
		"bad effort":       func(m map[string]any) { m["effort"] = "ludicrous" },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			f := newFixture(t, api, probeScript)
			mutate(f.req)
			res := f.run()
			if res.code != 2 || res.stderr == "" || res.stdout != "" {
				t.Errorf("code %d, stderr %q, stdout %q", res.code, res.stderr, res.stdout)
			}
		})
	}
	t.Run("bad json", func(t *testing.T) {
		f := newFixture(t, api, probeScript)
		p := filepath.Join(f.dir, "bad.json")
		os.WriteFile(p, []byte("{nope"), 0o644)
		var errb bytes.Buffer
		cmd := f.command("run", p)
		cmd.Stderr = &errb
		err := cmd.Run()
		if ee, ok := err.(*exec.ExitError); !ok || ee.ExitCode() != 2 || errb.Len() == 0 {
			t.Errorf("err %v, stderr %q", err, errb.String())
		}
	})
	if n := len(api.requests()); n != 0 {
		t.Errorf("%d requests made", n)
	}
}

const sleepScript = `#!/bin/sh
echo started >> "$(dirname "$0")/starts"
sleep 30
echo finished
`

func waitForLines(t *testing.T, path string, n int) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if b, _ := os.ReadFile(path); strings.Count(string(b), "\n") >= n {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("tool never started")
}

func checkResumes(t *testing.T, f *fixture, api *fakeAPI) {
	t.Helper()
	// Resume: no prompt, the interrupted call is reported and not re-run.
	delete(f.req, "prompt")
	delete(f.req, "timeout_seconds")
	res := f.run()
	if res.code != 0 {
		t.Fatalf("resume exit %d: %s", res.code, res.stderr)
	}
	if out := res.outcome(t); out["reason"] != "completed" || out["text"] != "recovered" {
		t.Errorf("resume outcome = %v", out)
	}
	reqs := api.requests()
	if len(reqs) != 2 || !strings.Contains(reqs[1], "may or may not have taken effect") {
		t.Errorf("second request does not report the interrupted call: %v", reqs)
	}
	if n := strings.Count(readFile(t, filepath.Join(f.dir, "starts")), "\n"); n != 1 {
		t.Errorf("tool started %d times", n)
	}
}

func TestSIGTERMCancelsAndResumes(t *testing.T) {
	api := newFakeAPI(t, toolReply("toolu_1", "probe", `{}`), textReply("recovered"))
	f := newFixture(t, api, sleepScript)
	cmd := f.command("run", f.writeReq())
	var out, errb bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errb
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	waitForLines(t, filepath.Join(f.dir, "starts"), 1)
	start := time.Now()
	cmd.Process.Signal(syscall.SIGTERM)
	res := finish(t, cmd, &out, &errb)
	if res.code != 0 {
		t.Fatalf("exit %d: %s", res.code, res.stderr)
	}
	if o := res.outcome(t); o["reason"] != "canceled" {
		t.Errorf("outcome = %v", o)
	}
	if time.Since(start) > 15*time.Second {
		t.Error("took too long to stop")
	}
	checkResumes(t, f, api)
}

func TestTimeout(t *testing.T) {
	api := newFakeAPI(t, toolReply("toolu_1", "probe", `{}`), textReply("recovered"))
	f := newFixture(t, api, sleepScript)
	f.req["timeout_seconds"] = 1
	res := f.run()
	if res.code != 0 {
		t.Fatalf("exit %d: %s", res.code, res.stderr)
	}
	if o := res.outcome(t); o["reason"] != "canceled" {
		t.Errorf("outcome = %v", o)
	}
	checkResumes(t, f, api)
}

func TestVersion(t *testing.T) {
	out, err := exec.Command(binary, "version").Output()
	if err != nil || !strings.HasPrefix(string(out), "agentloop ") {
		t.Errorf("%q %v", out, err)
	}
}
