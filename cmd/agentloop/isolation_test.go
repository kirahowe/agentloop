package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

var (
	envdumpOnce sync.Once
	envdumpPath string
	envdumpErr  error
)

// buildEnvdump compiles testdata/envdump, a tool that reports exactly what
// its process was given. A shell would add PWD, SHLVL and _ on its own, so
// the tests use this instead.
func buildEnvdump(t *testing.T) string {
	envdumpOnce.Do(func() {
		dir, err := os.MkdirTemp("", "envdump")
		if err != nil {
			envdumpErr = err
			return
		}
		envdumpPath = filepath.Join(dir, "envdump")
		if out, err := exec.Command("go", "build", "-o", envdumpPath, "./testdata/envdump").CombinedOutput(); err != nil {
			envdumpErr = fmt.Errorf("%v\n%s", err, out)
		}
	})
	if envdumpErr != nil {
		t.Fatal(envdumpErr)
	}
	return envdumpPath
}

type dump struct {
	Env   []string
	Args  []string
	Stdin string
	Fds   []string
	Cwd   string
	Pgid  int
	Pid   int
}

func readDumps(t *testing.T, dir string) []dump {
	files, _ := filepath.Glob(filepath.Join(dir, "dump-*.json"))
	var out []dump
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		var d dump
		if err := json.Unmarshal(b, &d); err != nil {
			t.Fatal(err)
		}
		out = append(out, d)
	}
	return out
}

// toolWorld runs one tool-use turn that calls the envdump tool, then ends.
func toolWorld(t *testing.T, toolEnv map[string]string, input string) (*chaosWorld, chaosRun, []dump) {
	w := newChaosWorld(t, newTestRand())
	w.turns = [][]string{{"toolu_iso_1"}}
	w.api.turns = w.turns
	w.req["tool_command"] = []string{buildEnvdump(t)}
	w.req["tool_env"] = toolEnv
	w.req["tools"] = []map[string]any{{
		"name": "probe", "description": "probe",
		"parameters": map[string]any{"type": "object"},
	}}
	// The API script names the tool "ledger"; make the probe answer to it.
	w.req["tools"].([]map[string]any)[0]["name"] = "ledger"
	r := w.run("go", nil, 0, 0)
	if r.code != 0 {
		t.Fatalf("exit %d: %s", r.code, r.stderr)
	}
	return w, r, readDumps(t, w.dir)
}

func sortedCopy(s []string) []string {
	c := append([]string{}, s...)
	sort.Strings(c)
	return c
}

// TestToolSeesExactlyToolEnv compares the tool process's whole environment
// with tool_env.
func TestToolSeesExactlyToolEnv(t *testing.T) {
	cases := []map[string]string{
		{},
		{"A": "1"},
		{"A": "1", "B": "x=y", "EMPTY": "", "UNI": "héllo 世界", "SP": " a b ", "NL": "line1\nline2", "PATH": "/usr/bin"},
		{"AGENTLOOP_API_KEY": "from-tool-env-not-the-real-key"},
		{"HOME": "/nonexistent", "LEAK_CANARY": "request-says-so"},
	}
	for i, env := range cases {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			_, _, dumps := toolWorld(t, env, "")
			if len(dumps) != 1 {
				t.Fatalf("%d dumps", len(dumps))
			}
			var want []string
			for k, v := range env {
				want = append(want, k+"="+v)
			}
			got := dumps[0].Env
			if strings.Join(sortedCopy(got), "\x00") != strings.Join(sortedCopy(want), "\x00") {
				t.Errorf("tool environment\n got  %q\n want %q", got, want)
			}
		})
	}
}

// TestToolIsolation: the key and the parent's environment never reach the
// tool, and the tool has no stray descriptors.
func TestToolIsolation(t *testing.T) {
	w, r, dumps := toolWorld(t, map[string]string{"X": "1"}, "")
	d := dumps[0]
	blob, _ := json.Marshal(d)
	for _, leak := range []string{chaosKey, "LEAK_CANARY", "parent-env"} {
		if strings.Contains(string(blob), leak) {
			t.Errorf("tool saw %q: %s", leak, blob)
		}
	}
	if strings.TrimSpace(d.Stdin) != `{"n":"toolu_iso_1"}` {
		t.Errorf("stdin = %q", d.Stdin)
	}
	if len(d.Args) != 2 || d.Args[1] != "ledger" {
		t.Errorf("argv = %q", d.Args)
	}
	if d.Cwd == "" || !strings.HasSuffix(d.Cwd, filepath.Base(w.dir)) {
		t.Errorf("cwd = %q, want tool_dir %q", d.Cwd, w.dir)
	}
	if got := strings.Join(sortedCopy(d.Fds), ","); got != "0,1,2" {
		t.Errorf("tool descriptors = %s, want 0,1,2", got)
	}
	if d.Pgid != d.Pid {
		t.Errorf("tool is not a process group leader (pgid %d, pid %d)", d.Pgid, d.Pid)
	}
	// The key did go to the API, and appears nowhere else.
	for name, s := range map[string]string{
		"stdout":  r.stdout,
		"stderr":  r.stderr,
		"session": readFileT(t, w.session),
	} {
		if strings.Contains(s, chaosKey) {
			t.Errorf("key in %s", name)
		}
	}
}

func readFileT(t *testing.T, p string) string {
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// TestKeyNeverLeaksOnErrors drives the failure paths of the model call and
// checks that the key is in none of the outputs.
func TestKeyNeverLeaksOnErrors(t *testing.T) {
	var seenKey []string
	var mu sync.Mutex
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		seenKey = append(seenKey, r.Header.Get("x-api-key")+r.Header.Get("Authorization"))
		mu.Unlock()
		switch r.URL.Query().Get("m") {
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(401)
		// A hostile server may even echo; the agent must still not add the key itself.
		fmt.Fprint(w, `{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}`)
	}))
	defer srv.Close()
	closed, _ := net.Listen("tcp", "127.0.0.1:0")
	closedURL := "http://" + closed.Addr().String()
	closed.Close()

	for name, base := range map[string]string{"401": srv.URL, "refused": closedURL} {
		t.Run(name, func(t *testing.T) {
			f := newValFixture(t)
			f.req["base_url"] = base
			f.req["tools"] = []any{}
			delete(f.req, "tool_command")
			r := f.exec(nil)
			if r.code != 0 {
				t.Fatalf("exit %d: %s", r.code, r.stderr)
			}
			out := r.outputLines()
			var o map[string]any
			json.Unmarshal(out[len(out)-1], &o)
			if o["reason"] != "model_error" {
				t.Fatalf("outcome %v", o)
			}
			for what, s := range map[string]string{"stdout": r.stdout, "stderr": r.stderr, "session": readFileT(t, f.session())} {
				if strings.Contains(s, chaosKey) {
					t.Errorf("key in %s: %s", what, s)
				}
			}
		})
	}
	mu.Lock()
	defer mu.Unlock()
	for _, k := range seenKey {
		if !strings.Contains(k, chaosKey) {
			t.Errorf("the API was not sent the key: %q", k)
		}
	}
}

// keyCmdFailures are the ways the key command can give no key, with the
// message the operator should see.
func keyCmdFailures(t *testing.T) map[string]struct {
	cmd  []string
	want string
} {
	dir := t.TempDir()
	loud := filepath.Join(dir, "loud.sh")
	os.WriteFile(loud, []byte("#!/bin/sh\necho "+chaosKey+"\necho 'keychain locked' >&2\nexit 1\n"), 0o755)
	return map[string]struct {
		cmd  []string
		want string
	}{
		"nonzero exit":              {[]string{"/bin/sh", "-c", "echo 'item not found' >&2; exit 1"}, "The api_key_command exited 1: item not found."},
		"exit, no text":             {[]string{"/bin/sh", "-c", "exit 3"}, "The api_key_command exited 3."},
		"stdout ignored on failure": {[]string{loud}, "The api_key_command exited 1: keychain locked."},
		"prints nothing":            {[]string{"/usr/bin/true"}, "The api_key_command printed no key."},
		"prints only whitespace":    {[]string{"/bin/sh", "-c", "printf ' \\n\\t\\n'"}, "The api_key_command printed no key."},
		"cannot start":              {[]string{filepath.Join(dir, "does-not-exist")}, "The api_key_command could not run:"},
		"not on PATH":               {[]string{"no-such-key-helper-xyz"}, "The api_key_command could not run:"},
	}
}

// TestNoKey: when the key command gives no key there is an unauthorized
// outcome, no HTTP request, and the session is left exactly as it was.
func TestNoKey(t *testing.T) {
	for name, c := range keyCmdFailures(t) {
		t.Run(name, func(t *testing.T) {
			f := newValFixture(t)
			f.req["api_key_command"] = c.cmd
			// An existing session with work pending must not be touched either.
			os.MkdirAll(filepath.Dir(f.session()), 0o755)
			before := `{"seq":1,"time":"2026-01-01T00:00:00Z","type":"run_started","run_started":{"prompt":"x"}}` + "\n"
			os.WriteFile(f.session(), []byte(before), 0o600)
			r := f.exec(nil)
			out := r.outputLines()
			if r.code != 0 || len(out) != 1 {
				t.Fatalf("exit %d, stdout %q, stderr %q", r.code, r.stdout, r.stderr)
			}
			var o struct {
				Type   string
				Reason string
				Error  struct{ Kind, Message string }
			}
			json.Unmarshal(out[0], &o)
			if o.Type != "outcome" || o.Reason != "model_error" || o.Error.Kind != "unauthorized" {
				t.Errorf("outcome %s", out[0])
			}
			if !strings.HasPrefix(o.Error.Message, c.want) {
				t.Errorf("message %q, want prefix %q", o.Error.Message, c.want)
			}
			if strings.Contains(r.stdout+r.stderr, chaosKey) {
				t.Errorf("the command's stdout leaked: %s %s", r.stdout, r.stderr)
			}
			if n := f.api.count(); n != 0 {
				t.Errorf("%d HTTP requests", n)
			}
			if got := readFileT(t, f.session()); got != before {
				t.Errorf("session changed: %q", got)
			}
		})
	}
}

// TestNoKeyChat: chat exits 2 with the same message and calls nothing.
func TestNoKeyChat(t *testing.T) {
	for name, c := range keyCmdFailures(t) {
		t.Run(name, func(t *testing.T) {
			w := newChaosWorld(t, newTestRand())
			w.req["api_key_command"] = c.cmd
			p := startChat(t, w, chaosEnv())
			code, signaled := p.wait(20 * time.Second)
			if signaled || code != 2 {
				t.Fatalf("exit %d (signaled %v), stderr %q", code, signaled, p.errb.String())
			}
			if !strings.Contains(p.errb.String(), c.want) {
				t.Errorf("stderr %q, want %q", p.errb.String(), c.want)
			}
			if strings.Contains(p.out.String()+p.errb.String(), chaosKey) {
				t.Error("the command's stdout leaked")
			}
			if n := w.api.count(); n != 0 {
				t.Errorf("%d HTTP requests", n)
			}
		})
	}
}

// buildKeycmd compiles testdata/keycmd.
func buildKeycmd(t *testing.T) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "keycmd")
	if out, err := exec.Command("go", "build", "-o", p, "./testdata/keycmd").CombinedOutput(); err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
	return p
}

// TestKeyCommandEnvironment: the key command gets PATH and HOME from
// agentloop's environment and nothing else, no stdin, and its own argv.
func TestKeyCommandEnvironment(t *testing.T) {
	keycmd := buildKeycmd(t)
	w := newChaosWorld(t, newTestRand())
	dumpPath := filepath.Join(t.TempDir(), "keyenv.json")
	w.req["api_key_command"] = []string{keycmd, chaosKey, dumpPath}
	w.turns = nil
	r := chaosExec(t, []string{"run", w.write("go", nil)}, append(chaosEnv(), "SECRET_PARENT_VAR=nope", "AGENTLOOP_API_KEY=nope"), nil)
	if r.code != 0 {
		t.Fatalf("exit %d: %s", r.code, r.stderr)
	}
	var d dump
	json.Unmarshal([]byte(readFileT(t, dumpPath)), &d)
	want := []string{"HOME=/nonexistent-home", "PATH=/usr/bin:/bin"}
	if got := sortedCopy(d.Env); strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Errorf("key command environment\n got  %q\n want %q", got, want)
	}
	if d.Stdin != "" {
		t.Errorf("key command stdin = %q", d.Stdin)
	}
	// Without HOME in agentloop's own environment, there is none to pass on.
	os.Remove(dumpPath)
	chaosExec(t, []string{"run", w.write("again", nil)}, []string{"PATH=/usr/bin:/bin"}, nil)
	d = dump{}
	json.Unmarshal([]byte(readFileT(t, dumpPath)), &d)
	if len(d.Env) != 1 || d.Env[0] != "PATH=/usr/bin:/bin" {
		t.Errorf("key command environment without HOME: %q", d.Env)
	}
}

// TestKeyCommandStdoutIsOnlyTheKey: the key is the trimmed stdout, and
// reaches the API and nothing else.
func TestKeyCommandKeyReachesOnlyTheAPI(t *testing.T) {
	f := newValFixture(t)
	f.req["api_key_command"] = []string{"/bin/sh", "-c", "printf '  \\n%s\\n\\n' " + chaosKey}
	f.req["tools"] = []any{}
	delete(f.req, "tool_command")
	var got string
	f.api.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Get("x-api-key")
		w.WriteHeader(401)
		fmt.Fprint(w, `{"type":"error","error":{"type":"authentication_error","message":"no"}}`)
	})
	r := f.exec(nil)
	if got != chaosKey {
		t.Errorf("the API was sent %q, want the trimmed key", got)
	}
	for what, s := range map[string]string{"stdout": r.stdout, "stderr": r.stderr, "session": readFileT(t, f.session())} {
		if strings.Contains(s, chaosKey) {
			t.Errorf("key in %s", what)
		}
	}
}

// TestFetchKeyTimeout runs the command in process with a short timeout, so
// the 30 second limit need not be waited for. A child that keeps the pipes
// open must not hold the wait past the deadline.
func TestFetchKeyTimeout(t *testing.T) {
	start := time.Now()
	_, err := fetchKey(context.Background(), []string{"/bin/sh", "-c", "sleep 30 & sleep 30"}, 300*time.Millisecond)
	if err == nil || !strings.Contains(err.Error(), "did not finish within") {
		t.Fatalf("err = %v", err)
	}
	if d := time.Since(start); d > 5*time.Second {
		t.Errorf("took %v", d)
	}
	if keyTimeout != 30*time.Second {
		t.Errorf("keyTimeout = %v", keyTimeout)
	}
}

func TestFetchKeyTrimsAndFails(t *testing.T) {
	key, err := fetchKey(context.Background(), []string{"printf", "  k-1 \n"}, time.Minute)
	if err != nil || key != "k-1" {
		t.Errorf("key %q, err %v", key, err)
	}
	long := strings.Repeat("e", 1000)
	_, err = fetchKey(context.Background(), []string{"/bin/sh", "-c", "echo " + long + " >&2; exit 4"}, time.Minute)
	if err == nil || !strings.HasPrefix(err.Error(), "The api_key_command exited 4: ...") || len(err.Error()) > 300 {
		t.Errorf("err = %v", err)
	}
}
