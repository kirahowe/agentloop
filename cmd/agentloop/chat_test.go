package main

import (
	"bytes"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

// chatProc is a running `agentloop chat` with a stdin the test controls.
type chatProc struct {
	t      *testing.T
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	out    *bytes.Buffer
	errb   *syncBuf
	done   chan error
	waited bool
}

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

func startChat(t *testing.T, w *chaosWorld, env []string) *chatProc {
	t.Helper()
	path := w.write("", nil)
	cmd := exec.Command(binary, "chat", path)
	cmd.Env = env
	in, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	p := &chatProc{t: t, cmd: cmd, stdin: in, out: &bytes.Buffer{}, errb: newSyncBuf(), done: make(chan error, 1)}
	cmd.Stdout = p.out
	cmd.Stderr = p.errb
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	go func() { p.done <- cmd.Wait() }()
	t.Cleanup(func() {
		if !p.waited {
			cmd.Process.Kill()
			<-p.done
		}
	})
	return p
}

func (p *chatProc) wait(timeout time.Duration) (code int, signaled bool) {
	p.t.Helper()
	select {
	case err := <-p.done:
		p.waited = true
		if ee, ok := err.(*exec.ExitError); ok {
			if ws, ok := ee.Sys().(syscall.WaitStatus); ok && ws.Signaled() {
				return -1, true
			}
			return ee.ExitCode(), false
		}
		return 0, false
	case <-time.After(timeout):
		p.t.Fatalf("chat did not exit in %v; stderr %q", timeout, p.errb.String())
	}
	return 0, false
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func chatWorld(t *testing.T, script string) *chaosWorld {
	w := newChaosWorld(t, newTestRand())
	w.turns = [][]string{{"toolu_chat_1"}}
	w.api.turns = w.turns
	os.WriteFile(filepath.Join(w.dir, "tool.sh"), []byte(script), 0o755)
	return w
}

func pidsOf(t *testing.T, path string) []int {
	raw, _ := os.ReadFile(path)
	var pids []int
	for _, f := range strings.Fields(string(raw)) {
		if n, err := strconv.Atoi(f); err == nil {
			pids = append(pids, n)
		}
	}
	return pids
}

func assertDead(t *testing.T, pids []int) {
	t.Helper()
	deadline := time.Now().Add(6 * time.Second)
	for _, pid := range pids {
		for syscall.Kill(pid, 0) == nil && time.Now().Before(deadline) {
			time.Sleep(10 * time.Millisecond)
		}
		if syscall.Kill(pid, 0) == nil {
			t.Errorf("process %d still running", pid)
			syscall.Kill(pid, syscall.SIGKILL)
		}
	}
}

func TestChatInputShapes(t *testing.T) {
	cases := map[string]struct {
		in       string
		requests int
	}{
		"empty stdin":               {"", 0},
		"only blank lines":          {"\n\n  \n\t\n", 0},
		"last line without newline": {"hello", 1},
		"two lines":                 {"a\nb\n", 2},
		"crlf":                      {"a\r\n", 1},
		"huge line":                 {strings.Repeat("x", 3<<20) + "\n", 1},
		"unicode":                   {"héllo 世界 \U0001F600\n", 1},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			w := chatWorld(t, ledgerScript)
			p := startChat(t, w, chaosEnv())
			io.WriteString(p.stdin, c.in)
			p.stdin.Close()
			if code, sig := p.wait(30 * time.Second); code != 0 || sig {
				t.Fatalf("exit %d (signaled %v); stderr %q", code, sig, p.errb.String())
			}
			// A chat turn that needs no tool: the script only calls the
			// tool when the conversation has no assistant turn yet, so the
			// requests counted are at least one per non-blank line.
			if n := w.api.count(); n < c.requests {
				t.Errorf("%d API requests, want at least %d", n, c.requests)
			}
			if c.requests == 0 && w.api.count() != 0 {
				t.Errorf("API called on empty input")
			}
		})
	}
}

func TestChatNoKey(t *testing.T) {
	w := chatWorld(t, ledgerScript)
	p := startChat(t, w, []string{"PATH=/usr/bin:/bin"})
	p.stdin.Close()
	if code, _ := p.wait(10 * time.Second); code != 2 {
		t.Errorf("exit %d, want 2; stderr %q", code, p.errb.String())
	}
}

func TestChatRefusesSessionInUse(t *testing.T) {
	w := chatWorld(t, ledgerScript)
	first := startChat(t, w, chaosEnv())
	waitFor(t, "the lock file", func() bool { _, err := os.Stat(w.session + ".lock"); return err == nil })
	time.Sleep(50 * time.Millisecond)
	second := startChat(t, w, chaosEnv())
	second.stdin.Close()
	if code, _ := second.wait(10 * time.Second); code != 1 || !strings.Contains(second.errb.String(), "in use") {
		t.Errorf("second chat: exit %d, stderr %q", code, second.errb.String())
	}
	first.stdin.Close()
	if code, _ := first.wait(10 * time.Second); code != 0 {
		t.Errorf("first chat: exit %d", code)
	}
}

const slowToolScript = `#!/bin/sh
cat > /dev/null
echo $$ >> "$PIDS"
sleep 30 &
echo $! >> "$PIDS"
wait
`

// TestChatSIGINTCancelsRunNotChat: SIGINT while a run is going cancels the
// run (and the tool with its children) and the chat carries on.
func TestChatSIGINTCancelsRunNotChat(t *testing.T) {
	w := chatWorld(t, slowToolScript)
	p := startChat(t, w, chaosEnv())
	io.WriteString(p.stdin, "go\n")
	waitFor(t, "the tool to start", func() bool { return len(pidsOf(t, w.pids)) >= 2 })
	p.cmd.Process.Signal(syscall.SIGINT)
	waitFor(t, "the run to be reported canceled", func() bool { return strings.Contains(p.errb.String(), "[canceled]") })
	assertDead(t, pidsOf(t, w.pids))
	// Still alive and taking input: a second line works.
	io.WriteString(p.stdin, "again\n")
	waitFor(t, "a second prompt", func() bool { return strings.Count(p.errb.String(), "> ") >= 3 })
	p.stdin.Close()
	if code, sig := p.wait(30 * time.Second); code != 0 || sig {
		t.Fatalf("exit %d signaled %v; stderr %q", code, sig, p.errb.String())
	}
	w.checkSession(false)
}

// TestChatSIGTERMDuringRun: SIGTERM stops the run and its tool tree, exits
// 130, and leaves a session that resumes.
func TestChatSIGTERMDuringRun(t *testing.T) {
	w := chatWorld(t, slowToolScript)
	p := startChat(t, w, chaosEnv())
	io.WriteString(p.stdin, "go\n")
	waitFor(t, "the tool to start", func() bool { return len(pidsOf(t, w.pids)) >= 2 })
	p.cmd.Process.Signal(syscall.SIGTERM)
	code, sig := p.wait(30 * time.Second)
	if code != 130 || sig {
		t.Errorf("exit %d signaled %v, want 130", code, sig)
	}
	assertDead(t, pidsOf(t, w.pids))
	// The log is sound: the interrupted call has its result.
	w.checkSession(false)
	sess := readSession(t, w.session)
	if len(sess.Pending()) != 0 {
		t.Errorf("pending after a clean SIGTERM: %v", sess.Pending())
	}
}

func TestChatSignalsWhileIdle(t *testing.T) {
	for _, sig := range []syscall.Signal{syscall.SIGINT, syscall.SIGTERM} {
		t.Run(sig.String(), func(t *testing.T) {
			w := chatWorld(t, ledgerScript)
			p := startChat(t, w, chaosEnv())
			waitFor(t, "the prompt", func() bool { return strings.Contains(p.errb.String(), "> ") })
			time.Sleep(50 * time.Millisecond)
			p.cmd.Process.Signal(sig)
			if code, _ := p.wait(10 * time.Second); code != 130 {
				t.Errorf("exit %d, want 130", code)
			}
		})
	}
}
