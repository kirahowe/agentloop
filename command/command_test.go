package command

import (
	"context"
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func sh(script string) *Tool {
	return &Tool{Argv: []string{"/bin/sh", "-c", script}, Env: []string{"PATH=/usr/bin:/bin"}}
}

func call(t *testing.T, tool *Tool, in string) (string, bool, error) {
	t.Helper()
	r, err := tool.Call(context.Background(), json.RawMessage(in))
	return r.Content, r.IsError, err
}

func TestEnvIsolation(t *testing.T) {
	t.Setenv("PARENT_SECRET", "leak")
	out, _, err := call(t, sh("env"), `{}`)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(out, "PARENT_SECRET") || strings.Contains(out, "HOME=") {
		t.Errorf("parent env leaked: %s", out)
	}
	tool := sh("env")
	tool.Env = nil
	out, _, _ = call(t, tool, `{}`)
	if strings.Contains(out, "PATH=") && strings.Contains(out, "PARENT") {
		t.Errorf("nil env leaked: %s", out)
	}
	tool = sh("echo $FOO")
	tool.Env = []string{"FOO=bar"}
	out, _, _ = call(t, tool, `{}`)
	if out != "bar" {
		t.Errorf("out = %q", out)
	}
}

func TestStdinAndDir(t *testing.T) {
	dir := t.TempDir()
	tool := sh("cat; pwd")
	tool.Dir = dir
	out, isErr, err := call(t, tool, `{"a":1}`)
	if err != nil || isErr {
		t.Fatal(err, isErr)
	}
	if !strings.HasPrefix(out, `{"a":1}`) || !strings.HasSuffix(out, "/"+dirBase(dir)) {
		t.Errorf("out = %q", out)
	}
}

func dirBase(p string) string { return p[strings.LastIndex(p, "/")+1:] }

func TestExitCodesAndStderr(t *testing.T) {
	out, isErr, err := call(t, sh("echo fine; echo warn >&2"), `{}`)
	if err != nil || isErr || out != "fine" {
		t.Errorf("ok: %q %v %v", out, isErr, err)
	}
	out, isErr, _ = call(t, sh("echo only-err >&2"), `{}`)
	if isErr || out != "only-err" {
		t.Errorf("stderr fallback: %q %v", out, isErr)
	}
	out, isErr, err = call(t, sh("echo o; echo e >&2; exit 3"), `{}`)
	if err != nil || !isErr || out != "o\ne" {
		t.Errorf("fail: %q %v %v", out, isErr, err)
	}
	out, isErr, _ = call(t, sh("exit 4"), `{}`)
	if !isErr || out != "exited 4" {
		t.Errorf("bare exit: %q %v", out, isErr)
	}
}

func TestOutputCap(t *testing.T) {
	tool := sh("head -c 100000 /dev/zero | tr '\\0' x")
	tool.MaxOutputBytes = 100
	out, isErr, err := call(t, tool, `{}`)
	if err != nil || isErr {
		t.Fatal(err, isErr)
	}
	if !strings.HasPrefix(out, strings.Repeat("x", 100)) || strings.Count(out, "x") != 100 || !strings.Contains(out, "cut") {
		t.Errorf("out = %q", out)
	}
}

func TestMissingBinary(t *testing.T) {
	tool := &Tool{Argv: []string{"/nonexistent/prog"}}
	if _, _, err := call(t, tool, `{}`); err == nil {
		t.Error("want error")
	}
}

func alive(pid int) bool {
	return syscall.Kill(pid, 0) == nil
}

func TestCancelKillsGrandchild(t *testing.T) {
	pidFile := t.TempDir() + "/pid"
	// The shell starts a background grandchild, records its pid, then waits.
	tool := sh("sleep 60 & echo $! > " + pidFile + "; wait")
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := tool.Call(ctx, json.RawMessage(`{}`))
		done <- err
	}()
	var pid int
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		b, _ := os.ReadFile(pidFile)
		if n, err := strconv.Atoi(strings.TrimSpace(string(b))); err == nil {
			pid = n
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if pid == 0 {
		t.Fatal("grandchild never started")
	}
	t.Cleanup(func() { syscall.Kill(pid, syscall.SIGKILL) })
	cancel()
	select {
	case err := <-done:
		if err != context.Canceled {
			t.Errorf("err = %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("Call did not return")
	}
	for i := 0; i < 100 && alive(pid); i++ {
		time.Sleep(20 * time.Millisecond)
	}
	if alive(pid) {
		t.Errorf("grandchild %d still running", pid)
	}
}
