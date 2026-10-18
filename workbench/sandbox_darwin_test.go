//go:build darwin

package workbench

import (
	"context"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
)

// These run real commands under real sandbox-exec, on files made here.

type world struct {
	box             *box
	allowed, denied string // a readable file, and one on the deny list
	outside         string // a directory outside scratch
}

func newWorld(t *testing.T, l Limits) *world {
	t.Helper()
	if err := Available(); err != nil {
		t.Skip(err)
	}
	base := Resolve(t.TempDir())
	w := &world{allowed: filepath.Join(base, "allowed", "notes.txt"), outside: filepath.Join(base, "outside")}
	denyDir := filepath.Join(base, "secret")
	w.denied = filepath.Join(denyDir, "key.txt")
	for _, d := range []string{filepath.Dir(w.allowed), denyDir, w.outside} {
		if err := os.MkdirAll(d, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	os.WriteFile(w.allowed, []byte("concert on 2027-03-14, 120 seats\n"), 0o600)
	os.WriteFile(w.denied, []byte("SECRET-KEY-6612\n"), 0o600)
	d := DefaultLimits
	if l.Timeout != 0 {
		d.Timeout = l.Timeout
	}
	if l.OutputBytes != 0 {
		d.OutputBytes = l.OutputBytes
	}
	if l.FileBytes != 0 {
		d.FileBytes = l.FileBytes
	}
	if l.ScratchBytes != 0 {
		d.ScratchBytes = l.ScratchBytes
	}
	b, err := newBox(base, []string{denyDir}, d)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(b.close)
	w.box = b
	return w
}

func (w *world) run(t *testing.T, command string) CommandResult {
	t.Helper()
	r, err := w.box.run(context.Background(), command)
	if err != nil {
		t.Fatalf("%s: %v", command, err)
	}
	return r
}

func TestSandboxAllowsWork(t *testing.T) {
	w := newWorld(t, Limits{})
	cases := map[string]string{
		"cat " + w.allowed:                                                    "120 seats",
		"grep -c seats " + w.allowed:                                          "1",
		"find " + filepath.Dir(w.allowed) + " -name '*.txt'":                  "notes.txt",
		`python3 -c 'print(sum(range(101)))'`:                                 "5050",
		`sqlite3 :memory: 'select 6*7'`:                                       "42",
		`echo '{"a":[1,2,3]}' | jq '.a|add'`:                                  "6",
		"echo kept > note && cat note && pwd":                                 w.box.scratch,
		"cat <<EOF\nheredoc\nEOF":                                             "heredoc",
		"mktemp":                                                              w.box.scratch,
		"echo out > /dev/stdout; echo err > /dev/stderr":                      "err",
		"python3 -c 'import json,csv,sqlite3,statistics; print(\"modules\")'": "modules",
	}
	if p, err := exec.LookPath("rg"); err == nil {
		cases[p+" -c seats "+w.allowed] = "1"
	}
	for cmd, want := range cases {
		r := w.run(t, cmd)
		if r.Exit != 0 || !strings.Contains(r.Output, want) {
			t.Errorf("%s: exit %d, output %q, want %q", cmd, r.Exit, r.Output, want)
		}
	}
}

// failsWithout runs command and checks it failed and printed nothing
// holding leak.
func (w *world) fails(t *testing.T, command, leak string) CommandResult {
	t.Helper()
	r := w.run(t, command)
	if r.Exit == 0 {
		t.Errorf("%s: exit 0, output %q", command, r.Output)
	}
	if leak != "" && strings.Contains(r.Output, leak) {
		t.Errorf("%s: output holds %q: %q", command, leak, r.Output)
	}
	return r
}

func TestSandboxDeniesReadsOnTheList(t *testing.T) {
	w := newWorld(t, Limits{})
	w.fails(t, "cat "+w.denied, "SECRET-KEY")
	w.fails(t, "ls "+filepath.Dir(w.denied), "key.txt")
	// The same file by other names.
	w.fails(t, "cat "+strings.Replace(w.denied, "/secret/", "/SECRET/", 1), "SECRET-KEY")
	w.fails(t, "cat /System/Volumes/Data"+w.denied, "SECRET-KEY")
	w.fails(t, "ln -s "+w.denied+" link && cat link", "SECRET-KEY")
	w.fails(t, "ln "+w.denied+" hard && cat hard", "SECRET-KEY")
	w.fails(t, "cd "+filepath.Dir(w.denied)+" && cat key.txt", "SECRET-KEY")
	var st syscall.Stat_t
	if err := syscall.Stat(w.denied, &st); err == nil {
		w.fails(t, "cat /.vol/"+strconv.Itoa(int(st.Dev))+"/"+strconv.FormatUint(st.Ino, 10), "SECRET-KEY")
	}
	// The probe and the session directory are hidden too.
	w.fails(t, "ls "+w.box.probe, "")
	w.fails(t, "ls "+w.box.root, "probe")
}

func TestSandboxWritesOnlyScratch(t *testing.T) {
	w := newWorld(t, Limits{})
	target := filepath.Join(w.outside, "written")
	w.fails(t, "echo x > "+target, "")
	w.fails(t, "touch "+target, "")
	w.fails(t, "mkdir "+target, "")
	w.fails(t, "cp "+w.allowed+" "+target, "")
	w.fails(t, "ln "+w.allowed+" hl && echo changed >> hl", "")
	w.fails(t, "rm "+w.allowed, "")
	w.fails(t, "mv "+w.allowed+" .", "")
	if _, err := os.Stat(target); err == nil {
		t.Errorf("%s was written", target)
	}
	if b, _ := os.ReadFile(w.allowed); !strings.Contains(string(b), "120 seats") {
		t.Errorf("%s was changed: %q", w.allowed, b)
	}
}

func TestSandboxHasNoNetwork(t *testing.T) {
	w := newWorld(t, Limits{Timeout: 10 * time.Second})
	// A listener on loopback that counts connections.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Skipf("cannot listen on loopback here: %v", err)
	}
	defer ln.Close()
	var conns atomic.Int32
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			conns.Add(1)
			c.Close()
		}
	}()
	addr := ln.Addr().String()
	host, port, _ := net.SplitHostPort(addr)
	w.fails(t, "curl -sS -m 5 http://"+addr+"/", "")
	w.fails(t, "nc -z -w 3 "+host+" "+port, "")
	w.fails(t, `python3 -c 'import socket; socket.create_connection(("`+host+`", `+port+`), 3)'`, "")
	w.fails(t, `python3 -c 'import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.sendto(b"x", ("`+host+`", `+port+`))'`, "")
	// A sandbox inside the sandbox cannot loosen it.
	w.fails(t, SandboxExec+" -p '(version 1)(allow default)' curl -sS -m 5 http://"+addr+"/", "")
	if n := conns.Load(); n != 0 {
		t.Errorf("the listener got %d connections", n)
	}

	w.fails(t, "dig +time=2 +tries=1 @127.0.0.1 example.com", "")

	// A unix socket outside the sandbox.
	dir, err := os.MkdirTemp("/tmp", "wb")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	sock := filepath.Join(dir, "s")
	ul, err := net.Listen("unix", sock)
	if err != nil {
		t.Skipf("cannot listen on a unix socket here: %v", err)
	}
	defer ul.Close()
	var uconns atomic.Int32
	go func() {
		for {
			c, err := ul.Accept()
			if err != nil {
				return
			}
			uconns.Add(1)
			c.Close()
		}
	}()
	w.fails(t, "echo hi | nc -U "+sock, "")
	w.fails(t, `python3 -c 'import socket; s=socket.socket(socket.AF_UNIX); s.connect("`+sock+`")'`, "")
	if n := uconns.Load(); n != 0 {
		t.Errorf("the unix socket got %d connections", n)
	}
}

// lookUp asks launchd for a mach service by name, as a client of the
// service would, and exits 0 when it is found.
const lookUp = `python3 -c 'import ctypes,sys
lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
port = ctypes.c_uint32()
bp = ctypes.c_uint32.in_dll(lib, "bootstrap_port")
sys.exit(lib.bootstrap_look_up(bp, sys.argv[1].encode(), ctypes.byref(port)))' `

func TestSandboxReachesNoSystemService(t *testing.T) {
	w := newWorld(t, Limits{Timeout: 20 * time.Second})
	// The services that would be a way out: DNS (mDNSResponder),
	// LaunchServices, AppleEvents, the pasteboard, Spotlight, user
	// notifications and launchd's job submission.
	for _, svc := range []string{"com.apple.dnssd.service", "com.apple.coreservices.launchservicesd",
		"com.apple.coreservices.appleevents", "com.apple.pasteboard.1", "com.apple.metadata.mds",
		"com.apple.usernoted.client", "com.apple.system.opendirectoryd.libinfo", "com.apple.cfprefsd.daemon"} {
		if err := exec.Command("/bin/sh", "-c", lookUp+svc).Run(); err != nil {
			t.Logf("%s is not found outside the sandbox either (%v); checked anyway", svc, err)
		}
		w.fails(t, lookUp+svc, "")
	}

	// LaunchServices, AppleEvents and the pasteboard. Each would be a way
	// out: a browser fetches a URL, an app sends what it is told.
	w.fails(t, "open https://example.invalid/", "")
	w.fails(t, "osascript -e 'tell application \"Finder\" to count windows'", "")
	w.fails(t, "echo WB-CLIP-MARKER | pbcopy", "")
	w.fails(t, "pbpaste", "")
	w.fails(t, "launchctl submit -l wb.test.never -- /usr/bin/true", "")
	w.fails(t, "crontab -l", "")
	if out, _ := exec.Command("launchctl", "list", "wb.test.never").CombinedOutput(); !strings.Contains(string(out), "Could not find") {
		t.Errorf("launchctl knows wb.test.never: %s", out)
	}
}

func TestSandboxContainsRunaways(t *testing.T) {
	w := newWorld(t, Limits{Timeout: 3 * time.Second, OutputBytes: 1000, FileBytes: 64 << 10})

	start := time.Now()
	r := w.run(t, "while :; do :; done")
	if !r.TimedOut || time.Since(start) > 10*time.Second {
		t.Errorf("a loop was not stopped: %+v after %s", r, time.Since(start))
	}

	start = time.Now()
	r = w.run(t, "f() { f | f & }; f")
	if time.Since(start) > 15*time.Second {
		t.Errorf("a fork bomb ran %s", time.Since(start))
	}
	if n := w.box.left(); n != 0 {
		t.Errorf("%d processes of the fork bomb were left", n)
	}

	// A process that leaves the command's group and session.
	r = w.run(t, `python3 -c 'import os,time
if os.fork() == 0:
    os.setsid()
    time.sleep(300)
print("parent done")'`)
	if r.Swept < 1 {
		t.Errorf("the escaped process was not swept: %+v", r)
	}
	if n := w.box.left(); n != 0 {
		t.Errorf("%d processes left after the sweep", n)
	}

	r = w.run(t, "yes | head -c 100000")
	if !r.Truncated || len(r.Output) != 1000 {
		t.Errorf("output not cut: %d bytes, truncated %v", len(r.Output), r.Truncated)
	}

	r = w.fails(t, "dd if=/dev/zero of=big bs=1024 count=200", "")
	if fi, err := os.Stat(filepath.Join(w.box.scratch, "big")); err == nil && fi.Size() > 64<<10 {
		t.Errorf("a file grew to %d bytes", fi.Size())
	}
}

func TestSandboxScratchLimit(t *testing.T) {
	w := newWorld(t, Limits{FileBytes: 64 << 10, ScratchBytes: 100 << 10})
	r := w.run(t, "for i in 1 2 3; do dd if=/dev/zero of=f$i bs=1024 count=60 2>/dev/null; done")
	if !r.ScratchFull {
		t.Fatalf("scratch over its limit was not noticed: %+v", r)
	}
	if r := w.run(t, "echo again"); !r.ScratchFull || strings.Contains(r.Output, "again") {
		t.Errorf("a command ran after scratch filled: %+v", r)
	}
}

func TestSweepLeavesOtherProcessesAlone(t *testing.T) {
	w := newWorld(t, Limits{})
	// A process in another sandbox that may write everywhere, and one in
	// none.
	other := exec.Command(SandboxExec, "-p", "(version 1)(allow default)", "/bin/sleep", "30")
	plain := exec.Command("/bin/sleep", "30")
	for _, c := range []*exec.Cmd{other, plain} {
		if err := c.Start(); err != nil {
			t.Fatal(err)
		}
		defer c.Process.Kill()
	}
	time.Sleep(200 * time.Millisecond)
	if n := w.box.sweep(); n != 0 {
		t.Fatalf("the sweep killed %d processes that are not the workbench's", n)
	}
	for _, c := range []*exec.Cmd{other, plain} {
		if err := c.Process.Signal(syscall.Signal(0)); err != nil {
			t.Errorf("pid %d is gone: %v", c.Process.Pid, err)
		}
	}
}
