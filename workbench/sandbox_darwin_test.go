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

	"golang.org/x/sys/unix"
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

// Another process's command line can hold a password or a token, and
// sysctl is where it is read from.
func TestSandboxCannotReadOtherProcessesArguments(t *testing.T) {
	w := newWorld(t, Limits{})
	marker := "WB-ARG-CANARY-7731"
	other := exec.Command("/bin/sleep", "60", marker)
	if err := other.Start(); err != nil {
		t.Fatal(err)
	}
	defer other.Process.Kill()
	time.Sleep(200 * time.Millisecond)
	// Through the kernel directly: KERN_PROCARGS2 is {1, 49, pid}.
	r := w.run(t, `python3 -c '
import ctypes, ctypes.util, sys
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
mib = (ctypes.c_int * 3)(1, 49, `+strconv.Itoa(other.Process.Pid)+`)
size = ctypes.c_size_t(1 << 20)
buf = ctypes.create_string_buffer(size.value)
if libc.sysctl(mib, 3, buf, ctypes.byref(size), None, 0) != 0:
    print("refused", ctypes.get_errno())
else:
    print("got", buf.raw[:size.value])
'`)
	if strings.Contains(r.Output, marker) {
		t.Errorf("another process's arguments were readable: %q", r.Output)
	}
	// And through sysctl(8), which is how anyone would try first.
	if out := w.run(t, "sysctl kern.procargs2").Output; strings.Contains(out, marker) {
		t.Errorf("sysctl(8) read them: %q", out)
	}
	// What the C library needs still works.
	if r := w.run(t, "sort -u /etc/shells > sorted; shasum -a 256 /etc/shells | cut -c1-6; python3 -c 'import os; print(os.cpu_count())'; perl -e 'print 42'"); r.Exit != 0 || !strings.Contains(r.Output, "42") {
		t.Errorf("the sysctl allowlist is too narrow: %+v", r)
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

// The limit stops a command while it runs, not only afterwards: a loop
// writing file after file, each inside the file size limit, would
// otherwise fill the disk.
func TestSandboxScratchLimitStopsACommandThatKeepsWriting(t *testing.T) {
	const limit = 4 << 20
	w := newWorld(t, Limits{Timeout: 30 * time.Second, FileBytes: 1 << 20, ScratchBytes: limit})
	start := time.Now()
	r := w.run(t, "i=0; while :; do dd if=/dev/zero of=f$i bs=1024 count=1000 2>/dev/null; i=$((i+1)); done")
	if !r.ScratchFull {
		t.Fatalf("the command was not stopped for filling the scratch directory: %+v", r)
	}
	if took := time.Since(start); took > 20*time.Second {
		t.Errorf("it ran for %s", took)
	}
	// It is stopped soon after the limit, not left running.
	if n, _ := size(w.box.scratch); n > limit+200<<20 {
		t.Errorf("the scratch directory grew to %d bytes, limit %d", n, limit)
	}
	if n := w.box.left(); n != 0 {
		t.Errorf("%d processes were left", n)
	}
	if r := w.run(t, "echo again"); !r.ScratchFull || strings.Contains(r.Output, "again") {
		t.Errorf("a command ran after scratch filled: %+v", r)
	}
}

// A command can delete its own scratch directory, and then there is
// nothing left to ask how full the disk is; and it can take the
// permissions off a directory it made, so that neither the size walk nor
// the cleanup can see inside.
func TestSandboxScratchCannotBeHiddenFromTheLimit(t *testing.T) {
	w := newWorld(t, Limits{Timeout: 10 * time.Second})
	if r := w.run(t, "mkdir -p sub && echo x > sub/f && chmod 000 sub"); r.Exit != 0 {
		t.Fatalf("setup: %+v", r)
	}
	if !w.box.overFull() {
		t.Error("a scratch directory that cannot be walked was taken as within its limit")
	}
	// close puts the permissions back and removes everything.
	root := w.box.root
	w.box.close()
	if w.box.leftover != nil {
		t.Errorf("the session directory was left behind: %v", w.box.leftover)
	}
	if _, err := os.Stat(root); !os.IsNotExist(err) {
		t.Errorf("stat %s = %v, want not exist", root, err)
	}

	// And with the scratch directory gone, the next command is refused
	// rather than left unbounded.
	w2 := newWorld(t, Limits{Timeout: 10 * time.Second})
	if r := w2.run(t, "cd / && rm -rf "+w2.box.scratch); r.Exit != 0 {
		t.Fatalf("the scratch directory could not be removed by its own command: %+v", r)
	}
	if !w2.box.overFull() {
		t.Error("a missing scratch directory was taken as within its limit")
	}
}

// A command can unlink a file it is still writing, so no walk of the
// scratch directory sees its size. The floor under the filesystem's free
// space is what stops that.
func TestSandboxStopsACommandWritingUnlinkedFiles(t *testing.T) {
	w := newWorld(t, Limits{Timeout: 30 * time.Second, FileBytes: 1 << 30, ScratchBytes: 1 << 30})
	var st unix.Statfs_t
	if err := unix.Statfs(w.box.scratch, &st); err != nil {
		t.Fatal(err)
	}
	free := int64(st.Bavail) * int64(st.Bsize)
	// A floor just under what is free now: the next few megabytes cross it.
	w.box.limits.MinFreeBytes = free - 8<<20
	start := time.Now()
	r := w.run(t, `python3 -c '
import os
f = open("gone", "wb")
os.unlink("gone")
block = b"x" * (1 << 20)
while True:
    f.write(block)
    f.flush()
'`)
	if !r.ScratchFull {
		t.Fatalf("the command was not stopped as the disk filled: %+v", r)
	}
	if took := time.Since(start); took > 20*time.Second {
		t.Errorf("it ran for %s", took)
	}
	if n := w.box.left(); n != 0 {
		t.Errorf("%d processes were left", n)
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
