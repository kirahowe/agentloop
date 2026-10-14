package command

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
	"unicode/utf8"
)

var (
	propTrials = flag.Int("command.trials", 0, "number of output property trials (0 means the default)")
	propSeed   = flag.Int64("command.seed", 0, "seed for the property tests (0 means time based; it is logged)")
)

func propSeedValue(t *testing.T) int64 {
	seed := *propSeed
	if seed == 0 {
		seed = time.Now().UnixNano() & 0xffffffff
	}
	t.Logf("seed = %d (rerun with -command.seed=%d)", seed, seed)
	return seed
}

// genBytes makes n bytes of one of several flavours.
func genBytes(rng *rand.Rand, n int) []byte {
	b := make([]byte, n)
	switch rng.Intn(7) {
	case 0: // plain ASCII
		for i := range b {
			b[i] = byte('a' + rng.Intn(26))
		}
	case 1: // all bytes, so invalid UTF-8 and NUL
		rng.Read(b)
	case 2: // whitespace at the edges, text in the middle
		for i := range b {
			b[i] = byte('a' + rng.Intn(26))
		}
		for i := 0; i < len(b) && i < 3; i++ {
			b[i] = " \n\t\r"[rng.Intn(4)]
		}
		for i := 0; i < len(b) && i < 3; i++ {
			b[len(b)-1-i] = " \n\t\r"[rng.Intn(4)]
		}
	case 3: // all whitespace
		for i := range b {
			b[i] = " \n\t\r\v\f"[rng.Intn(6)]
		}
	case 4: // multi-byte runes, so caps can split one
		s := strings.Repeat("é世\U0001F600", n)
		b = []byte(s)[:n]
	case 5: // lines
		for i := range b {
			b[i] = "ab\n"[rng.Intn(3)]
		}
	default: // NULs and high bytes only
		for i := range b {
			b[i] = []byte{0, 0xff, 0xc3, 0x28}[rng.Intn(4)]
		}
	}
	return b
}

// genSize picks sizes clustered on the cap.
func genSize(rng *rand.Rand, max int) int {
	switch rng.Intn(6) {
	case 0:
		return 0
	case 1:
		return rng.Intn(20)
	case 2:
		return max - 1
	case 3:
		return max
	case 4:
		return max + 1
	}
	return max + rng.Intn(3*max)
}

func trim(s string) string { return strings.TrimSpace(s) }

// docText is the documented rule for one stream: the first max bytes,
// trimmed, with a note if there was more.
func docText(b []byte, max int) string {
	cut := len(b) > max
	if cut {
		b = b[:max]
	}
	s := trim(string(b))
	if cut {
		s += fmt.Sprintf("\n[output cut at %d bytes]", max)
	}
	return trim(s)
}

// TestOutputProperties: for random stdout, stderr and exit status, the
// result is what the package comment and DESIGN.md say.
func TestOutputProperties(t *testing.T) {
	n := *propTrials
	if n == 0 {
		n = 400
		if testing.Short() {
			n = 80
		}
	}
	seed := propSeedValue(t)
	rng := rand.New(rand.NewSource(seed))
	dir := t.TempDir()
	outFile, errFile := filepath.Join(dir, "out"), filepath.Join(dir, "err")
	counts := map[string]int{}
	for i := 0; i < n; i++ {
		max := 1 + rng.Intn(64)
		if rng.Intn(4) == 0 {
			max = 0 // the default cap
		}
		eff := max
		if eff == 0 {
			eff = DefaultMaxOutputBytes
		}
		var out, errb []byte
		if max == 0 {
			// Around the default cap only occasionally: it is a megabyte.
			if rng.Intn(10) == 0 {
				out = genBytes(rng, eff+rng.Intn(3)-1)
			} else {
				out = genBytes(rng, rng.Intn(200))
			}
			errb = genBytes(rng, rng.Intn(200))
		} else {
			out, errb = genBytes(rng, genSize(rng, max)), genBytes(rng, genSize(rng, max))
		}
		code := []int{0, 0, 0, 0, 0, 1, 2, 3, 42, 127, 255, -1}[rng.Intn(12)]
		os.WriteFile(outFile, out, 0o644)
		os.WriteFile(errFile, errb, 0o644)
		script := `cat "$1"; cat "$2" >&2; if [ "$3" = -1 ]; then kill -9 $$; fi; exit "$3"`
		tool := &Tool{
			Argv:           []string{"/bin/sh", "-c", script, "sh", outFile, errFile, strconv.Itoa(code)},
			Env:            []string{"PATH=/usr/bin:/bin"},
			MaxOutputBytes: max,
		}
		res, err := tool.Call(context.Background(), json.RawMessage(`{}`))
		desc := fmt.Sprintf("trial %d: max=%d out=%d err=%d code=%d", i, max, len(out), len(errb), code)
		if err != nil {
			t.Fatalf("%s: unexpected Go error %v", desc, err)
		}
		if res.IsError != (code != 0) {
			t.Fatalf("%s: IsError = %v", desc, res.IsError)
		}
		var want string
		if code == 0 {
			if trim(string(clip(out, eff))) == "" {
				want = docText(errb, eff)
			} else {
				want = docText(out, eff)
			}
			counts["ok"]++
		} else {
			want = trim(docText(out, eff) + "\n" + docText(errb, eff))
			if want == "" {
				want = fmt.Sprintf("exited %d", code)
				if code == -1 {
					want = "exited: signal: killed"
				}
			}
			counts["fail"]++
		}
		if res.Content != want {
			t.Fatalf("%s:\n got  %q\n want %q", desc, res.Content, want)
		}
		// Properties that do not depend on the oracle.
		if len(res.Content) > 2*eff+2*len("\n[output cut at 1048576 bytes]")+1 {
			t.Fatalf("%s: content is %d bytes", desc, len(res.Content))
		}
		if res.Content != trim(res.Content) {
			t.Fatalf("%s: content is not trimmed: %q", desc, res.Content)
		}
		if len(out) > eff && code == 0 && !strings.Contains(res.Content, "[output cut") && trim(string(clip(out, eff))) != "" {
			t.Fatalf("%s: stdout was cut and the note is missing", desc)
		}
		if len(out) <= eff && len(errb) <= eff && strings.Contains(res.Content, "[output cut") {
			t.Fatalf("%s: a note about a cut that did not happen", desc)
		}
		// Whatever the bytes, the content must survive being logged as JSON.
		b, err := json.Marshal(res.Content)
		if err != nil || !json.Valid(b) {
			t.Fatalf("%s: content does not marshal: %v", desc, err)
		}
		if utf8.ValidString(res.Content) {
			counts["utf8"]++
		}
	}
	t.Logf("%d trials: %v", n, counts)
}

func clip(b []byte, max int) []byte {
	if len(b) > max {
		return b[:max]
	}
	return b
}

// --- cancellation kills the whole tree ---

func waitPids(t *testing.T, file string, n int, within time.Duration) []int {
	t.Helper()
	deadline := time.Now().Add(within)
	for time.Now().Before(deadline) {
		if pids := readPids(file); len(pids) >= n {
			return pids
		}
		time.Sleep(2 * time.Millisecond)
	}
	return readPids(file)
}

func readPids(file string) []int {
	raw, _ := os.ReadFile(file)
	var pids []int
	for _, f := range strings.Fields(string(raw)) {
		if p, err := strconv.Atoi(f); err == nil {
			pids = append(pids, p)
		}
	}
	return pids
}

// mustDie waits for every pid to be gone, then fails for those that are
// not. It kills the stragglers by pid so the test leaves nothing behind.
func mustDie(t *testing.T, what string, pids []int) {
	t.Helper()
	deadline := time.Now().Add(6 * time.Second)
	for _, pid := range pids {
		for alive(pid) && time.Now().Before(deadline) {
			time.Sleep(5 * time.Millisecond)
		}
		if alive(pid) {
			t.Errorf("%s: process %d survived", what, pid)
			syscall.Kill(pid, syscall.SIGKILL)
		}
	}
}

// Each script records every pid it creates in $PIDS, then blocks.
var treeScripts = map[string]string{
	"background and wait":     `echo $$ >> "$PIDS"; sleep 60 & echo $! >> "$PIDS"; wait`,
	"double fork":             `echo $$ >> "$PIDS"; ( ( sleep 60 & echo $! >> "$PIDS" ) ); sleep 60`,
	"nohup child":             `echo $$ >> "$PIDS"; nohup sleep 60 >/dev/null 2>&1 & echo $! >> "$PIDS"; wait`,
	"ignores SIGTERM":         `trap '' TERM; echo $$ >> "$PIDS"; sleep 60 & echo $! >> "$PIDS"; while :; do sleep 1; done`,
	"grandchild ignores TERM": `echo $$ >> "$PIDS"; ( trap '' TERM; echo $BASHPID $$ >> "$PIDS"; sleep 60 ) & echo $! >> "$PIDS"; wait`,
	"nested shells":           `echo $$ >> "$PIDS"; sh -c 'echo $$ >> "$PIDS"; sh -c "echo \$\$ >> \"$PIDS\"; sleep 60" & wait' & wait`,
	"pipeline":                `echo $$ >> "$PIDS"; sleep 60 | sleep 60 | cat`,
	"exec replaces shell":     `echo $$ >> "$PIDS"; exec sleep 60`,
}

func TestCancelKillsTree(t *testing.T) {
	seed := propSeedValue(t)
	rounds := 3
	if testing.Short() {
		rounds = 1
	}
	i := 0
	for name, script := range treeScripts {
		name, script := name, script
		rng := rand.New(rand.NewSource(seed + int64(i)))
		i++
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			for round := 0; round < rounds; round++ {
				pidFile := filepath.Join(t.TempDir(), "pids")
				tool := &Tool{
					Argv: []string{"/bin/sh", "-c", script},
					Env:  []string{"PIDS=" + pidFile, "PATH=/usr/bin:/bin"},
				}
				ctx, cancel := context.WithCancel(context.Background())
				done := make(chan error, 1)
				go func() { _, err := tool.Call(ctx, json.RawMessage(`{}`)); done <- err }()
				// Cancel at a random moment: before, during or after the tree exists.
				time.Sleep(time.Duration(rng.Intn(40)) * time.Millisecond)
				if rng.Intn(2) == 0 {
					want := 2
					if name == "exec replaces shell" || name == "pipeline" {
						want = 1
					}
					waitPids(t, pidFile, want, 3*time.Second)
				}
				start := time.Now()
				cancel()
				select {
				case err := <-done:
					if err != context.Canceled {
						t.Errorf("err = %v", err)
					}
				case <-time.After(10 * time.Second):
					t.Fatalf("Call did not return")
				}
				if d := time.Since(start); d > killGrace+2*time.Second {
					t.Errorf("took %v to stop", d)
				}
				// SIGKILL is asynchronous and a killed process stays visible as a
				// zombie until it is reaped, so "gone" is checked with a wait.
				mustDie(t, name, readPids(pidFile))
			}
		})
	}
}

// TestNothingLeftAfterNormalExit: a tool that exits but leaves a background
// child must not leave that child running behind the call.
func TestNothingLeftAfterNormalExit(t *testing.T) {
	pidFile := filepath.Join(t.TempDir(), "pids")
	// The child has its stdout and stderr closed, so only the group
	// sweep can stop it.
	tool := &Tool{
		Argv: []string{"/bin/sh", "-c", `sleep 60 >/dev/null 2>&1 & echo $! >> "$PIDS"; echo finished`},
		Env:  []string{"PIDS=" + pidFile, "PATH=/usr/bin:/bin"},
	}
	res, err := tool.Call(context.Background(), json.RawMessage(`{}`))
	if err != nil || res.IsError || res.Content != "finished" {
		t.Fatalf("result %+v, err %v", res, err)
	}
	pids := readPids(pidFile)
	if len(pids) != 1 {
		t.Fatalf("pids = %v", pids)
	}
	mustDie(t, "background child", pids)
}

// TestBackgroundChildHoldingOutput: a tool that exits 0 while a background
// child still holds its stdout open has finished. Its output is the result;
// the call is not a failure.
func TestBackgroundChildHoldingOutput(t *testing.T) {
	pidFile := filepath.Join(t.TempDir(), "pids")
	tool := &Tool{
		Argv: []string{"/bin/sh", "-c", `echo useful; sleep 60 & echo $! >> "$PIDS"`},
		Env:  []string{"PIDS=" + pidFile, "PATH=/usr/bin:/bin"},
	}
	start := time.Now()
	res, err := tool.Call(context.Background(), json.RawMessage(`{}`))
	t.Logf("took %v", time.Since(start))
	if err != nil {
		t.Fatalf("a successful tool is reported as failed: %v", err)
	}
	if res.IsError || res.Content != "useful" {
		t.Errorf("result %+v", res)
	}
	mustDie(t, "background child", readPids(pidFile))
}

// TestSetsidEscapes documents the limit of process groups: a program that
// starts its own session leaves the group and survives the sweep. It is a
// known limit, not an expectation; the test only records what happens and
// cleans up by pid.
func TestSetsidEscapes(t *testing.T) {
	if _, err := os.Stat("/usr/bin/perl"); err != nil {
		t.Skip("no perl")
	}
	pidFile := filepath.Join(t.TempDir(), "pids")
	tool := &Tool{
		Argv: []string{"/bin/sh", "-c", `/usr/bin/perl -MPOSIX -e 'POSIX::setsid(); open(F, ">>", $ENV{PIDS}); print F "$$\n"; close F; exec "sleep", "60"' & wait`},
		Env:  []string{"PIDS=" + pidFile, "PATH=/usr/bin:/bin"},
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { tool.Call(ctx, json.RawMessage(`{}`)); close(done) }()
	pids := waitPids(t, pidFile, 1, 5*time.Second)
	cancel()
	<-done
	time.Sleep(100 * time.Millisecond)
	for _, p := range pids {
		if alive(p) {
			t.Logf("known limit: process %d, which called setsid, survived cancellation", p)
			syscall.Kill(p, syscall.SIGKILL)
		}
	}
}

func openFDs(t *testing.T) int {
	t.Helper()
	d, err := os.Open("/dev/fd")
	if err != nil {
		t.Skip("no /dev/fd")
	}
	defer d.Close()
	names, _ := d.Readdirnames(-1)
	return len(names)
}

// TestNoLeaks: many calls of every kind leave no descriptors and no
// goroutines behind.
func TestNoLeaks(t *testing.T) {
	calls := 250
	if testing.Short() {
		calls = 80
	}
	seed := propSeedValue(t)
	rng := rand.New(rand.NewSource(seed))
	dir := t.TempDir()
	run := func(i int) {
		var script string
		cancelAfter := time.Duration(-1)
		switch rng.Intn(8) {
		case 0:
			script = `echo ok`
		case 1:
			script = `echo oops >&2; exit 3`
		case 2:
			script = `head -c 3000000 /dev/zero | tr '\0' x`
		case 3:
			script = `sleep 60 & wait`
			cancelAfter = time.Duration(rng.Intn(30)) * time.Millisecond
		case 4:
			script = `sleep 60`
			cancelAfter = time.Duration(rng.Intn(30)) * time.Millisecond
		case 5:
			script = `exit 0`
		case 6:
			script = `kill -9 $$`
		default:
			script = `sleep 60 >/dev/null 2>&1 & echo done`
		}
		tool := &Tool{Argv: []string{"/bin/sh", "-c", script}, Env: []string{"PATH=/usr/bin:/bin"}, Dir: dir, MaxOutputBytes: 1000}
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		if cancelAfter >= 0 {
			go func() { time.Sleep(cancelAfter); cancel() }()
		}
		tool.Call(ctx, json.RawMessage(strings.Repeat(" ", rng.Intn(100000))+`{}`))
	}
	// Warm up lazily created things (the runtime's poller, the os/exec
	// internals) so that the baseline includes them.
	for i := 0; i < 5; i++ {
		run(i)
	}
	time.Sleep(100 * time.Millisecond)
	fdsBefore, grBefore := openFDs(t), runtime.NumGoroutine()
	for i := 0; i < calls; i++ {
		run(i)
	}
	// Goroutines of cancelled calls and the sleeping cancellers need a moment.
	var fdsAfter, grAfter int
	for try := 0; try < 100; try++ {
		fdsAfter, grAfter = openFDs(t), runtime.NumGoroutine()
		if fdsAfter <= fdsBefore && grAfter <= grBefore+1 {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Logf("%d calls: fds %d -> %d, goroutines %d -> %d", calls, fdsBefore, fdsAfter, grBefore, grAfter)
	if fdsAfter > fdsBefore {
		t.Errorf("descriptors leaked: %d -> %d", fdsBefore, fdsAfter)
	}
	if grAfter > grBefore+1 {
		t.Errorf("goroutines leaked: %d -> %d", grBefore, grAfter)
	}
}

// TestLargeInputToToolThatDoesNotRead: a tool that never reads its stdin
// still completes, and its exit is what is reported.
func TestLargeInputToToolThatDoesNotRead(t *testing.T) {
	input := `{"x":"` + strings.Repeat("a", 8<<20) + `"}`
	res, err := (&Tool{Argv: []string{"/bin/sh", "-c", "echo ignored"}, Env: []string{"PATH=/usr/bin:/bin"}}).
		Call(context.Background(), json.RawMessage(input))
	if err != nil || res.IsError || res.Content != "ignored" {
		t.Errorf("result %+v err %v", res, err)
	}
	// And one that reads it all gets it all.
	res, err = (&Tool{Argv: []string{"/bin/sh", "-c", "wc -c"}, Env: []string{"PATH=/usr/bin:/bin"}}).
		Call(context.Background(), json.RawMessage(input))
	if err != nil || strings.TrimSpace(res.Content) != strconv.Itoa(len(input)) {
		t.Errorf("result %+v err %v", res, err)
	}
}

// TestDeadlineIsNotCancellation: a context deadline stops the tree too and
// is reported as the deadline, which the agent turns into a tool timeout.
func TestDeadlineStopsTree(t *testing.T) {
	pidFile := filepath.Join(t.TempDir(), "pids")
	tool := &Tool{
		Argv: []string{"/bin/sh", "-c", `echo $$ >> "$PIDS"; sleep 60 & echo $! >> "$PIDS"; wait`},
		Env:  []string{"PIDS=" + pidFile, "PATH=/usr/bin:/bin"},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()
	_, err := tool.Call(ctx, json.RawMessage(`{}`))
	if err != context.DeadlineExceeded {
		t.Errorf("err = %v", err)
	}
	pids := readPids(pidFile)
	if len(pids) != 2 {
		t.Fatalf("pids = %v", pids)
	}
	mustDie(t, "deadline", pids)
}
