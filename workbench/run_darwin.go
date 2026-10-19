//go:build darwin

package workbench

/*
#include <stdlib.h>
#include <sys/types.h>

// sandbox_check is libsystem_sandbox's, declared in no public header. It
// returns 0 when the process may do the operation, 1 when it may not.
int sandbox_check(pid_t pid, const char *operation, int type, ...);

static int wb_sandboxed(pid_t pid) { return sandbox_check(pid, NULL, 0); }

// SANDBOX_FILTER_PATH is 1; SANDBOX_CHECK_NO_REPORT keeps the check out
// of the system log.
static int wb_may_write(pid_t pid, const char *path) {
	return sandbox_check(pid, "file-write-create", 1 | 0x40000000, path);
}
*/
import "C"

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"sync"
	"syscall"
	"time"
	"unsafe"

	"golang.org/x/sys/unix"
)

// SandboxExec is the program that applies a profile. It is fixed: a
// request cannot point the workbench at another one.
const SandboxExec = "/usr/bin/sandbox-exec"

// commandPath is what a command finds programs on.
const commandPath = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

// A box is one session's sandbox: a directory of its own holding the
// scratch directory and a probe directory, and the profile that lets
// commands write the first and not the second. The pair is how the
// processes of this sandbox are told from every other process (see
// sweep).
type box struct {
	root, scratch, probe string
	profile              string
	limits               Limits
	started              time.Time
	mu                   sync.Mutex
	full                 bool
}

// newBox makes a session directory under base and the profile for it.
// deny is the caller's deny list, already resolved.
func newBox(base string, deny []string, limits Limits) (*box, error) {
	if _, err := os.Stat(SandboxExec); err != nil {
		return nil, fmt.Errorf("%s is not available: %w", SandboxExec, err)
	}
	base, err := filepath.EvalSymlinks(base)
	if err != nil {
		return nil, err
	}
	root, err := os.MkdirTemp(base, "agentloop-workbench-")
	if err != nil {
		return nil, err
	}
	b := &box{root: root, scratch: filepath.Join(root, "scratch"), probe: filepath.Join(root, "probe"),
		limits: limits, started: time.Now()}
	for _, d := range []string{b.scratch, b.probe} {
		if err := os.Mkdir(d, 0o700); err != nil {
			os.RemoveAll(root)
			return nil, err
		}
	}
	// The session directory itself is denied, so a command cannot read
	// the probe, and other sessions' directories are denied by the
	// default "agentloop-" prefix when they sit in the temporary
	// directory.
	all := append(append([]string{}, deny...), root)
	b.profile, err = Profile(Sandbox{Scratch: b.scratch, DenyRead: all})
	if err != nil {
		os.RemoveAll(root)
		return nil, err
	}
	return b, nil
}

// A CommandResult is what one command did.
type CommandResult struct {
	Output    string
	Truncated bool
	Exit      int
	TimedOut  bool
	// Swept is how many processes of this sandbox were still running
	// after the command and were killed.
	Swept int
	// ScratchFull is set when the scratch directory has grown past its
	// limit; no further command runs.
	ScratchFull bool
}

// run runs one command under the sandbox, in the scratch directory, with
// an environment of only PATH, HOME, TMPDIR (the scratch directory) and
// LANG, and stdin empty. The limits are set by a shell outside the
// sandbox just before it becomes sandbox-exec, so they cover everything
// the command starts and cannot be raised from inside: the number of
// processes this user may have (the current count plus Processes), CPU
// seconds per process and the size of a file. The scratch directory's
// size and the free space on its filesystem are watched while the
// command runs, and the command is stopped when either goes past its
// limit, since a loop writing file after file could fill the disk before
// a check after the command. The command starts in a
// session of its own. Its process group is killed when it ends or times
// out, and then every process left in this sandbox is killed too,
// whatever group or session it moved to.
func (b *box) run(ctx context.Context, command string) (CommandResult, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.full {
		return CommandResult{ScratchFull: true}, nil
	}
	nproc, err := userProcesses()
	if err != nil {
		return CommandResult{}, err
	}
	cpu := int(b.limits.Timeout / time.Second)
	if cpu < 1 {
		cpu = 1
	}
	const wrapper = `ulimit -u "$1" && ulimit -t "$2" && ulimit -f "$3" && exec ` + SandboxExec + ` -p "$4" /bin/sh -c "$5"`
	ctx, cancel := context.WithTimeout(ctx, b.limits.Timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "/bin/sh", "-c", wrapper, "workbench",
		strconv.Itoa(nproc+b.limits.Processes), strconv.Itoa(cpu),
		// bash's ulimit -f counts 1024-byte blocks.
		strconv.FormatInt(max(1, b.limits.FileBytes/1024), 10),
		b.profile, command)
	cmd.Dir = b.scratch
	cmd.Env = []string{"PATH=" + commandPath, "HOME=" + homeDir(), "TMPDIR=" + b.scratch, "LANG=en_US.UTF-8"}
	out := &capped{max: b.limits.OutputBytes}
	cmd.Stdout, cmd.Stderr = out, out
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	cmd.Cancel = func() error { return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL) }
	cmd.WaitDelay = time.Second
	// The scratch directory is watched while the command runs: a loop
	// that writes file after file, each inside the file size limit,
	// would otherwise fill the disk before the check after the command.
	watching := make(chan struct{})
	go func() {
		defer close(watching)
		for {
			select {
			case <-ctx.Done():
				return
			case <-time.After(100 * time.Millisecond):
				if b.overFull() {
					b.full = true
					cancel()
					return
				}
			}
		}
	}()
	err = cmd.Run()
	cancel()
	<-watching
	res := CommandResult{Exit: cmd.ProcessState.ExitCode()}
	res.Swept = b.sweep()
	res.TimedOut = errors.Is(ctx.Err(), context.DeadlineExceeded)
	if err != nil && !res.TimedOut && cmd.ProcessState == nil {
		return res, err
	}
	res.Output, res.Truncated = out.String(), out.over
	if b.full || b.overFull() {
		b.full, res.ScratchFull = true, true
		// A command stopped for filling the scratch directory was not
		// stopped for running too long.
		res.TimedOut = false
	}
	return res, nil
}

// close kills what is left in the sandbox and removes the session
// directory.
func (b *box) close() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.sweep()
	os.RemoveAll(b.root)
}

// sweep kills every process left from this sandbox's commands, and
// returns how many it killed.
//
// It stops before it kills. A single pass that stops and kills lets a
// child forked while the pass ran survive and go on forking, and a fork
// bomb forks faster than the processes can be listed. Stopping first
// converges instead: a stopped process cannot fork, so each pass leaves
// fewer running, and when a pass finds every process of the sandbox
// stopped, killing them ends it. The passes go on until the sandbox is
// empty or ten seconds have passed.
func (b *box) sweep() int {
	killed := 0
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		pids, groups, running := b.snapshot()
		if len(pids) == 0 {
			return killed
		}
		for _, g := range groups {
			syscall.Kill(-g, syscall.SIGSTOP)
		}
		for _, pid := range pids {
			syscall.Kill(pid, syscall.SIGSTOP)
		}
		if running > 0 {
			// Something was still running when the list was made, so
			// there may be children it forked that are not on it.
			continue
		}
		for _, g := range groups {
			syscall.Kill(-g, syscall.SIGKILL)
		}
		for _, pid := range pids {
			if syscall.Kill(pid, syscall.SIGKILL) == nil {
				killed++
			}
		}
	}
	return killed
}

// snapshot is this sandbox's processes, the groups to kill whole, and how
// many of those processes were not stopped.
func (b *box) snapshot() (pids, groups []int, running int) {
	procs, err := unix.SysctlKinfoProcSlice("kern.proc.uid", os.Getuid())
	if err != nil {
		return nil, nil, 0
	}
	pids, groups = b.members(procs)
	mine := map[int]bool{}
	for _, pid := range pids {
		mine[pid] = true
	}
	for _, p := range procs {
		if mine[int(p.Proc.P_pid)] && p.Proc.P_stat != sStop {
			running++
		}
	}
	return pids, groups, running
}

// members are the pids among procs that are in this sandbox, and the
// process groups to kill whole.
//
// A process is in this sandbox when sandbox_check says it is sandboxed,
// may write the scratch directory and may not write the probe beside it.
// It is plainly outside when it is not sandboxed, or when it may write
// the probe: no profile of this package's making allows that. A
// sandboxed process that may write neither is undecided, which happens
// to a process that is on its way out while the pass runs.
//
// A group is killed whole when one of its processes is in this sandbox
// and none is plainly outside. The undecided ones do not stop it: a
// command runs in a session of its own, and a process can only join a
// group in its own session, so nothing of anyone else's can be in a
// group of the command's.
func (b *box) members(procs []unix.KinfoProc) (pids, groups []int) {
	scratch, probe := C.CString(b.scratch), C.CString(b.probe)
	defer C.free(unsafe.Pointer(scratch))
	defer C.free(unsafe.Pointer(probe))
	self, own := os.Getpid(), syscall.Getpgrp()
	// ours[pg] is true once a member is seen in the group, and the group
	// is in outside[pg] once anything else is.
	ours, outside := map[int]bool{}, map[int]bool{}
	for _, p := range procs {
		pid := C.pid_t(p.Proc.P_pid)
		pg := int(p.Eproc.Pgid)
		if p.Proc.P_stat == sZomb {
			continue
		}
		if int(pid) == self || pid <= 1 || C.wb_sandboxed(pid) == 0 {
			outside[pg] = true
			continue
		}
		switch {
		case C.wb_may_write(pid, probe) == 0:
			outside[pg] = true
		case C.wb_may_write(pid, scratch) == 0:
			pids = append(pids, int(pid))
			ours[pg] = true
		}
	}
	for g := range ours {
		if g > 1 && g != own && !outside[g] {
			groups = append(groups, g)
		}
	}
	return pids, groups
}

// left is how many processes of this sandbox are running, for tests.
func (b *box) left() int {
	pids, _, _ := b.snapshot()
	return len(pids)
}

// The process states this package looks at: stopped, and exited but not
// yet reaped.
const (
	sStop = 4
	sZomb = 5
)

// userProcesses is how many processes this user has now.
func userProcesses() (int, error) {
	procs, err := unix.SysctlKinfoProcSlice("kern.proc.uid", os.Getuid())
	return len(procs), err
}

func homeDir() string {
	h, err := os.UserHomeDir()
	if err != nil {
		return "/"
	}
	return h
}

// overFull reports whether the scratch directory has grown past its
// limit, or its filesystem is nearly full. The free-space floor is what
// bounds a command that holds its files unlinked, where no walk can see
// their size, and it is what the limit is really for: the operator's
// disk must not fill up.
func (b *box) overFull() bool {
	if size(b.scratch) > b.limits.ScratchBytes {
		return true
	}
	var st unix.Statfs_t
	if err := unix.Statfs(b.scratch, &st); err != nil {
		return false
	}
	return int64(st.Bavail)*int64(st.Bsize) < b.limits.MinFreeBytes
}

// size is the bytes of the files under dir.
func size(dir string) int64 {
	var n int64
	filepath.WalkDir(dir, func(_ string, d fs.DirEntry, err error) error {
		if err == nil && d.Type().IsRegular() {
			if info, err := d.Info(); err == nil {
				n += info.Size()
			}
		}
		return nil
	})
	return n
}

// capped keeps the first max bytes written to it and counts the rest.
type capped struct {
	mu   sync.Mutex
	buf  bytes.Buffer
	max  int
	over bool
}

func (c *capped) Write(p []byte) (int, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if room := c.max - c.buf.Len(); room > 0 {
		if len(p) > room {
			c.buf.Write(p[:room])
			c.over = true
		} else {
			c.buf.Write(p)
		}
	} else if len(p) > 0 {
		c.over = true
	}
	return len(p), nil
}

func (c *capped) String() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.buf.String()
}

// Available reports whether this system can run the workbench.
func Available() error {
	_, err := os.Stat(SandboxExec)
	return err
}
