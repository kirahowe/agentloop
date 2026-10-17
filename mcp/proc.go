package mcp

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"syscall"
	"time"
)

// killGrace is how long the server gets to exit after its stdin closes,
// and again after SIGTERM, before its group is killed.
const killGrace = 2 * time.Second

// proc is the server process, in a process group of its own whose leader
// is a sentinel, as in the command package: the sentinel is reaped only
// after the group has been swept, so the sweep can never reach a group
// that reused the id.
type proc struct {
	cmd      *exec.Cmd
	sentinel *exec.Cmd
	release  io.Closer
	pgid     int
	// Our ends of the server's stdin and stdout.
	in  *os.File
	out *os.File
	// The read end of the server's stderr, when it is copied to a writer,
	// and closed when the copy ends.
	errR    *os.File
	errDone chan struct{}

	exited  chan struct{}
	waitErr error
}

// start runs the server and returns it with the pipe to read its stdout
// and the pipe to write its stdin.
func start(srv Server) (*proc, io.ReadCloser, io.WriteCloser, error) {
	sentinel, release, err := startSentinel()
	if err != nil {
		return nil, nil, nil, err
	}
	p := &proc{sentinel: sentinel, release: release, pgid: sentinel.Process.Pid, exited: make(chan struct{})}
	fail := func(err error) (*proc, io.ReadCloser, io.WriteCloser, error) {
		p.sweep()
		return nil, nil, nil, err
	}
	// Pipes made here, rather than by exec, so that Wait never closes the
	// ends the session reads and writes.
	inR, inW, err := os.Pipe()
	if err != nil {
		return fail(err)
	}
	outR, outW, err := os.Pipe()
	if err != nil {
		inR.Close()
		inW.Close()
		return fail(err)
	}
	p.in, p.out = inW, outR
	cmd := exec.Command(srv.Argv[0], srv.Argv[1:]...)
	// Never nil: a nil Env inherits the caller's environment.
	cmd.Env = append([]string{}, srv.Env...)
	cmd.Dir = srv.Dir
	cmd.Stdin, cmd.Stdout = inR, outW
	// Stderr too is a pipe made here and copied, so that Wait returns
	// when the server exits even if something it left holds stderr.
	var errW *os.File
	if srv.Stderr != nil {
		if p.errR, errW, err = os.Pipe(); err != nil {
			inR.Close()
			inW.Close()
			outR.Close()
			outW.Close()
			return fail(err)
		}
		cmd.Stderr = errW
		p.errDone = make(chan struct{})
		go func() {
			io.Copy(srv.Stderr, p.errR)
			close(p.errDone)
		}()
	}
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true, Pgid: p.pgid}
	err = cmd.Start()
	inR.Close()
	outW.Close()
	if errW != nil {
		errW.Close()
	}
	if err != nil {
		inW.Close()
		outR.Close()
		if p.errR != nil {
			p.errR.Close()
		}
		return fail(fmt.Errorf("start %s: %w", srv.Argv[0], err))
	}
	p.cmd = cmd
	go func() {
		p.waitErr = cmd.Wait()
		close(p.exited)
	}()
	return p, outR, inW, nil
}

func (p *proc) hasExited() bool {
	select {
	case <-p.exited:
		return true
	default:
		return false
	}
}

func (p *proc) exitedWithin(d time.Duration) bool {
	select {
	case <-p.exited:
		return true
	case <-time.After(d):
		return false
	}
}

// stop closes the server's stdin, gives it killGrace to exit, then
// SIGTERM and killGrace more, and then kills its whole group and the
// server itself. It returns within about four times killGrace whatever
// the server does.
func (p *proc) stop() {
	p.in.Close()
	if !p.exitedWithin(killGrace) {
		syscall.Kill(-p.pgid, syscall.SIGTERM)
		p.cmd.Process.Signal(syscall.SIGTERM)
		p.exitedWithin(killGrace)
	}
	p.sweep()
	// A server that left the group escaped the sweep. It is our child
	// and, until exited is closed, unreaped, so its pid is still its own.
	p.cmd.Process.Kill()
	select {
	case <-p.exited:
	case <-time.After(killGrace):
	}
	p.out.Close()
	if p.errR != nil {
		// What the server wrote before it exited is copied out before
		// stop returns, so a caller can close the writer after it.
		select {
		case <-p.errDone:
		case <-time.After(killGrace):
		}
		p.errR.Close()
	}
}

// sweep kills the group while the sentinel still holds its id, then lets
// the sentinel go.
func (p *proc) sweep() {
	syscall.Kill(-p.pgid, syscall.SIGKILL)
	p.release.Close()
	p.sentinel.Wait()
}

// sentinelPath is the program that holds the server's process group: it
// reads its stdin, a pipe, until the client closes it.
const sentinelPath = "/bin/cat"

func startSentinel() (*exec.Cmd, io.Closer, error) {
	s := exec.Command(sentinelPath)
	s.Env = []string{}
	s.Dir = "/"
	s.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	w, err := s.StdinPipe()
	if err != nil {
		return nil, nil, err
	}
	if err := s.Start(); err != nil {
		return nil, nil, fmt.Errorf("start the process group's sentinel: %w", err)
	}
	return s, w, nil
}
