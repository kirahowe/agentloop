// Package command provides an agentloop.Tool that runs a program.
//
// The program gets exactly the environment it is configured with: nothing
// is inherited from the calling process. It speaks one of two protocols.
//
// Plain: the call's input JSON on stdin, and stdout as the result (stderr
// when stdout is empty, and both when the program fails).
//
// Envelope: one JSON object on stdin, {"input", "labels", "call_id"}, and
// exactly one on stdout, {"content", "is_error", "labels"}. Anything else
// (a non-zero exit, output past the cap, or stdout that is not one valid
// envelope with known labels) gives the model only "The tool <name>
// failed.", and what went wrong, with stderr, goes to the operator's log.
package command

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"strings"
	"syscall"
	"time"

	"github.com/kirahowe/agentloop"
)

// DefaultMaxOutputBytes caps each of stdout and stderr when
// Tool.MaxOutputBytes is zero.
const DefaultMaxOutputBytes = 1 << 20

// killGrace is how long a process group gets after SIGTERM before SIGKILL.
const killGrace = 2 * time.Second

// A Tool runs Argv once per call.
type Tool struct {
	// Def is what the model is told about the tool.
	Def agentloop.ToolSpec
	// Argv is the program and its arguments.
	Argv []string
	// Env is the whole environment, as K=V. Nil means empty.
	Env []string
	// Dir is the working directory. Empty means the current one.
	Dir string
	// MaxOutputBytes caps what is kept of stdout and of stderr. Zero means
	// DefaultMaxOutputBytes.
	MaxOutputBytes int
	// Envelope selects the envelope protocol. KnownLabel then says which
	// labels a result may carry; nil allows none.
	Envelope   bool
	KnownLabel func(string) bool
}

var _ agentloop.CallTool = (*Tool)(nil)

// Spec returns the tool's definition.
func (t *Tool) Spec() agentloop.ToolSpec { return t.Def }

// Call runs the program with the plain protocol's stdin, or with an
// envelope that has no call id and no labels.
func (t *Tool) Call(ctx context.Context, input json.RawMessage) (agentloop.Result, error) {
	return t.CallWith(ctx, agentloop.Call{Input: input})
}

// CallWith runs the program for one call.
func (t *Tool) CallWith(ctx context.Context, call agentloop.Call) (agentloop.Result, error) {
	if t.Envelope {
		return t.envelope(ctx, call)
	}
	if len(t.Argv) == 0 {
		return agentloop.Result{}, errors.New("command tool has no argv")
	}
	if err := ctx.Err(); err != nil {
		return agentloop.Result{}, err
	}
	stdout, stderr, waitErr, err := t.run(ctx, call.Input)
	if err != nil {
		return agentloop.Result{}, err
	}
	if err := ctx.Err(); err != nil {
		return agentloop.Result{}, err
	}
	var exitErr *exec.ExitError
	switch {
	// ErrWaitDelay means the program exited with status 0 but a process it
	// left behind kept the output pipes open. The program succeeded and
	// that process was just swept, so this is a result, not a failure.
	case waitErr == nil || errors.Is(waitErr, exec.ErrWaitDelay):
		out := text(stdout)
		if strings.TrimSpace(stdout.String()) == "" {
			out = text(stderr)
		}
		return agentloop.Result{Content: out}, nil
	case errors.As(waitErr, &exitErr):
		out := strings.TrimSpace(strings.TrimSpace(text(stdout)) + "\n" + strings.TrimSpace(text(stderr)))
		if out == "" {
			out = fmt.Sprintf("exited %d", exitErr.ExitCode())
			if exitErr.ExitCode() < 0 {
				out = "exited: " + exitErr.Error()
			}
		}
		return agentloop.Result{Content: out, IsError: true}, nil
	default:
		return agentloop.Result{}, waitErr
	}
}

// run runs the program with stdin, under the context, and returns what it
// wrote and how it ended. err is set only when it could not start.
func (t *Tool) run(ctx context.Context, stdin []byte) (stdout, stderr *capped, waitErr, err error) {
	max := t.MaxOutputBytes
	if max <= 0 {
		max = DefaultMaxOutputBytes
	}
	cmd := exec.Command(t.Argv[0], t.Argv[1:]...)
	cmd.Env = append([]string{}, t.Env...)
	cmd.Dir = t.Dir
	cmd.Stdin = bytes.NewReader(stdin)
	stdout, stderr = &capped{max: max}, &capped{max: max}
	cmd.Stdout, cmd.Stderr = stdout, stderr
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	// If something outside the group keeps the pipes open, do not wait on it.
	cmd.WaitDelay = killGrace
	if err := cmd.Start(); err != nil {
		return nil, nil, nil, fmt.Errorf("start %s: %w", t.Argv[0], err)
	}
	pgid := cmd.Process.Pid

	done := make(chan struct{})
	killed := make(chan struct{})
	go func() {
		defer close(killed)
		select {
		case <-done:
		case <-ctx.Done():
			syscall.Kill(-pgid, syscall.SIGTERM)
			select {
			case <-done:
			case <-time.After(killGrace):
			}
			// Always sweep the group, whatever the leader did.
			syscall.Kill(-pgid, syscall.SIGKILL)
		}
	}()
	waitErr = cmd.Wait()
	close(done)
	<-killed
	// Leave nothing of the call running behind it.
	syscall.Kill(-pgid, syscall.SIGKILL)
	return stdout, stderr, waitErr, nil
}

func text(c *capped) string {
	s := strings.TrimSpace(c.String())
	if c.cut {
		s += fmt.Sprintf("\n[output cut at %d bytes]", c.max)
	}
	return strings.TrimSpace(s)
}

// capped keeps the first max bytes written and discards the rest.
type capped struct {
	buf bytes.Buffer
	max int
	cut bool
}

func (c *capped) Write(p []byte) (int, error) {
	room := c.max - c.buf.Len()
	if room < len(p) {
		c.cut = true
		if room > 0 {
			c.buf.Write(p[:room])
		}
		return len(p), nil
	}
	return c.buf.Write(p)
}

func (c *capped) String() string { return c.buf.String() }
