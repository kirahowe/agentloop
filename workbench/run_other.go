//go:build !darwin

package workbench

import (
	"context"
	"errors"
)

// errUnsupported is why the workbench cannot run here: it needs macOS's
// sandbox-exec.
var errUnsupported = errors.New("the workbench needs macOS sandbox-exec")

type box struct{ scratch string }

func newBox(string, []string, Limits) (*box, error) { return nil, errUnsupported }

// A CommandResult is what one command did.
type CommandResult struct {
	Output      string
	Truncated   bool
	Exit        int
	TimedOut    bool
	Swept       int
	ScratchFull bool
}

func (b *box) run(context.Context, string) (CommandResult, error) {
	return CommandResult{}, errUnsupported
}
func (b *box) close() {}

// Available reports whether this system can run the workbench.
func Available() error { return errUnsupported }
