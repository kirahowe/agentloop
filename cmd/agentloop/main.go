// Command agentloop runs one agent from a JSON request.
//
//	agentloop run  REQUEST.json   run one agent to an outcome
//	agentloop chat REQUEST.json   talk to it on the terminal
//	agentloop version
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/internal/request"
	"github.com/kirahowe/agentloop/internal/sessionfile"
	"github.com/kirahowe/agentloop/internal/usage"
)

// version is set at build time with -ldflags "-X main.version=...".
var version = "dev"

// errInterrupted ends a chat that SIGTERM stopped: exit 130.
var errInterrupted = errors.New("interrupted")

func main() {
	os.Exit(realMain(os.Args[1:], os.Stdin, os.Stdout, os.Stderr))
}

func realMain(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "usage: agentloop run|chat REQUEST.json | agentloop version")
		return 2
	}
	var err error
	switch args[0] {
	case "version":
		fmt.Fprintln(stdout, "agentloop", version)
		return 0
	case "run", "chat":
		if len(args) != 2 {
			fmt.Fprintf(stderr, "usage: agentloop %s REQUEST.json\n", args[0])
			return 2
		}
		// SIGTERM and SIGINT mean "cancel the run and still report it". The
		// handler goes in before anything else so that no signal after the
		// process starts can kill it without an outcome line.
		ctx, stop := context.Background(), func() {}
		if args[0] == "run" {
			ctx, stop = signal.NotifyContext(ctx, syscall.SIGTERM, syscall.SIGINT)
		}
		defer stop()
		var req *request.Request
		req, err = request.Load(args[1])
		if err == nil {
			if args[0] == "run" {
				err = runCmd(ctx, req, stdout, stderr)
			} else {
				err = chatCmd(req, stdin, stdout, stderr)
			}
		}
	default:
		fmt.Fprintf(stderr, "agentloop: unknown command %q\n", args[0])
		return 2
	}
	if err == nil {
		return 0
	}
	if errors.Is(err, errInterrupted) {
		fmt.Fprintln(stderr)
		return 130
	}
	fmt.Fprintln(stderr, "agentloop:", err)
	var ue *usage.Error
	if errors.As(err, &ue) {
		return 2
	}
	return 1
}

// outcomeLine is the last line of a run.
type outcomeLine struct {
	Type string `json:"type"`
	agentloop.Outcome
	Session string `json:"session"`
}

func noKeyOutcome(err error) agentloop.Outcome {
	return agentloop.Outcome{
		Reason: agentloop.ReasonModelError,
		Error:  &agentloop.ModelError{Kind: agentloop.Unauthorized, Message: err.Error()},
	}
}

func runCmd(ctx context.Context, req *request.Request, stdout, stderr io.Writer) error {
	key, err := request.FetchKey(ctx, req.APIKeyCommand, request.KeyTimeout)
	var ke *request.KeyError
	if errors.As(err, &ke) && ke.Canceled {
		// A signal arrived while the key command ran. The run below sees the
		// canceled context and ends as canceled without calling the model,
		// the same as a signal at any other point before the first call.
		err = nil
	}
	if err != nil {
		return writeJSON(stdout, outcomeLine{Type: "outcome", Outcome: noKeyOutcome(err), Session: req.Session})
	}
	if req.Prompt == "" {
		// Nothing to resume if there is no session yet; say so without
		// creating the directory or a lock for it.
		if !sessionfile.Exists(req.Session) {
			return usage.Errorf("run has no prompt and there is no session to resume")
		}
	}
	sess, unlock, err := sessionfile.Open(req.Session)
	if err != nil {
		return err
	}
	defer unlock()
	// With no prompt the run resumes the session. Only a session with no
	// conversation at all leaves nothing to resume; a finished one just
	// completes again, so a supervisor can restart a run that was cut off
	// just before it ended.
	if req.Prompt == "" && len(sess.Messages()) == 0 {
		return usage.Errorf("run has no prompt and the session has no conversation to resume")
	}

	if req.TimeoutSeconds > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, request.Seconds(req.TimeoutSeconds))
		defer cancel()
	}

	agent := req.Agent(key)
	var writeErr error
	agent.Observe = func(ev agentloop.Event) {
		if ev.Entry != nil && writeErr == nil {
			writeErr = writeJSON(stdout, ev.Entry)
		}
	}
	out, err := agent.Run(ctx, sess, req.Prompt)
	if err != nil {
		return err
	}
	if writeErr != nil {
		return writeErr
	}
	return writeJSON(stdout, outcomeLine{Type: "outcome", Outcome: *out, Session: req.Session})
}

func writeJSON(w io.Writer, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	_, err = w.Write(append(b, '\n'))
	return err
}

func chatCmd(req *request.Request, stdin io.Reader, stdout, stderr io.Writer) error {
	key, err := request.FetchKey(context.Background(), req.APIKeyCommand, request.KeyTimeout)
	if err != nil {
		return usage.Errorf("%v", err)
	}
	sess, unlock, err := sessionfile.Open(req.Session)
	if err != nil {
		return err
	}
	defer unlock()
	agent := req.Agent(key)

	var mu sync.Mutex
	var cancelRun context.CancelFunc // set while a run is going
	canceled := false
	terminating := false // SIGTERM arrived during a run
	sigs := make(chan os.Signal, 4)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		for sig := range sigs {
			mu.Lock()
			// The first signal during a run cancels the run, so its tool
			// and the tool's children are stopped and the call is logged.
			// SIGINT then returns to the prompt; SIGTERM leaves the chat
			// when the run has stopped. Any other signal, or a second
			// one, leaves at once.
			if cancelRun != nil && !canceled {
				canceled = true
				terminating = sig == syscall.SIGTERM
				cancelRun()
				mu.Unlock()
				continue
			}
			mu.Unlock()
			fmt.Fprintln(stderr)
			os.Exit(130)
		}
	}()

	streamed := false
	agent.Observe = func(ev agentloop.Event) {
		switch {
		case ev.Delta != nil:
			if ev.Delta.Type == agentloop.TextPart {
				streamed = true
				io.WriteString(stdout, ev.Delta.Text)
			}
		case ev.Entry != nil:
			showEntry(stderr, ev.Entry)
		}
	}

	in := newLineReader(stdin, stderr)
	for {
		line, rerr := in.readLine("> ")
		line = strings.TrimSpace(line)
		if line != "" {
			ctx, cancel := context.WithCancel(context.Background())
			mu.Lock()
			cancelRun, canceled = cancel, false
			mu.Unlock()
			streamed = false
			out, err := agent.Run(ctx, sess, line)
			mu.Lock()
			cancelRun = nil
			stopping := terminating
			mu.Unlock()
			cancel()
			if streamed {
				fmt.Fprintln(stdout)
			}
			if err != nil {
				return err
			}
			if stopping {
				return errInterrupted
			}
			if out.Reason != agentloop.ReasonCompleted {
				fmt.Fprintf(stderr, "[%s%s]\n", out.Reason, outcomeDetail(out))
			}
		}
		if rerr != nil {
			if rerr != io.EOF {
				return rerr
			}
			fmt.Fprintln(stderr)
			return nil
		}
	}
}

func outcomeDetail(o *agentloop.Outcome) string {
	switch {
	case o.Error != nil:
		return ": " + o.Error.Error()
	case o.Limit != "":
		return ": " + o.Limit
	case len(o.Held) > 0:
		return fmt.Sprintf(": %d call(s) held", len(o.Held))
	}
	return ""
}

// showEntry prints tool calls and their results in one short line each.
func showEntry(w io.Writer, e *agentloop.Entry) {
	switch {
	case e.Message != nil:
		for _, c := range e.Message.Message.ToolCalls() {
			fmt.Fprintf(w, "[tool] %s %s\n", c.Name, request.Brief(string(c.Input)))
		}
	case e.ToolResult != nil:
		mark := "->"
		if e.ToolResult.Result.IsError {
			mark = "-> error:"
		}
		fmt.Fprintf(w, "[tool] %s %s\n", mark, request.Brief(e.ToolResult.Result.Content))
	}
}
