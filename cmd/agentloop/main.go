// Command agentloop runs one agent from a JSON request.
//
//	agentloop run REQUEST.json   run one agent to an outcome
//	agentloop acp REQUEST.json   serve it to an ACP client on stdin and stdout
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
	"syscall"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/internal/request"
	"github.com/kirahowe/agentloop/internal/sessionfile"
	"github.com/kirahowe/agentloop/internal/usage"
)

// version is set at build time with -ldflags "-X main.version=...".
var version = "dev"

func main() {
	os.Exit(realMain(os.Args[1:], os.Stdin, os.Stdout, os.Stderr))
}

func realMain(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "usage: agentloop run|acp REQUEST.json | agentloop version")
		return 2
	}
	var err error
	switch args[0] {
	case "version":
		fmt.Fprintln(stdout, "agentloop", version)
		return 0
	case "acp":
		if len(args) != 2 {
			fmt.Fprintln(stderr, "usage: agentloop acp REQUEST.json")
			return 2
		}
		var req *request.Request
		req, err = request.Load(args[1], request.ACP)
		if err == nil {
			err = acpCmd(req, stdin, stdout, stderr)
		}
	case "run":
		if len(args) != 2 {
			fmt.Fprintln(stderr, "usage: agentloop run REQUEST.json")
			return 2
		}
		// SIGTERM and SIGINT mean "cancel the run and still report it". The
		// handler goes in before anything else so that no signal after the
		// process starts can kill it without an outcome line.
		ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
		defer stop()
		var req *request.Request
		req, err = request.Load(args[1], request.Run)
		if err == nil {
			err = runCmd(ctx, req, stdout, stderr)
		}
	default:
		fmt.Fprintf(stderr, "agentloop: unknown command %q\n", args[0])
		return 2
	}
	if err == nil {
		return 0
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
	// Labels are the session's labels when the run ended.
	Labels []string `json:"labels"`
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
		return writeJSON(stdout, outcomeLine{Type: "outcome", Outcome: noKeyOutcome(err), Session: req.Session, Labels: storedLabels(req.Session)})
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

	if !req.Restricted() && sess.Policed() {
		// Carrying a labelled or held session on with no policy would
		// drop everything the policy stood for.
		return usage.Errorf("the session was run under a policy, and this request has none")
	}
	agent := req.Agent(key)
	resolved, err := resolve(ctx, agent, sess, req.Resolve)
	if err != nil {
		return err
	}
	for _, e := range resolved {
		if err := writeJSON(stdout, e); err != nil {
			return err
		}
	}

	if req.TimeoutSeconds > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, request.Seconds(req.TimeoutSeconds))
		defer cancel()
	}

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
	return writeJSON(stdout, outcomeLine{Type: "outcome", Outcome: *out, Session: req.Session, Labels: sess.Labels()})
}

// resolve records a person's decisions on held calls, from the request's
// resolve list, and returns the entries that wrote. It checks every one
// against the session before writing any, so a bad list writes nothing.
// A call already resolved the same way is passed over (see Agent.Resolve).
func resolve(ctx context.Context, agent *agentloop.Agent, sess *agentloop.Session, list []agentloop.Resolution) ([]agentloop.Entry, error) {
	for _, r := range list {
		if err := agent.CheckResolve(ctx, sess, r); err != nil {
			return nil, usage.Errorf("resolve: %v", err)
		}
	}
	before := len(sess.Entries())
	for _, r := range list {
		if err := agent.Resolve(ctx, sess, r); err != nil {
			return nil, err
		}
	}
	return sess.Entries()[before:], nil
}

// storedLabels reads the labels of the session at path without locking
// it, for an outcome that never opened it. A session that does not exist
// or cannot be read has none that can be told.
func storedLabels(path string) []string {
	if !sessionfile.Exists(path) {
		return []string{}
	}
	sess, err := agentloop.OpenSession(agentloop.NewFileStore(path))
	if err != nil {
		return []string{}
	}
	return sess.Labels()
}

func writeJSON(w io.Writer, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	_, err = w.Write(append(b, '\n'))
	return err
}
