// Command agentloop runs one agent from a JSON request.
//
//	agentloop run REQUEST.json   run one agent to an outcome
//	agentloop acp REQUEST.json   serve it to an ACP client on stdin and stdout
//	agentloop otel SESSION.jsonl  convert a session log to OpenTelemetry traces
//	agentloop mcp-tools          list a remote MCP server's tools (input on stdin)
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
	"github.com/kirahowe/agentloop/anthropic"
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
		fmt.Fprintln(stderr, "usage: agentloop run|acp REQUEST.json | agentloop otel SESSION.jsonl | agentloop mcp-tools | agentloop version")
		return 2
	}
	var err error
	switch args[0] {
	case "version":
		fmt.Fprintln(stdout, "agentloop", version)
		return 0
	case "otel":
		err = otelCmd(args[1:], stdout)
	case "mcp-tools":
		if len(args) != 1 {
			fmt.Fprintln(stderr, "usage: agentloop mcp-tools < {\"url\": ..., \"token\": ...}")
			return 2
		}
		err = mcpToolsCmd(context.Background(), stdin, stdout)
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
	// Labels are the session's labels when the run ended. Nil (left out)
	// means they could not be read, which is not the same as none.
	Labels *[]string `json:"labels,omitempty"`
	// CostUSD is what the run's model calls cost, in US dollars. Left out
	// (never 0) when the run made no call or any call's model is unpriced.
	// It is on this line alone: the prices belong to the provider package,
	// which agentloop.Outcome cannot import, and a price table that moves
	// should not rewrite what an old run_ended entry says.
	CostUSD *float64 `json:"cost_usd,omitempty"`
	// ToolsError says why the MCP tools could not be set up, for reason
	// tools_unavailable.
	ToolsError *request.ToolsError `json:"tools_error,omitempty"`
}

// priceFunc is the price lookup for a provider's models. Only Anthropic's
// are priced; OpenAI's adapter reports no prices.
func priceFunc(provider string) func(string) *agentloop.Prices {
	if provider == "anthropic" {
		return anthropic.PricesFor
	}
	return func(string) *agentloop.Prices { return nil }
}

// reasonToolsUnavailable ends a run whose MCP tools could not be set up:
// a server did not start or lacks a tool, or a definition is not the one
// the request or the session pinned. The run never starts, and the
// outcome's tools_error says why.
const reasonToolsUnavailable agentloop.Reason = "tools_unavailable"

func toolsUnavailable(stdout io.Writer, req *request.Request, sess *agentloop.Session, te *request.ToolsError) error {
	return writeJSON(stdout, outcomeLine{Type: "outcome", Outcome: agentloop.Outcome{Reason: reasonToolsUnavailable},
		Session: req.Session, Labels: ptr(sess.Labels()), ToolsError: te})
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

	// MCP servers are not started once a signal has come, and a signal
	// while they start stops them: the outcome says canceled, as it
	// would from the run itself.
	canceled := func() error {
		return writeJSON(stdout, outcomeLine{Type: "outcome", Outcome: agentloop.Outcome{Reason: agentloop.ReasonCanceled}, Session: req.Session, Labels: ptr(sess.Labels())})
	}
	if len(req.MCPServers) > 0 && ctx.Err() != nil {
		return canceled()
	}
	servers, err := req.StartMCP(ctx, stderr)
	if len(req.MCPServers) > 0 && ctx.Err() != nil {
		servers.Close()
		return canceled()
	}
	var te *request.ToolsError
	if errors.As(err, &te) {
		return toolsUnavailable(stdout, req, sess, te)
	}
	if err != nil {
		return err
	}
	defer servers.Close()
	agent := req.Agent(key, servers)
	// A changed tool definition stops the run before anything is
	// written.
	var pe *agentloop.PinError
	if err := sess.CheckPins(agent.Pins); errors.As(err, &pe) {
		return toolsUnavailable(stdout, req, sess, req.PinsError(pe))
	} else if err != nil {
		return err
	}
	if req.TimeoutSeconds > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, request.Seconds(req.TimeoutSeconds))
		defer cancel()
	}

	var writeErr error
	var entries []agentloop.Entry
	agent.Observe = func(ev agentloop.Event) {
		if ev.Entry != nil {
			entries = append(entries, *ev.Entry)
			if writeErr == nil {
				writeErr = writeJSON(stdout, ev.Entry)
			}
		}
	}
	out, err := agent.Run(ctx, sess, req.Prompt)
	if err != nil {
		return err
	}
	if writeErr != nil {
		return writeErr
	}
	return writeJSON(stdout, outcomeLine{Type: "outcome", Outcome: *out, Session: req.Session, Labels: ptr(sess.Labels()), CostUSD: runCost(callsOf(entries), priceFunc(req.Provider))})
}

// storedLabels reads the labels of the session at path without locking
// it, for an outcome that never opened it. A session that does not exist
// has none; one that cannot be read gives nil, so the outcome leaves its
// labels out rather than claim it has none.
func storedLabels(path string) *[]string {
	if !sessionfile.Exists(path) {
		return ptr([]string{})
	}
	sess, err := agentloop.OpenSession(agentloop.NewFileStore(path))
	if err != nil {
		return nil
	}
	return ptr(sess.Labels())
}

func ptr[T any](v T) *T { return &v }

func writeJSON(w io.Writer, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	_, err = w.Write(append(b, '\n'))
	return err
}
