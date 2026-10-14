// Command agentloop runs one agent from a JSON request.
//
//	agentloop run  REQUEST.json   run one agent to an outcome
//	agentloop chat REQUEST.json   talk to it on the terminal
//	agentloop version
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"

	"agentloop"
	"agentloop/anthropic"
	"agentloop/command"
	"agentloop/openai"
)

// version is set at build time with -ldflags "-X main.version=...".
var version = "dev"

const defaultKeyEnv = "AGENTLOOP_API_KEY"

// usageError is an unusable request or command line: exit 2.
type usageError struct{ msg string }

func (e *usageError) Error() string { return e.msg }

func usagef(format string, args ...any) error {
	return &usageError{fmt.Sprintf(format, args...)}
}

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
		var req *request
		req, err = loadRequest(args[1])
		if err == nil {
			if args[0] == "run" {
				err = runCmd(req, stdout, stderr)
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
	fmt.Fprintln(stderr, "agentloop:", err)
	var ue *usageError
	if errors.As(err, &ue) {
		return 2
	}
	return 1
}

// request is the JSON document that says what to run.
type request struct {
	Provider           string               `json:"provider"`
	Model              string               `json:"model"`
	Effort             string               `json:"effort"`
	BaseURL            string               `json:"base_url"`
	APIKeyEnv          string               `json:"api_key_env"`
	RefusalFallback    *bool                `json:"refusal_fallback"`
	System             string               `json:"system"`
	Prompt             string               `json:"prompt"`
	Session            string               `json:"session"`
	Tools              []agentloop.ToolSpec `json:"tools"`
	ToolCommand        []string             `json:"tool_command"`
	ToolEnv            map[string]string    `json:"tool_env"`
	ToolDir            string               `json:"tool_dir"`
	ToolTimeoutSeconds float64              `json:"tool_timeout_seconds"`
	TimeoutSeconds     float64              `json:"timeout_seconds"`
	Limits             agentloop.Limits     `json:"limits"`
	Retry              *struct {
		Retries        int     `json:"retries"`
		BackoffSeconds float64 `json:"backoff_seconds"`
		MaxWaitSeconds float64 `json:"max_wait_seconds"`
	} `json:"retry"`
}

func loadRequest(path string) (*request, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, usagef("cannot read request: %v", err)
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	var r request
	if err := dec.Decode(&r); err != nil {
		return nil, usagef("bad request %s: %v", path, err)
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, usagef("bad request %s: trailing data after the JSON object", path)
	}
	if err := r.validate(); err != nil {
		return nil, err
	}
	return &r, nil
}

func (r *request) validate() error {
	switch r.Provider {
	case "anthropic", "openai":
	case "":
		return usagef("request has no provider")
	default:
		return usagef("unknown provider %q (want anthropic or openai)", r.Provider)
	}
	if r.Model == "" {
		return usagef("request has no model")
	}
	switch agentloop.Effort(r.Effort) {
	case "", agentloop.EffortLow, agentloop.EffortMedium, agentloop.EffortHigh, agentloop.EffortXHigh, agentloop.EffortMax:
	default:
		return usagef("unknown effort %q", r.Effort)
	}
	if r.Session == "" {
		return usagef("request has no session path")
	}
	if len(r.Tools) > 0 && len(r.ToolCommand) == 0 {
		return usagef("request has tools but no tool_command")
	}
	seen := map[string]bool{}
	for _, t := range r.Tools {
		if t.Name == "" {
			return usagef("a tool has no name")
		}
		if seen[t.Name] {
			return usagef("two tools are named %q", t.Name)
		}
		seen[t.Name] = true
	}
	if r.ToolTimeoutSeconds < 0 || r.TimeoutSeconds < 0 {
		return usagef("timeouts cannot be negative")
	}
	return nil
}

func (r *request) keyEnv() string {
	if r.APIKeyEnv != "" {
		return r.APIKeyEnv
	}
	return defaultKeyEnv
}

func seconds(s float64) time.Duration { return time.Duration(s * float64(time.Second)) }

// refusalFallbackModels get the fallback on unless the request says otherwise.
var refusalFallbackModels = map[string]bool{
	"claude-opus-5-5":   true,
	"claude-opus-5":     true,
	"claude-fable-5-1":  true,
	"claude-sonnet-5-5": true,
}

func (r *request) newModel(key string) agentloop.Model {
	if r.Provider == "openai" {
		return openai.New(openai.Config{APIKey: key, BaseURL: r.BaseURL})
	}
	fallback := refusalFallbackModels[r.Model]
	if r.RefusalFallback != nil {
		fallback = *r.RefusalFallback
	}
	return anthropic.New(anthropic.Config{APIKey: key, BaseURL: r.BaseURL, RefusalFallback: fallback})
}

// agent builds the agent. The key goes to the model and nowhere else: the
// tools get exactly tool_env.
func (r *request) agent(key string) *agentloop.Agent {
	env := make([]string, 0, len(r.ToolEnv))
	for k, v := range r.ToolEnv {
		env = append(env, k+"="+v)
	}
	sort.Strings(env)
	var tools []agentloop.Tool
	for _, spec := range r.Tools {
		argv := append(append([]string{}, r.ToolCommand...), spec.Name)
		tools = append(tools, &command.Tool{Def: spec, Argv: argv, Env: env, Dir: r.ToolDir})
	}
	a := &agentloop.Agent{
		Model:       r.newModel(key),
		ModelID:     r.Model,
		System:      r.System,
		Tools:       tools,
		Effort:      agentloop.Effort(r.Effort),
		Limits:      r.Limits,
		ToolTimeout: seconds(r.ToolTimeoutSeconds),
	}
	if r.Retry != nil {
		a.Retry = &agentloop.RetryPolicy{
			Retries: r.Retry.Retries,
			Backoff: seconds(r.Retry.BackoffSeconds),
			MaxWait: seconds(r.Retry.MaxWaitSeconds),
		}
	}
	return a
}

func (r *request) openSession() (*agentloop.Session, error) {
	if err := os.MkdirAll(filepath.Dir(r.Session), 0o755); err != nil {
		return nil, fmt.Errorf("create session directory: %w", err)
	}
	return agentloop.OpenSession(agentloop.NewFileStore(r.Session))
}

// outcomeLine is the last line of a run.
type outcomeLine struct {
	Type string `json:"type"`
	agentloop.Outcome
	Session string `json:"session"`
}

func missingKeyOutcome(name string) agentloop.Outcome {
	return agentloop.Outcome{
		Reason: agentloop.ReasonModelError,
		Error: &agentloop.ModelError{
			Kind:    agentloop.Unauthorized,
			Message: fmt.Sprintf("No API key in $%s.", name),
		},
	}
}

func runCmd(req *request, stdout, stderr io.Writer) error {
	key := os.Getenv(req.keyEnv())
	if key == "" {
		return writeJSON(stdout, outcomeLine{Type: "outcome", Outcome: missingKeyOutcome(req.keyEnv()), Session: req.Session})
	}
	sess, err := req.openSession()
	if err != nil {
		return err
	}
	if req.Prompt == "" && len(sess.Pending()) == 0 {
		msgs := sess.Messages()
		if len(msgs) == 0 || msgs[len(msgs)-1].Role != agentloop.User {
			return usagef("run has no prompt and nothing is pending in the session")
		}
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()
	if req.TimeoutSeconds > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, seconds(req.TimeoutSeconds))
		defer cancel()
	}

	agent := req.agent(key)
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

func chatCmd(req *request, stdin io.Reader, stdout, stderr io.Writer) error {
	key := os.Getenv(req.keyEnv())
	if key == "" {
		return usagef("No API key in $%s.", req.keyEnv())
	}
	sess, err := req.openSession()
	if err != nil {
		return err
	}
	agent := req.agent(key)

	var mu sync.Mutex
	var cancelRun context.CancelFunc // set while a run is going
	canceled := false
	sigs := make(chan os.Signal, 4)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		for sig := range sigs {
			mu.Lock()
			if sig == syscall.SIGINT && cancelRun != nil && !canceled {
				canceled = true
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

	in := bufio.NewReader(stdin)
	for {
		fmt.Fprint(stderr, "> ")
		line, rerr := in.ReadString('\n')
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
			mu.Unlock()
			cancel()
			if streamed {
				fmt.Fprintln(stdout)
			}
			if err != nil {
				return err
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
			fmt.Fprintf(w, "[tool] %s %s\n", c.Name, brief(string(c.Input)))
		}
	case e.ToolResult != nil:
		mark := "->"
		if e.ToolResult.Result.IsError {
			mark = "-> error:"
		}
		fmt.Fprintf(w, "[tool] %s %s\n", mark, brief(e.ToolResult.Result.Content))
	}
}

func brief(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	const max = 120
	if r := []rune(s); len(r) > max {
		return string(r[:max]) + "..."
	}
	return s
}
