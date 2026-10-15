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
	"io/fs"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
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

// keyTimeout is how long api_key_command may run.
const keyTimeout = 30 * time.Second

// errInterrupted ends a chat that SIGTERM stopped: exit 130.
var errInterrupted = errors.New("interrupted")

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
		// SIGTERM and SIGINT mean "cancel the run and still report it". The
		// handler goes in before anything else so that no signal after the
		// process starts can kill it without an outcome line.
		ctx, stop := context.Background(), func() {}
		if args[0] == "run" {
			ctx, stop = signal.NotifyContext(ctx, syscall.SIGTERM, syscall.SIGINT)
		}
		defer stop()
		var req *request
		req, err = loadRequest(args[1])
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
	APIKeyCommand      []string             `json:"api_key_command"`
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
	if len(r.APIKeyCommand) == 0 {
		return usagef("request has no api_key_command")
	}
	if len(r.Tools) > 0 && len(r.ToolCommand) == 0 {
		return usagef("request has tools but no tool_command")
	}
	seen := map[string]bool{}
	for _, t := range r.Tools {
		if t.Name == "" {
			return usagef("a tool has no name")
		}
		if !toolName.MatchString(t.Name) {
			return usagef("tool name %q is not 1 to 64 letters, digits, - or _", t.Name)
		}
		if seen[t.Name] {
			return usagef("two tools are named %q", t.Name)
		}
		seen[t.Name] = true
		if p := bytes.TrimSpace(t.Parameters); len(p) > 0 && p[0] != '{' {
			return usagef("tool %q: parameters must be a JSON Schema object", t.Name)
		}
	}
	if r.ToolTimeoutSeconds < 0 || r.TimeoutSeconds < 0 {
		return usagef("timeouts cannot be negative")
	}
	if r.Limits.MaxTurns < 0 || r.Limits.MaxToolCalls < 0 || r.Limits.MaxTokens < 0 {
		return usagef("limits cannot be negative")
	}
	if rt := r.Retry; rt != nil && (rt.Retries < 0 || rt.BackoffSeconds < 0 || rt.MaxWaitSeconds < 0) {
		return usagef("retry values cannot be negative")
	}
	if r.BaseURL != "" {
		u, err := url.Parse(r.BaseURL)
		if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
			return usagef("base_url %q is not an http or https URL", r.BaseURL)
		}
	}
	if err := r.checkNUL(); err != nil {
		return err
	}
	// Compile every tool's schema now, with the core's own rules, by running
	// an agent that has nothing to do: a bad schema is an unusable request,
	// found before a session is touched.
	sess, err := agentloop.OpenSession(agentloop.NewMemoryStore())
	if err != nil {
		return err
	}
	if _, err := r.agent("").Run(context.Background(), sess, ""); err != nil {
		return usagef("bad tool definition: %v", err)
	}
	return nil
}

var toolName = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)

// checkNUL refuses strings that no file name, argument or environment
// entry can hold.
func (r *request) checkNUL() error {
	bad := func(s string) bool { return strings.ContainsRune(s, 0) }
	if bad(r.Session) || bad(r.ToolDir) {
		return usagef("session and tool_dir cannot contain a NUL character")
	}
	for _, a := range r.APIKeyCommand {
		if bad(a) {
			return usagef("api_key_command cannot contain a NUL character")
		}
	}
	if len(r.APIKeyCommand) > 0 && r.APIKeyCommand[0] == "" {
		return usagef("api_key_command starts with an empty program name")
	}
	for _, a := range r.ToolCommand {
		if bad(a) {
			return usagef("tool_command cannot contain a NUL character")
		}
	}
	if len(r.ToolCommand) > 0 && r.ToolCommand[0] == "" {
		return usagef("tool_command starts with an empty program name")
	}
	for k, v := range r.ToolEnv {
		if k == "" || strings.ContainsRune(k, '=') || bad(k) || bad(v) {
			return usagef("tool_env entry %q is not a usable environment variable", k)
		}
	}
	return nil
}

// keyError is why the key command gave no key. Its message is plain text for
// the operator and never holds the command's stdout.
type keyError struct {
	msg      string
	canceled bool // agentloop was told to stop while the command ran
}

func (e *keyError) Error() string { return e.msg }

// fetchKey runs the key command: no shell, an empty stdin, and an
// environment of only PATH and HOME from this process. Its trimmed stdout is
// the key. The key is returned to the caller and goes nowhere else.
func fetchKey(ctx context.Context, argv []string, timeout time.Duration) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	cmd.Env = nil
	for _, k := range []string{"PATH", "HOME"} {
		if v, ok := os.LookupEnv(k); ok {
			cmd.Env = append(cmd.Env, k+"="+v)
		}
	}
	if cmd.Env == nil {
		cmd.Env = []string{}
	}
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	// Stop the command's children too, or one that holds the pipes open
	// would keep the wait going past the deadline.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error { return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL) }
	cmd.WaitDelay = time.Second
	err := cmd.Run()
	if ctx.Err() != nil {
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return "", &keyError{msg: fmt.Sprintf("The api_key_command did not finish within %d seconds.", int(timeout.Seconds()))}
		}
		return "", &keyError{msg: "The api_key_command was interrupted.", canceled: true}
	}
	if err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			what := fmt.Sprintf("exited %d", ee.ExitCode())
			if ee.ExitCode() < 0 {
				what = "was killed (" + ee.String() + ")"
			}
			msg := "The api_key_command " + what
			if tail := stderrTail(stderr.Bytes()); tail != "" {
				msg += ": " + tail
			}
			return "", &keyError{msg: msg + "."}
		}
		return "", &keyError{msg: fmt.Sprintf("The api_key_command could not run: %v.", err)}
	}
	key := strings.TrimSpace(stdout.String())
	if key == "" {
		return "", &keyError{msg: "The api_key_command printed no key."}
	}
	return key, nil
}

// stderrTail is the end of what the key command wrote to stderr, on one line.
func stderrTail(b []byte) string {
	s := strings.Join(strings.Fields(strings.ToValidUTF8(string(b), "")), " ")
	const max = 200
	if r := []rune(s); len(r) > max {
		return "..." + string(r[len(r)-max:])
	}
	return strings.TrimSuffix(s, ".")
}

// seconds converts a request value to a duration. A huge value saturates
// instead of overflowing into a negative (instantly expired) deadline, and
// a tiny positive one is not rounded down to zero, which means "none".
func seconds(s float64) time.Duration {
	const max = 100 * 365 * 24 * 3600.0
	if s <= 0 {
		return 0
	}
	if s > max {
		s = max
	}
	if d := time.Duration(s * float64(time.Second)); d > 0 {
		return d
	}
	return 1
}

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

// openSession opens the session and takes an exclusive lock on it for as
// long as the process lives (or until the returned function is called).
// Two processes on one session would each run its pending calls and log
// over the other's entries; the lock is released by the OS if the process
// is killed.
func (r *request) openSession() (*agentloop.Session, func(), error) {
	if err := os.MkdirAll(filepath.Dir(r.Session), 0o755); err != nil {
		return nil, nil, usagef("cannot create the session directory: %v", err)
	}
	if fi, err := os.Stat(r.Session); err == nil && fi.IsDir() {
		return nil, nil, usagef("session %s is a directory", r.Session)
	}
	lock, err := os.OpenFile(r.Session+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, nil, usagef("cannot lock the session: %v", err)
	}
	if err := syscall.Flock(int(lock.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		lock.Close()
		return nil, nil, fmt.Errorf("session %s is in use by another agentloop", r.Session)
	}
	sess, err := agentloop.OpenSession(agentloop.NewFileStore(r.Session))
	if err != nil {
		lock.Close()
		return nil, nil, err
	}
	return sess, func() { lock.Close() }, nil
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

func runCmd(ctx context.Context, req *request, stdout, stderr io.Writer) error {
	key, err := fetchKey(ctx, req.APIKeyCommand, keyTimeout)
	var ke *keyError
	if errors.As(err, &ke) && ke.canceled {
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
		if _, err := os.Stat(req.Session); errors.Is(err, fs.ErrNotExist) {
			return usagef("run has no prompt and there is no session to resume")
		}
	}
	sess, unlock, err := req.openSession()
	if err != nil {
		return err
	}
	defer unlock()
	// With no prompt the run resumes the session. Only a session with no
	// conversation at all leaves nothing to resume; a finished one just
	// completes again, so a supervisor can restart a run that was cut off
	// just before it ended.
	if req.Prompt == "" && len(sess.Messages()) == 0 {
		return usagef("run has no prompt and the session has no conversation to resume")
	}

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
	key, err := fetchKey(context.Background(), req.APIKeyCommand, keyTimeout)
	if err != nil {
		return usagef("%v", err)
	}
	sess, unlock, err := req.openSession()
	if err != nil {
		return err
	}
	defer unlock()
	agent := req.agent(key)

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
