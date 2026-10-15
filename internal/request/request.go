// Package request is the JSON document that says what to run, and what it
// takes to turn that document into an agent: loading, validation, the API
// key command and agent construction. Nothing here touches a session.
package request

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/url"
	"os"
	"os/exec"
	"regexp"
	"slices"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/anthropic"
	"github.com/kirahowe/agentloop/command"
	"github.com/kirahowe/agentloop/internal/usage"
	"github.com/kirahowe/agentloop/openai"
)

// KeyTimeout is how long api_key_command may run.
const KeyTimeout = 30 * time.Second

// Request is the JSON document that says what to run.
type Request struct {
	Provider           string               `json:"provider"`
	Model              string               `json:"model"`
	Effort             string               `json:"effort"`
	BaseURL            string               `json:"base_url"`
	APIKeyCommand      []string             `json:"api_key_command"`
	RefusalFallback    *bool                `json:"refusal_fallback"`
	System             string               `json:"system"`
	Prompt             string               `json:"prompt"`
	Session            string               `json:"session"`
	SessionDir         string               `json:"session_dir"`
	Tools              []agentloop.ToolSpec `json:"tools"`
	ToolCommand        []string             `json:"tool_command"`
	ToolEnv            map[string]string    `json:"tool_env"`
	ToolDir            string               `json:"tool_dir"`
	ToolTimeoutSeconds float64              `json:"tool_timeout_seconds"`
	TimeoutSeconds     float64              `json:"timeout_seconds"`
	Limits             agentloop.Limits     `json:"limits"`
	ToolKinds          map[string]string    `json:"tool_kinds"`
	ContextWindow      int64                `json:"context_window"`
	Retry              *struct {
		Retries        int     `json:"retries"`
		BackoffSeconds float64 `json:"backoff_seconds"`
		MaxWaitSeconds float64 `json:"max_wait_seconds"`
	} `json:"retry"`
}

// Mode is the command a request is for. The two read the same document
// with different rules about which fields it must and must not have.
type Mode int

const (
	// Run is `agentloop run`: one session file, named by session.
	Run Mode = iota
	// ACP is `agentloop acp`: the client opens sessions, each a file in
	// session_dir, and sends the prompts. A deadline is refused, because
	// it would end a turn as cancelled, which ACP keeps for the client.
	ACP
)

// ToolKinds are the values tool_kinds may give a tool: ACP's tool kinds.
var ToolKinds = []string{"read", "edit", "delete", "move", "search", "execute", "think", "fetch", "switch_mode", "other"}

// Load reads and validates the request in the file at path for a mode.
// Every error is a *usage.Error.
func Load(path string, mode Mode) (*Request, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, usage.Errorf("cannot read request: %v", err)
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	var r Request
	if err := dec.Decode(&r); err != nil {
		return nil, usage.Errorf("bad request %s: %v", path, err)
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, usage.Errorf("bad request %s: trailing data after the JSON object", path)
	}
	if err := r.validate(mode); err != nil {
		return nil, err
	}
	return &r, nil
}

func (r *Request) validate(mode Mode) error {
	switch r.Provider {
	case "anthropic", "openai":
	case "":
		return usage.Errorf("request has no provider")
	default:
		return usage.Errorf("unknown provider %q (want anthropic or openai)", r.Provider)
	}
	if r.Model == "" {
		return usage.Errorf("request has no model")
	}
	switch agentloop.Effort(r.Effort) {
	case "", agentloop.EffortLow, agentloop.EffortMedium, agentloop.EffortHigh, agentloop.EffortXHigh, agentloop.EffortMax:
	default:
		return usage.Errorf("unknown effort %q", r.Effort)
	}
	if err := r.validateMode(mode); err != nil {
		return err
	}
	if len(r.APIKeyCommand) == 0 {
		return usage.Errorf("request has no api_key_command")
	}
	if len(r.Tools) > 0 && len(r.ToolCommand) == 0 {
		return usage.Errorf("request has tools but no tool_command")
	}
	seen := map[string]bool{}
	for _, t := range r.Tools {
		if t.Name == "" {
			return usage.Errorf("a tool has no name")
		}
		if !toolName.MatchString(t.Name) {
			return usage.Errorf("tool name %q is not 1 to 64 letters, digits, - or _", t.Name)
		}
		if seen[t.Name] {
			return usage.Errorf("two tools are named %q", t.Name)
		}
		seen[t.Name] = true
		if p := bytes.TrimSpace(t.Parameters); len(p) > 0 && p[0] != '{' {
			return usage.Errorf("tool %q: parameters must be a JSON Schema object", t.Name)
		}
	}
	if r.ToolTimeoutSeconds < 0 || r.TimeoutSeconds < 0 {
		return usage.Errorf("timeouts cannot be negative")
	}
	if r.Limits.MaxTurns < 0 || r.Limits.MaxToolCalls < 0 || r.Limits.MaxTokens < 0 {
		return usage.Errorf("limits cannot be negative")
	}
	if rt := r.Retry; rt != nil && (rt.Retries < 0 || rt.BackoffSeconds < 0 || rt.MaxWaitSeconds < 0) {
		return usage.Errorf("retry values cannot be negative")
	}
	if r.BaseURL != "" {
		u, err := url.Parse(r.BaseURL)
		if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
			return usage.Errorf("base_url %q is not an http or https URL", r.BaseURL)
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
	if _, err := r.Agent("").Run(context.Background(), sess, ""); err != nil {
		return usage.Errorf("bad tool definition: %v", err)
	}
	return nil
}

// validateMode checks the fields that differ between the commands.
func (r *Request) validateMode(mode Mode) error {
	if mode == Run {
		if r.Session == "" {
			return usage.Errorf("request has no session path")
		}
		for _, f := range []field{
			{"session_dir", r.SessionDir != ""}, {"tool_kinds", r.ToolKinds != nil}, {"context_window", r.ContextWindow != 0},
		} {
			if f.set {
				return usage.Errorf("%s is only for agentloop acp", f.name)
			}
		}
		return nil
	}
	if r.SessionDir == "" {
		return usage.Errorf("request has no session_dir")
	}
	for _, f := range []field{
		{"session", r.Session != ""}, {"prompt", r.Prompt != ""}, {"timeout_seconds", r.TimeoutSeconds != 0},
	} {
		if f.set {
			return usage.Errorf("%s is not used by agentloop acp", f.name)
		}
	}
	if r.ContextWindow < 0 {
		return usage.Errorf("context_window cannot be negative")
	}
	tools := map[string]bool{}
	for _, t := range r.Tools {
		tools[t.Name] = true
	}
	names := slices.Sorted(maps.Keys(r.ToolKinds))
	for _, name := range names {
		kind := r.ToolKinds[name]
		if !tools[name] {
			return usage.Errorf("tool_kinds names %q, which is not one of the tools", name)
		}
		if !slices.Contains(ToolKinds, kind) {
			return usage.Errorf("tool_kinds gives %q the kind %q (want one of %s)", name, kind, strings.Join(ToolKinds, ", "))
		}
	}
	return nil
}

type field struct {
	name string
	set  bool
}

var toolName = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)

// checkNUL refuses strings that no file name, argument or environment
// entry can hold.
func (r *Request) checkNUL() error {
	bad := func(s string) bool { return strings.ContainsRune(s, 0) }
	if bad(r.Session) || bad(r.SessionDir) || bad(r.ToolDir) {
		return usage.Errorf("session, session_dir and tool_dir cannot contain a NUL character")
	}
	for _, a := range r.APIKeyCommand {
		if bad(a) {
			return usage.Errorf("api_key_command cannot contain a NUL character")
		}
	}
	if len(r.APIKeyCommand) > 0 && r.APIKeyCommand[0] == "" {
		return usage.Errorf("api_key_command starts with an empty program name")
	}
	for _, a := range r.ToolCommand {
		if bad(a) {
			return usage.Errorf("tool_command cannot contain a NUL character")
		}
	}
	if len(r.ToolCommand) > 0 && r.ToolCommand[0] == "" {
		return usage.Errorf("tool_command starts with an empty program name")
	}
	for k, v := range r.ToolEnv {
		if k == "" || strings.ContainsRune(k, '=') || bad(k) || bad(v) {
			return usage.Errorf("tool_env entry %q is not a usable environment variable", k)
		}
	}
	return nil
}

// KeyError is why the key command gave no key. Its message is plain text for
// the operator and never holds the command's stdout.
type KeyError struct {
	msg      string
	Canceled bool // agentloop was told to stop while the command ran
}

func (e *KeyError) Error() string { return e.msg }

// FetchKey runs the key command: no shell, an empty stdin, and an
// environment of only PATH and HOME from this process. Its trimmed stdout is
// the key. The key is returned to the caller and goes nowhere else.
func FetchKey(ctx context.Context, argv []string, timeout time.Duration) (string, error) {
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
			return "", &KeyError{msg: fmt.Sprintf("The api_key_command did not finish within %d seconds.", int(timeout.Seconds()))}
		}
		return "", &KeyError{msg: "The api_key_command was interrupted.", Canceled: true}
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
			return "", &KeyError{msg: msg + "."}
		}
		return "", &KeyError{msg: fmt.Sprintf("The api_key_command could not run: %v.", err)}
	}
	key := strings.TrimSpace(stdout.String())
	if key == "" {
		return "", &KeyError{msg: "The api_key_command printed no key."}
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

// Seconds converts a request value to a duration. A huge value saturates
// instead of overflowing into a negative (instantly expired) deadline, and
// a tiny positive one is not rounded down to zero, which means "none".
func Seconds(s float64) time.Duration {
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

func (r *Request) newModel(key string) agentloop.Model {
	if r.Provider == "openai" {
		return openai.New(openai.Config{APIKey: key, BaseURL: r.BaseURL})
	}
	fallback := refusalFallbackModels[r.Model]
	if r.RefusalFallback != nil {
		fallback = *r.RefusalFallback
	}
	return anthropic.New(anthropic.Config{APIKey: key, BaseURL: r.BaseURL, RefusalFallback: fallback})
}

// Agent builds the agent. The key goes to the model and nowhere else: the
// tools get exactly tool_env.
func (r *Request) Agent(key string) *agentloop.Agent {
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
		ToolTimeout: Seconds(r.ToolTimeoutSeconds),
	}
	if r.Retry != nil {
		a.Retry = &agentloop.RetryPolicy{
			Retries: r.Retry.Retries,
			Backoff: Seconds(r.Retry.BackoffSeconds),
			MaxWait: Seconds(r.Retry.MaxWaitSeconds),
		}
	}
	return a
}

// Brief squeezes s onto one short line.
func Brief(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	const max = 120
	if r := []rune(s); len(r) > max {
		return string(r[:max]) + "..."
	}
	return s
}
