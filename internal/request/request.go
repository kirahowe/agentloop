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
	"reflect"
	"regexp"
	"slices"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/anthropic"
	"github.com/kirahowe/agentloop/command"
	"github.com/kirahowe/agentloop/internal/strictjson"
	"github.com/kirahowe/agentloop/internal/usage"
	"github.com/kirahowe/agentloop/openai"
	"github.com/kirahowe/agentloop/quarantine"
	"github.com/kirahowe/agentloop/trifecta"
)

// displayRefPrefix is what a display_ref_prefix may be: a kind, a colon
// and a path of plain segments, so the ref the model is told is one
// relay can read back.
var displayRefPrefix = regexp.MustCompile(`^[a-z]{1,20}:[a-z0-9._/-]{0,80}$`)

// KeyTimeout is how long api_key_command may run.
const KeyTimeout = 30 * time.Second

// Request is the JSON document that says what to run.
type Request struct {
	Provider           string                 `json:"provider"`
	Model              string                 `json:"model"`
	Effort             string                 `json:"effort"`
	BaseURL            string                 `json:"base_url"`
	APIKeyCommand      []string               `json:"api_key_command"`
	RefusalFallback    *bool                  `json:"refusal_fallback"`
	System             string                 `json:"system"`
	Prompt             string                 `json:"prompt"`
	Session            string                 `json:"session"`
	SessionDir         string                 `json:"session_dir"`
	Tools              []Tool                 `json:"tools"`
	MCPServers         map[string]MCPServer   `json:"mcp_servers"`
	ToolCommand        []string               `json:"tool_command"`
	ToolEnv            map[string]string      `json:"tool_env"`
	ToolDir            string                 `json:"tool_dir"`
	ToolTimeoutSeconds float64                `json:"tool_timeout_seconds"`
	TimeoutSeconds     float64                `json:"timeout_seconds"`
	Limits             agentloop.Limits       `json:"limits"`
	ToolKinds          map[string]string      `json:"tool_kinds"`
	Title              string                 `json:"title"`
	NewSessionsOnly    *bool                  `json:"new_sessions_only"`
	Policy             *Policy                `json:"policy"`
	ToolProtocol       string                 `json:"tool_protocol"`
	Resolve            []agentloop.Resolution `json:"resolve"`
	Quarantine         *Quarantine            `json:"quarantine"`
	DisplayRefPrefix   string                 `json:"display_ref_prefix"`
	Retry              *struct {
		Retries        int     `json:"retries"`
		BackoffSeconds float64 `json:"backoff_seconds"`
		MaxWaitSeconds float64 `json:"max_wait_seconds"`
	} `json:"retry"`
}

// Tool is one tool of the request. A command tool (no protocol) gives its
// name, description and parameters, and runs tool_command. An MCP tool
// ("protocol": "mcp-stdio", or "mcp-http" for a remote server) gives its
// name, the server it comes from and the server's name for it; the server
// gives its description and parameters.
type Tool struct {
	agentloop.ToolSpec
	Protocol string `json:"protocol"`
	Server   string `json:"server"`
	Tool     string `json:"tool"`
	// AllowContent lets an MCP tool's results hold these kinds of
	// non-text content (see mcp.ToolConfig.AllowContent).
	AllowContent []string `json:"allow_content"`
	// Digest, when given, pins an MCP tool's definition (see mcp.Digest):
	// a server that lists another fails the start.
	Digest string `json:"digest"`
	// FixedArgs and HiddenArgs narrow an MCP tool's arguments (see
	// mcp.ToolConfig).
	FixedArgs  map[string]json.RawMessage `json:"fixed_args"`
	HiddenArgs []string                   `json:"hidden_args"`
}

// MCP reports whether the tool comes from an MCP server.
func (t Tool) MCP() bool {
	return t.Protocol == ToolProtocolMCPStdio || t.Protocol == ToolProtocolMCPHTTP
}

// MCPServer is an MCP server run over stdio for the length of the run, or
// a remote one reached over streamable HTTP.
type MCPServer struct {
	// Command is the server's argv. An argument may hold
	// CredentialsPlaceholder.
	Command []string `json:"command"`
	// URL is a remote server's endpoint, in place of Command: https, or
	// http to a loopback address.
	URL string `json:"url"`
	// CredentialsCommand, when given, is run once at start. For a stdio
	// server, what it prints is written to a file only this user can
	// read, which is removed when agentloop exits, and
	// CredentialsPlaceholder in Command is replaced by that file's path.
	// For a remote server what it prints, trimmed, is the bearer token,
	// kept only in memory; it is run again when the server refuses the
	// token.
	CredentialsCommand []string `json:"credentials_command"`
	// SendLabels sends the session's labels with each call.
	SendLabels bool `json:"send_labels"`
}

// ToolProtocolMCPStdio is the protocol of a tool from an MCP server run
// over stdio, and ToolProtocolMCPHTTP of one from a remote server.
const (
	ToolProtocolMCPStdio = "mcp-stdio"
	ToolProtocolMCPHTTP  = "mcp-http"
)

// CredentialsPlaceholder in an MCP server's command is replaced by the
// path of the file holding what its credentials_command printed.
const CredentialsPlaceholder = "{credentials_file}"

// Policy is the request's trifecta policy: the session's starting labels,
// each tool's class and labels, and the labels the session may never
// hold. {"unrestricted": true} instead turns the policy off; it is for
// agentloop's own examples and tests.
type Policy struct {
	Unrestricted *bool                    `json:"unrestricted"`
	Labels       []string                 `json:"labels"`
	Tools        map[string]trifecta.Tool `json:"tools"`
	ForbidLabels []string                 `json:"forbid_labels"`
}

// Quarantine turns on the quarantined reader, read_quarantined: the
// reader model, on the request's provider with its base_url and key, and
// the tools it may read through.
type Quarantine struct {
	Model           string `json:"model"`
	Effort          string `json:"effort"`
	MaxOutputTokens int    `json:"max_output_tokens"`
	// Retries is how many times the reader is asked again after an
	// answer that is not valid. Nil means DefaultQuarantineRetries.
	Retries *int     `json:"retries"`
	Tools   []string `json:"tools"`
}

// DefaultQuarantineRetries is the reader's retries when the request
// gives none.
const DefaultQuarantineRetries = 1

// MaxQuarantineRetries bounds quarantine.retries, since each one is a
// model call.
const MaxQuarantineRetries = 5

// Restricted reports whether the request has a real policy.
func (r *Request) Restricted() bool {
	return r.Policy != nil && r.Policy.Unrestricted == nil
}

// ToolProtocolEnvelope is the tool_protocol for envelope tools; the plain
// protocol is the empty string.
const ToolProtocolEnvelope = "envelope"

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
	// encoding/json takes "POLICY" for "policy" and lets a repeated key
	// replace the first; a request names each field once, exactly.
	if err := strictjson.Check(raw, reflect.TypeFor[Request]()); err != nil {
		return nil, usage.Errorf("bad request %s: %v", path, err)
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
	if r.DisplayRefPrefix != "" && !displayRefPrefix.MatchString(r.DisplayRefPrefix) {
		return usage.Errorf("display_ref_prefix must be a kind and a path, such as shown:2026-10-18/, of at most 100 characters")
	}
	seen := map[string]bool{}
	for _, t := range r.Tools {
		if t.Name == "" {
			return usage.Errorf("a tool has no name")
		}
		if t.Name == quarantine.ToolName {
			return usage.Errorf("%s is built in; give a quarantine block instead of a tool by that name", quarantine.ToolName)
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
	if err := r.validateMCP(); err != nil {
		return err
	}
	if len(r.commandTools()) > 0 && len(r.ToolCommand) == 0 {
		return usage.Errorf("request has tools but no tool_command")
	}
	if err := r.validatePolicy(); err != nil {
		return err
	}
	if err := r.validateQuarantine(); err != nil {
		return err
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
	a, err := r.Agent("", r.stubMCP())
	if err != nil {
		return usage.Errorf("bad quarantine: %v", err)
	}
	if _, err := a.Run(context.Background(), sess, ""); err != nil {
		return usage.Errorf("bad tool definition: %v", err)
	}
	return nil
}

// validatePolicy fails closed: a request with tools must say how they are
// policed, every tool needs a known class, and every label must be one
// agentloop knows, so a typo cannot drop one.
func (r *Request) validatePolicy() error {
	switch r.ToolProtocol {
	case "", ToolProtocolEnvelope:
	default:
		return usage.Errorf("unknown tool_protocol %q (want %q or none)", r.ToolProtocol, ToolProtocolEnvelope)
	}
	p := r.Policy
	if p == nil {
		if len(r.Tools) > 0 {
			return usage.Errorf(`request has tools but no policy (give one, or {"unrestricted": true})`)
		}
		return nil
	}
	if p.Unrestricted != nil {
		if !*p.Unrestricted {
			return usage.Errorf(`policy.unrestricted can only be true; leave it out for a real policy`)
		}
		if p.Labels != nil || p.Tools != nil || p.ForbidLabels != nil {
			return usage.Errorf("an unrestricted policy cannot have labels, tools or forbid_labels")
		}
		return nil
	}
	if len(r.commandTools()) > 0 && r.ToolProtocol != ToolProtocolEnvelope {
		return usage.Errorf(`a request with a policy and command tools needs "tool_protocol": %q`, ToolProtocolEnvelope)
	}
	if err := trifecta.CheckLabels(p.Labels); err != nil {
		return usage.Errorf("policy.labels: %v", err)
	}
	if err := trifecta.CheckLabels(p.ForbidLabels); err != nil {
		return usage.Errorf("policy.forbid_labels: %v", err)
	}
	// A session that would start holding a label it may not hold is not
	// started at all.
	if bad := agentloop.Forbidden(p.Labels, p.ForbidLabels); len(bad) > 0 {
		return usage.Errorf("policy.labels holds %v, which policy.forbid_labels forbids, so the session cannot start", bad)
	}
	tools := map[string]bool{}
	for _, name := range r.toolNames() {
		tools[name] = true
		if _, ok := p.Tools[name]; !ok {
			return usage.Errorf("policy.tools has no entry for the tool %q", name)
		}
	}
	for _, name := range slices.Sorted(maps.Keys(p.Tools)) {
		e := p.Tools[name]
		if !tools[name] {
			return usage.Errorf("policy.tools names %q, which is not one of the tools", name)
		}
		if !trifecta.KnownClass(e.Class) {
			return usage.Errorf("policy.tools gives %q the class %q (want one of %v)", name, e.Class, trifecta.Classes)
		}
		if err := trifecta.CheckLabels(e.Labels); err != nil {
			return usage.Errorf("policy.tools %q: %v", name, err)
		}
		if bad := agentloop.Forbidden(e.Labels, p.ForbidLabels); len(bad) > 0 {
			return usage.Errorf("policy.tools gives %q the labels %v, which policy.forbid_labels forbids", name, bad)
		}
	}
	return nil
}

// commandTools are the tools that run tool_command.
func (r *Request) commandTools() []Tool {
	var out []Tool
	for _, t := range r.Tools {
		if !t.MCP() {
			out = append(out, t)
		}
	}
	return out
}

// toolNames are the names of every tool the agent has: the request's
// tools, and read_quarantined when there is a quarantine block.
func (r *Request) toolNames() []string {
	var names []string
	for _, t := range r.Tools {
		names = append(names, t.Name)
	}
	if r.Quarantine != nil {
		names = append(names, quarantine.ToolName)
	}
	return names
}

// validateQuarantine checks the quarantine block. The reader needs a
// real policy, since its whole purpose is the labels; it reads only
// through tools of class read or egress, which bring content in and
// change nothing (an egress call is decided under the caller's labels,
// see trifecta's decideReader); and it cannot read through itself.
func (r *Request) validateQuarantine() error {
	q := r.Quarantine
	if q == nil {
		return nil
	}
	if !r.Restricted() {
		return usage.Errorf("quarantine needs a policy: the reader exists to keep labels off the session")
	}
	if q.Model == "" {
		return usage.Errorf("quarantine has no model")
	}
	switch agentloop.Effort(q.Effort) {
	case "", agentloop.EffortLow, agentloop.EffortMedium, agentloop.EffortHigh, agentloop.EffortXHigh, agentloop.EffortMax:
	default:
		return usage.Errorf("unknown quarantine.effort %q", q.Effort)
	}
	if q.MaxOutputTokens < 0 {
		return usage.Errorf("quarantine.max_output_tokens cannot be negative")
	}
	if q.Retries != nil && (*q.Retries < 0 || *q.Retries > MaxQuarantineRetries) {
		return usage.Errorf("quarantine.retries must be from 0 to %d", MaxQuarantineRetries)
	}
	if len(q.Tools) == 0 {
		return usage.Errorf("quarantine.tools names no tool to read through")
	}
	if e := r.Policy.Tools[quarantine.ToolName]; e.Class != trifecta.Read {
		return usage.Errorf("policy.tools must give %s the class read", quarantine.ToolName)
	}
	given := map[string]bool{}
	for _, t := range r.Tools {
		given[t.Name] = true
	}
	seen := map[string]bool{}
	for _, name := range q.Tools {
		switch {
		case name == quarantine.ToolName:
			return usage.Errorf("quarantine.tools cannot name %s itself", quarantine.ToolName)
		case seen[name]:
			return usage.Errorf("quarantine.tools names %q twice", name)
		case !given[name]:
			return usage.Errorf("quarantine.tools names %q, which is not one of the tools", name)
		case r.Policy.Tools[name].Class != trifecta.Read && r.Policy.Tools[name].Class != trifecta.Egress:
			return usage.Errorf("quarantine.tools names %q, whose class is %q; the reader reads only through tools of class read or egress", name, r.Policy.Tools[name].Class)
		}
		seen[name] = true
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
			{"session_dir", r.SessionDir != ""}, {"tool_kinds", r.ToolKinds != nil}, {"title", r.Title != ""},
			{"new_sessions_only", r.NewSessionsOnly != nil},
		} {
			if f.set {
				return usage.Errorf("%s is only for agentloop acp", f.name)
			}
		}
		if r.Resolve != nil && !r.Restricted() {
			return usage.Errorf("resolve needs a policy: a person's allow takes the call's labels from it")
		}
		seen := map[string]bool{}
		for _, res := range r.Resolve {
			if res.CallID == "" {
				return usage.Errorf("a resolve entry has no call_id")
			}
			if seen[res.CallID] {
				return usage.Errorf("resolve names the call %q twice", res.CallID)
			}
			seen[res.CallID] = true
			if res.Verdict != agentloop.Allow && res.Verdict != agentloop.Deny {
				return usage.Errorf("resolve gives the call %q the verdict %q (want allow or deny)", res.CallID, res.Verdict)
			}
			if res.InputSHA256 == "" {
				return usage.Errorf("resolve gives the call %q no input_sha256", res.CallID)
			}
		}
		return nil
	}
	if r.SessionDir == "" {
		return usage.Errorf("request has no session_dir")
	}
	for _, f := range []field{
		{"session", r.Session != ""}, {"prompt", r.Prompt != ""}, {"timeout_seconds", r.TimeoutSeconds != 0},
		{"resolve", r.Resolve != nil},
	} {
		if f.set {
			return usage.Errorf("%s is not used by agentloop acp", f.name)
		}
	}
	if r.NewSessionsOnly != nil && !*r.NewSessionsOnly {
		return usage.Errorf(`"new_sessions_only": false is refused; leave the field out to allow reopening sessions`)
	}
	tools := map[string]bool{}
	for _, name := range r.toolNames() {
		tools[name] = true
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
	out, err := secretCommand(ctx, "api_key_command", argv, timeout)
	if err != nil {
		return "", err
	}
	key := strings.TrimSpace(string(out))
	if key == "" {
		return "", &KeyError{msg: "The api_key_command printed no key."}
	}
	return key, nil
}

// baseEnv is PATH and HOME from this process, and nothing else.
func baseEnv() []string {
	env := []string{}
	for _, k := range []string{"PATH", "HOME"} {
		if v, ok := os.LookupEnv(k); ok {
			env = append(env, k+"="+v)
		}
	}
	return env
}

// secretCommand runs a command whose stdout is a secret, as FetchKey
// describes, and returns its stdout. Its errors name the command by the
// request field it came from and never hold its stdout.
func secretCommand(ctx context.Context, name string, argv []string, timeout time.Duration) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	cmd.Env = baseEnv()
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
			return nil, &KeyError{msg: fmt.Sprintf("The %s did not finish within %d seconds.", name, int(timeout.Seconds()))}
		}
		return nil, &KeyError{msg: fmt.Sprintf("The %s was interrupted.", name), Canceled: true}
	}
	if err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			what := fmt.Sprintf("exited %d", ee.ExitCode())
			if ee.ExitCode() < 0 {
				what = "was killed (" + ee.String() + ")"
			}
			msg := "The " + name + " " + what
			if tail := stderrTail(stderr.Bytes()); tail != "" {
				msg += ": " + tail
			}
			return nil, &KeyError{msg: msg + "."}
		}
		return nil, &KeyError{msg: fmt.Sprintf("The %s could not run: %v.", name, err)}
	}
	return stdout.Bytes(), nil
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
	return r.newModelFor(key, r.Model)
}

// newModelFor is the provider's model for one model id, on the request's
// base_url and key.
func (r *Request) newModelFor(key, model string) agentloop.Model {
	if r.Provider == "openai" {
		return openai.New(openai.Config{APIKey: key, BaseURL: r.BaseURL})
	}
	fallback := refusalFallbackModels[model]
	if r.RefusalFallback != nil {
		fallback = *r.RefusalFallback
	}
	return anthropic.New(anthropic.Config{APIKey: key, BaseURL: r.BaseURL, RefusalFallback: fallback})
}

// Agent builds the agent, with the MCP tools and pins of m, which may be
// nil when the request has no MCP servers. The key goes to the model and
// nowhere else: the tools get exactly tool_env. It fails only when the
// quarantined reader cannot be built, which Load has already ruled out
// for a request it accepted.
func (r *Request) Agent(key string, m *MCP) (*agentloop.Agent, error) {
	env := make([]string, 0, len(r.ToolEnv))
	for k, v := range r.ToolEnv {
		env = append(env, k+"="+v)
	}
	sort.Strings(env)
	var tools []agentloop.Tool
	for _, t := range r.commandTools() {
		argv := append(append([]string{}, r.ToolCommand...), t.Name)
		tools = append(tools, &command.Tool{
			Def: t.ToolSpec, Argv: argv, Env: env, Dir: r.ToolDir,
			Envelope: r.ToolProtocol == ToolProtocolEnvelope, KnownLabel: trifecta.KnownLabel,
		})
	}
	if m != nil {
		tools = append(tools, m.Tools...)
	}
	retry := agentloop.DefaultRetry
	if r.Retry != nil {
		retry = agentloop.RetryPolicy{
			Retries: r.Retry.Retries,
			Backoff: Seconds(r.Retry.BackoffSeconds),
			MaxWait: Seconds(r.Retry.MaxWaitSeconds),
		}
	}
	var readers map[string][]string
	reader, err := r.reader(key, tools, retry)
	if err != nil {
		return nil, err
	}
	if reader != nil {
		tools = append(tools, reader)
		readers = map[string][]string{quarantine.ToolName: slices.Clone(r.Quarantine.Tools)}
	}
	a := &agentloop.Agent{
		Model:       r.newModel(key),
		ModelID:     r.Model,
		System:      r.System,
		Tools:       tools,
		Effort:      agentloop.Effort(r.Effort),
		Limits:      r.Limits,
		ToolTimeout: Seconds(r.ToolTimeoutSeconds),
		// Where displays are kept, so the model can be told the ref.
		DisplayRefPrefix: r.DisplayRefPrefix,
	}
	if a.ToolTimeout == 0 && len(r.MCPServers) > 0 {
		a.ToolTimeout = DefaultMCPToolTimeout
	}
	if r.Restricted() {
		a.Labels = slices.Clone(r.Policy.Labels)
		a.Forbid = slices.Clone(r.Policy.ForbidLabels)
		a.Policy = trifecta.Policy{Tools: maps.Clone(r.Policy.Tools), Forbid: slices.Clone(r.Policy.ForbidLabels), Readers: readers}
	}
	if r.Retry != nil {
		a.Retry = &retry
	}
	if m != nil && len(m.Pins) > 0 {
		a.Pins = maps.Clone(m.Pins)
	}
	return a, nil
}

// reader is the read_quarantined tool, reading through the named tools
// among tools, or nil when the request has no quarantine block.
func (r *Request) reader(key string, tools []agentloop.Tool, retry agentloop.RetryPolicy) (*quarantine.Reader, error) {
	q := r.Quarantine
	if q == nil {
		return nil, nil
	}
	var inner []agentloop.Tool
	labels := map[string][]string{}
	for _, t := range tools {
		if name := t.Spec().Name; slices.Contains(q.Tools, name) {
			inner = append(inner, t)
			labels[name] = slices.Clone(r.Policy.Tools[name].Labels)
		}
	}
	retries := DefaultQuarantineRetries
	if q.Retries != nil {
		retries = *q.Retries
	}
	return quarantine.NewReader(quarantine.Reader{
		Model:           r.newModelFor(key, q.Model),
		ModelID:         q.Model,
		Effort:          agentloop.Effort(q.Effort),
		MaxOutputTokens: q.MaxOutputTokens,
		Retries:         retries,
		Retry:           retry,
		Inner:           inner,
		Labels:          labels,
	})
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
