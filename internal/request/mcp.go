package request

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/internal/usage"
	"github.com/kirahowe/agentloop/mcp"
)

// MCPStartTimeout bounds starting one MCP server: its credentials
// command, then connecting and listing its tools.
const MCPStartTimeout = 30 * time.Second

// DefaultMCPToolTimeout is the tool timeout of a request with MCP servers
// that gives none: a call to a server that never answers must end.
const DefaultMCPToolTimeout = 120 * time.Second

// A ToolsError says why the MCP tools could not be set up: a server did
// not start or lacks a tool, or a tool's definition is not the one the
// request or the session pinned. Message is for the operator and never
// holds a secret.
type ToolsError struct {
	Server string `json:"server,omitempty"`
	// Tool is the tool's name in the request.
	Tool    string `json:"tool,omitempty"`
	Pinned  string `json:"pinned,omitempty"`
	Got     string `json:"got,omitempty"`
	Message string `json:"message"`
}

func (e *ToolsError) Error() string { return e.Message }

// PinsError is the ToolsError for a session that pinned another
// definition of a tool.
func (r *Request) PinsError(pe *agentloop.PinError) *ToolsError {
	e := &ToolsError{Tool: pe.Tool, Pinned: pe.Pinned, Got: pe.Got,
		Message: fmt.Sprintf("The definition of the tool %s is not the one this session pinned (pinned %s, now %s). The session cannot go on with it: use a new session.", pe.Tool, pe.Pinned, pe.Got)}
	for _, t := range r.Tools {
		if t.Name == pe.Tool {
			e.Server = t.Server
		}
	}
	return e
}

// digestPattern is a SHA-256 in lowercase hex, as Digest gives it.
var digestPattern = regexp.MustCompile(`^[0-9a-f]{64}$`)

// contentKinds are the values allow_content may hold.
var contentKinds = []string{mcp.Image, mcp.Audio, mcp.ResourceLink, mcp.Resource}

// validateMCP checks mcp_servers and the MCP tools. A server's tools
// take their class and labels from the policy, so they need a real one.
func (r *Request) validateMCP() error {
	used := map[string]bool{}
	remote := map[[2]string]string{}
	for _, t := range r.Tools {
		switch t.Protocol {
		case "":
			if t.Server != "" || t.Tool != "" || t.AllowContent != nil || t.Digest != "" {
				return usage.Errorf("tool %q: server, tool, allow_content and digest are only for %q tools", t.Name, ToolProtocolMCPStdio)
			}
			continue
		case ToolProtocolMCPStdio:
		default:
			return usage.Errorf("tool %q: unknown protocol %q (want %q or none)", t.Name, t.Protocol, ToolProtocolMCPStdio)
		}
		if t.Description != "" || t.Parameters != nil {
			return usage.Errorf("tool %q: an MCP tool's description and parameters come from its server; leave them out", t.Name)
		}
		if _, ok := r.MCPServers[t.Server]; !ok {
			return usage.Errorf("tool %q names the MCP server %q, which mcp_servers does not have", t.Name, t.Server)
		}
		if t.Tool == "" || strings.ContainsRune(t.Tool, 0) {
			return usage.Errorf("tool %q: give the server's name for it as tool", t.Name)
		}
		if t.Digest != "" && !digestPattern.MatchString(t.Digest) {
			return usage.Errorf("tool %q: digest is not a SHA-256 in lowercase hex", t.Name)
		}
		key := [2]string{t.Server, t.Tool}
		if other, dup := remote[key]; dup {
			return usage.Errorf("tools %q and %q are both the tool %q of the MCP server %q", other, t.Name, t.Tool, t.Server)
		}
		remote[key] = t.Name
		used[t.Server] = true
		seen := map[string]bool{}
		for _, k := range t.AllowContent {
			if !slices.Contains(contentKinds, k) {
				return usage.Errorf("tool %q: allow_content has %q (want some of %s)", t.Name, k, strings.Join(contentKinds, ", "))
			}
			if seen[k] {
				return usage.Errorf("tool %q: allow_content names %q twice", t.Name, k)
			}
			seen[k] = true
		}
	}
	if len(r.MCPServers) > 0 && !r.Restricted() {
		return usage.Errorf("mcp_servers need a policy: an MCP tool's class and labels come only from it")
	}
	for _, name := range slices.Sorted(maps.Keys(r.MCPServers)) {
		s := r.MCPServers[name]
		if !toolName.MatchString(name) {
			return usage.Errorf("MCP server name %q is not 1 to 64 letters, digits, - or _", name)
		}
		if !used[name] {
			return usage.Errorf("the MCP server %q gives none of the tools", name)
		}
		if err := checkArgv(s.Command, "mcp_servers."+name+".command"); err != nil {
			return err
		}
		if s.Command == nil {
			return usage.Errorf("the MCP server %q has no command", name)
		}
		holds := false
		for i, a := range s.Command {
			if strings.Contains(a, CredentialsPlaceholder) {
				if i == 0 {
					return usage.Errorf("mcp_servers.%s.command cannot run %s", name, CredentialsPlaceholder)
				}
				holds = true
			}
		}
		if s.CredentialsCommand == nil {
			if holds {
				return usage.Errorf("mcp_servers.%s.command has %s but there is no credentials_command", name, CredentialsPlaceholder)
			}
			continue
		}
		if err := checkArgv(s.CredentialsCommand, "mcp_servers."+name+".credentials_command"); err != nil {
			return err
		}
		if !holds {
			return usage.Errorf("mcp_servers.%s has a credentials_command, but no argument of its command holds %s", name, CredentialsPlaceholder)
		}
	}
	return nil
}

// checkArgv refuses an argv that is empty (but present), starts with an
// empty program name, or holds a NUL.
func checkArgv(argv []string, field string) error {
	if argv == nil {
		return nil
	}
	if len(argv) == 0 || argv[0] == "" {
		return usage.Errorf("%s has no program", field)
	}
	for _, a := range argv {
		if strings.ContainsRune(a, 0) {
			return usage.Errorf("%s cannot contain a NUL character", field)
		}
	}
	return nil
}

// stubMCP stands in for the MCP tools when nothing is started: a tool by
// each alias that takes any object and never runs.
func (r *Request) stubMCP() *MCP {
	m := &MCP{}
	for _, t := range r.Tools {
		if !t.MCP() {
			continue
		}
		m.Tools = append(m.Tools, agentloop.Func(
			agentloop.ToolSpec{Name: t.Name, Parameters: json.RawMessage(`{"type":"object"}`)},
			func(context.Context, json.RawMessage) (agentloop.Result, error) {
				return agentloop.Result{}, errors.New("not started")
			}))
	}
	return m
}

// MCP is the request's MCP servers, running.
type MCP struct {
	// Tools are the MCP tools, by alias.
	Tools []agentloop.Tool
	// Pins are their definitions' digests, by alias.
	Pins map[string]string

	clients []*mcp.Client
	dir     string // the credentials files' directory, or ""
}

// RemoveCredentials removes the credentials files, for an exit that
// cannot wait for Close.
func (m *MCP) RemoveCredentials() {
	if m != nil && m.dir != "" {
		os.RemoveAll(m.dir)
	}
}

// Close stops every server and removes the credentials files.
func (m *MCP) Close() {
	if m == nil {
		return
	}
	var wg sync.WaitGroup
	for _, c := range m.clients {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c.Close()
		}()
	}
	wg.Wait()
	if m.dir != "" {
		os.RemoveAll(m.dir)
	}
}

// StartMCP starts the request's MCP servers, in name order, and checks
// their tools. Each server gets PATH and HOME from this process and
// nothing else of its environment, runs in tool_dir, and writes its
// stderr to log. Every pin and every place a server's annotations
// disagree with the policy goes to log too. A server that dies is
// started again for the next call. It returns nil when the request has
// no servers.
func (r *Request) StartMCP(ctx context.Context, log io.Writer) (*MCP, error) {
	if len(r.MCPServers) == 0 {
		return nil, nil
	}
	log = &lockedWriter{w: log}
	SweepCredentials(os.TempDir())
	m := &MCP{Pins: map[string]string{}}
	ok := false
	defer func() {
		if !ok {
			m.Close()
		}
	}()
	for _, name := range slices.Sorted(maps.Keys(r.MCPServers)) {
		c, err := r.startServer(ctx, name, m, log)
		if err != nil {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			return nil, r.toolsError(name, err)
		}
		m.clients = append(m.clients, c)
		m.Tools = append(m.Tools, c.Tools()...)
		digests := c.Digests()
		for _, alias := range slices.Sorted(maps.Keys(digests)) {
			m.Pins[alias] = digests[alias]
			fmt.Fprintf(log, "agentloop: mcp server %s: tool %s pinned at sha256 %s\n", name, alias, digests[alias])
		}
		for _, w := range c.Warnings() {
			fmt.Fprintf(log, "agentloop: mcp server %s: warning: %s (the policy decides; annotations are not used)\n", name, w)
		}
	}
	ok = true
	return m, nil
}

func (r *Request) startServer(ctx context.Context, name string, m *MCP, log io.Writer) (*mcp.Client, error) {
	ctx, cancel := context.WithTimeout(ctx, MCPStartTimeout)
	defer cancel()
	s := r.MCPServers[name]
	argv := slices.Clone(s.Command)
	if s.CredentialsCommand != nil {
		path, err := m.writeCredentials(ctx, name, s.CredentialsCommand)
		if err != nil {
			return nil, err
		}
		for i := range argv {
			argv[i] = strings.ReplaceAll(argv[i], CredentialsPlaceholder, path)
		}
	}
	tools := map[string]mcp.ToolConfig{}
	for _, t := range r.Tools {
		if t.MCP() && t.Server == name {
			tools[t.Tool] = mcp.ToolConfig{Alias: t.Name, Policy: r.Policy.Tools[t.Name], AllowContent: slices.Clone(t.AllowContent), Digest: t.Digest}
		}
	}
	c, err := mcp.Connect(ctx, mcp.Server{
		Name: name, Argv: argv, Env: baseEnv(), Dir: r.ToolDir, Stderr: log,
		SendLabels: s.SendLabels, Restart: true, StartTimeout: MCPStartTimeout, Tools: tools,
	})
	if err != nil {
		return nil, err
	}
	return c, nil
}

// toolsError says why a server did not start.
func (r *Request) toolsError(server string, err error) *ToolsError {
	var drift *mcp.DriftError
	if errors.As(err, &drift) {
		alias := drift.Tool
		for _, t := range r.Tools {
			if t.MCP() && t.Server == server && t.Tool == drift.Tool {
				alias = t.Name
			}
		}
		return &ToolsError{Server: server, Tool: alias, Pinned: drift.Pinned, Got: drift.Got,
			Message: fmt.Sprintf("The MCP server %s gives a definition of the tool %s that is not the one the request pins (pinned %s, got %s).", server, alias, drift.Pinned, drift.Got)}
	}
	return &ToolsError{Server: server, Message: fmt.Sprintf("The MCP server %s did not start: %v", server, err)}
}

// writeCredentials runs a server's credentials command and writes what
// it printed to a file only this user can read, in a directory made for
// the run. It returns the file's path.
func (m *MCP) writeCredentials(ctx context.Context, name string, argv []string) (string, error) {
	out, err := secretCommand(ctx, "credentials_command of the MCP server "+name, argv, MCPStartTimeout)
	if err != nil {
		return "", err
	}
	if len(out) == 0 {
		return "", fmt.Errorf("the credentials_command of the MCP server %s printed nothing", name)
	}
	if m.dir == "" {
		// MkdirTemp makes the directory 0700. The pid file says whose it
		// is, so a run that could not remove it is swept up by a later one.
		if m.dir, err = os.MkdirTemp("", credentialsPrefix); err != nil {
			return "", err
		}
		if err := os.WriteFile(filepath.Join(m.dir, "pid"), fmt.Appendf(nil, "%d\n", os.Getpid()), 0o600); err != nil {
			return "", err
		}
	}
	path := filepath.Join(m.dir, name+".credentials")
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return "", err
	}
	if _, err := f.Write(out); err != nil {
		f.Close()
		return "", err
	}
	return path, f.Close()
}

// lockedWriter lets several servers' stderr copies share one writer.
type lockedWriter struct {
	mu sync.Mutex
	w  io.Writer
}

func (l *lockedWriter) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.w.Write(p)
}

const credentialsPrefix = "agentloop-mcp-"

var (
	credentialsDir  = regexp.MustCompile(`^` + credentialsPrefix + `[0-9]+$`)
	credentialsFile = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}\.credentials$`)
	pidContent      = regexp.MustCompile(`^[1-9][0-9]{0,9}\n$`)
)

// SweepCredentials removes the credentials directories in root that a
// run which has since died left behind (killed, say, before it could
// remove its own). It removes a directory only when its name, its owner,
// its files and its pid file all have the shape agentloop gives them and
// no process has that pid. Anything else is left alone.
func SweepCredentials(root string) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return
	}
	for _, e := range entries {
		if !credentialsDir.MatchString(e.Name()) {
			continue
		}
		dir := filepath.Join(root, e.Name())
		if pid, ok := staleOwner(dir); ok && !alive(pid) {
			os.RemoveAll(dir)
		}
	}
}

// staleOwner is the pid in a credentials directory that has agentloop's
// shape, and whether it has it.
func staleOwner(dir string) (int, bool) {
	st, err := os.Lstat(dir)
	if err != nil || !st.IsDir() || !ownedByMe(st) {
		return 0, false
	}
	files, err := os.ReadDir(dir)
	if err != nil {
		return 0, false
	}
	pid := 0
	for _, f := range files {
		info, err := f.Info()
		if err != nil || !info.Mode().IsRegular() || !ownedByMe(info) {
			return 0, false
		}
		switch {
		case f.Name() == "pid":
			b, err := os.ReadFile(filepath.Join(dir, "pid"))
			if err != nil || !pidContent.Match(b) {
				return 0, false
			}
			pid, err = strconv.Atoi(strings.TrimSpace(string(b)))
			if err != nil {
				return 0, false
			}
		case credentialsFile.MatchString(f.Name()):
		default:
			return 0, false
		}
	}
	return pid, pid > 0
}

func ownedByMe(info os.FileInfo) bool {
	st, ok := info.Sys().(*syscall.Stat_t)
	return ok && int(st.Uid) == os.Getuid()
}

// alive reports whether a process has the pid. A process of another user
// counts as alive.
func alive(pid int) bool {
	return !errors.Is(syscall.Kill(pid, 0), syscall.ESRCH)
}
