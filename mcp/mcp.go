// Package mcp gives an agent the tools of one MCP server, run as a child
// process and spoken to over stdio, as agentloop.Tool values.
//
// The caller's configuration is the authority. It names the tools the
// agent may have (an allowlist) and gives each its trifecta class, labels
// and always_hold. A tool the server lists that is not on the allowlist is
// ignored, and an allowlisted tool the server lacks is an error. The
// server's tool annotations are only linted against the configured class
// (Warnings); they never decide anything.
//
// Each allowed tool's definition (name, description, input and output
// schema) is pinned by its SHA-256 (Digest) when the client connects. A
// definition that differs from the digest the caller gave, or on Reconnect
// from the one pinned, is an error.
//
// The client declares no capabilities and handles no requests from the
// server: no sampling, elicitation or roots. A result that asks for input
// fails the call, and a request from the server breaks the connection
// until Reconnect.
//
// A result is text. Image, audio, resource_link and embedded resource
// content fail the call unless the tool allows the kind, and even then the
// model sees only a placeholder. A tool with an outputSchema shows the
// model only its structuredContent, validated against that schema and
// serialised again. Whenever a result cannot be used the model is told
// only "The tool <name> failed.", and the reason goes to Result.Log.
//
// The server process gets exactly the argv, environment and directory it
// is configured with, and runs in a process group of its own that is
// killed when the client closes.
package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/modelcontextprotocol/go-sdk/jsonrpc"
	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/santhosh-tekuri/jsonschema/v6"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/trifecta"
)

// The _meta keys agentloop uses, under its own reverse-DNS prefix.
const (
	// LabelsKey holds the session's labels on a request (when the server
	// is sent them) and the labels a result adds on a response.
	LabelsKey = "com.kirahowe.agentloop/labels"
	// CallIDKey holds agentloop's id for the call on a request.
	CallIDKey = "com.kirahowe.agentloop/call-id"
)

// DefaultMaxMessageBytes caps one message from the server when
// Server.MaxMessageBytes is zero.
const DefaultMaxMessageBytes = 1 << 20

// The kinds of non-text content a tool can be allowed to return.
const (
	Image        = "image"
	Audio        = "audio"
	ResourceLink = "resource_link"
	Resource     = "resource"
)

// A Server is one MCP server and the tools an agent may use from it.
type Server struct {
	// Name is the operator's name for the server, used in errors.
	Name string
	// Argv is the program and its arguments.
	Argv []string
	// Env is the whole environment, as K=V. Nil means empty: nothing is
	// inherited.
	Env []string
	// Dir is the working directory. Empty means the current one.
	Dir string
	// Stderr receives what the server writes to stderr, for the
	// operator. Nil discards it.
	Stderr io.Writer
	// SendLabels sends the session's labels and the call id in each
	// call's _meta. Off, the server learns neither.
	SendLabels bool
	// Tools is the allowlist: each tool the agent may call, by the name
	// the server gives it.
	Tools map[string]ToolConfig
	// MaxMessageBytes caps one JSON-RPC message from the server. Zero
	// means DefaultMaxMessageBytes.
	MaxMessageBytes int
}

// A ToolConfig is one allowed tool.
type ToolConfig struct {
	// Policy is the tool's class, labels and always_hold.
	Policy trifecta.Tool
	// Digest pins the tool's definition. Empty accepts whatever the
	// server lists, and Client.Digests says what that was.
	Digest string
	// AllowContent lists the non-text content kinds a result may carry
	// (Image, Audio, ResourceLink, Resource). The model sees a
	// placeholder for each, never the content.
	AllowContent []string
}

// A DriftError says an allowed tool's definition is not the pinned one.
type DriftError struct {
	Server, Tool, Pinned, Got string
}

func (e *DriftError) Error() string {
	return fmt.Sprintf("mcp server %s: the definition of %s changed (pinned %s, got %s)", e.Server, e.Tool, e.Pinned, e.Got)
}

// Failed is all the model is told of a call whose result cannot be used.
func Failed(name string) string { return fmt.Sprintf("The tool %s failed.", name) }

// NotRun is what the model is told of a call that was never sent,
// because the server stopped or broke the protocol earlier.
func NotRun(name string) string {
	return fmt.Sprintf("The tool %s was not run: its MCP server is not running and must be reconnected.", name)
}

// Stopped is what the model is told when the server died during a call.
func Stopped(name string) string {
	return fmt.Sprintf("The MCP server stopped while the call to %s was running, so it may or may not have taken effect. It was not run again.", name)
}

// A Client is a connection to one server and the tools it gives.
type Client struct {
	srv      Server
	pins     map[string]string
	warnings []string
	tools    []*tool

	// lock guards conn. It is a channel so that a call can stop waiting
	// for it when its context ends.
	lock chan struct{}
	conn *conn
}

// Connect starts the server, lists its tools and checks them against the
// allowlist and the digests given.
func Connect(ctx context.Context, srv Server) (*Client, error) {
	if err := checkServer(srv); err != nil {
		return nil, err
	}
	cn, listed, err := dial(ctx, srv)
	if err != nil {
		return nil, err
	}
	c := &Client{srv: srv, pins: map[string]string{}, conn: cn, lock: make(chan struct{}, 1)}
	if err := c.adopt(listed); err != nil {
		cn.close()
		return nil, err
	}
	return c, nil
}

func checkServer(srv Server) error {
	if len(srv.Argv) == 0 || srv.Argv[0] == "" {
		return fmt.Errorf("mcp server %s has no program", srv.Name)
	}
	for _, a := range srv.Argv {
		if strings.ContainsRune(a, 0) {
			return fmt.Errorf("mcp server %s: an argument holds a NUL", srv.Name)
		}
	}
	if len(srv.Tools) == 0 {
		return fmt.Errorf("mcp server %s allows no tools", srv.Name)
	}
	for name, tc := range srv.Tools {
		if err := trifecta.CheckLabels(tc.Policy.Labels); err != nil {
			return fmt.Errorf("mcp server %s, tool %s: %w", srv.Name, name, err)
		}
		if !trifecta.KnownClass(tc.Policy.Class) {
			return fmt.Errorf("mcp server %s, tool %s: unknown class %q", srv.Name, name, tc.Policy.Class)
		}
		for _, k := range tc.AllowContent {
			if !slices.Contains([]string{Image, Audio, ResourceLink, Resource}, k) {
				return fmt.Errorf("mcp server %s, tool %s: unknown content kind %q", srv.Name, name, k)
			}
		}
	}
	return nil
}

// adopt builds the client's tools from the first listing.
func (c *Client) adopt(listed map[string]*sdk.Tool) error {
	names := make([]string, 0, len(c.srv.Tools))
	for name := range c.srv.Tools {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		cfg := c.srv.Tools[name]
		def, ok := listed[name]
		if !ok {
			return fmt.Errorf("mcp server %s does not have the tool %s", c.srv.Name, name)
		}
		d, err := Digest(def)
		if err != nil {
			return fmt.Errorf("mcp server %s, tool %s: %w", c.srv.Name, name, err)
		}
		if cfg.Digest != "" && cfg.Digest != d {
			return &DriftError{Server: c.srv.Name, Tool: name, Pinned: cfg.Digest, Got: d}
		}
		t, err := newTool(c, name, cfg, def)
		if err != nil {
			return fmt.Errorf("mcp server %s, tool %s: %w", c.srv.Name, name, err)
		}
		c.pins[name] = d
		c.tools = append(c.tools, t)
		c.warnings = append(c.warnings, Lint(name, cfg.Policy, def.Annotations)...)
	}
	return nil
}

// Tools returns the allowed tools, by name.
func (c *Client) Tools() []agentloop.Tool {
	out := make([]agentloop.Tool, len(c.tools))
	for i, t := range c.tools {
		out[i] = t
	}
	return out
}

// Digests returns the pinned digest of each allowed tool, by name.
func (c *Client) Digests() map[string]string {
	out := make(map[string]string, len(c.pins))
	for k, v := range c.pins {
		out[k] = v
	}
	return out
}

// Warnings returns where the server's annotations contradict the
// configured classes and labels, for the operator.
func (c *Client) Warnings() []string { return slices.Clone(c.warnings) }

// Reconnect stops the server, if it is running, and starts it again. Every
// allowed tool must have the definition pinned when the client first
// connected; to accept a changed one, Connect again with its new digest.
// Nothing reconnects on its own.
func (c *Client) Reconnect(ctx context.Context) error {
	if err := c.acquire(ctx); err != nil {
		return err
	}
	defer c.release()
	if c.conn != nil {
		c.conn.close()
		c.conn = nil
	}
	cn, listed, err := dial(ctx, c.srv)
	if err != nil {
		return err
	}
	for _, t := range c.tools {
		def, ok := listed[t.name]
		if !ok {
			cn.close()
			return fmt.Errorf("mcp server %s does not have the tool %s", c.srv.Name, t.name)
		}
		d, err := Digest(def)
		if err != nil {
			cn.close()
			return err
		}
		if d != c.pins[t.name] {
			cn.close()
			return &DriftError{Server: c.srv.Name, Tool: t.name, Pinned: c.pins[t.name], Got: d}
		}
	}
	c.conn = cn
	return nil
}

// Close stops the server and everything in its process group.
func (c *Client) Close() error {
	c.acquire(context.Background())
	defer c.release()
	if c.conn != nil {
		c.conn.close()
		c.conn = nil
	}
	return nil
}

func (c *Client) acquire(ctx context.Context) error {
	select {
	case c.lock <- struct{}{}:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (c *Client) release() { <-c.lock }

// current is the connection, waiting for a Reconnect or Close in progress
// no longer than ctx allows.
func (c *Client) current(ctx context.Context) (*conn, error) {
	if err := c.acquire(ctx); err != nil {
		return nil, err
	}
	defer c.release()
	return c.conn, nil
}

// conn is one running server and the session with it.
type conn struct {
	cs *sdk.ClientSession
	p  *proc

	mu         sync.Mutex
	broken     string // why no further call can be made, or ""
	violations []string
}

// violation records a protocol violation and breaks the connection, so
// that no further call is sent on it.
func (cn *conn) violation(why string) {
	cn.mu.Lock()
	if n := len(cn.violations); n == 0 || cn.violations[n-1] != why {
		cn.violations = append(cn.violations, why)
	}
	if cn.broken == "" {
		cn.broken = why
	}
	cn.mu.Unlock()
}

func (cn *conn) violationsSince(n int) []string {
	cn.mu.Lock()
	defer cn.mu.Unlock()
	return slices.Clone(cn.violations[n:])
}

func (cn *conn) violationCount() int {
	cn.mu.Lock()
	defer cn.mu.Unlock()
	return len(cn.violations)
}

func (cn *conn) breakWith(why string) {
	cn.mu.Lock()
	if cn.broken == "" {
		cn.broken = why
	}
	cn.mu.Unlock()
}

func (cn *conn) brokenReason() string {
	cn.mu.Lock()
	defer cn.mu.Unlock()
	if cn.broken != "" {
		return cn.broken
	}
	if cn.p.hasExited() {
		return "the server exited"
	}
	return ""
}

func (cn *conn) close() {
	done := make(chan struct{})
	go func() {
		cn.cs.Close()
		close(done)
	}()
	cn.p.stop()
	select {
	case <-done:
	case <-time.After(killGrace):
	}
}

// dial starts the server, connects and lists its tools by name.
func dial(ctx context.Context, srv Server) (*conn, map[string]*sdk.Tool, error) {
	p, r, w, err := start(srv)
	if err != nil {
		return nil, nil, fmt.Errorf("mcp server %s: %w", srv.Name, err)
	}
	cn := &conn{p: p}
	max := srv.MaxMessageBytes
	if max <= 0 {
		max = DefaultMaxMessageBytes
	}
	client := sdk.NewClient(&sdk.Implementation{Name: "agentloop", Version: "1"}, &sdk.ClientOptions{
		// A nil value would advertise roots. No handler is set, so no
		// capability is added to this.
		Capabilities: &sdk.ClientCapabilities{},
		// The SDK would otherwise answer input requests itself and retry.
		MultiRoundTrip: &sdk.MultiRoundTripOptions{Disabled: true},
	})
	client.AddReceivingMiddleware(refuseRequests(cn))
	cs, err := client.Connect(ctx, &sdk.IOTransport{Reader: r, Writer: w, MaxLineLength: max}, nil)
	if err != nil {
		p.stop()
		return nil, nil, fmt.Errorf("mcp server %s: connect: %w", srv.Name, err)
	}
	cn.cs = cs
	listed := map[string]*sdk.Tool{}
	params := &sdk.ListToolsParams{}
	seen := map[string]bool{"": true}
	for pages := 1; ; pages++ {
		if pages > maxPages {
			cn.close()
			return nil, nil, fmt.Errorf("mcp server %s lists its tools in more than %d pages", srv.Name, maxPages)
		}
		res, err := cs.ListTools(ctx, params)
		if err != nil {
			cn.close()
			return nil, nil, fmt.Errorf("mcp server %s: list tools: %w", srv.Name, err)
		}
		for _, t := range res.Tools {
			if _, dup := listed[t.Name]; dup {
				cn.close()
				return nil, nil, fmt.Errorf("mcp server %s lists the tool %s twice", srv.Name, t.Name)
			}
			listed[t.Name] = t
			if len(listed) > maxTools {
				cn.close()
				return nil, nil, fmt.Errorf("mcp server %s lists more than %d tools", srv.Name, maxTools)
			}
		}
		if res.NextCursor == "" {
			break
		}
		if seen[res.NextCursor] {
			cn.close()
			return nil, nil, fmt.Errorf("mcp server %s: list tools repeats a cursor", srv.Name)
		}
		seen[res.NextCursor] = true
		params = &sdk.ListToolsParams{Cursor: res.NextCursor}
	}
	return cn, listed, nil
}

// The most pages and tools a listing may have.
const (
	maxPages = 100
	maxTools = 1000
)

// refuseRequests answers any request from the server, other than a ping,
// with an error and breaks the connection (see conn.violation).
// Notifications
// pass, and with no handlers set the SDK drops them.
func refuseRequests(cn *conn) sdk.Middleware {
	return func(next sdk.MethodHandler) sdk.MethodHandler {
		return func(ctx context.Context, method string, req sdk.Request) (sdk.Result, error) {
			if method == "ping" || strings.HasPrefix(method, "notifications/") {
				return next(ctx, method, req)
			}
			cn.violation(fmt.Sprintf("the server sent a %s request, which agentloop does not take", method))
			return nil, fmt.Errorf("agentloop takes no %s requests", method)
		}
	}
}

// tool is one allowed tool of a client.
type tool struct {
	c      *Client
	name   string
	cfg    ToolConfig
	spec   agentloop.ToolSpec
	output *jsonschema.Schema
}

var _ agentloop.CallTool = (*tool)(nil)

func newTool(c *Client, name string, cfg ToolConfig, def *sdk.Tool) (*tool, error) {
	params, err := canonicalValue(def.InputSchema)
	if err != nil {
		return nil, fmt.Errorf("inputSchema: %w", err)
	}
	// The agent compiles it again when it starts; a bad one fails here
	// first, with the server named.
	if _, err := compileSchema("mcp-input:"+name, params); err != nil {
		return nil, fmt.Errorf("inputSchema: %w", err)
	}
	t := &tool{c: c, name: name, cfg: cfg, spec: agentloop.ToolSpec{Name: name, Description: def.Description, Parameters: params}}
	if def.OutputSchema != nil {
		raw, err := canonicalValue(def.OutputSchema)
		if err != nil {
			return nil, fmt.Errorf("outputSchema: %w", err)
		}
		if t.output, err = compileSchema("mcp-output:"+name, raw); err != nil {
			return nil, fmt.Errorf("outputSchema: %w", err)
		}
	}
	return t, nil
}

func (t *tool) Spec() agentloop.ToolSpec { return t.spec }

// Call runs the tool with no call id and no labels.
func (t *tool) Call(ctx context.Context, input json.RawMessage) (agentloop.Result, error) {
	return t.CallWith(ctx, agentloop.Call{Input: input})
}

// CallWith makes one tools/call request. It never retries and never
// reconnects.
func (t *tool) CallWith(ctx context.Context, call agentloop.Call) (agentloop.Result, error) {
	failed := func(why string) (agentloop.Result, error) {
		return agentloop.Result{Content: Failed(t.name), IsError: true, Log: why}, nil
	}
	if err := ctx.Err(); err != nil {
		return agentloop.Result{}, err
	}
	notRun := func(why string) (agentloop.Result, error) {
		return agentloop.Result{Content: NotRun(t.name), IsError: true, Log: why + "; only Reconnect starts it again"}, nil
	}
	cn, err := t.c.current(ctx)
	if err != nil {
		return agentloop.Result{Log: "stopped while waiting for a Reconnect or Close: " + err.Error()}, err
	}
	if cn == nil {
		return notRun("the server is not connected")
	}
	if why := cn.brokenReason(); why != "" {
		return notRun(why)
	}
	input := call.Input
	if len(strings.TrimSpace(string(input))) == 0 {
		input = json.RawMessage(`{}`)
	}
	params := &sdk.CallToolParams{Name: t.name, Arguments: input}
	if t.c.srv.SendLabels {
		params.Meta = RequestMeta(call)
	}
	before := cn.violationCount()
	// The call ends when the server exits, even if something it left
	// behind holds its stdout open. A response written just before the
	// exit gets exitWait to arrive.
	callCtx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	go func() {
		select {
		case <-cn.p.exited:
			select {
			case <-time.After(exitWait):
				cancel(errServerExited)
			case <-callCtx.Done():
			}
		case <-callCtx.Done():
		}
	}()
	res, err := cn.cs.CallTool(callCtx, params)
	if ctxErr := ctx.Err(); ctxErr != nil {
		// The agent says what a timeout or a cancel means.
		return agentloop.Result{Log: "stopped: " + ctxErr.Error()}, ctxErr
	}
	if v := cn.violationsSince(before); len(v) > 0 {
		return failed(strings.Join(v, "\n"))
	}
	if err != nil {
		var wire *jsonrpc.Error
		if errors.As(err, &wire) {
			return failed("the server answered with an error: " + err.Error())
		}
		// Anything else means the connection is gone (an EOF, the
		// server's exit, an unreadable message), and with it any
		// knowledge of what the call did.
		if cause := context.Cause(callCtx); cause != nil {
			err = fmt.Errorf("%w: %v", cause, err)
		}
		cn.breakWith("the connection was lost during a call: " + err.Error())
		return agentloop.Result{Content: Stopped(t.name), IsError: true, Log: err.Error()}, nil
	}
	return ToResult(t.name, t.cfg, t.output, res), nil
}

// exitWait is how long a call waits for its response after the server
// has exited.
const exitWait = 200 * time.Millisecond

var errServerExited = errors.New("the server exited")

// RequestMeta is the _meta sent with a call when the server is sent
// labels. The labels are sorted and never null.
func RequestMeta(call agentloop.Call) sdk.Meta {
	labels := slices.Clone(call.Labels)
	if labels == nil {
		labels = []string{}
	}
	sort.Strings(labels)
	return sdk.Meta{LabelsKey: labels, CallIDKey: call.ID}
}
