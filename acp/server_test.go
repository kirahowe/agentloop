package acpagent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	acp "github.com/caelis-labs/acp-go-sdk"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/fake"
)

// The server tests drive a real Server through the SDK's client side over
// in-memory pipes, and record every line on the wire in order.

// wire records the lines written in one direction, in the order they were
// written.
type wire struct {
	mu    sync.Mutex
	dst   io.Writer
	buf   []byte
	lines [][]byte
}

func (w *wire) Write(p []byte) (int, error) {
	w.mu.Lock()
	w.buf = append(w.buf, p...)
	for {
		i := bytes.IndexByte(w.buf, '\n')
		if i < 0 {
			break
		}
		w.lines = append(w.lines, bytes.Clone(w.buf[:i]))
		w.buf = w.buf[i+1:]
	}
	w.mu.Unlock()
	return w.dst.Write(p)
}

// Close closes the pipe beneath, so the reader at the other end sees EOF.
func (w *wire) Close() error {
	if c, ok := w.dst.(io.Closer); ok {
		return c.Close()
	}
	return nil
}

func (w *wire) snapshot() [][]byte {
	w.mu.Lock()
	defer w.mu.Unlock()
	return append([][]byte(nil), w.lines...)
}

// message is one JSON-RPC line.
type message struct {
	ID     json.RawMessage `json:"id"`
	Method string          `json:"method"`
	Params json.RawMessage `json:"params"`
	Result json.RawMessage `json:"result"`
	Error  json.RawMessage `json:"error"`
}

// tb is what the harness needs of a test: a *testing.T or a *rapid.T.
type tb interface {
	Helper()
	Fatal(args ...any)
	Fatalf(format string, args ...any)
	Cleanup(func())
}

func parseLines(t tb, lines [][]byte) []message {
	t.Helper()
	out := make([]message, len(lines))
	for i, l := range lines {
		var raw map[string]json.RawMessage
		if err := json.Unmarshal(l, &raw); err != nil {
			t.Fatalf("line %d is not a JSON object: %q", i, l)
		}
		if string(raw["jsonrpc"]) != `"2.0"` {
			t.Fatalf("line %d is not JSON-RPC 2.0: %q", i, l)
		}
		json.Unmarshal(l, &out[i])
	}
	return out
}

// testClient is the client side: it keeps every update and answers any
// permission request with permit, or with cancelled when permit is nil.
type testClient struct {
	mu       sync.Mutex
	updates  []acp.SessionNotification
	onUpdate func(acp.SessionNotification)
	permit   func(context.Context, acp.RequestPermissionRequest) (acp.RequestPermissionResponse, error)
	asked    []acp.RequestPermissionRequest
}

func (c *testClient) RequestPermission(ctx context.Context, req acp.RequestPermissionRequest) (acp.RequestPermissionResponse, error) {
	c.mu.Lock()
	c.asked = append(c.asked, req)
	permit := c.permit
	c.mu.Unlock()
	if permit != nil {
		return permit(ctx, req)
	}
	return acp.RequestPermissionResponse{Outcome: acp.NewRequestPermissionOutcomeCancelled()}, nil
}

func (c *testClient) SessionUpdate(_ context.Context, n acp.SessionNotification) error {
	c.mu.Lock()
	c.updates = append(c.updates, n)
	f := c.onUpdate
	c.mu.Unlock()
	if f != nil {
		f(n)
	}
	return nil
}

type harness struct {
	t                 tb
	srv               *Server
	conn              *acp.AgentSideConnection
	client            *acp.ClientSideConnection
	cl                *testClient
	toAgent, toClient *wire
	dir               string
	log               *bytes.Buffer
	logMu             *sync.Mutex
}

type lockedWriter struct {
	mu *sync.Mutex
	w  io.Writer
}

func (l lockedWriter) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.w.Write(p)
}

// serve starts a server with the agent and a client connected to it, and
// initializes the connection.
func serve(t testing.TB, agent *agentloop.Agent, opts Options, notices bool) *harness {
	t.Helper()
	return serveIn(t, t.TempDir(), agent, opts, notices)
}

// serveIn is serve with the session directory given, for tests that are
// not a *testing.T. Everything is shut down at the test's cleanup.
func serveIn(t tb, dir string, agent *agentloop.Agent, opts Options, notices bool) *harness {
	t.Helper()
	h := &harness{t: t, dir: dir, log: &bytes.Buffer{}, logMu: &sync.Mutex{}, cl: &testClient{}}
	if opts.ContextWindow > 0 {
		// The window comes from the model's description, never from the
		// options, so a window the test asks for is the model's.
		a := *agent
		a.Model = describedModel{a.Model, agentloop.ModelInfo{ID: a.ModelID, ContextWindow: opts.ContextWindow}, nil}
		agent = &a
	}
	h.srv = NewServer(Config{Version: "test", SessionDir: h.dir, Agent: agent, Options: opts,
		Log: lockedWriter{h.logMu, h.log}})
	aR, aW := io.Pipe() // client to agent
	cR, cW := io.Pipe() // agent to client
	h.toAgent = &wire{dst: aW}
	h.toClient = &wire{dst: cW}
	conn, err := h.srv.Connect(h.toClient, aR)
	if err != nil {
		t.Fatal(err)
	}
	h.conn = conn
	h.client = acp.NewClientSideConnection(h.cl, h.toAgent, cR)
	t.Cleanup(func() {
		h.client.Close()
		h.toAgent.Close()
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		h.srv.Shutdown(ctx)
		<-conn.Done()
	})
	req := acp.InitializeRequest{ProtocolVersion: acp.ProtocolVersion(acp.WireProtocolVersion)}
	if notices {
		req.ClientCapabilities.Session = &acp.ClientSessionCapabilities{Notices: &acp.NoticeCapabilities{}}
	}
	resp, err := h.client.Initialize(h.ctx(), req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.ProtocolVersion != 1 || !resp.AgentCapabilities.LoadSession {
		t.Fatalf("initialize: %+v", resp)
	}
	return h
}

func (h *harness) ctx() context.Context {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	h.t.Cleanup(cancel)
	return ctx
}

func (h *harness) newSession() acp.SessionId {
	h.t.Helper()
	resp, err := h.client.NewSession(h.ctx(), acp.NewSessionRequest{Cwd: "/", McpServers: []acp.McpServer{}})
	if err != nil {
		h.t.Fatal(err)
	}
	return resp.SessionId
}

func (h *harness) prompt(id acp.SessionId, text string) (acp.PromptResponse, error) {
	return h.client.Prompt(h.ctx(), acp.PromptRequest{SessionId: id, Prompt: []acp.ContentBlock{acp.TextBlock(text)}})
}

func (h *harness) logged() string {
	h.logMu.Lock()
	defer h.logMu.Unlock()
	return h.log.String()
}

// turnOnWire finds the n-th session/prompt request for the session on the
// wire and returns the updates the agent sent for that session from the
// request to its response, the response, and the updates after it.
func (h *harness) turnOnWire(id acp.SessionId, n int) (during []json.RawMessage, resp message, after []json.RawMessage) {
	h.t.Helper()
	var reqID json.RawMessage
	seen := 0
	for _, m := range parseLines(h.t, h.toAgent.snapshot()) {
		if m.Method == "session/prompt" && strings.Contains(string(m.Params), `"`+string(id)+`"`) {
			if seen == n {
				reqID = m.ID
			}
			seen++
		}
	}
	if reqID == nil {
		h.t.Fatalf("no prompt %d on the wire", n)
	}
	answered := false
	for _, m := range parseLines(h.t, h.toClient.snapshot()) {
		switch {
		case m.Method == "session/update":
			var p struct {
				SessionID string          `json:"sessionId"`
				Update    json.RawMessage `json:"update"`
			}
			json.Unmarshal(m.Params, &p)
			if p.SessionID != string(id) {
				continue
			}
			if answered {
				after = append(after, p.Update)
			} else {
				during = append(during, p.Update)
			}
		case m.Method == "" && bytes.Equal(m.ID, reqID):
			resp, answered = m, true
		}
	}
	if !answered {
		h.t.Fatalf("prompt %s has no response on the wire", reqID)
	}
	return during, resp, after
}

func asJSON(t tb, ups []Update) []string {
	t.Helper()
	out := make([]string, len(ups))
	for i, u := range ups {
		b, err := json.Marshal(u)
		if err != nil {
			t.Fatal(err)
		}
		out[i] = string(b)
	}
	return out
}

func rawStrings(raws []json.RawMessage) []string {
	out := make([]string, len(raws))
	for i, r := range raws {
		var b bytes.Buffer
		json.Compact(&b, r)
		out[i] = b.String()
	}
	return out
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// goldenAgent is the agent of a golden case, with a clock that does not
// move and retries that do not wait. record, if set, gets every event.
func goldenAgent(c goldenCase, record func(agentloop.Event)) *agentloop.Agent {
	tool := c.tool
	if tool == nil {
		tool = echoOK
	}
	return &agentloop.Agent{
		Model:   c.model,
		ModelID: fake.ModelID,
		Tools: []agentloop.Tool{agentloop.Func(
			agentloop.ToolSpec{Name: "echo", Description: "echo", Parameters: json.RawMessage(echoSchema)},
			func(ctx context.Context, in json.RawMessage) (agentloop.Result, error) {
				return tool(ctx, func() {}, in)
			},
		)},
		Policy:  c.policy,
		Limits:  c.limits,
		Now:     func() time.Time { return goldenStart },
		Sleep:   func(ctx context.Context, d time.Duration) error { return ctx.Err() },
		Observe: record,
	}
}

// TestServerSendsTheTranslation runs the golden cases through the server,
// with and without the client's notices capability: a reply, a tool loop,
// a denied call, a retry, a refusal, a model error, a held call and a
// limit. What reaches the client for the turn is exactly the translation
// of the run's events, in order, every update validates and carries the
// session id, and the response is Respond of the outcome, after every
// update and with none following it.
func TestServerSendsTheTranslation(t *testing.T) {
	for _, c := range goldenCases() {
		if c.name == "cancelled_call" {
			continue // the tool cancels the test's own context; see the cancel tests
		}
		for _, notices := range []bool{false, true} {
			name := c.name + map[bool]string{false: "", true: "/notices"}[notices]
			t.Run(name, func(t *testing.T) {
				c := c
				c.model = goldenCases()[indexOf(c.name)].model // a fresh script
				var mu sync.Mutex
				var events []agentloop.Event
				var out *agentloop.Outcome
				agent := goldenAgent(c, func(ev agentloop.Event) {
					mu.Lock()
					defer mu.Unlock()
					events = append(events, ev)
					if ev.Entry != nil && ev.Entry.RunEnded != nil {
						out = ev.Entry.RunEnded
					}
				})
				opts := Options{Kinds: map[string]acp.ToolKind{"echo": acp.ToolKindRead}, ContextWindow: 200000}
				h := serve(t, agent, opts, notices)
				id := h.newSession()
				resp, err := h.prompt(id, "Say hi.")

				mu.Lock()
				defer mu.Unlock()
				opts.Notices = notices
				want := asJSON(t, translateAll(opts, events))
				during, wireResp, after := h.turnOnWire(id, 0)
				if got := rawStrings(during); !equalStrings(got, want) {
					t.Fatalf("updates on the wire:\n%s\nwant the translation:\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
				}
				if len(after) != 0 {
					t.Fatalf("%d updates after the response", len(after))
				}
				for _, u := range during {
					var up Update
					if err := json.Unmarshal(u, &up); err != nil {
						t.Fatal(err)
					}
					if err := validate("SessionNotification", map[string]any{"sessionId": id, "update": json.RawMessage(u)}); err != nil {
						t.Fatal(err)
					}
				}
				noticeSeen := strings.Contains(strings.Join(want, ""), `"notice"`)
				if noticeSeen && !notices {
					t.Fatal("a notice for a client that cannot show them")
				}
				if c.name == "retry_then_success" && notices && !noticeSeen {
					t.Fatal("no retry notice for a client that shows them")
				}

				stop, rpcErr := Respond(out)
				if rpcErr != nil {
					var re *acp.RequestError
					if !errors.As(err, &re) || re.Code != rpcErr.Code || re.Message != rpcErr.Message {
						t.Fatalf("got %v, %v; want the error %v", resp, err, rpcErr)
					}
					if wireResp.Error == nil {
						t.Fatalf("the response on the wire is not an error")
					}
				} else if err != nil || resp.StopReason != stop {
					t.Fatalf("got %v, %v; want %s", resp, err, stop)
				} else if err := validate("PromptResponse", resp); err != nil {
					t.Fatal(err)
				}
			})
		}
	}
}

func indexOf(name string) int {
	for i, c := range goldenCases() {
		if c.name == name {
			return i
		}
	}
	panic(name)
}

// blockingTool is a tool that signals when it starts and then waits for
// its context to end or for release.
type blockingTool struct {
	started chan string
	release chan struct{}
}

func newBlockingTool() *blockingTool {
	return &blockingTool{started: make(chan string, 16), release: make(chan struct{})}
}

func (b *blockingTool) tool() agentloop.Tool {
	return agentloop.Func(
		agentloop.ToolSpec{Name: "echo", Description: "echo", Parameters: json.RawMessage(echoSchema)},
		func(ctx context.Context, in json.RawMessage) (agentloop.Result, error) {
			b.started <- string(in)
			select {
			case <-ctx.Done():
				return agentloop.Result{}, ctx.Err()
			case <-b.release:
				return agentloop.Result{Content: "released"}, nil
			}
		})
}

func statusOf(raw json.RawMessage) (kind, status string) {
	var u struct {
		SessionUpdate string `json:"sessionUpdate"`
		Status        string `json:"status"`
	}
	json.Unmarshal(raw, &u)
	return u.SessionUpdate, u.Status
}

// TestSDKDeliversCancelDuringPrompt pins what the design left to check
// (section 3.3): the SDK runs session/cancel while the session/prompt
// handler is still running, and keeps notifications and the response in
// order. The tool returns only when its context ends, so the turn can end
// only if the cancel got through; the update that reports the cancelled
// call must be on the wire before the response, and nothing after it.
func TestSDKDeliversCancelDuringPrompt(t *testing.T) {
	bt := newBlockingTool()
	agent := &agentloop.Agent{
		Model: fake.Script(fake.ToolUse("Running it.", fake.Call("call_1", "echo", `{"text":"hi"}`))),
		Tools: []agentloop.Tool{bt.tool()},
	}
	h := serve(t, agent, Options{}, false)
	id := h.newSession()
	go func() {
		<-bt.started
		h.client.Cancel(context.Background(), acp.CancelNotification{SessionId: id})
	}()
	resp, err := h.prompt(id, "go")
	if err != nil || resp.StopReason != acp.StopReasonCancelled {
		t.Fatalf("got %+v, %v; want cancelled", resp, err)
	}
	during, _, after := h.turnOnWire(id, 0)
	var statuses []string
	for _, u := range during {
		if kind, status := statusOf(u); kind == "tool_call" || kind == "tool_call_update" {
			statuses = append(statuses, status)
		}
	}
	if !equalStrings(statuses, []string{"pending", "in_progress", "failed"}) {
		t.Fatalf("call statuses before the response: %v", statuses)
	}
	time.Sleep(50 * time.Millisecond)
	if _, _, after = h.turnOnWire(id, 0); len(after) != 0 {
		t.Fatalf("%d updates after the response", len(after))
	}
}

// TestSecondPromptLeavesTheFirstRunning is a regression test for what the
// SDK does with a second session/prompt for a session: it cancels the
// first prompt's request context. A run on that context would be cut off
// by a prompt the server refuses; the run must go on and end normally.
func TestSecondPromptLeavesTheFirstRunning(t *testing.T) {
	bt := newBlockingTool()
	agent := &agentloop.Agent{
		Model: fake.Script(fake.ToolUse("", fake.Call("call_1", "echo", `{"text":"hi"}`)), fake.Text("Done.")),
		Tools: []agentloop.Tool{bt.tool()},
	}
	h := serve(t, agent, Options{}, false)
	id := h.newSession()
	first := make(chan error, 1)
	var firstResp acp.PromptResponse
	go func() {
		var err error
		firstResp, err = h.prompt(id, "go")
		first <- err
	}()
	<-bt.started
	_, err := h.prompt(id, "and again")
	var re *acp.RequestError
	if !errors.As(err, &re) || re.Code != -32603 || !strings.Contains(re.Message, "already running") {
		t.Fatalf("second prompt: %v", err)
	}
	close(bt.release)
	if err := <-first; err != nil || firstResp.StopReason != acp.StopReasonEndTurn {
		t.Fatalf("first prompt: %+v, %v", firstResp, err)
	}
}

func TestPromptForUnknownSession(t *testing.T) {
	h := serve(t, &agentloop.Agent{Model: fake.Script()}, Options{}, false)
	_, err := h.prompt("00000000-0000-4000-8000-000000000000", "hi")
	var re *acp.RequestError
	if !errors.As(err, &re) || re.Code != -32002 {
		t.Fatalf("err = %v", err)
	}
}

// A prompt the server cannot take is refused without changing the
// session, which takes the next prompt.
func TestRefusedPromptLeavesSessionUsable(t *testing.T) {
	h := serve(t, &agentloop.Agent{Model: fake.Script(fake.Text("Hello."))}, Options{}, false)
	id := h.newSession()
	_, err := h.client.Prompt(h.ctx(), acp.PromptRequest{SessionId: id,
		Prompt: []acp.ContentBlock{acp.ImageBlock("aGk=", "image/png")}})
	var re *acp.RequestError
	if !errors.As(err, &re) || re.Code != -32602 {
		t.Fatalf("image prompt: %v", err)
	}
	if resp, err := h.prompt(id, "hi"); err != nil || resp.StopReason != acp.StopReasonEndTurn {
		t.Fatalf("%+v, %v", resp, err)
	}
}

func TestSessionsAreSeparate(t *testing.T) {
	h := serve(t, &agentloop.Agent{Model: fake.Script(fake.Text("One."), fake.Text("Two."))}, Options{}, false)
	a, b := h.newSession(), h.newSession()
	if a == b {
		t.Fatal("two sessions with one id")
	}
	for _, id := range []acp.SessionId{a, b} {
		if resp, err := h.prompt(id, "hi"); err != nil || resp.StopReason != acp.StopReasonEndTurn {
			t.Fatalf("%+v, %v", resp, err)
		}
		if _, err := os.Stat(SessionPath(h.dir, id)); err != nil {
			t.Fatal(err)
		}
	}
}

// Client MCP servers are not connected, and the operator is told so on
// stderr, once per session/new.
func TestMCPServersAreIgnored(t *testing.T) {
	h := serve(t, &agentloop.Agent{Model: fake.Script()}, Options{}, false)
	_, err := h.client.NewSession(h.ctx(), acp.NewSessionRequest{Cwd: "/", McpServers: []acp.McpServer{
		{Stdio: &acp.McpServerStdio{Name: "files", Command: "/bin/true", Args: []string{}, Env: []acp.EnvVariable{}}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if got := h.logged(); strings.Count(got, "ignoring 1 MCP server") != 1 {
		t.Fatalf("log: %q", got)
	}
}

// flockFree reports whether a fresh lock on the session succeeds.
func flockFree(t tb, path string) bool {
	t.Helper()
	f, err := os.OpenFile(path+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	return syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB) == nil
}

// Losing the client cancels the run in flight, answers it cancelled,
// releases every session's lock and refuses new sessions.
func TestShutdownCancelsAndReleases(t *testing.T) {
	bt := newBlockingTool()
	agent := &agentloop.Agent{
		Model: fake.Script(fake.ToolUse("", fake.Call("call_1", "echo", `{"text":"hi"}`))),
		Tools: []agentloop.Tool{bt.tool()},
	}
	h := serve(t, agent, Options{}, false)
	busy, idle := h.newSession(), h.newSession()
	for _, id := range []acp.SessionId{busy, idle} {
		if flockFree(t, SessionPath(h.dir, id)) {
			t.Fatal("an open session is not locked")
		}
	}
	answered := make(chan acp.PromptResponse, 1)
	go func() {
		resp, _ := h.prompt(busy, "go")
		answered <- resp
	}()
	<-bt.started
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	h.srv.Shutdown(ctx)
	if resp := <-answered; resp.StopReason != acp.StopReasonCancelled {
		t.Fatalf("the run in flight stopped %q", resp.StopReason)
	}
	for _, id := range []acp.SessionId{busy, idle} {
		if !flockFree(t, SessionPath(h.dir, id)) {
			t.Fatalf("session %s is still locked", id)
		}
		if _, open := h.srv.State(id); open {
			t.Fatalf("session %s is still open", id)
		}
	}
	_, err := h.client.NewSession(h.ctx(), acp.NewSessionRequest{Cwd: "/", McpServers: []acp.McpServer{}})
	if err == nil {
		t.Fatal("a new session after shutdown")
	}
}

// describedModel is a model whose provider describes it as info, or fails
// with err.
type describedModel struct {
	agentloop.Model
	info agentloop.ModelInfo
	err  error
}

func (d describedModel) Describe(context.Context, string) (agentloop.ModelInfo, error) {
	return d.info, d.err
}

// notes is every update the client has received so far.
func (c *testClient) notes() []acp.SessionNotification {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]acp.SessionNotification(nil), c.updates...)
}

// A conversation that may not hold private: a tool result carrying it is
// refused, the client sees only the refusal and the operator's log says
// which call it was; a session that holds it anyway is not run, and its
// prompt is answered with an error.
func TestForbiddenLabelsInAConversation(t *testing.T) {
	secret := agentloop.Func(
		agentloop.ToolSpec{Name: "secret", Description: "secret", Parameters: json.RawMessage(`{"type":"object"}`)},
		func(context.Context, json.RawMessage) (agentloop.Result, error) {
			return agentloop.Result{Content: "SECRET-TEXT", Labels: []string{"private"}}, nil
		})
	m := fake.Script(fake.ToolUse("", fake.Call("c1", "secret", `{}`)), fake.Text("done"))
	h := serve(t, &agentloop.Agent{Model: m, Tools: []agentloop.Tool{secret}, Forbid: []string{"private"}}, Options{}, false)
	id := h.newSession()
	if resp, err := h.prompt(id, "hi"); err != nil || resp.StopReason != acp.StopReasonEndTurn {
		t.Fatalf("%+v, %v", resp, err)
	}
	for _, l := range h.toClient.snapshot() {
		if bytes.Contains(l, []byte("SECRET-TEXT")) {
			t.Fatalf("the client saw the refused result: %s", l)
		}
	}
	if !strings.Contains(h.logged(), "refused the result of call c1 to secret") {
		t.Fatalf("log %q", h.logged())
	}

	// An agent that would seed a forbidden label opens no session.
	seeded := serve(t, &agentloop.Agent{Model: noModel(t), Labels: []string{"private"}, Forbid: []string{"private"}}, Options{}, false)
	if _, err := seeded.client.NewSession(seeded.ctx(), acp.NewSessionRequest{Cwd: "/", McpServers: []acp.McpServer{}}); err == nil {
		t.Fatal("a session was opened holding a forbidden label")
	}

	// A session already holding private, from an agent without the rule,
	// loads, but is not run.
	dir := t.TempDir()
	before := serveIn(t, dir, &agentloop.Agent{Model: fake.Script(fake.Text("hi")), Labels: []string{"private"}}, Options{}, false)
	oid := before.newSession()
	if _, err := before.prompt(oid, "hi"); err != nil {
		t.Fatal(err)
	}
	if _, err := before.client.CloseSession(before.ctx(), acp.CloseSessionRequest{SessionId: oid}); err != nil {
		t.Fatal(err)
	}
	after := serveIn(t, dir, &agentloop.Agent{Model: noModel(t), Forbid: []string{"private"}}, Options{}, false)
	if _, err := after.client.LoadSession(after.ctx(), acp.LoadSessionRequest{SessionId: oid, Cwd: "/", McpServers: []acp.McpServer{}}); err != nil {
		t.Fatal(err)
	}
	_, err := after.prompt(oid, "more")
	var re *acp.RequestError
	if !errors.As(err, &re) || re.Code != -32603 || !strings.Contains(re.Message, "may not hold") {
		t.Fatalf("prompt: %v", err)
	}
}

func noModel(t *testing.T) agentloop.Model {
	return fake.New(func(*agentloop.Request) (*agentloop.Response, error) {
		t.Error("the model was called")
		return nil, errors.New("no model")
	})
}

// A prompt whose content brings a label the session may not hold is
// refused before the label is written: the session stays as it was, and a
// prompt without it runs.
func TestAPromptBringingAForbiddenLabelIsRefusedBeforeItIsRecorded(t *testing.T) {
	h := serve(t, &agentloop.Agent{Model: fake.Script(fake.Text("Hello.")), Forbid: []string{"contaminated"}}, Options{}, false)
	id := h.newSession()
	_, err := h.client.Prompt(h.ctx(), acp.PromptRequest{SessionId: id, Prompt: []acp.ContentBlock{
		acp.TextBlock("read this"), acp.ResourceLinkBlock("page", "https://example.com/x")}})
	var re *acp.RequestError
	if !errors.As(err, &re) || !strings.Contains(re.Message, "may not hold") {
		t.Fatalf("prompt: %v", err)
	}
	if resp, err := h.prompt(id, "hi"); err != nil || resp.StopReason != acp.StopReasonEndTurn {
		t.Fatalf("%+v, %v", resp, err)
	}
	b, err := os.ReadFile(SessionPath(h.dir, id))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), "contaminated") {
		t.Fatalf("the session took the label: %s", b)
	}
}
