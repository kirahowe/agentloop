package acpagent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	acp "github.com/caelis-labs/acp-go-sdk"
	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/fake"
	"github.com/kirahowe/agentloop/internal/scenario"
)

// --- harness ---

func (h *harness) load(id acp.SessionId) error {
	_, err := h.client.LoadSession(h.ctx(), acp.LoadSessionRequest{SessionId: id, Cwd: "/", McpServers: []acp.McpServer{}})
	return err
}

func (h *harness) resume(id acp.SessionId) error {
	_, err := h.client.ResumeSession(h.ctx(), acp.ResumeSessionRequest{SessionId: id, Cwd: "/"})
	return err
}

func (h *harness) close(id acp.SessionId) error {
	_, err := h.client.CloseSession(h.ctx(), acp.CloseSessionRequest{SessionId: id})
	return err
}

// mark is how many lines the client has been sent so far.
func (h *harness) mark() int { return len(h.toClient.snapshot()) }

// since splits what the client was sent after mark at the first response:
// the session's updates before it, the response, and the session's
// updates after it.
func (h *harness) since(mark int, id acp.SessionId) (before []json.RawMessage, resp message, after []json.RawMessage) {
	h.t.Helper()
	answered := false
	for _, m := range parseLines(h.t, h.toClient.snapshot()[mark:]) {
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
				before = append(before, p.Update)
			}
		case m.Method == "" && !answered:
			resp, answered = m, true
		}
	}
	if !answered {
		h.t.Fatalf("no response on the wire after line %d", mark)
	}
	return before, resp, after
}

// sessionUpdates is every update the client got for the session, in order.
func (h *harness) sessionUpdates(id acp.SessionId) []Update {
	h.cl.mu.Lock()
	defer h.cl.mu.Unlock()
	var out []Update
	for _, n := range h.cl.updates {
		if n.SessionId == id {
			out = append(out, n.Update)
		}
	}
	return out
}

func isRPC(err error, code int, msg string) bool {
	var re *acp.RequestError
	return errors.As(err, &re) && re.Code == code && (msg == "" || re.Message == msg)
}

func inUse(err error) bool { return isRPC(err, -32603, "session in use") }

// tree is every file under root, with its contents, but the lock files
// (which a lock attempt may create, empty, and never writes to).
func tree(t tb, root string) map[string]string {
	t.Helper()
	out := map[string]string{}
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(root, p)
		if d.IsDir() {
			out[rel+"/"] = ""
			return nil
		}
		if strings.HasSuffix(p, ".lock") {
			return nil
		}
		b, err := os.ReadFile(p)
		out[rel] = string(b)
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func sameTree(a, b map[string]string) (string, bool) {
	for k, v := range a {
		if w, ok := b[k]; !ok || w != v {
			return k, false
		}
	}
	for k := range b {
		if _, ok := a[k]; !ok {
			return k, false
		}
	}
	return "", true
}

func textAgent(replies ...string) *agentloop.Agent {
	var steps []any
	for _, r := range replies {
		steps = append(steps, fake.Text(r))
	}
	return &agentloop.Agent{Model: fake.Script(steps...)}
}

// --- scripted cases ---

// A loaded session's history reaches the client before the load response,
// exactly as Replay gives it, and nothing follows the response. The next
// prompt goes on from the log, with message ids that follow its entries.
func TestLoadReplaysBeforeResponding(t *testing.T) {
	c := goldenCases()[indexOf("tool_loop")]
	opts := Options{Kinds: map[string]acp.ToolKind{"echo": acp.ToolKindRead}, ContextWindow: 1000}
	dir := t.TempDir()
	first := serveIn(t, dir, goldenAgent(c, nil), opts, false)
	id := first.newSession()
	if resp, err := first.prompt(id, "go"); err != nil || resp.StopReason != acp.StopReasonEndTurn {
		t.Fatalf("%+v, %v", resp, err)
	}
	if err := first.close(id); err != nil {
		t.Fatal(err)
	}
	entries, err := agentloop.NewFileStore(SessionPath(dir, id)).Load()
	if err != nil {
		t.Fatal(err)
	}

	h := serveIn(t, dir, &agentloop.Agent{Model: fake.Script(fake.Text("Again."))}, opts, false)
	m := h.mark()
	if err := h.load(id); err != nil {
		t.Fatal(err)
	}
	time.Sleep(20 * time.Millisecond)
	before, resp, after := h.since(m, id)
	if resp.Error != nil || string(resp.Result) != "{}" {
		t.Fatalf("load response %s %s", resp.Result, resp.Error)
	}
	want := asJSON(t, Replay(opts, entries))
	if got := rawStrings(before); !equalStrings(got, want) {
		t.Fatalf("replayed:\n%s\nwant:\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
	}
	if len(after) != 0 || len(want) == 0 {
		t.Fatalf("%d replayed, %d after the response", len(want), len(after))
	}
	if !strings.Contains(strings.Join(want, ""), `"user_message_chunk"`) {
		t.Fatal("the prompt was not replayed")
	}

	if resp, err := h.prompt(id, "more"); err != nil || resp.StopReason != acp.StopReasonEndTurn {
		t.Fatalf("%+v, %v", resp, err)
	}
	ups := h.sessionUpdates(id)
	last := ups[len(ups)-1]
	for _, u := range ups {
		if u.AgentMessageChunk != nil {
			last = u
		}
	}
	all, _ := agentloop.NewFileStore(SessionPath(dir, id)).Load()
	var lastAssistant int
	for _, e := range all {
		if e.Message != nil && e.Message.Message.Role == agentloop.Assistant {
			lastAssistant = e.Seq
		}
	}
	if got := *last.AgentMessageChunk.MessageId; got != assistantID(lastAssistant) {
		t.Fatalf("the new reply streams as %s, its entry is %s", got, assistantID(lastAssistant))
	}
}

// session/resume opens the session without sending its history.
func TestResumeSendsNoHistory(t *testing.T) {
	dir := t.TempDir()
	first := serveIn(t, dir, textAgent("One."), Options{}, false)
	id := first.newSession()
	if _, err := first.prompt(id, "hi"); err != nil {
		t.Fatal(err)
	}
	if err := first.close(id); err != nil {
		t.Fatal(err)
	}
	h := serveIn(t, dir, textAgent("Two."), Options{}, false)
	if err := h.resume(id); err != nil {
		t.Fatal(err)
	}
	if n := len(h.sessionUpdates(id)); n != 0 {
		t.Fatalf("%d updates on resume", n)
	}
	if resp, err := h.prompt(id, "again"); err != nil || resp.StopReason != acp.StopReasonEndTurn {
		t.Fatalf("%+v, %v", resp, err)
	}
	entries, _ := agentloop.NewFileStore(SessionPath(dir, id)).Load()
	var said []string
	for _, e := range entries {
		if e.Message != nil && e.Message.Message.Role == agentloop.Assistant {
			said = append(said, e.Message.Message.Parts[0].Text)
		}
	}
	if strings.Join(said, " ") != "One. Two." {
		t.Fatalf("the log holds %v", said)
	}
}

// Ids agentloop could not have made, and well-formed ids with no log, are
// unknown sessions; so is closing a session that is not open.
func TestLoadUnknownSessions(t *testing.T) {
	h := serve(t, textAgent(), Options{}, false)
	for _, id := range []acp.SessionId{NewSessionID([16]byte{9}), "../x", "nope", ""} {
		if err := h.load(id); !isRPC(err, -32002, "") {
			t.Errorf("load %q: %v", id, err)
		}
		if err := h.resume(id); !isRPC(err, -32002, "") {
			t.Errorf("resume %q: %v", id, err)
		}
		if err := h.close(id); !isRPC(err, -32002, "") {
			t.Errorf("close %q: %v", id, err)
		}
	}
}

// Regression: session/new made no file until the first prompt was logged,
// so a session the client had an id for could be neither listed nor
// loaded once its connection was gone.
func TestNewSessionCanBeLoadedBeforeAnyPrompt(t *testing.T) {
	dir := t.TempDir()
	first := serveIn(t, dir, textAgent(), Options{}, false)
	id := first.newSession()
	if _, err := os.Stat(SessionPath(dir, id)); err != nil {
		t.Fatalf("no session file: %v", err)
	}
	if err := first.close(id); err != nil {
		t.Fatal(err)
	}
	h := serveIn(t, dir, textAgent("Hi."), Options{}, false)
	list, err := h.client.ListSessions(h.ctx(), acp.ListSessionsRequest{})
	if err != nil || len(list.Sessions) != 1 || list.Sessions[0].SessionId != id || list.Sessions[0].UpdatedAt != nil {
		t.Fatalf("%+v, %v", list, err)
	}
	if err := h.load(id); err != nil {
		t.Fatal(err)
	}
	if resp, err := h.prompt(id, "hi"); err != nil || resp.StopReason != acp.StopReasonEndTurn {
		t.Fatalf("%+v, %v", resp, err)
	}
}

// Loading or resuming a session this connection already has open, while
// it is idle, takes it up again: what acp-tck's load and resume cases do.
func TestReloadOnTheSameConnection(t *testing.T) {
	h := serve(t, textAgent("One.", "Two."), Options{}, false)
	id := h.newSession()
	if err := h.load(id); err != nil {
		t.Fatalf("load of a new session: %v", err)
	}
	if _, err := h.prompt(id, "hi"); err != nil {
		t.Fatal(err)
	}
	m := h.mark()
	if err := h.load(id); err != nil {
		t.Fatal(err)
	}
	before, _, _ := h.since(m, id)
	entries, _ := agentloop.NewFileStore(SessionPath(h.dir, id)).Load()
	if got, want := rawStrings(before), asJSON(t, Replay(Options{}, entries)); !equalStrings(got, want) {
		t.Fatalf("replayed %v, want %v", got, want)
	}
	if err := h.resume(id); err != nil {
		t.Fatal(err)
	}
	if resp, err := h.prompt(id, "again"); err != nil || resp.StopReason != acp.StopReasonEndTurn {
		t.Fatalf("%+v, %v", resp, err)
	}
	if st, open := h.srv.State(id); !open || st != Idle {
		t.Fatalf("%s %v", st, open)
	}
}

func TestListSessions(t *testing.T) {
	dir := t.TempDir()
	h := serveIn(t, dir, textAgent("One.", "Two."), Options{}, false)
	a, b := h.newSession(), h.newSession()
	for _, id := range []acp.SessionId{a, b} {
		if _, err := h.prompt(id, "hi"); err != nil {
			t.Fatal(err)
		}
	}
	os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("x"), 0o600)
	os.Mkdir(filepath.Join(dir, string(NewSessionID([16]byte{3}))+".jsonl"), 0o755)

	list, err := h.client.ListSessions(h.ctx(), acp.ListSessionsRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if err := validate("ListSessionsResponse", list); err != nil {
		t.Fatal(err)
	}
	if len(list.Sessions) != 2 || list.Sessions[0].SessionId != b || list.Sessions[1].SessionId != a {
		t.Fatalf("%+v", list.Sessions)
	}
	for _, s := range list.Sessions {
		if s.Cwd != h.srv.cfg.Cwd || !filepath.IsAbs(s.Cwd) || s.UpdatedAt == nil {
			t.Fatalf("%+v", s)
		}
	}
	list, err = h.client.ListSessions(h.ctx(), acp.ListSessionsRequest{Cwd: acp.Ptr(filepath.Join(dir, "elsewhere"))})
	if err != nil || list.Sessions == nil || len(list.Sessions) != 0 {
		t.Fatalf("%+v, %v", list, err)
	}
	if _, err := h.client.ListSessions(h.ctx(), acp.ListSessionsRequest{Cursor: acp.Ptr("x")}); !isRPC(err, -32602, "") {
		t.Fatalf("cursor: %v", err)
	}
}

// A missing session directory lists nothing rather than failing.
func TestListWithNoSessionDir(t *testing.T) {
	h := serveIn(t, filepath.Join(t.TempDir(), "none"), textAgent(), Options{}, false)
	list, err := h.client.ListSessions(h.ctx(), acp.ListSessionsRequest{})
	if err != nil || list.Sessions == nil || len(list.Sessions) != 0 {
		t.Fatalf("%+v, %v", list, err)
	}
}

// session/close on an idle session lets go of it at once; on a running
// one it cancels the turn, which is answered cancelled, and answers only
// once the session is let go of. A closed session is no longer open.
func TestCloseSession(t *testing.T) {
	for _, running := range []bool{false, true} {
		t.Run(fmt.Sprint("running=", running), func(t *testing.T) {
			bt := newBlockingTool()
			agent := &agentloop.Agent{
				Model: fake.Script(fake.ToolUse("", fake.Call("call_1", "echo", `{"text":"hi"}`))),
				Tools: []agentloop.Tool{bt.tool()},
			}
			h := serve(t, agent, Options{}, false)
			id := h.newSession()
			answered := make(chan acp.StopReason, 1)
			if running {
				go func() {
					resp, _ := h.prompt(id, "go")
					answered <- resp.StopReason
				}()
				<-bt.started
			}
			if err := h.close(id); err != nil {
				t.Fatal(err)
			}
			if !flockFree(t, SessionPath(h.dir, id)) {
				t.Fatal("the session is still locked after close answered")
			}
			if _, open := h.srv.State(id); open {
				t.Fatal("the session is still open")
			}
			if running {
				if stop := <-answered; stop != acp.StopReasonCancelled {
					t.Fatalf("the turn stopped %q", stop)
				}
			}
			if _, err := h.prompt(id, "hi"); !isRPC(err, -32002, "") {
				t.Fatalf("prompt after close: %v", err)
			}
			if err := h.load(id); err != nil {
				t.Fatalf("load after close: %v", err)
			}
		})
	}
}

// --- properties ---

// guard keeps a rapid failure inside the server's goroutines from killing
// the test binary: the scenario's checks call Fatalf wherever the agent
// calls them, which is on the run's goroutine. A panic there is caught,
// kept, and reported on the test's own goroutine by check.
type guard struct {
	mu  sync.Mutex
	msg string
}

func (g *guard) catch(r any) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.msg == "" {
		g.msg = fmt.Sprint(r)
	}
}

func (g *guard) check(t *rapid.T) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.msg != "" {
		t.Fatalf("in the server: %s", g.msg)
	}
}

type guardModel struct {
	agentloop.Model
	g *guard
}

func (m guardModel) Generate(ctx context.Context, req *agentloop.Request, obs func(agentloop.Delta)) (resp *agentloop.Response, err error) {
	defer func() {
		if r := recover(); r != nil {
			m.g.catch(r)
			resp, err = nil, errors.New("the test failed")
		}
	}()
	return m.Model.Generate(ctx, req, obs)
}

type guardTool struct {
	agentloop.Tool
	g *guard
}

func (t guardTool) Call(ctx context.Context, in json.RawMessage) (res agentloop.Result, err error) {
	defer func() {
		if r := recover(); r != nil {
			t.g.catch(r)
			res, err = agentloop.Result{}, errors.New("the test failed")
		}
	}()
	return t.Tool.Call(ctx, in)
}

type guardCallTool struct {
	guardTool
	ct agentloop.CallTool
}

func (t guardCallTool) CallWith(ctx context.Context, c agentloop.Call) (res agentloop.Result, err error) {
	defer func() {
		if r := recover(); r != nil {
			t.g.catch(r)
			res, err = agentloop.Result{}, errors.New("the test failed")
		}
	}()
	return t.ct.CallWith(ctx, c)
}

func (g *guard) agent(a *agentloop.Agent) *agentloop.Agent {
	b := *a
	b.Model = guardModel{a.Model, g}
	b.Tools = nil
	for _, t := range a.Tools {
		if ct, ok := t.(agentloop.CallTool); ok {
			b.Tools = append(b.Tools, guardCallTool{guardTool{t, g}, ct})
			continue
		}
		b.Tools = append(b.Tools, guardTool{t, g})
	}
	if p := a.Policy; p != nil {
		b.Policy = agentloop.PolicyFunc(func(ctx context.Context, c agentloop.ToolCall, v agentloop.View) (d agentloop.Decision) {
			defer func() {
				if r := recover(); r != nil {
					g.catch(r)
					d = agentloop.Decision{Verdict: agentloop.Deny, Reason: "the test failed"}
				}
			}()
			return p.Decide(ctx, c, v)
		})
	}
	return &b
}

func drawID(t *rapid.T) acp.SessionId {
	var b [16]byte
	copy(b[:], rapid.SliceOfN(rapid.Byte(), 16, 16).Draw(t, "idbytes"))
	return NewSessionID(b)
}

// checkReplay holds what a load sends: exactly Replay of the log as it
// was, every update valid, all of it before the response and nothing
// after.
func checkReplay(t *rapid.T, h *harness, mark int, id acp.SessionId, opts Options, entries []agentloop.Entry) {
	time.Sleep(time.Millisecond)
	before, resp, after := h.since(mark, id)
	if resp.Error != nil {
		t.Fatalf("load: %s", resp.Error)
	}
	want := Replay(opts, entries)
	if got, w := rawStrings(before), asJSON(t, want); !equalStrings(got, w) {
		t.Fatalf("replayed:\n%s\nwant:\n%s", strings.Join(got, "\n"), strings.Join(w, "\n"))
	}
	if len(after) != 0 {
		t.Fatalf("%d updates after the load response", len(after))
	}
	for _, u := range want {
		if err := validateUpdate(u); err != nil {
			t.Fatal(err)
		}
	}
}

// TestPropertyResumeAnywhere is P4.1. A generated scenario runs with its
// crash faults (the store dies at random appends, before or after the
// line is durable) and cancellations, and the log it leaves is handed to
// a session file. A fresh server loads it, which replays exactly that
// log, and the conversation is then driven over ACP alone, one prompt at
// a time, as the core harness drains it: every prompt is answered with
// Respond of its run's outcome, and the conversation completes. The
// client answers every question about a held call with cancelled, which
// denies it. Throughout, the
// scenario's own checks hold for every model call, policy decision and
// tool run the server makes, and at rest the log reloads to the same
// state, kept everything persisted before, ran no call twice, and
// settled every call the crashes left open.
func TestPropertyResumeAnywhere(t *testing.T) {
	root := t.TempDir()
	rapid.Check(t, func(t *rapid.T) {
		sc := scenario.New(t, scenario.GenConfig(t), scenario.Faults{
			Crash:  true,
			Cancel: rapid.Bool().Draw(t, "cancel"),
			Wild:   rapid.Bool().Draw(t, "wild"),
		})
		sc.Faulty()
		entries, err := sc.Store.Load()
		if err != nil {
			t.Fatal(err)
		}
		dir, err := os.MkdirTemp(root, "s")
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { os.RemoveAll(dir) })
		id := drawID(t)
		sc.HandOver(SessionPath(dir, id))
		sc.PrepareDrain()

		g := &guard{}
		var mu sync.Mutex
		var out *agentloop.Outcome
		agent := g.agent(sc.Agent())
		agent.Observe = func(ev agentloop.Event) {
			if ev.Entry != nil && ev.Entry.RunEnded != nil {
				mu.Lock()
				out = ev.Entry.RunEnded
				mu.Unlock()
			}
		}
		opts := genOptions(t)
		h := serveIn(t, dir, agent, opts, opts.Notices)
		m := h.mark()
		if err := h.load(id); err != nil {
			t.Fatalf("load: %v", err)
		}
		checkReplay(t, h, m, id, opts, entries)

		complete := false
		for i := 0; i < sc.DrainRuns() && !complete; i++ {
			mu.Lock()
			out = nil
			mu.Unlock()
			resp, err := h.prompt(id, "continue")
			g.check(t)
			mu.Lock()
			o := out
			mu.Unlock()
			if o == nil {
				t.Fatalf("prompt %d: no run ended (%+v, %v)", i, resp, err)
			}
			stop, rpcErr := Respond(o)
			switch {
			case rpcErr != nil:
				if !isRPC(err, rpcErr.Code, rpcErr.Message) {
					t.Fatalf("prompt %d: got %+v, %v; want the error %v", i, resp, err, rpcErr)
				}
			case err != nil || resp.StopReason != stop:
				t.Fatalf("prompt %d: got %+v, %v; want %s", i, resp, err, stop)
			}
			if o.Reason == agentloop.ReasonHeld {
				break
			}
			complete = o.Reason == agentloop.ReasonCompleted
		}
		for _, u := range h.sessionUpdates(id) {
			if err := validateUpdate(u); err != nil {
				t.Fatal(err)
			}
		}
		if !complete {
			es, _ := agentloop.NewFileStore(SessionPath(dir, id)).Load()
			if s, err := agentloop.OpenSession(agentloop.NewMemoryStore(es...)); err != nil || len(s.Held()) == 0 {
				t.Fatalf("the conversation did not complete and nothing is held (%v)", err)
			}
		}
		sc.CheckAtRest(complete)
	})
}

// TestPropertyReplayMatchesLive is P4.2, P1.3 end to end: a generated
// scenario is driven over a real server from an empty session, a few
// prompts, and the session is closed. A second connection loads it. Per
// messageId, the chunk text the client saw live equals what the load
// replays, and each call ends with the same status, content, input and
// output; the replay is exactly Replay of the log on disk.
func TestPropertyReplayMatchesLive(t *testing.T) {
	root := t.TempDir()
	rapid.Check(t, func(t *rapid.T) {
		sc := scenario.New(t, scenario.GenConfig(t), scenario.Faults{Wild: rapid.Bool().Draw(t, "wild")})
		dir, err := os.MkdirTemp(root, "s")
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { os.RemoveAll(dir) })
		id := drawID(t)
		sc.HandOver(SessionPath(dir, id))
		g := &guard{}
		opts := genOptions(t)

		live := serveIn(t, dir, g.agent(sc.Agent()), opts, opts.Notices)
		if err := live.resume(id); err != nil {
			t.Fatalf("resume: %v", err)
		}
		for i, n := 0, rapid.IntRange(1, 4).Draw(t, "prompts"); i < n; i++ {
			live.prompt(id, rapid.SampledFrom([]string{"go", "more", "x\ny"}).Draw(t, "prompt"))
			g.check(t)
		}
		if err := live.close(id); err != nil {
			t.Fatal(err)
		}
		liveUps := live.sessionUpdates(id)
		entries, err := agentloop.NewFileStore(SessionPath(dir, id)).Load()
		if err != nil {
			t.Fatal(err)
		}

		h := serveIn(t, dir, g.agent(sc.Agent()), opts, opts.Notices)
		m := h.mark()
		if err := h.load(id); err != nil {
			t.Fatalf("load: %v", err)
		}
		checkReplay(t, h, m, id, opts, entries)
		// Both streams as the client decoded them: a value the wire
		// carries as null, say, reads back the same way on both sides.
		checkLiveIsReplay(t, read(t, liveUps), read(t, h.sessionUpdates(id)), entries)
		sc.CheckAtRest(false)
	})
}

// TestPropertyExclusive is P4.3 within one process. Connection A has a
// session open, idle or with a turn in flight; then loads and resumes of
// it come from A itself and from connection B, a second server in the
// same process on the same directory, in a random order. Every attempt
// from B fails with "session in use", as does one from A while its turn
// runs; A may take its idle session up again. No attempt changes
// anything on disk. Once A closes the session, B can load it.
func TestPropertyExclusive(t *testing.T) {
	root := t.TempDir()
	rapid.Check(t, func(t *rapid.T) {
		dir, err := os.MkdirTemp(root, "x")
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { os.RemoveAll(dir) })
		bt := newBlockingTool()
		history := rapid.Bool().Draw(t, "history")
		script := []any{fake.ToolUse("", fake.Call("call_1", "echo", `{"text":"hi"}`)), fake.Text("Done.")}
		if history {
			script = append([]any{fake.Text("Hello.")}, script...)
		}
		a := serveIn(t, dir, &agentloop.Agent{Model: fake.Script(script...), Tools: []agentloop.Tool{bt.tool()}}, Options{}, false)
		b := serveIn(t, dir, textAgent("From B."), Options{}, false)

		id := a.newSession()
		if history {
			if _, err := a.prompt(id, "hi"); err != nil {
				t.Fatal(err)
			}
		}
		busy := rapid.Bool().Draw(t, "busy")
		turn := make(chan error, 1)
		if busy {
			go func() {
				_, err := a.prompt(id, "go")
				turn <- err
			}()
			<-bt.started
		}
		disk := tree(t, dir)
		for i, n := 0, rapid.IntRange(1, 6).Draw(t, "attempts"); i < n; i++ {
			from := rapid.SampledFrom([]string{"A", "B"}).Draw(t, "from")
			how := rapid.SampledFrom([]string{"load", "resume"}).Draw(t, "how")
			h := map[string]*harness{"A": a, "B": b}[from]
			var err error
			if how == "load" {
				err = h.load(id)
			} else {
				err = h.resume(id)
			}
			if from == "A" && !busy {
				if err != nil {
					t.Fatalf("A could not take up its idle session: %v", err)
				}
			} else if !inUse(err) {
				t.Fatalf("%s %s of a session open in A (busy %v): %v", from, how, busy, err)
			}
			if path, same := sameTree(disk, tree(t, dir)); !same {
				t.Fatalf("%s %s changed %s", from, how, path)
			}
		}
		if busy {
			close(bt.release)
			if err := <-turn; err != nil {
				t.Fatal(err)
			}
		}
		if _, open := b.srv.State(id); open {
			t.Fatal("B has the session open")
		}
		if err := a.close(id); err != nil {
			t.Fatal(err)
		}
		if err := b.load(id); err != nil {
			t.Fatalf("B cannot load the session A closed: %v", err)
		}
	})
}

// TestPropertyIDsAreSafe is P4.4 over the server. The session directory
// sits beside files a crafted id might reach (a log of the same name one
// level up, another directory's log). Random strings sent as sessionId to
// session/load, session/resume, session/close and session/prompt never
// change, create or lock anything outside session_dir, and only the
// valid ids of logs inside it load.
func TestPropertyIDsAreSafe(t *testing.T) {
	root := t.TempDir()
	rapid.Check(t, func(t *rapid.T) {
		base, err := os.MkdirTemp(root, "p")
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { os.RemoveAll(base) })
		dir := filepath.Join(base, "sessions")
		stored := NewSessionID([16]byte{7})
		outside := NewSessionID([16]byte{8})
		log := `{"seq":1,"time":"2026-10-15T09:00:00Z","type":"run_started","run_started":{"prompt":"hi"}}` + "\n"
		for _, p := range []string{SessionPath(dir, stored), filepath.Join(base, string(outside)+".jsonl"),
			filepath.Join(base, "other", string(outside)+".jsonl"), filepath.Join(base, string(stored)+".jsonl")} {
			os.MkdirAll(filepath.Dir(p), 0o755)
			if err := os.WriteFile(p, []byte(log), 0o600); err != nil {
				t.Fatal(err)
			}
		}
		h := serveIn(t, dir, textAgent(), Options{}, false)
		before := tree(t, base)

		for i, n := 0, rapid.IntRange(1, 5).Draw(t, "n"); i < n; i++ {
			raw := genSessionIDString().Draw(t, "id")
			if rapid.IntRange(0, 4).Draw(t, "pick") == 0 {
				raw = rapid.SampledFrom([]string{string(stored), string(outside), "../" + string(outside), "other/../" + string(stored)}).Draw(t, "known")
			}
			id := acp.SessionId(raw)
			err := h.load(id)
			if id == stored {
				if err != nil {
					t.Fatalf("load of the stored session: %v", err)
				}
				if err := h.close(id); err != nil {
					t.Fatal(err)
				}
			} else if !isRPC(err, -32002, "") {
				// The SDK refuses some strings as params before the
				// server sees them; those never reach a path either.
				if !isRPC(err, -32602, "") {
					t.Fatalf("load %q: %v", id, err)
				}
			}
			if err := h.resume(id); id != stored && !isRPC(err, -32002, "") && !isRPC(err, -32602, "") {
				t.Fatalf("resume %q: %v", id, err)
			} else if id == stored {
				if err != nil {
					t.Fatalf("resume of the stored session: %v", err)
				}
				h.close(id)
			}
			if err := h.close(id); !isRPC(err, -32002, "") && !isRPC(err, -32602, "") {
				t.Fatalf("close %q: %v", id, err)
			}
			if _, err := h.prompt(id, "hi"); !isRPC(err, -32002, "") && !isRPC(err, -32602, "") {
				t.Fatalf("prompt %q: %v", id, err)
			}
		}
		if path, same := sameTree(before, tree(t, base)); !same {
			t.Fatalf("%s changed", path)
		}
		err = filepath.WalkDir(base, func(p string, d fs.DirEntry, err error) error {
			if strings.HasSuffix(p, ".lock") && p != SessionPath(dir, stored)+".lock" {
				return fmt.Errorf("a lock file at %s", p)
			}
			return err
		})
		if err != nil {
			t.Fatal(err)
		}
	})
}

// A server that opens only new sessions refuses to load, resume or list
// any, even one it made on this connection or one left by an earlier
// process, and the log is not touched. A new session works as ever.
func TestNewSessionsOnlyRefusesReopening(t *testing.T) {
	dir := t.TempDir()
	old := serveIn(t, dir, textAgent("Before."), Options{}, false)
	before := old.newSession()
	if _, err := old.prompt(before, "hi"); err != nil {
		t.Fatal(err)
	}
	if err := old.close(before); err != nil {
		t.Fatal(err)
	}
	path := SessionPath(dir, before)
	logBefore, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	h := serveConfigured(t, dir, textAgent("Fresh."), Options{}, false, true)
	id := h.newSession()
	if r, err := h.prompt(id, "hello"); err != nil || r.StopReason != acp.StopReasonEndTurn {
		t.Fatalf("prompt: %+v %v", r, err)
	}
	for _, sid := range []acp.SessionId{before, id} {
		for name, err := range map[string]error{"load": h.load(sid), "resume": h.resume(sid)} {
			var re *acp.RequestError
			if !errors.As(err, &re) || re.Code != -32601 {
				t.Fatalf("%s %s: %v", name, sid, err)
			}
		}
	}
	_, err = h.client.ListSessions(h.ctx(), acp.ListSessionsRequest{})
	var re *acp.RequestError
	if !errors.As(err, &re) || re.Code != -32601 {
		t.Fatalf("list: %v", err)
	}
	if logAfter, _ := os.ReadFile(path); string(logAfter) != string(logBefore) {
		t.Fatal("the old session's log changed")
	}
}
