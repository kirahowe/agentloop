package acpagent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	acp "github.com/caelis-labs/acp-go-sdk"
	"pgregory.net/rapid"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/fake"
	"github.com/kirahowe/agentloop/trifecta"
)

// Only the allow option this agent offered allows; every other answer,
// and no answer, denies.
func TestPropertyPermissionVerdict(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		var resp acp.RequestPermissionResponse
		var err error
		switch rapid.IntRange(0, 3).Draw(t, "shape") {
		case 0:
			id := rapid.SampledFrom([]acp.PermissionOptionId{OptionAllowOnce, OptionRejectOnce, "allow_always", "Allow_once", "", "allow_once "}).Draw(t, "option")
			resp.Outcome = acp.NewRequestPermissionOutcomeSelected(id)
		case 1:
			resp.Outcome = acp.NewRequestPermissionOutcomeCancelled()
		case 2:
			err = errors.New("connection closed")
			resp.Outcome = acp.NewRequestPermissionOutcomeSelected(OptionAllowOnce)
		}
		cancelled := rapid.Bool().Draw(t, "cancelled")
		got := PermissionVerdict(resp, err, cancelled)
		want := agentloop.Deny
		if !cancelled && err == nil && resp.Outcome.Selected != nil && resp.Outcome.Selected.OptionId == OptionAllowOnce {
			want = agentloop.Allow
		}
		if got != want {
			t.Fatalf("%+v, %v gave %s", resp, err, got)
		}
	})
}

func TestPermissionRequestShowsTheWholeInput(t *testing.T) {
	long := `{"body":"` + strings.Repeat("é‮x", 200) + `"}`
	held := agentloop.HeldCall{ToolCall: agentloop.ToolCall{ID: "c1", Name: "post", Input: json.RawMessage(long)}}
	req := PermissionRequest("s1", held, map[string]acp.ToolKind{"post": acp.ToolKindFetch}, []string{"contaminated", "private"})
	if err := req.Validate(); err != nil {
		t.Fatal(err)
	}
	var opts []string
	for _, o := range req.Options {
		opts = append(opts, string(o.OptionId)+"/"+string(o.Kind))
	}
	if !reflect.DeepEqual(opts, []string{"allow_once/allow_once", "reject_once/reject_once"}) {
		t.Fatalf("options %v", opts)
	}
	raw, _ := json.Marshal(req.ToolCall.RawInput)
	if string(raw) != long && !jsonEqual(raw, []byte(long)) {
		t.Fatalf("rawInput %s", raw)
	}
	if string(req.ToolCall.Meta["agentloop/labels"]) != `["contaminated","private"]` || *req.ToolCall.Kind != acp.ToolKindFetch {
		t.Fatalf("%+v", req.ToolCall)
	}
}

func jsonEqual(a, b []byte) bool {
	var x, y any
	json.Unmarshal(a, &x)
	json.Unmarshal(b, &y)
	return reflect.DeepEqual(x, y)
}

// holdAgent posts once; the session starts private and contaminated, so
// the egress call is held.
func holdAgent(ran *[]string) *agentloop.Agent {
	post := agentloop.Func(agentloop.ToolSpec{Name: "post"}, func(_ context.Context, in json.RawMessage) (agentloop.Result, error) {
		*ran = append(*ran, string(in))
		return agentloop.Result{Content: "posted"}, nil
	})
	n := 0
	return &agentloop.Agent{
		Model: fake.New(func(req *agentloop.Request) (*agentloop.Response, error) {
			last := req.Messages[len(req.Messages)-1]
			if p := last.Parts[0]; p.Type == agentloop.ToolResultPart {
				return fake.Text("result: " + p.ToolResult.Content), nil
			}
			n++
			return fake.ToolUse("", fake.Call(fmt.Sprintf("c%d", n), "post", `{"body":"secret"}`)), nil
		}),
		ModelID: fake.ModelID,
		Tools:   []agentloop.Tool{post},
		Labels:  []string{trifecta.Private, trifecta.Contaminated},
		Policy:  trifecta.Policy{Tools: map[string]trifecta.Tool{"post": {Class: trifecta.Egress, Labels: []string{trifecta.Private}}}},
	}
}

func lastResult(t *testing.T, h *harness, id acp.SessionId) agentloop.ToolResultEntry {
	t.Helper()
	h.srv.mu.Lock()
	ss := h.srv.sessions[id]
	h.srv.mu.Unlock()
	var res *agentloop.ToolResultEntry
	for _, e := range ss.sess.Entries() {
		if e.Type == agentloop.EntryToolResult {
			res = e.ToolResult
		}
	}
	if res == nil {
		t.Fatal("no result")
	}
	return *res
}

func TestHeldCallAskedAndAllowed(t *testing.T) {
	var ran []string
	h := serve(t, holdAgent(&ran), Options{}, false)
	h.cl.permit = func(context.Context, acp.RequestPermissionRequest) (acp.RequestPermissionResponse, error) {
		return acp.RequestPermissionResponse{Outcome: acp.NewRequestPermissionOutcomeSelected(OptionAllowOnce)}, nil
	}
	id := h.newSession()
	resp, err := h.prompt(id, "post it")
	if err != nil || resp.StopReason != acp.StopReasonEndTurn {
		t.Fatalf("%+v %v", resp, err)
	}
	if len(ran) != 1 || len(h.cl.asked) != 1 {
		t.Fatalf("ran %v, asked %d", ran, len(h.cl.asked))
	}
	asked := h.cl.asked[0]
	if asked.SessionId != id || *asked.ToolCall.Name != "post" || string(asked.ToolCall.Meta["agentloop/labels"]) != `["contaminated","private"]` {
		t.Fatalf("asked %+v", asked)
	}
	if r := lastResult(t, h, id); r.Result.Content != "posted" {
		t.Fatalf("result %+v", r)
	}
	// The prompt is in the conversation once.
	h.srv.mu.Lock()
	msgs := h.srv.sessions[id].sess.Messages()
	h.srv.mu.Unlock()
	n := 0
	for _, m := range msgs {
		if m.Role == agentloop.User && m.Text() == "post it" {
			n++
		}
	}
	if n != 1 {
		t.Fatalf("the prompt is in the conversation %d times", n)
	}
}

func TestHeldCallRejectedOrAnsweredWrongly(t *testing.T) {
	for _, answer := range []acp.RequestPermissionOutcome{
		acp.NewRequestPermissionOutcomeSelected(OptionRejectOnce),
		acp.NewRequestPermissionOutcomeSelected("allow_always"),
		acp.NewRequestPermissionOutcomeCancelled(),
	} {
		var ran []string
		h := serve(t, holdAgent(&ran), Options{}, false)
		h.cl.permit = func(context.Context, acp.RequestPermissionRequest) (acp.RequestPermissionResponse, error) {
			return acp.RequestPermissionResponse{Outcome: answer}, nil
		}
		id := h.newSession()
		resp, err := h.prompt(id, "post it")
		if err != nil || resp.StopReason != acp.StopReasonEndTurn {
			t.Fatalf("%+v %v", resp, err)
		}
		if len(ran) != 0 {
			t.Fatalf("%+v: the call ran", answer)
		}
		if r := lastResult(t, h, id); r.Result.Content != NotApproved || !r.Result.IsError {
			t.Fatalf("result %+v", r)
		}
	}
}

// A cancel while the client is being asked denies the call and ends the
// turn cancelled.
func TestCancelWhileAskingDenies(t *testing.T) {
	var ran []string
	h := serve(t, holdAgent(&ran), Options{}, false)
	asking := make(chan struct{})
	h.cl.permit = func(ctx context.Context, _ acp.RequestPermissionRequest) (acp.RequestPermissionResponse, error) {
		close(asking)
		<-ctx.Done()
		return acp.RequestPermissionResponse{}, ctx.Err()
	}
	id := h.newSession()
	go func() {
		<-asking
		h.client.Cancel(context.Background(), acp.CancelNotification{SessionId: id})
	}()
	resp, err := h.prompt(id, "post it")
	if err != nil || resp.StopReason != acp.StopReasonCancelled {
		t.Fatalf("%+v %v", resp, err)
	}
	if len(ran) != 0 {
		t.Fatal("the call ran")
	}
	h.srv.mu.Lock()
	ss := h.srv.sessions[id]
	h.srv.mu.Unlock()
	var d *agentloop.DecisionEntry
	for _, e := range ss.sess.Entries() {
		if e.Type == agentloop.EntryDecision {
			d = e.Decision
		}
	}
	if d == nil || !d.Resolved || d.Verdict != agentloop.Deny {
		t.Fatalf("decision %+v", d)
	}
}

// The request's starting labels are in a session's log from the moment it
// is opened, new or loaded, and a prompt with a resource contaminates it.
func TestSeedAndPromptLabels(t *testing.T) {
	var ran []string
	a := holdAgent(&ran)
	a.Labels = []string{trifecta.Private}
	a.Model = fake.New(func(*agentloop.Request) (*agentloop.Response, error) { return fake.Text("ok"), nil })
	h := serve(t, a, Options{}, false)
	id := h.newSession()
	labels := func() []string {
		h.srv.mu.Lock()
		defer h.srv.mu.Unlock()
		return h.srv.sessions[id].sess.Labels()
	}
	if got := labels(); !reflect.DeepEqual(got, []string{"private"}) {
		t.Fatalf("labels %v", got)
	}
	_, err := h.client.Prompt(h.ctx(), acp.PromptRequest{SessionId: id, Prompt: []acp.ContentBlock{
		acp.TextBlock("read this"), acp.ResourceLinkBlock("page", "https://example.com/x"),
	}})
	if err != nil {
		t.Fatal(err)
	}
	if got := labels(); !slices.Equal(got, []string{"contaminated", "private"}) {
		t.Fatalf("labels %v", got)
	}
}

func TestPromptLabels(t *testing.T) {
	if PromptLabels([]acp.ContentBlock{acp.TextBlock("hi")}) != nil {
		t.Error("text contaminated the session")
	}
	if got := PromptLabels([]acp.ContentBlock{acp.TextBlock("hi"), acp.ResourceLinkBlock("n", "file:///x")}); !slices.Equal(got, []string{"contaminated"}) {
		t.Errorf("resource link: %v", got)
	}
	res := acp.ContentBlock{Resource: &acp.ContentBlockResource{Resource: acp.EmbeddedResourceResource{
		TextResourceContents: &acp.TextResourceContents{Uri: "file:///x", Text: "t"},
	}}}
	if got := PromptLabels([]acp.ContentBlock{res}); !slices.Equal(got, []string{"contaminated"}) {
		t.Errorf("resource: %v", got)
	}
}

// A session loaded by a server whose request has starting labels takes
// them on as it is loaded, before any prompt.
func TestLoadSeedsLabels(t *testing.T) {
	dir := t.TempDir()
	quiet := &agentloop.Agent{Model: fake.Script(), ModelID: fake.ModelID}
	h := serveIn(t, dir, quiet, Options{}, false)
	id := h.newSession()
	if _, err := h.client.CloseSession(h.ctx(), acp.CloseSessionRequest{SessionId: id}); err != nil {
		t.Fatal(err)
	}
	seeded := &agentloop.Agent{Model: fake.Script(), ModelID: fake.ModelID, Labels: []string{trifecta.Contaminated}}
	h2 := serveIn(t, dir, seeded, Options{}, false)
	if _, err := h2.client.LoadSession(h2.ctx(), acp.LoadSessionRequest{SessionId: id, Cwd: "/", McpServers: []acp.McpServer{}}); err != nil {
		t.Fatal(err)
	}
	s, err := agentloop.OpenSession(agentloop.NewFileStore(SessionPath(dir, id)))
	if err != nil {
		t.Fatal(err)
	}
	if got := s.Labels(); !slices.Equal(got, []string{"contaminated"}) {
		t.Fatalf("labels %v", got)
	}
}

// Regression: an allow that arrives after the turn was cancelled must not
// count; the SDK can hand back the client's answer after the cancel.
func TestAllowAfterCancelIsADeny(t *testing.T) {
	var ran []string
	h := serve(t, holdAgent(&ran), Options{}, false)
	var id acp.SessionId
	h.cl.permit = func(ctx context.Context, _ acp.RequestPermissionRequest) (acp.RequestPermissionResponse, error) {
		h.client.Cancel(context.Background(), acp.CancelNotification{SessionId: id})
		deadline := time.Now().Add(5 * time.Second)
		for st, _ := h.srv.State(id); st != Cancelling && time.Now().Before(deadline); st, _ = h.srv.State(id) {
			time.Sleep(100 * time.Microsecond)
		}
		return acp.RequestPermissionResponse{Outcome: acp.NewRequestPermissionOutcomeSelected(OptionAllowOnce)}, nil
	}
	id = h.newSession()
	resp, err := h.prompt(id, "post it")
	if err != nil || resp.StopReason != acp.StopReasonCancelled {
		t.Fatalf("%+v %v", resp, err)
	}
	h.srv.mu.Lock()
	ss := h.srv.sessions[id]
	h.srv.mu.Unlock()
	info, _ := ss.sess.Call("c1")
	if len(ran) != 0 || !info.Resolved || info.Decision.Verdict != agentloop.Deny {
		t.Fatalf("ran %v, call %+v", ran, info)
	}
}

// Regression: a server for an unrestricted request refuses to load or
// resume a session a policy has marked.
func TestUnrestrictedServerRefusesAPolicedSession(t *testing.T) {
	dir := t.TempDir()
	var ran []string
	h := serveIn(t, dir, holdAgent(&ran), Options{}, false)
	id := h.newSession()
	if _, err := h.client.CloseSession(h.ctx(), acp.CloseSessionRequest{SessionId: id}); err != nil {
		t.Fatal(err)
	}
	h2 := serveIn(t, dir, &agentloop.Agent{Model: fake.Script(), ModelID: fake.ModelID}, Options{}, false)
	h2.srv.cfg.Unrestricted = true
	if err := h2.load(id); err == nil {
		t.Fatal("loaded a policed session without a policy")
	}
	if _, err := h2.client.ResumeSession(h2.ctx(), acp.ResumeSessionRequest{SessionId: id, Cwd: "/"}); err == nil {
		t.Fatal("resumed a policed session without a policy")
	}
}

// Regression: the SDK can return the client's answer after the turn was
// cancelled; an allow then is a deny.
func TestAllowAnswerAfterCancelDenies(t *testing.T) {
	resp := acp.RequestPermissionResponse{Outcome: acp.NewRequestPermissionOutcomeSelected(OptionAllowOnce)}
	if PermissionVerdict(resp, nil, true) != agentloop.Deny {
		t.Fatal("an allow after a cancel counted")
	}
	if PermissionVerdict(resp, nil, false) != agentloop.Allow {
		t.Fatal("an allow did not count")
	}
}

// Regression: a label change with no tool call or usage update (a seed, a
// resource in the prompt) was invisible to a client. It now comes as a
// session_info_update with the labels in _meta.
func TestLabelChangesAreShown(t *testing.T) {
	a := &agentloop.Agent{Model: fake.New(func(*agentloop.Request) (*agentloop.Response, error) { return fake.Text("ok"), nil }),
		ModelID: fake.ModelID, Labels: []string{trifecta.Private}}
	h := serve(t, a, Options{}, false)
	id := h.newSession()
	if _, err := h.client.Prompt(h.ctx(), acp.PromptRequest{SessionId: id, Prompt: []acp.ContentBlock{
		acp.TextBlock("read this"), acp.ResourceLinkBlock("page", "https://example.com/x"),
	}}); err != nil {
		t.Fatal(err)
	}
	var shown [][]string
	for _, n := range h.cl.notes() {
		if u := n.Update.SessionInfoUpdate; u != nil {
			var l []string
			json.Unmarshal(u.Meta["agentloop/labels"], &l)
			shown = append(shown, l)
		}
	}
	want := [][]string{{"private"}, {"contaminated", "private"}}
	if !reflect.DeepEqual(shown, want) {
		t.Fatalf("labels shown %v, want %v", shown, want)
	}
}
