package main

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/kirahowe/agentloop/internal/acpschema"
)

// responseDefs maps a request method to the schema definition its result
// must satisfy.
var responseDefs = map[string]string{
	"initialize":     "InitializeResponse",
	"session/new":    "NewSessionResponse",
	"session/load":   "LoadSessionResponse",
	"session/resume": "ResumeSessionResponse",
	"session/list":   "ListSessionsResponse",
	"session/close":  "CloseSessionResponse",
	"session/prompt": "PromptResponse",
}

func validateAs(t *testing.T, def string, v any) {
	t.Helper()
	if err := acpschema.Validate(def, v); err != nil {
		t.Error(err)
	}
}

// awaitResponse waits for the response whose id is raw and returns its line.
func (p *acpProc) awaitResponse(raw string) []byte {
	p.t.Helper()
	deadline := time.After(15 * time.Second)
	seen := 0
	for {
		p.mu.Lock()
		lines, more := p.lines, p.more
		p.mu.Unlock()
		for ; seen < len(lines); seen++ {
			var l struct {
				ID     json.RawMessage `json:"id"`
				Method string          `json:"method"`
			}
			if json.Unmarshal(lines[seen], &l) == nil && l.Method == "" && string(l.ID) == raw {
				return lines[seen]
			}
		}
		select {
		case <-more:
		case <-p.exited:
			p.t.Fatalf("agentloop exited %d waiting for response %s; stderr: %s", p.code, raw, p.stderr.String())
		case <-deadline:
			p.t.Fatalf("no response %s; stderr: %s", raw, p.stderr.String())
		}
	}
}

// rateLimited is a retryable failure: the first model call of a replay gets
// it, so the loop emits a retry notice that a client without
// session.notices must not see.
func rateLimited(w http.ResponseWriter) {
	w.Header().Set("Retry-After", "0")
	http.Error(w, `{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`, 429)
}

// TestClientTraces replays the client half of each recorded session in
// acp/testdata/clients against the real binary.
func TestClientTraces(t *testing.T) {
	files, err := filepath.Glob("../../acp/testdata/clients/*.jsonl")
	if err != nil || len(files) == 0 {
		t.Fatalf("no client traces: %v", err)
	}
	for _, file := range files {
		t.Run(strings.TrimSuffix(filepath.Base(file), ".jsonl"), func(t *testing.T) {
			replayTrace(t, file)
		})
	}
}

func replayTrace(t *testing.T, file string) {
	raw, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	api := newFakeAPI(t, rateLimited, toolReply("toolu_1", "probe", `{"q":"hello"}`), textReply("all done"))
	dir := t.TempDir()
	r := newACPReq(t, dir, api.URL, probeScript, keycmdBinary(t))
	r.req["retry"] = map[string]any{"retries": 1}
	p := startACP(t, r.write(t))

	var sessionID string
	methods := map[string]string{} // raw request id -> method
	var order []string
	var stopReasons []string
	sawNotices := false

	for n, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		var msg map[string]any
		if err := json.Unmarshal([]byte(line), &msg); err != nil {
			t.Fatalf("trace line %d: %v", n+1, err)
		}
		method, _ := msg["method"].(string)
		if params, ok := msg["params"].(map[string]any); ok {
			if _, has := params["sessionId"]; has && sessionID != "" {
				params["sessionId"] = sessionID
			}
			if method == "initialize" {
				if caps, ok := params["clientCapabilities"].(map[string]any); ok {
					if s, ok := caps["session"].(map[string]any); ok {
						_, sawNotices = s["notices"]
					}
				}
			}
		}
		rawID, hasID := msg["id"]
		idJSON, _ := json.Marshal(rawID)
		if hasID {
			if _, dup := methods[string(idJSON)]; dup {
				t.Fatalf("trace line %d reuses request id %s", n+1, idJSON)
			}
			methods[string(idJSON)] = method
			order = append(order, string(idJSON))
		}
		p.send(msg)
		if !hasID {
			continue
		}
		resp := p.awaitResponse(string(idJSON))
		var parsed struct {
			Result json.RawMessage `json:"result"`
			Error  json.RawMessage `json:"error"`
		}
		json.Unmarshal(resp, &parsed)
		if parsed.Error != nil {
			t.Fatalf("%s: error response %s", method, parsed.Error)
		}
		switch method {
		case "session/new":
			var res struct {
				SessionID string `json:"sessionId"`
			}
			json.Unmarshal(parsed.Result, &res)
			sessionID = res.SessionID
		case "session/prompt":
			var res struct {
				StopReason string `json:"stopReason"`
			}
			json.Unmarshal(parsed.Result, &res)
			stopReasons = append(stopReasons, res.StopReason)
		}
	}
	p.stdin.Close()
	if code, ok := p.wait(5 * time.Second); !ok || code != 0 {
		t.Fatalf("exit %d (exited %v); stderr: %s", code, ok, p.stderr.String())
	}

	if n := len(api.requests()); n != 3 {
		t.Errorf("%d model calls, want 3 (a failure, a tool call, an answer)", n)
	}

	// Nothing but JSON-RPC on stdout.
	p.checkFraming(t)

	for _, s := range stopReasons {
		if s != "end_turn" {
			t.Errorf("stop reasons %v, want all end_turn", stopReasons)
		}
	}
	if len(stopReasons) == 0 {
		t.Error("the trace holds no prompt")
	}

	responses := map[string]int{}
	p.mu.Lock()
	lines := append([][]byte(nil), p.lines...)
	p.mu.Unlock()
	for _, line := range lines {
		var m struct {
			ID     json.RawMessage `json:"id"`
			Method string          `json:"method"`
			Params map[string]any  `json:"params"`
			Result any             `json:"result"`
		}
		if err := json.Unmarshal(line, &m); err != nil {
			t.Fatalf("%q: %v", line, err)
		}
		if m.Method != "" {
			// The agent may notify with session/update and nothing else:
			// fs, terminal and elicitation are advertised but unused.
			if m.Method != "session/update" || m.ID != nil {
				t.Errorf("the agent sent %s to the client: %s", m.Method, line)
				continue
			}
			validateAs(t, "SessionNotification", m.Params)
			if u, _ := m.Params["update"].(map[string]any); u != nil && !sawNotices && u["sessionUpdate"] == "notice" {
				t.Errorf("a notice went to a client without session.notices: %s", line)
			}
			continue
		}
		id := string(m.ID)
		method, ok := methods[id]
		if !ok {
			t.Errorf("a response to no request: %s", line)
			continue
		}
		responses[id]++
		if def, ok := responseDefs[method]; ok {
			validateAs(t, def, m.Result)
		}
	}
	for _, id := range order {
		if responses[id] != 1 {
			t.Errorf("request %s (%s) got %d responses, want 1", id, methods[id], responses[id])
		}
	}
}
