package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"agentloop"
)

type server struct {
	*httptest.Server
	mu     sync.Mutex
	bodies []string
	hdrs   []http.Header
	hits   int
}

func (s *server) lastBody(t *testing.T) map[string]any {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	var m map[string]any
	if err := json.Unmarshal([]byte(s.bodies[len(s.bodies)-1]), &m); err != nil {
		t.Fatal(err)
	}
	return m
}

func newServer(t *testing.T, h http.HandlerFunc) *server {
	s := &server{}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		s.mu.Lock()
		s.hits++
		s.bodies = append(s.bodies, string(b))
		s.hdrs = append(s.hdrs, r.Header.Clone())
		s.mu.Unlock()
		h(w, r)
	}))
	t.Cleanup(s.Close)
	return s
}

func sse(w http.ResponseWriter, events ...string) {
	w.Header().Set("Content-Type", "text/event-stream")
	for _, e := range events {
		var h struct {
			Type string `json:"type"`
		}
		json.Unmarshal([]byte(e), &h)
		if h.Type != "" {
			fmt.Fprintf(w, "event: %s\n", h.Type)
		}
		fmt.Fprintf(w, "data: %s\n\n", e)
	}
}

func respJSON(status, output, extra string) string {
	return fmt.Sprintf(`{"id":"resp_1","object":"response","created_at":1,"status":%q,"model":"gpt-served","output":%s,"parallel_tool_calls":true,"tool_choice":"auto","tools":[]%s}`, status, output, extra)
}

const usageJSON = `,"usage":{"input_tokens":100,"input_tokens_details":{"cached_tokens":30},"output_tokens":20,"output_tokens_details":{"reasoning_tokens":5},"total_tokens":120}`

func completed(output string) []string {
	return []string{
		`{"type":"response.created","sequence_number":0,"response":` + respJSON("in_progress", "[]", "") + `}`,
		`{"type":"response.completed","sequence_number":9,"response":` + respJSON("completed", output, usageJSON) + `}`,
	}
}

func model(s *server) *Model {
	return New(Config{APIKey: "sk-test", BaseURL: s.URL, HTTPClient: s.Client()})
}

func gen(t *testing.T, m *Model, req *agentloop.Request) (*agentloop.Response, []agentloop.Delta) {
	t.Helper()
	var ds []agentloop.Delta
	if req.Model == "" {
		req.Model = "gpt-test"
	}
	r, err := m.Generate(context.Background(), req, func(d agentloop.Delta) { ds = append(ds, d) })
	if err != nil {
		t.Fatal(err)
	}
	return r, ds
}

func TestRequestShape(t *testing.T) {
	s := newServer(t, func(w http.ResponseWriter, r *http.Request) { sse(w, completed("[]")...) })
	schema := `{"type":"object","properties":{"b":{"type":"string"},"a":{"type":"integer"}},"required":["a"],"additionalProperties":true}`
	m := New(Config{APIKey: "sk-test", BaseURL: s.URL, HTTPClient: s.Client(), DefaultMaxTokens: 777})
	gen(t, m, &agentloop.Request{
		Model: "gpt-x", System: "be brief", Effort: agentloop.EffortHigh,
		Messages: []agentloop.Message{agentloop.Text(agentloop.User, "hi")},
		Tools:    []agentloop.ToolSpec{{Name: "t", Description: "d", Parameters: json.RawMessage(schema)}},
	})
	b := s.lastBody(t)
	if b["store"] != false || b["stream"] != true || b["model"] != "gpt-x" || b["instructions"] != "be brief" {
		t.Fatalf("body %v", b)
	}
	if inc := b["include"].([]any); len(inc) != 1 || inc[0] != "reasoning.encrypted_content" {
		t.Fatalf("include %v", inc)
	}
	if r := b["reasoning"].(map[string]any); r["effort"] != "high" || r["summary"] != "auto" {
		t.Fatalf("reasoning %v", r)
	}
	if b["max_output_tokens"] != float64(777) {
		t.Fatalf("max %v", b["max_output_tokens"])
	}
	tool := b["tools"].([]any)[0].(map[string]any)
	if tool["type"] != "function" || tool["name"] != "t" || tool["description"] != "d" || tool["strict"] != false {
		t.Fatalf("tool %v", tool)
	}
	// The schema must be byte-identical in the request.
	if !strings.Contains(s.bodies[0], `"parameters":`+schema) {
		t.Fatalf("schema altered: %s", s.bodies[0])
	}
	in := b["input"].([]any)[0].(map[string]any)
	if in["role"] != "user" || in["content"] != "hi" {
		t.Fatalf("input %v", in)
	}
	if s.hdrs[0].Get("Authorization") != "Bearer sk-test" {
		t.Fatalf("auth %v", s.hdrs[0])
	}
}

func TestEffortAndMaxTokens(t *testing.T) {
	s := newServer(t, func(w http.ResponseWriter, r *http.Request) { sse(w, completed("[]")...) })
	m := model(s)
	for e, want := range map[agentloop.Effort]string{
		agentloop.EffortLow: "low", agentloop.EffortMedium: "medium", agentloop.EffortHigh: "high",
		agentloop.EffortXHigh: "xhigh", agentloop.EffortMax: "max",
	} {
		gen(t, m, &agentloop.Request{Effort: e, MaxOutputTokens: 55})
		b := s.lastBody(t)
		if got := b["reasoning"].(map[string]any)["effort"]; got != want {
			t.Errorf("%s -> %v", e, got)
		}
		if b["max_output_tokens"] != float64(55) {
			t.Errorf("max %v", b["max_output_tokens"])
		}
	}
	gen(t, m, &agentloop.Request{})
	b := s.lastBody(t)
	if _, ok := b["max_output_tokens"]; ok {
		t.Error("max_output_tokens should be omitted")
	}
	if _, ok := b["reasoning"].(map[string]any)["effort"]; ok {
		t.Error("effort should be omitted")
	}
}

const reasoningItem = `{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"thinking A"},{"type":"summary_text","text":"thinking B"}],"encrypted_content":"gAAAA+/==opaque"}`
const messageItem = `{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","annotations":[],"text":"Hello there"}]}`
const callItem = `{"id":"fc_1","type":"function_call","status":"completed","call_id":"call_9","name":"lookup","arguments":"{\"q\":\"x\"}"}`

func TestTextReasoningAndToolCall(t *testing.T) {
	out := "[" + reasoningItem + "," + messageItem + "," + callItem + "]"
	s := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		sse(w,
			`{"type":"response.created","sequence_number":0,"response":`+respJSON("in_progress", "[]", "")+`}`,
			`{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[]}}`,
			`{"type":"response.reasoning_summary_part.added","sequence_number":2,"item_id":"rs_1","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":""}}`,
			`{"type":"response.reasoning_summary_text.delta","sequence_number":3,"item_id":"rs_1","output_index":0,"summary_index":0,"delta":"thinking A"}`,
			`{"type":"response.reasoning_summary_part.added","sequence_number":4,"item_id":"rs_1","output_index":0,"summary_index":1,"part":{"type":"summary_text","text":""}}`,
			`{"type":"response.reasoning_summary_text.delta","sequence_number":5,"item_id":"rs_1","output_index":0,"summary_index":1,"delta":"thinking B"}`,
			`{"type":"response.output_item.done","sequence_number":6,"output_index":0,"item":`+reasoningItem+`}`,
			`{"type":"response.output_item.added","sequence_number":7,"output_index":1,"item":{"id":"msg_1","type":"message","status":"in_progress","role":"assistant","content":[]}}`,
			`{"type":"response.output_text.delta","sequence_number":8,"item_id":"msg_1","output_index":1,"content_index":0,"delta":"Hello ","logprobs":[]}`,
			`{"type":"response.output_text.delta","sequence_number":9,"item_id":"msg_1","output_index":1,"content_index":0,"delta":"there","logprobs":[]}`,
			`{"type":"response.output_item.done","sequence_number":10,"output_index":1,"item":`+messageItem+`}`,
			`{"type":"response.function_call_arguments.delta","sequence_number":11,"item_id":"fc_1","output_index":2,"delta":"{}"}`,
			`{"type":"response.output_item.done","sequence_number":12,"output_index":2,"item":`+callItem+`}`,
			`{"type":"response.completed","sequence_number":13,"response":`+respJSON("completed", out, usageJSON)+`}`)
	})
	r, ds := gen(t, model(s), &agentloop.Request{})
	wantD := []agentloop.Delta{
		{Type: agentloop.ReasoningPart, Text: "thinking A"},
		{Type: agentloop.ReasoningPart, Text: "\n\n"},
		{Type: agentloop.ReasoningPart, Text: "thinking B"},
		{Type: agentloop.TextPart, Text: "Hello "},
		{Type: agentloop.TextPart, Text: "there"},
	}
	if fmt.Sprint(ds) != fmt.Sprint(wantD) {
		t.Fatalf("deltas %v", ds)
	}
	p := r.Message.Parts
	if len(p) != 3 || p[0].Type != agentloop.ReasoningPart || p[0].Text != "thinking A\n\nthinking B" ||
		p[1].Type != agentloop.TextPart || p[1].Text != "Hello there" || p[2].Type != agentloop.ToolCallPart {
		t.Fatalf("parts %+v", p)
	}
	tc := p[2].ToolCall
	if tc.ID != "call_9" || tc.Name != "lookup" || string(tc.Input) != `{"q":"x"}` {
		t.Fatalf("call %+v", tc)
	}
	if r.Stop != agentloop.StopToolUse || r.Model != "gpt-served" {
		t.Fatalf("stop %v model %v", r.Stop, r.Model)
	}
	if r.Usage != (agentloop.Usage{InputTokens: 70, OutputTokens: 20, CacheReadTokens: 30}) {
		t.Fatalf("usage %+v", r.Usage)
	}
	n := r.Message.Native
	if n.Provider != "openai" || n.Model != "gpt-served" {
		t.Fatalf("native %+v", n)
	}
	var items []json.RawMessage
	if err := json.Unmarshal(n.Data, &items); err != nil || len(items) != 3 || string(items[0]) != reasoningItem {
		t.Fatalf("data %s err %v", n.Data, err)
	}
}

func TestEndTurnAndNilObserver(t *testing.T) {
	s := newServer(t, func(w http.ResponseWriter, r *http.Request) { sse(w, completed("["+messageItem+"]")...) })
	r, err := model(s).Generate(context.Background(), &agentloop.Request{Model: "m"}, nil)
	if err != nil || r.Stop != agentloop.StopEndTurn || r.Message.Text() != "Hello there" {
		t.Fatalf("%v %+v", err, r)
	}
}

func TestNativeReplayIsByteFaithful(t *testing.T) {
	s := newServer(t, func(w http.ResponseWriter, r *http.Request) { sse(w, completed("[]")...) })
	// Odd key order and unicode must survive unchanged.
	odd := `{"type":"reasoning","id":"rs_7","encrypted_content":"AbC+/=é\\n","summary":[],"future_field":{"z":1,"a":[1,2]}}`
	msgItem := `{"type":"message","id":"msg_7","role":"assistant","status":"completed","content":[{"type":"output_text","text":"yo","annotations":[]}]}`
	data := "[" + odd + "," + msgItem + "," + callItem + "]"
	history := []agentloop.Message{
		agentloop.Text(agentloop.User, "q"),
		{Role: agentloop.Assistant,
			Parts:  []agentloop.Part{{Type: agentloop.TextPart, Text: "IGNORED when native present"}},
			Native: &agentloop.Native{Provider: "openai", Model: "some-other-model", Data: json.RawMessage(data)}},
		{Role: agentloop.User, Parts: []agentloop.Part{{Type: agentloop.ToolResultPart, ToolResult: &agentloop.ToolResult{CallID: "call_9", Content: "ok"}}}},
	}
	gen(t, model(s), &agentloop.Request{Messages: history})
	body := s.bodies[0]
	for _, it := range []string{odd, msgItem, callItem} {
		if !strings.Contains(body, it) {
			t.Fatalf("item not replayed byte for byte: %s\nin %s", it, body)
		}
	}
	if strings.Contains(body, "IGNORED") {
		t.Fatal("parts rendered despite native")
	}
	in := s.lastBody(t)["input"].([]any)
	if len(in) != 5 {
		t.Fatalf("input items %d", len(in))
	}
}

func TestForeignAssistantAndToolResults(t *testing.T) {
	s := newServer(t, func(w http.ResponseWriter, r *http.Request) { sse(w, completed("[]")...) })
	history := []agentloop.Message{
		agentloop.Text(agentloop.User, "q"),
		{Role: agentloop.Assistant,
			Parts: []agentloop.Part{
				{Type: agentloop.ReasoningPart, Text: "secret thoughts"},
				{Type: agentloop.TextPart, Text: "calling"},
				{Type: agentloop.ToolCallPart, ToolCall: &agentloop.ToolCall{ID: "c1", Name: "n", Input: json.RawMessage(`{"a":1}`)}},
				{Type: agentloop.ToolCallPart, ToolCall: &agentloop.ToolCall{ID: "c2", Name: "m"}},
			},
			Native: &agentloop.Native{Provider: "anthropic", Model: "claude", Data: json.RawMessage(`[{"type":"thinking"}]`)}},
		{Role: agentloop.User, Parts: []agentloop.Part{
			{Type: agentloop.ToolResultPart, ToolResult: &agentloop.ToolResult{CallID: "c1", Content: "fine"}},
			{Type: agentloop.ToolResultPart, ToolResult: &agentloop.ToolResult{CallID: "c2", Content: "boom", IsError: true}},
			{Type: agentloop.TextPart, Text: "continue"},
		}},
	}
	gen(t, model(s), &agentloop.Request{Messages: history})
	if strings.Contains(s.bodies[0], "secret thoughts") || strings.Contains(s.bodies[0], "thinking") {
		t.Fatal("foreign reasoning leaked")
	}
	var got []map[string]any
	for _, x := range s.lastBody(t)["input"].([]any) {
		got = append(got, x.(map[string]any))
	}
	want := []map[string]any{
		{"type": "message", "role": "user", "content": "q"},
		{"type": "message", "role": "assistant", "content": "calling"},
		{"type": "function_call", "call_id": "c1", "name": "n", "arguments": `{"a":1}`},
		{"type": "function_call", "call_id": "c2", "name": "m", "arguments": "{}"},
		{"type": "function_call_output", "call_id": "c1", "output": "fine"},
		{"type": "function_call_output", "call_id": "c2", "output": "Error: boom"},
		{"type": "message", "role": "user", "content": "continue"},
	}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("got  %v\nwant %v", got, want)
	}
}

func TestIncompleteAndRefusal(t *testing.T) {
	for _, c := range []struct {
		name, reason, output string
		stop                 agentloop.Stop
		refusal              string
	}{
		{"max", "max_output_tokens", "[" + messageItem + "]", agentloop.StopMaxTokens, ""},
		{"filter", "content_filter", "[]", agentloop.StopRefusal, ""},
		{"refusal", "", `[{"id":"msg_r","type":"message","status":"completed","role":"assistant","content":[{"type":"refusal","refusal":"I can't help"}]}]`, agentloop.StopRefusal, "I can't help"},
	} {
		t.Run(c.name, func(t *testing.T) {
			s := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				if c.reason == "" {
					sse(w, completed(c.output)...)
					return
				}
				sse(w, `{"type":"response.incomplete","sequence_number":1,"response":`+
					respJSON("incomplete", c.output, `,"incomplete_details":{"reason":"`+c.reason+`"}`)+`}`)
			})
			r, _ := gen(t, model(s), &agentloop.Request{})
			if r.Stop != c.stop || r.Refusal != c.refusal {
				t.Fatalf("%v %q", r.Stop, r.Refusal)
			}
			if c.name == "refusal" && r.Message.Text() != "I can't help" {
				t.Fatalf("text %q", r.Message.Text())
			}
		})
	}
}

func TestErrorClassification(t *testing.T) {
	cases := []struct {
		name    string
		status  int
		code    string
		typ     string
		headers map[string]string
		kind    agentloop.ErrorKind
	}{
		{"quota", 429, "insufficient_quota", "insufficient_quota", nil, agentloop.Billing},
		{"quota type only", 429, "", "insufficient_quota", nil, agentloop.Billing},
		{"rate", 429, "rate_limit_exceeded", "requests", map[string]string{"retry-after-ms": "1500", "x-ratelimit-reset-requests": "1s", "x-ratelimit-reset-tokens": "6m0s"}, agentloop.RateLimited},
		{"401", 401, "invalid_api_key", "invalid_request_error", nil, agentloop.Unauthorized},
		{"403", 403, "", "", nil, agentloop.Unauthorized},
		{"404", 404, "model_not_found", "invalid_request_error", nil, agentloop.NotFound},
		{"500", 500, "server_error", "server_error", nil, agentloop.Overloaded},
		{"503", 503, "", "", nil, agentloop.Overloaded},
		{"400", 400, "", "invalid_request_error", nil, agentloop.InvalidRequest},
		{"422", 422, "", "", nil, agentloop.InvalidRequest},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				for k, v := range c.headers {
					w.Header().Set(k, v)
				}
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(c.status)
				fmt.Fprintf(w, `{"error":{"message":"msg here","type":%q,"code":%q,"param":null}}`, c.typ, c.code)
			})
			before := time.Now()
			_, err := model(s).Generate(context.Background(), &agentloop.Request{Model: "m"}, nil)
			var me *agentloop.ModelError
			if !errors.As(err, &me) {
				t.Fatalf("not a ModelError: %v", err)
			}
			if me.Kind != c.kind || me.Status != c.status || me.Err == nil || !strings.Contains(me.Message, "msg here") {
				t.Fatalf("%+v", me)
			}
			if s.hits != 1 {
				t.Fatalf("server hit %d times", s.hits)
			}
			if c.name == "rate" {
				if d := me.RetryAfter.Sub(before); d < 1400*time.Millisecond || d > 3*time.Second {
					t.Errorf("retry after %v", d)
				}
				if d := me.ResetsAt.Sub(before); d < 5*time.Minute+55*time.Second || d > 6*time.Minute+10*time.Second {
					t.Errorf("resets in %v", d)
				}
			} else if !me.RetryAfter.IsZero() || !me.ResetsAt.IsZero() {
				t.Errorf("unexpected times %+v", me)
			}
		})
	}
}

func TestRetryAfterSecondsAndReset(t *testing.T) {
	h := http.Header{}
	h.Set("retry-after", "7")
	h.Set("x-ratelimit-reset-requests", "20ms")
	now := time.Now()
	if got := retryAfter(h, now); got != now.Add(7*time.Second) {
		t.Fatal(got)
	}
	if got := resetsAt(h, now); got != now.Add(20*time.Millisecond) {
		t.Fatal(got)
	}
	if !resetsAt(http.Header{}, now).IsZero() {
		t.Fatal("expected zero")
	}
}

func TestStreamErrors(t *testing.T) {
	for _, c := range []struct {
		name  string
		event string
		kind  agentloop.ErrorKind
	}{
		{"failed", `{"type":"response.failed","sequence_number":1,"response":` + respJSON("failed", "[]", `,"error":{"code":"rate_limit_exceeded","message":"slow down"}`) + `}`, agentloop.RateLimited},
		{"failed server", `{"type":"response.failed","sequence_number":1,"response":` + respJSON("failed", "[]", `,"error":{"code":"server_error","message":"oops"}`) + `}`, agentloop.Overloaded},
		{"error event", `{"type":"error","sequence_number":1,"code":"insufficient_quota","message":"no money","param":null}`, agentloop.Billing},
		{"error object", `{"error":{"type":"invalid_request_error","code":"invalid_prompt","message":"bad"}}`, agentloop.InvalidRequest},
	} {
		t.Run(c.name, func(t *testing.T) {
			s := newServer(t, func(w http.ResponseWriter, r *http.Request) { sse(w, c.event) })
			_, err := model(s).Generate(context.Background(), &agentloop.Request{Model: "m"}, nil)
			var me *agentloop.ModelError
			if !errors.As(err, &me) || me.Kind != c.kind || me.Message == "" || me.Err == nil {
				t.Fatalf("%v %+v", err, me)
			}
		})
	}
}

func TestTruncatedStream(t *testing.T) {
	s := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		sse(w, `{"type":"response.output_text.delta","sequence_number":1,"item_id":"m","output_index":0,"content_index":0,"delta":"hi","logprobs":[]}`)
	})
	_, err := model(s).Generate(context.Background(), &agentloop.Request{Model: "m"}, nil)
	var me *agentloop.ModelError
	if !errors.As(err, &me) || me.Kind != agentloop.Unreachable {
		t.Fatalf("%v", err)
	}
}

func TestNetworkError(t *testing.T) {
	s := newServer(t, func(w http.ResponseWriter, r *http.Request) {})
	m := model(s)
	s.Close()
	_, err := m.Generate(context.Background(), &agentloop.Request{Model: "m"}, nil)
	var me *agentloop.ModelError
	if !errors.As(err, &me) || me.Kind != agentloop.Unreachable || !me.Transient() {
		t.Fatalf("%v", err)
	}
}

func TestContextCancelIsNotClassified(t *testing.T) {
	s := newServer(t, func(w http.ResponseWriter, r *http.Request) { <-r.Context().Done() })
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	_, err := model(s).Generate(ctx, &agentloop.Request{Model: "m"}, nil)
	var me *agentloop.ModelError
	if err == nil || errors.As(err, &me) || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("%v", err)
	}
}

func TestNoEnvLeak(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-env-leak")
	t.Setenv("OPENAI_BASE_URL", "http://127.0.0.1:1/leak")
	t.Setenv("OPENAI_ORG_ID", "org-leak")
	t.Setenv("OPENAI_PROJECT_ID", "proj-leak")
	t.Setenv("OPENAI_CUSTOM_HEADERS", "X-Leak: yes")
	t.Setenv("OPENAI_ADMIN_KEY", "admin-leak")
	s := newServer(t, func(w http.ResponseWriter, r *http.Request) { sse(w, completed("[]")...) })
	gen(t, model(s), &agentloop.Request{})
	h := s.hdrs[0]
	if h.Get("Authorization") != "Bearer sk-test" {
		t.Fatalf("auth %q", h.Get("Authorization"))
	}
	for _, k := range []string{"Openai-Organization", "Openai-Project", "X-Leak"} {
		if h.Get(k) != "" {
			t.Errorf("%s leaked: %q", k, h.Get(k))
		}
	}
	for k, v := range h {
		if strings.Contains(strings.Join(v, ","), "leak") {
			t.Errorf("header %s=%v", k, v)
		}
	}
}

func TestNoSDKRetries(t *testing.T) {
	s := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
		fmt.Fprint(w, `{"error":{"message":"down","type":"server_error","code":null}}`)
	})
	_, err := model(s).Generate(context.Background(), &agentloop.Request{Model: "m"}, nil)
	if err == nil || s.hits != 1 {
		t.Fatalf("hits %d err %v", s.hits, err)
	}
}

func TestProvider(t *testing.T) {
	if New(Config{}).Provider() != "openai" {
		t.Fatal("provider")
	}
}
