package anthropic

import (
	"bytes"
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

type recorded struct {
	header http.Header
	body   []byte
}

// server replies to each request with handler and records what it got.
type server struct {
	*httptest.Server
	mu   sync.Mutex
	reqs []recorded
}

func newServer(t *testing.T, handler http.HandlerFunc) *server {
	t.Helper()
	s := &server{}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		s.mu.Lock()
		s.reqs = append(s.reqs, recorded{r.Header.Clone(), b})
		s.mu.Unlock()
		handler(w, r)
	}))
	t.Cleanup(s.Close)
	return s
}

func (s *server) last(t *testing.T) recorded {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.reqs) == 0 {
		t.Fatal("server got no request")
	}
	return s.reqs[len(s.reqs)-1]
}

func (s *server) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.reqs)
}

func sse(w http.ResponseWriter, events ...[2]string) {
	w.Header().Set("Content-Type", "text/event-stream")
	for _, e := range events {
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", e[0], e[1])
	}
}

func ev(name, data string) [2]string { return [2]string{name, data} }

func start(model string) [2]string {
	return ev("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"`+model+`","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"cache_creation_input_tokens":3,"cache_read_input_tokens":4,"output_tokens":1}}}`)
}

func stop(reason, details string) [][2]string {
	return [][2]string{
		ev("message_delta", `{"type":"message_delta","delta":{"stop_reason":"`+reason+`","stop_sequence":null,"stop_details":`+details+`},"usage":{"output_tokens":7}}`),
		ev("message_stop", `{"type":"message_stop"}`),
	}
}

func textStream(w http.ResponseWriter, model, text string) {
	evs := [][2]string{
		start(model),
		ev("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
		ev("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":`+jsonStr(text)+`}}`),
		ev("content_block_stop", `{"type":"content_block_stop","index":0}`),
	}
	evs = append(evs, stop("end_turn", "null")...)
	sse(w, evs...)
}

func jsonStr(s string) string { b, _ := json.Marshal(s); return string(b) }

func model(s *server, cfg Config) *Model {
	cfg.APIKey = "test-key"
	cfg.BaseURL = s.URL
	return New(cfg)
}

func body(t *testing.T, r recorded) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(r.body, &m); err != nil {
		t.Fatalf("body: %v\n%s", err, r.body)
	}
	return m
}

func userReq(text string) *agentloop.Request {
	return &agentloop.Request{Model: "claude-x", Messages: []agentloop.Message{agentloop.Text(agentloop.User, text)}}
}

func TestRequestShape(t *testing.T) {
	s := newServer(t, func(w http.ResponseWriter, r *http.Request) { textStream(w, "claude-x", "hi") })
	m := model(s, Config{})
	req := userReq("hello")
	req.System = "be brief"
	req.Effort = agentloop.EffortXHigh
	req.Tools = []agentloop.ToolSpec{{
		Name:        "get",
		Description: "fetch a thing",
		Parameters: json.RawMessage(`{"type":"object","properties":{"id":{"type":"string","minLength":1},"opts":{"type":"object","additionalProperties":false}},` +
			`"required":["id"],"additionalProperties":false,"$defs":{"x":{"enum":[1,2]}}}`),
	}}
	var got []agentloop.Delta
	resp, err := m.Generate(context.Background(), req, func(d agentloop.Delta) { got = append(got, d) })
	if err != nil {
		t.Fatal(err)
	}
	r := s.last(t)
	if k := r.header.Get("X-Api-Key"); k != "test-key" {
		t.Errorf("x-api-key = %q", k)
	}
	b := body(t, r)
	if b["model"] != "claude-x" || b["max_tokens"] != float64(64000) || b["stream"] != true {
		t.Errorf("basics: %s", r.body)
	}
	if !jsonEq(b["thinking"], map[string]any{"type": "adaptive", "display": "summarized"}) {
		t.Errorf("thinking = %v", b["thinking"])
	}
	if !jsonEq(b["output_config"], map[string]any{"effort": "xhigh"}) {
		t.Errorf("output_config = %v", b["output_config"])
	}
	if !jsonEq(b["cache_control"], map[string]any{"type": "ephemeral"}) {
		t.Errorf("cache_control = %v", b["cache_control"])
	}
	if !jsonEq(b["system"], []any{map[string]any{"type": "text", "text": "be brief"}}) {
		t.Errorf("system = %v", b["system"])
	}
	if _, ok := b["fallbacks"]; ok {
		t.Error("fallbacks sent without RefusalFallback")
	}
	if h := r.header.Get("Anthropic-Beta"); strings.Contains(h, "fallback") {
		t.Errorf("beta header %q", h)
	}
	tools := b["tools"].([]any)
	wantSchema := map[string]any{}
	json.Unmarshal([]byte(`{"type":"object","properties":{"id":{"type":"string","minLength":1},"opts":{"type":"object","additionalProperties":false}},`+
		`"required":["id"],"additionalProperties":false,"$defs":{"x":{"enum":[1,2]}}}`), &wantSchema)
	tool := tools[0].(map[string]any)
	if tool["name"] != "get" || tool["description"] != "fetch a thing" || !jsonEq(tool["input_schema"], wantSchema) {
		t.Errorf("tool = %v", tool)
	}
	if len(got) != 1 || got[0].Type != agentloop.TextPart || got[0].Text != "hi" {
		t.Errorf("deltas = %v", got)
	}
	if resp.Message.Text() != "hi" || resp.Stop != agentloop.StopEndTurn || resp.Model != "claude-x" {
		t.Errorf("resp = %+v", resp)
	}
	wantUsage := agentloop.Usage{InputTokens: 10, OutputTokens: 7, CacheReadTokens: 4, CacheWriteTokens: 3}
	if resp.Usage != wantUsage {
		t.Errorf("usage = %+v", resp.Usage)
	}
}

func jsonEq(a, b any) bool {
	x, _ := json.Marshal(a)
	y, _ := json.Marshal(b)
	return bytes.Equal(x, y)
}

func TestMaxTokensAndNilObserver(t *testing.T) {
	s := newServer(t, func(w http.ResponseWriter, r *http.Request) { textStream(w, "claude-x", "hi") })
	m := model(s, Config{DefaultMaxTokens: 1000})
	if _, err := m.Generate(context.Background(), userReq("a"), nil); err != nil {
		t.Fatal(err)
	}
	if v := body(t, s.last(t))["max_tokens"]; v != float64(1000) {
		t.Errorf("default = %v", v)
	}
	req := userReq("a")
	req.MaxOutputTokens = 55
	if _, err := m.Generate(context.Background(), req, nil); err != nil {
		t.Fatal(err)
	}
	if v := body(t, s.last(t))["max_tokens"]; v != float64(55) {
		t.Errorf("request = %v", v)
	}
}

func TestRefusalFallback(t *testing.T) {
	s := newServer(t, func(w http.ResponseWriter, r *http.Request) { textStream(w, "claude-x", "hi") })
	m := model(s, Config{RefusalFallback: true})
	if _, err := m.Generate(context.Background(), userReq("a"), nil); err != nil {
		t.Fatal(err)
	}
	r := s.last(t)
	if v := body(t, r)["fallbacks"]; v != "default" {
		t.Errorf("fallbacks = %v", v)
	}
	if h := r.header.Get("Anthropic-Beta"); !strings.Contains(h, "server-side-fallback-2026-07-01") {
		t.Errorf("beta header %q", h)
	}
}

const sig = "EqQBCkYIARgCKkD+abc/123=="

func thinkingToolStream(w http.ResponseWriter) {
	evs := [][2]string{
		start("claude-served"),
		ev("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`),
		ev("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me "}}`),
		ev("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"think."}}`),
		ev("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"`+sig+`"}}`),
		ev("content_block_stop", `{"type":"content_block_stop","index":0}`),
		ev("content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"REDACTED=="}}`),
		ev("content_block_stop", `{"type":"content_block_stop","index":1}`),
		ev("content_block_start", `{"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}`),
		ev("content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"Calling <it> & done"}}`),
		ev("content_block_stop", `{"type":"content_block_stop","index":2}`),
		ev("content_block_start", `{"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"toolu_1","name":"get","input":{}}}`),
		ev("content_block_delta", `{"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{\"id\":"}}`),
		ev("content_block_delta", `{"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"\"abc\"}"}}`),
		ev("content_block_stop", `{"type":"content_block_stop","index":3}`),
	}
	evs = append(evs, stop("tool_use", "null")...)
	sse(w, evs...)
}

func TestThinkingToolUseAndNativeRoundTrip(t *testing.T) {
	s := newServer(t, func(w http.ResponseWriter, r *http.Request) { thinkingToolStream(w) })
	m := model(s, Config{})
	var deltas []agentloop.Delta
	resp, err := m.Generate(context.Background(), userReq("go"), func(d agentloop.Delta) { deltas = append(deltas, d) })
	if err != nil {
		t.Fatal(err)
	}
	if resp.Stop != agentloop.StopToolUse || resp.Model != "claude-served" {
		t.Fatalf("resp = %+v", resp)
	}
	parts := resp.Message.Parts
	if len(parts) != 4 {
		t.Fatalf("parts = %+v", parts)
	}
	if parts[0].Type != agentloop.ReasoningPart || parts[0].Text != "Let me think." {
		t.Errorf("reasoning = %+v", parts[0])
	}
	if parts[1].Type != agentloop.ReasoningPart || parts[1].Text != "" {
		t.Errorf("redacted = %+v", parts[1])
	}
	if parts[2].Text != "Calling <it> & done" {
		t.Errorf("text = %q", parts[2].Text)
	}
	tc := parts[3].ToolCall
	if tc == nil || tc.ID != "toolu_1" || tc.Name != "get" || string(tc.Input) != `{"id":"abc"}` {
		t.Errorf("tool call = %+v", tc)
	}
	var sawReason bool
	for _, d := range deltas {
		if d.Type == agentloop.ReasoningPart && d.Text == "Let me " {
			sawReason = true
		}
	}
	if !sawReason {
		t.Errorf("deltas = %v", deltas)
	}
	if resp.Message.Native == nil || resp.Message.Native.Provider != "anthropic" || resp.Message.Native.Model != "claude-served" {
		t.Fatalf("native = %+v", resp.Message.Native)
	}

	// Replay: the assistant turn's content must appear verbatim.
	var native struct {
		Content []json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(resp.Message.Native.Data, &native); err != nil {
		t.Fatal(err)
	}
	if len(native.Content) != 4 {
		t.Fatalf("native content = %s", resp.Message.Native.Data)
	}
	req := userReq("go")
	req.Messages = append(req.Messages, resp.Message, agentloop.Message{Role: agentloop.User, Parts: []agentloop.Part{
		{Type: agentloop.ToolResultPart, ToolResult: &agentloop.ToolResult{CallID: "toolu_1", Content: "oops", IsError: true}},
	}})
	if _, err := m.Generate(context.Background(), req, nil); err != nil {
		t.Fatal(err)
	}
	r := s.last(t)
	var sent struct {
		Messages []struct {
			Role    string            `json:"role"`
			Content []json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(r.body, &sent); err != nil {
		t.Fatal(err)
	}
	if len(sent.Messages) != 3 || sent.Messages[1].Role != "assistant" {
		t.Fatalf("messages = %s", r.body)
	}
	for i, c := range sent.Messages[1].Content {
		if i >= len(native.Content) {
			t.Fatalf("extra block %s", c)
		}
		// Byte-identical, apart from the HTML escaping json.Marshal applies
		// to a raw message.
		got := strings.NewReplacer(`\u003c`, "<", `\u003e`, ">", `\u0026`, "&").Replace(string(c))
		if got != string(native.Content[i]) {
			t.Errorf("block %d changed:\n got %s\nwant %s", i, c, native.Content[i])
		}
	}
	if len(sent.Messages[1].Content) != 4 {
		t.Errorf("assistant blocks = %d", len(sent.Messages[1].Content))
	}
	if !bytes.Contains(r.body, []byte(`"signature":"`+sig+`"`)) || !bytes.Contains(r.body, []byte(`"data":"REDACTED=="`)) {
		t.Errorf("thinking blocks lost: %s", r.body)
	}
	var last map[string]any
	json.Unmarshal(sent.Messages[2].Content[0], &last)
	if last["type"] != "tool_result" || last["tool_use_id"] != "toolu_1" || last["is_error"] != true || !jsonEq(last["content"], []any{map[string]any{"type": "text", "text": "oops"}}) {
		t.Errorf("tool_result = %v", last)
	}
}

func TestForeignAssistantAndMerging(t *testing.T) {
	s := newServer(t, func(w http.ResponseWriter, r *http.Request) { textStream(w, "claude-x", "ok") })
	m := model(s, Config{})
	req := &agentloop.Request{Model: "claude-x", Messages: []agentloop.Message{
		agentloop.Text(agentloop.User, "one"),
		agentloop.Text(agentloop.User, "two"),
		{Role: agentloop.Assistant, Parts: []agentloop.Part{
			{Type: agentloop.ReasoningPart, Text: "secret thoughts"},
			{Type: agentloop.TextPart, Text: "calling"},
			{Type: agentloop.ToolCallPart, ToolCall: &agentloop.ToolCall{ID: "c1", Name: "get", Input: json.RawMessage(`{"id":"z"}`)}},
		}, Native: &agentloop.Native{Provider: "openai", Model: "gpt", Data: json.RawMessage(`{"junk":true}`)}},
		{Role: agentloop.User, Parts: []agentloop.Part{
			{Type: agentloop.ToolResultPart, ToolResult: &agentloop.ToolResult{CallID: "c1", Content: "fine"}},
		}},
		agentloop.Text(agentloop.User, "next"),
	}}
	if _, err := m.Generate(context.Background(), req, nil); err != nil {
		t.Fatal(err)
	}
	r := s.last(t)
	if bytes.Contains(r.body, []byte("secret thoughts")) || bytes.Contains(r.body, []byte("junk")) {
		t.Errorf("foreign reasoning or native leaked: %s", r.body)
	}
	var sent struct {
		Messages []struct {
			Role    string           `json:"role"`
			Content []map[string]any `json:"content"`
		} `json:"messages"`
	}
	json.Unmarshal(r.body, &sent)
	if len(sent.Messages) != 3 {
		t.Fatalf("messages = %s", r.body)
	}
	if sent.Messages[0].Role != "user" || len(sent.Messages[0].Content) != 2 {
		t.Errorf("user merge: %v", sent.Messages[0])
	}
	a := sent.Messages[1].Content
	if len(a) != 2 || a[0]["type"] != "text" || a[1]["type"] != "tool_use" || a[1]["id"] != "c1" || !jsonEq(a[1]["input"], map[string]any{"id": "z"}) {
		t.Errorf("assistant = %v", a)
	}
	u := sent.Messages[2].Content
	if len(u) != 2 || u[0]["type"] != "tool_result" || u[1]["type"] != "text" {
		t.Errorf("last user = %v", u)
	}
}

func TestStopReasons(t *testing.T) {
	cases := []struct {
		reason, details string
		want            agentloop.Stop
		refusal         string
	}{
		{"end_turn", "null", agentloop.StopEndTurn, ""},
		{"stop_sequence", "null", agentloop.StopEndTurn, ""},
		{"tool_use", "null", agentloop.StopToolUse, ""},
		{"max_tokens", "null", agentloop.StopMaxTokens, ""},
		{"model_context_window_exceeded", "null", agentloop.StopMaxTokens, ""},
		{"pause_turn", "null", agentloop.StopPause, ""},
		{"refusal", `{"type":"refusal","category":"cyber","explanation":"no malware"}`, agentloop.StopRefusal, "cyber: no malware"},
	}
	for _, c := range cases {
		t.Run(c.reason, func(t *testing.T) {
			s := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				sse(w, append([][2]string{start("claude-x")}, stop(c.reason, c.details)...)...)
			})
			resp, err := model(s, Config{}).Generate(context.Background(), userReq("a"), nil)
			if err != nil {
				t.Fatal(err)
			}
			if resp.Stop != c.want || resp.Refusal != c.refusal {
				t.Errorf("stop = %q refusal = %q", resp.Stop, resp.Refusal)
			}
		})
	}
}

func TestErrorClassification(t *testing.T) {
	future := time.Now().Add(2 * time.Hour).UTC().Truncate(time.Second)
	farther := future.Add(time.Hour)
	past := time.Now().Add(-time.Hour).UTC().Truncate(time.Second)
	cases := []struct {
		name    string
		status  int
		typ     string
		msg     string
		headers map[string]string
		want    agentloop.ErrorKind
	}{
		{"rate", 429, "rate_limit_error", "slow down", map[string]string{"Retry-After": "30"}, agentloop.RateLimited},
		{"overloaded 529", 529, "overloaded_error", "busy", nil, agentloop.Overloaded},
		{"overloaded type", 200, "overloaded_error", "busy", nil, agentloop.Overloaded},
		{"500", 500, "api_error", "boom", nil, agentloop.Overloaded},
		{"503", 503, "", "down", nil, agentloop.Overloaded},
		{"401", 401, "authentication_error", "bad key", nil, agentloop.Unauthorized},
		{"403", 403, "permission_error", "no", nil, agentloop.Unauthorized},
		{"402", 402, "billing_error", "pay", nil, agentloop.Billing},
		{"credit", 400, "invalid_request_error", "Your credit balance is too low to access the API", nil, agentloop.Billing},
		{"404", 404, "not_found_error", "model: nope", nil, agentloop.NotFound},
		{"400", 400, "invalid_request_error", "bad field", nil, agentloop.InvalidRequest},
		{"413", 413, "request_too_large", "huge", nil, agentloop.InvalidRequest},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				for k, v := range c.headers {
					w.Header().Set(k, v)
				}
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(c.status)
				fmt.Fprintf(w, `{"type":"error","error":{"type":%q,"message":%q}}`, c.typ, c.msg)
			})
			if c.status == 200 {
				s = newServer(t, func(w http.ResponseWriter, r *http.Request) {
					sse(w, start("claude-x"), ev("error", fmt.Sprintf(`{"type":"error","error":{"type":%q,"message":%q}}`, c.typ, c.msg)))
				})
			}
			_, err := model(s, Config{}).Generate(context.Background(), userReq("a"), nil)
			var me *agentloop.ModelError
			if !errors.As(err, &me) {
				t.Fatalf("err = %v", err)
			}
			if me.Kind != c.want || me.Message != c.msg || me.Err == nil {
				t.Errorf("got %+v", me)
			}
			if c.status != 200 && me.Status != c.status {
				t.Errorf("status = %d", me.Status)
			}
			if c.name == "rate" {
				if d := time.Until(me.RetryAfter); d < 25*time.Second || d > 31*time.Second {
					t.Errorf("RetryAfter in %v", d)
				}
			}
		})
	}

	t.Run("headers", func(t *testing.T) {
		s := newServer(t, func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Retry-After", future.Format(http.TimeFormat))
			w.Header().Set("anthropic-ratelimit-requests-reset", future.Format(time.RFC3339))
			w.Header().Set("anthropic-ratelimit-tokens-reset", farther.Format(time.RFC3339))
			w.Header().Set("anthropic-ratelimit-input-tokens-reset", past.Format(time.RFC3339))
			w.WriteHeader(429)
			fmt.Fprint(w, `{"type":"error","error":{"type":"rate_limit_error","message":"x"}}`)
		})
		_, err := model(s, Config{}).Generate(context.Background(), userReq("a"), nil)
		me := agentloop.AsModelError(err)
		if me.Kind != agentloop.RateLimited || !me.RetryAfter.Equal(future) || !me.ResetsAt.Equal(farther) {
			t.Errorf("got %+v", me)
		}
	})
}

func TestNoRetries(t *testing.T) {
	s := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(529)
		fmt.Fprint(w, `{"type":"error","error":{"type":"overloaded_error","message":"busy"}}`)
	})
	_, err := model(s, Config{}).Generate(context.Background(), userReq("a"), nil)
	if agentloop.AsModelError(err).Kind != agentloop.Overloaded {
		t.Fatalf("err = %v", err)
	}
	if n := s.count(); n != 1 {
		t.Errorf("server hit %d times", n)
	}
}

func TestMidStreamError(t *testing.T) {
	s := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		sse(w, start("claude-x"),
			ev("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
			ev("error", `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`))
	})
	_, err := model(s, Config{}).Generate(context.Background(), userReq("a"), nil)
	var me *agentloop.ModelError
	if !errors.As(err, &me) || me.Kind != agentloop.Overloaded || me.Message != "Overloaded" {
		t.Fatalf("err = %v", err)
	}
}

func TestUnreachable(t *testing.T) {
	s := newServer(t, func(w http.ResponseWriter, r *http.Request) {})
	m := model(s, Config{})
	s.Close()
	_, err := m.Generate(context.Background(), userReq("a"), nil)
	var me *agentloop.ModelError
	if !errors.As(err, &me) || me.Kind != agentloop.Unreachable {
		t.Fatalf("err = %v", err)
	}
}

func TestTruncatedStream(t *testing.T) {
	s := newServer(t, func(w http.ResponseWriter, r *http.Request) { sse(w, start("claude-x")) })
	_, err := model(s, Config{}).Generate(context.Background(), userReq("a"), nil)
	if agentloop.AsModelError(err).Kind != agentloop.Unreachable {
		t.Fatalf("err = %v", err)
	}
}

func TestContextDone(t *testing.T) {
	s := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	})
	ctx, cancel := context.WithCancel(context.Background())
	go func() { time.Sleep(50 * time.Millisecond); cancel() }()
	_, err := model(s, Config{}).Generate(ctx, userReq("a"), nil)
	var me *agentloop.ModelError
	if err == nil || errors.As(err, &me) || !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %#v", err)
	}
}

func TestNoEnvLeakage(t *testing.T) {
	s := newServer(t, func(w http.ResponseWriter, r *http.Request) { textStream(w, "claude-x", "hi") })
	t.Setenv("ANTHROPIC_API_KEY", "bogus-env-key")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "bogus-token")
	t.Setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:1")
	t.Setenv("ANTHROPIC_PROFILE", "nope")
	if _, err := model(s, Config{}).Generate(context.Background(), userReq("a"), nil); err != nil {
		t.Fatal(err)
	}
	h := s.last(t).header
	if h.Get("X-Api-Key") != "test-key" {
		t.Errorf("x-api-key = %q", h.Get("X-Api-Key"))
	}
	if h.Get("Authorization") != "" {
		t.Errorf("authorization = %q", h.Get("Authorization"))
	}
}
