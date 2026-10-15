package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"reflect"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	sdk "github.com/anthropics/anthropic-sdk-go"

	"github.com/kirahowe/agentloop"
)

// ---------------------------------------------------------------- generators

var textBits = []string{
	"", " ", "\n", "a", "hello world", "<>&", `"quoted"`, `back\slash`, "line\nbreak", "tab\t",
	"  ", "héllo", "日本語", "😀🎉", "‮RTL", "é", "\x00nul", "\x7f", "</script>",
	`{"json":1}`, `A`, "\r\n", "  leading", "trailing  ",
}

func genText(r *rand.Rand) string {
	if r.Intn(400) == 0 {
		return strings.Repeat("x<é>&😀", 4000+r.Intn(4000)) // ~300KB
	}
	var b strings.Builder
	for i, n := 0, r.Intn(5); i < n; i++ {
		b.WriteString(textBits[r.Intn(len(textBits))])
	}
	return b.String()
}

func genNonBlank(r *rand.Rand) string {
	for {
		if s := genText(r); strings.TrimSpace(s) != "" {
			return s
		}
	}
}

func genJSON(r *rand.Rand, depth int) any {
	k := r.Intn(8)
	if depth <= 0 && k >= 6 {
		k = r.Intn(6)
	}
	switch k {
	case 0:
		return nil
	case 1:
		return r.Intn(2) == 0
	case 2:
		return r.Intn(2000) - 1000
	case 3:
		return json.Number([]string{"1.0", "1e30", "-0", "0.1", "12345678901234567890", "1E-7"}[r.Intn(6)])
	case 4, 5:
		return genText(r)
	case 6:
		a := make([]any, r.Intn(4))
		for i := range a {
			a[i] = genJSON(r, depth-1)
		}
		return a
	}
	return genObject(r, depth-1)
}

func genObject(r *rand.Rand, depth int) map[string]any {
	m := map[string]any{}
	for i, n := 0, r.Intn(4); i < n; i++ {
		m[genText(r)] = genJSON(r, depth)
	}
	return m
}

func mustJSON(v any) json.RawMessage {
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		panic(err)
	}
	return bytes.TrimSpace(b.Bytes())
}

// decode parses JSON keeping numbers exact.
func decode(raw []byte) (any, error) {
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	var v any
	if err := d.Decode(&v); err != nil {
		return nil, err
	}
	if d.More() {
		return nil, errors.New("trailing data")
	}
	return v, nil
}

func sameJSON(a, b []byte) bool {
	x, err1 := decode(a)
	y, err2 := decode(b)
	return err1 == nil && err2 == nil && reflect.DeepEqual(x, y)
}

// blk is one content block of a wire response.
type blk struct {
	kind  string // thinking, redacted_thinking, text, tool_use
	text  string // thinking / text
	sig   string
	data  string
	id    string
	name  string
	input json.RawMessage
}

func (b blk) wire() map[string]any {
	switch b.kind {
	case "thinking":
		return map[string]any{"type": "thinking", "thinking": b.text, "signature": b.sig}
	case "redacted_thinking":
		return map[string]any{"type": "redacted_thinking", "data": b.data}
	case "text":
		return map[string]any{"type": "text", "text": b.text}
	}
	in, _ := decode(b.input)
	return map[string]any{"type": "tool_use", "id": b.id, "name": b.name, "input": in}
}

var idSeq int

func genID(r *rand.Rand) string {
	idSeq++
	return fmt.Sprintf("toolu_%d_%x", idSeq, r.Uint32())
}

func genBlocks(r *rand.Rand) []blk {
	var out []blk
	for i, n := 0, r.Intn(5); i < n; i++ {
		switch r.Intn(4) {
		case 0:
			out = append(out, blk{kind: "thinking", text: genText(r), sig: "Eq" + genB64(r)})
		case 1:
			out = append(out, blk{kind: "redacted_thinking", data: genB64(r)})
		case 2:
			out = append(out, blk{kind: "text", text: genNonBlank(r)})
		default:
			out = append(out, blk{kind: "tool_use", id: genID(r), name: "tool_" + fmt.Sprint(r.Intn(5)), input: mustJSON(genObject(r, 3))})
		}
	}
	return out
}

func genB64(r *rand.Rand) string {
	const al = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
	b := make([]byte, 1+r.Intn(200))
	for i := range b {
		b[i] = al[r.Intn(len(al))]
	}
	return string(b)
}

// splitRunes splits s into 1..n pieces on rune boundaries.
func splitRunes(r *rand.Rand, s string) []string {
	rs := []rune(s)
	var out []string
	for len(rs) > 0 {
		n := 1 + r.Intn(len(rs))
		if n > 40 {
			n = 1 + r.Intn(40)
		}
		out = append(out, string(rs[:n]))
		rs = rs[n:]
	}
	return out
}

// events renders blocks as the SSE events of one streamed response.
func events(r *rand.Rand, model string, blocks []blk, stopReason string) [][2]string {
	evs := [][2]string{start(model)}
	for i, b := range blocks {
		var first string
		var deltas []string
		d := func(typ, field, val string) string {
			return fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":%q,%q:%s}}`, i, typ, field, jsonStr(val))
		}
		switch b.kind {
		case "thinking":
			first = `{"type":"thinking","thinking":"","signature":""}`
			for _, p := range splitRunes(r, b.text) {
				deltas = append(deltas, d("thinking_delta", "thinking", p))
			}
			for _, p := range splitRunes(r, b.sig) {
				deltas = append(deltas, d("signature_delta", "signature", p))
			}
		case "redacted_thinking":
			first = fmt.Sprintf(`{"type":"redacted_thinking","data":%s}`, jsonStr(b.data))
		case "text":
			first = `{"type":"text","text":""}`
			for _, p := range splitRunes(r, b.text) {
				deltas = append(deltas, d("text_delta", "text", p))
			}
		case "tool_use":
			first = fmt.Sprintf(`{"type":"tool_use","id":%s,"name":%s,"input":{}}`, jsonStr(b.id), jsonStr(b.name))
			for _, p := range splitRunes(r, string(b.input)) {
				deltas = append(deltas, d("input_json_delta", "partial_json", p))
			}
		}
		evs = append(evs, ev("content_block_start", fmt.Sprintf(`{"type":"content_block_start","index":%d,"content_block":%s}`, i, first)))
		for _, x := range deltas {
			evs = append(evs, ev("content_block_delta", x))
		}
		evs = append(evs, ev("content_block_stop", fmt.Sprintf(`{"type":"content_block_stop","index":%d}`, i)))
	}
	return append(evs, stop(stopReason, "null")...)
}

// render serialises events, optionally interleaving pings and comments.
func render(r *rand.Rand, evs [][2]string, noise bool) []byte {
	var b bytes.Buffer
	for _, e := range evs {
		if noise && r.Intn(2) == 0 {
			b.WriteString("event: ping\ndata: {\"type\": \"ping\"}\n\n")
		}
		if noise && r.Intn(4) == 0 {
			b.WriteString(": keepalive comment\n\n")
		}
		fmt.Fprintf(&b, "event: %s\ndata: %s\n\n", e[0], e[1])
	}
	return b.Bytes()
}

// writeChunked writes raw in random-sized pieces, flushing between them.
func writeChunked(w http.ResponseWriter, r *rand.Rand, raw []byte, maxChunk int) {
	w.Header().Set("Content-Type", "text/event-stream")
	fl, _ := w.(http.Flusher)
	for len(raw) > 0 {
		n := 1 + r.Intn(maxChunk)
		if n > len(raw) {
			n = len(raw)
		}
		w.Write(raw[:n])
		if fl != nil {
			fl.Flush()
		}
		raw = raw[n:]
	}
}

// ------------------------------------------------------------------ capture

// capture is a reusable server that records the last request and replies
// with whatever reply does.
type capture struct {
	*httptest.Server
	mu    sync.Mutex
	body  []byte
	hdr   http.Header
	hits  int
	reply func(w http.ResponseWriter, r *http.Request)
}

func newCapture(t testing.TB) *capture {
	c := &capture{}
	c.reply = func(w http.ResponseWriter, r *http.Request) { textStream(w, "claude-x", "ok") }
	c.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := readAll(r)
		c.mu.Lock()
		c.body, c.hdr = b, r.Header.Clone()
		c.hits++
		f := c.reply
		c.mu.Unlock()
		f(w, r)
	}))
	t.Cleanup(c.Close)
	return c
}

func readAll(r *http.Request) ([]byte, error) {
	var b bytes.Buffer
	_, err := b.ReadFrom(r.Body)
	return b.Bytes(), err
}

func (c *capture) setReply(f func(w http.ResponseWriter, r *http.Request)) {
	c.mu.Lock()
	c.reply = f
	c.mu.Unlock()
}

func (c *capture) model() *Model { return New(Config{APIKey: "k", BaseURL: c.URL}) }

func (c *capture) lastBody() []byte {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.body
}

func (c *capture) count() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.hits
}

// ------------------------------------------------- 1. request validity

var idRe = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

type wireBlock = map[string]any

type wireMsg struct {
	Role    string      `json:"role"`
	Content []wireBlock `json:"content"`
}

// checkBody is a strict structural check of a Messages API request body.
func checkBody(raw []byte) error {
	var top map[string]json.RawMessage
	if err := json.Unmarshal(raw, &top); err != nil {
		return fmt.Errorf("not JSON: %w", err)
	}
	allowed := map[string]bool{"model": true, "max_tokens": true, "messages": true, "stream": true, "thinking": true,
		"output_config": true, "cache_control": true, "system": true, "tools": true, "fallbacks": true}
	for k := range top {
		if !allowed[k] {
			return fmt.Errorf("unexpected top-level key %q", k)
		}
	}
	var msgs []wireMsg
	if err := json.Unmarshal(top["messages"], &msgs); err != nil {
		return fmt.Errorf("messages: %w", err)
	}
	if len(msgs) == 0 {
		return errors.New("no messages")
	}
	var pending map[string]bool // tool_use ids awaiting results from the previous assistant message
	seenUse := map[string]bool{}
	for i, m := range msgs {
		want := "user"
		if i%2 == 1 {
			want = "assistant"
		}
		if m.Role != want {
			return fmt.Errorf("message %d role %q, want %q (no alternation / first not user)", i, m.Role, want)
		}
		if len(m.Content) == 0 {
			return fmt.Errorf("message %d has no content", i)
		}
		results := map[string]bool{}
		sawNonResult := false
		var uses []string
		for j, b := range m.Content {
			typ, _ := b["type"].(string)
			switch typ {
			case "text":
				s, ok := b["text"].(string)
				if !ok || strings.TrimSpace(s) == "" {
					return fmt.Errorf("message %d block %d: blank text block %q", i, j, s)
				}
				sawNonResult = true
			case "thinking":
				if _, ok := b["thinking"].(string); !ok {
					return fmt.Errorf("message %d block %d: thinking without text", i, j)
				}
				if s, _ := b["signature"].(string); s == "" {
					return fmt.Errorf("message %d block %d: thinking without signature", i, j)
				}
				if m.Role != "assistant" {
					return fmt.Errorf("thinking in user message")
				}
			case "redacted_thinking":
				if s, _ := b["data"].(string); s == "" || m.Role != "assistant" {
					return fmt.Errorf("message %d block %d: bad redacted_thinking", i, j)
				}
			case "tool_use":
				id, _ := b["id"].(string)
				name, _ := b["name"].(string)
				_, isObj := b["input"].(map[string]any)
				if !idRe.MatchString(id) || name == "" || !isObj || m.Role != "assistant" {
					return fmt.Errorf("message %d block %d: bad tool_use %v", i, j, b)
				}
				if seenUse[id] {
					return fmt.Errorf("duplicate tool_use id %q", id)
				}
				seenUse[id] = true
				uses = append(uses, id)
			case "tool_result":
				id, _ := b["tool_use_id"].(string)
				if m.Role != "user" || !idRe.MatchString(id) {
					return fmt.Errorf("message %d block %d: bad tool_result %v", i, j, b)
				}
				if sawNonResult {
					return fmt.Errorf("message %d block %d: tool_result after other content", i, j)
				}
				if !pending[id] || results[id] {
					return fmt.Errorf("message %d: tool_result for %q does not match a pending tool_use", i, id)
				}
				results[id] = true
				if c, ok := b["content"]; ok {
					cs, ok := c.([]any)
					if !ok || len(cs) == 0 {
						return fmt.Errorf("tool_result content must be non-empty array when present: %v", c)
					}
					for _, x := range cs {
						xm, _ := x.(map[string]any)
						if s, _ := xm["text"].(string); xm["type"] != "text" || strings.TrimSpace(s) == "" {
							return fmt.Errorf("tool_result has blank text block %v", xm)
						}
					}
				}
				if v, ok := b["is_error"]; ok {
					if _, isBool := v.(bool); !isBool {
						return fmt.Errorf("is_error not bool")
					}
				}
			default:
				return fmt.Errorf("message %d block %d: unknown type %q", i, j, typ)
			}
			if typ != "tool_result" {
				sawNonResult = true
			}
		}
		if m.Role == "user" && len(pending) != len(results) {
			return fmt.Errorf("message %d answers %d of %d tool_use blocks", i, len(results), len(pending))
		}
		if m.Role == "assistant" {
			pending = map[string]bool{}
			for _, id := range uses {
				pending[id] = true
			}
		} else {
			pending = nil
		}
	}
	if len(pending) != 0 {
		return errors.New("last assistant message has unanswered tool_use")
	}
	if msgs[len(msgs)-1].Role != "user" {
		return errors.New("last message is not user")
	}
	return nil
}

// convo builds a random, well-formed neutral conversation that ends on a
// user turn. Every tool call is answered by results in the next user turn(s).
func genConvo(r *rand.Rand) []agentloop.Message {
	var msgs []agentloop.Message
	userText := func(must bool) agentloop.Message {
		if must {
			return agentloop.Text(agentloop.User, genNonBlank(r))
		}
		return agentloop.Text(agentloop.User, genText(r))
	}
	var pending []agentloop.ToolCall
	turns := 1 + r.Intn(6)
	for turn := 0; turn < turns; turn++ {
		// user side
		if len(pending) == 0 {
			msgs = append(msgs, userText(turn == 0 || turn == turns-1))
			if r.Intn(4) == 0 {
				msgs = append(msgs, userText(false))
			}
		} else {
			res := make([]agentloop.Part, len(pending))
			for i, c := range pending {
				res[i] = agentloop.Part{Type: agentloop.ToolResultPart, ToolResult: &agentloop.ToolResult{
					CallID: c.ID, Content: genText(r), IsError: r.Intn(3) == 0}}
			}
			r.Shuffle(len(res), func(i, j int) { res[i], res[j] = res[j], res[i] })
			// split across 1..3 user messages, with stray text parts anywhere
			for len(res) > 0 {
				n := 1 + r.Intn(len(res))
				part := append([]agentloop.Part(nil), res[:n]...)
				res = res[n:]
				if r.Intn(3) == 0 {
					p := agentloop.Part{Type: agentloop.TextPart, Text: genText(r)}
					at := r.Intn(len(part) + 1)
					part = append(part[:at], append([]agentloop.Part{p}, part[at:]...)...)
				}
				msgs = append(msgs, agentloop.Message{Role: agentloop.User, Parts: part})
			}
			pending = nil
		}
		if turn == turns-1 {
			break
		}
		// assistant side
		var m agentloop.Message
		switch r.Intn(3) {
		case 0: // own provider, from a real-looking native record
			m = nativeTurn(r, "claude-"+fmt.Sprint(r.Intn(3)), genBlocks(r))
		default: // foreign or native-less
			m.Role = agentloop.Assistant
			for i, n := 0, r.Intn(5); i < n; i++ {
				switch r.Intn(3) {
				case 0:
					m.Parts = append(m.Parts, agentloop.Part{Type: agentloop.ReasoningPart, Text: genText(r)})
				case 1:
					m.Parts = append(m.Parts, agentloop.Part{Type: agentloop.TextPart, Text: genText(r)})
				default:
					in := json.RawMessage(mustJSON(genObject(r, 2)))
					if r.Intn(5) == 0 {
						in = nil
					}
					m.Parts = append(m.Parts, agentloop.Part{Type: agentloop.ToolCallPart, ToolCall: &agentloop.ToolCall{ID: genID(r) + []string{"", "", ":x y", "/é.日"}[r.Intn(4)], Name: "t", Input: in}})
				}
			}
			if r.Intn(2) == 0 {
				m.Native = &agentloop.Native{Provider: "openai", Model: "gpt-x", Data: json.RawMessage(`[{"type":"reasoning","id":"rs_1"}]`)}
			}
		}
		msgs = append(msgs, m)
		pending = m.ToolCalls()
		if m.Native != nil && m.Native.Provider == Provider {
			pending = nil
			for _, b := range nativeToolUses(m) {
				pending = append(pending, b)
			}
		}
	}
	return msgs
}

func nativeToolUses(m agentloop.Message) []agentloop.ToolCall { return m.ToolCalls() }

// nativeTurn builds an assistant message the way the adapter's response()
// would: Native.Data is the API's message JSON.
func nativeTurn(r *rand.Rand, model string, blocks []blk) agentloop.Message {
	wire := make([]map[string]any, len(blocks))
	for i, b := range blocks {
		wire[i] = b.wire()
	}
	data := mustJSON(map[string]any{"id": "msg_1", "type": "message", "role": "assistant", "model": model, "content": wire,
		"stop_reason": "end_turn", "stop_sequence": nil, "usage": map[string]any{"input_tokens": 1, "output_tokens": 1}})
	var bm sdk.BetaMessage
	if err := json.Unmarshal(data, &bm); err != nil {
		panic(err)
	}
	return response(&bm).Message
}

func checkConvo(t testing.TB, c *capture, msgs []agentloop.Message) {
	t.Helper()
	req := &agentloop.Request{Model: "claude-x", Messages: msgs}
	if _, err := c.model().Generate(context.Background(), req, nil); err != nil {
		dump, _ := json.Marshal(msgs)
		t.Fatalf("Generate: %v\nconversation: %.400s", err, dump)
	}
	body := c.lastBody()
	if err := checkBody(body); err != nil {
		dump, _ := json.Marshal(msgs)
		t.Fatalf("invalid request: %v\nconversation: %.600s\nbody: %.600s", err, dump, body)
	}
}

func TestRequestValidityProperty(t *testing.T) {
	c := newCapture(t)
	for seed := int64(0); seed < 400; seed++ {
		r := rand.New(rand.NewSource(seed))
		msgs := genConvo(r)
		t.Run(fmt.Sprint(seed), func(t *testing.T) { checkConvo(t, c, msgs) })
		if t.Failed() {
			break
		}
	}
}

func FuzzRequestValidity(f *testing.F) {
	c := newCapture(f)
	for _, s := range []int64{0, 1, 2, 3, 7, 42, 1000} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, seed int64) {
		checkConvo(t, c, genConvo(rand.New(rand.NewSource(seed))))
	})
}

// Hand-written minimal cases for the bugs the generator found.
func TestRequestValidityRegressions(t *testing.T) {
	call := func(id string) agentloop.Message {
		return agentloop.Message{Role: agentloop.Assistant, Parts: []agentloop.Part{{Type: agentloop.ToolCallPart, ToolCall: &agentloop.ToolCall{ID: id, Name: "t", Input: json.RawMessage(`{}`)}}}}
	}
	result := func(id, content string) agentloop.Part {
		return agentloop.Part{Type: agentloop.ToolResultPart, ToolResult: &agentloop.ToolResult{CallID: id, Content: content}}
	}
	txt := func(s string) agentloop.Part { return agentloop.Part{Type: agentloop.TextPart, Text: s} }
	cases := map[string][]agentloop.Message{
		"text before result in one message": {agentloop.Text(agentloop.User, "go"), call("c1"),
			{Role: agentloop.User, Parts: []agentloop.Part{txt("note"), result("c1", "ok")}}},
		"text message then result message": {agentloop.Text(agentloop.User, "go"), call("c1"),
			agentloop.Text(agentloop.User, "note"), {Role: agentloop.User, Parts: []agentloop.Part{result("c1", "ok")}}},
		"empty tool result": {agentloop.Text(agentloop.User, "go"), call("c1"),
			{Role: agentloop.User, Parts: []agentloop.Part{result("c1", "")}}},
		"whitespace tool result": {agentloop.Text(agentloop.User, "go"), call("c1"),
			{Role: agentloop.User, Parts: []agentloop.Part{result("c1", " \n")}}},
		"whitespace user text": {agentloop.Text(agentloop.User, "go"), agentloop.Text(agentloop.Assistant, "hm"),
			agentloop.Text(agentloop.User, "  "), agentloop.Text(agentloop.User, "real")},
		"null tool input": {agentloop.Text(agentloop.User, "go"),
			{Role: agentloop.Assistant, Parts: []agentloop.Part{{Type: agentloop.ToolCallPart, ToolCall: &agentloop.ToolCall{ID: "c1", Name: "t", Input: json.RawMessage(`null`)}}}},
			{Role: agentloop.User, Parts: []agentloop.Part{result("c1", "ok")}}},
	}
	c := newCapture(t)
	for name, msgs := range cases {
		t.Run(name, func(t *testing.T) { checkConvo(t, c, msgs) })
	}
}

// ------------------------------------------------- 2. replay fidelity

func TestReplayFidelityProperty(t *testing.T) {
	c := newCapture(t)
	for seed := int64(0); seed < 300; seed++ {
		r := rand.New(rand.NewSource(seed))
		blocks := genBlocks(r)
		if len(blocks) == 0 {
			continue
		}
		raw := render(r, events(r, "claude-served", blocks, "end_turn"), r.Intn(2) == 0)
		c.setReply(func(w http.ResponseWriter, _ *http.Request) { writeChunked(w, r, raw, 1+r.Intn(300)) })
		resp, err := c.model().Generate(context.Background(), userReq("go"), nil)
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		// Replay in the next request, followed by results for any tool_use.
		next := []agentloop.Message{agentloop.Text(agentloop.User, "go"), resp.Message}
		var results []agentloop.Part
		for _, tc := range resp.Message.ToolCalls() {
			results = append(results, agentloop.Part{Type: agentloop.ToolResultPart, ToolResult: &agentloop.ToolResult{CallID: tc.ID, Content: "r"}})
		}
		if len(results) == 0 {
			results = []agentloop.Part{{Type: agentloop.TextPart, Text: "more"}}
		}
		next = append(next, agentloop.Message{Role: agentloop.User, Parts: results})
		c.setReply(func(w http.ResponseWriter, _ *http.Request) { textStream(w, "claude-x", "ok") })
		if _, err := c.model().Generate(context.Background(), &agentloop.Request{Model: "claude-x", Messages: next}, nil); err != nil {
			t.Fatalf("seed %d replay: %v", seed, err)
		}
		var sent struct {
			Messages []struct {
				Content []json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		if err := json.Unmarshal(c.lastBody(), &sent); err != nil {
			t.Fatal(err)
		}
		got := sent.Messages[1].Content
		if len(got) != len(blocks) {
			t.Fatalf("seed %d: replayed %d blocks, want %d\n%s", seed, len(got), len(blocks), c.lastBody())
		}
		for i, b := range blocks {
			gb := got[i]
			wb, _ := json.Marshal(b.wire())
			if !sameJSON(gb, wb) {
				t.Fatalf("seed %d block %d differs after decoding:\n got %s\nwant %s", seed, i, gb, wb)
			}
		}
		// The neutral reading agrees with the wire.
		parts := resp.Message.Parts
		if len(parts) != len(blocks) {
			t.Fatalf("seed %d: %d parts for %d blocks", seed, len(parts), len(blocks))
		}
		for i, b := range blocks {
			switch b.kind {
			case "thinking":
				if parts[i].Type != agentloop.ReasoningPart || parts[i].Text != b.text {
					t.Errorf("seed %d part %d: %+v", seed, i, parts[i])
				}
			case "redacted_thinking":
				if parts[i].Type != agentloop.ReasoningPart || parts[i].Text != "" {
					t.Errorf("seed %d part %d: %+v", seed, i, parts[i])
				}
			case "text":
				if parts[i].Type != agentloop.TextPart || parts[i].Text != b.text {
					t.Errorf("seed %d part %d: %+v", seed, i, parts[i])
				}
			case "tool_use":
				tc := parts[i].ToolCall
				if tc == nil || tc.ID != b.id || tc.Name != b.name || !sameJSON(tc.Input, b.input) {
					t.Errorf("seed %d part %d: %+v", seed, i, parts[i])
				}
			}
		}
	}
}

// ------------------------------------------------- 3. streaming robustness

type outcome struct {
	resp   *agentloop.Response
	deltas []agentloop.Delta
	err    error
}

func run(c *capture, raw []byte, serve func(w http.ResponseWriter, raw []byte)) outcome {
	c.setReply(func(w http.ResponseWriter, _ *http.Request) { serve(w, raw) })
	var o outcome
	o.resp, o.err = c.model().Generate(context.Background(), userReq("go"), func(d agentloop.Delta) { o.deltas = append(o.deltas, d) })
	return o
}

func TestStreamSplittingProperty(t *testing.T) {
	c := newCapture(t)
	for seed := int64(0); seed < 100; seed++ {
		r := rand.New(rand.NewSource(seed))
		blocks := genBlocks(r)
		evs := events(r, "claude-served", blocks, "end_turn")
		plain := render(r, evs, false)
		noisy := render(r, evs, true)
		whole := run(c, plain, func(w http.ResponseWriter, raw []byte) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Write(raw)
		})
		if whole.err != nil {
			t.Fatalf("seed %d: unsplit stream failed: %v", seed, whole.err)
		}
		for variant, raw := range map[string][]byte{"plain": plain, "noisy": noisy} {
			for _, max := range []int{1, 7, 64, 1000} {
				if (max == 1 && len(raw) > 3000) || (max == 7 && len(raw) > 30000) {
					continue
				}
				got := run(c, raw, func(w http.ResponseWriter, raw []byte) { writeChunked(w, r, raw, max) })
				if got.err != nil {
					t.Fatalf("seed %d %s max %d: %v", seed, variant, max, got.err)
				}
				if !reflect.DeepEqual(got.resp, whole.resp) {
					t.Fatalf("seed %d %s max %d: response differs\n got %+v\nwant %+v", seed, variant, max, got.resp, whole.resp)
				}
				if !reflect.DeepEqual(got.deltas, whole.deltas) {
					t.Fatalf("seed %d %s max %d: deltas differ", seed, variant, max)
				}
			}
		}
		// Deltas add up to the parts.
		var txtD, thD, txtP, thP strings.Builder
		for _, d := range whole.deltas {
			if d.Type == agentloop.TextPart {
				txtD.WriteString(d.Text)
			} else {
				thD.WriteString(d.Text)
			}
		}
		for _, p := range whole.resp.Message.Parts {
			switch p.Type {
			case agentloop.TextPart:
				txtP.WriteString(p.Text)
			case agentloop.ReasoningPart:
				thP.WriteString(p.Text)
			}
		}
		if txtD.String() != txtP.String() || thD.String() != thP.String() {
			t.Fatalf("seed %d: deltas do not add up to parts", seed)
		}
	}
}

func TestTruncatedStreamsAreUnreachable(t *testing.T) {
	c := newCapture(t)
	for seed := int64(0); seed < 40; seed++ {
		r := rand.New(rand.NewSource(seed))
		full := render(r, events(r, "claude-served", genBlocks(r), "end_turn"), true)
		cuts := []int{0, 1, len(full) - 1, len(full) - 2, len(full) / 2}
		for i := 0; i < 6; i++ {
			cuts = append(cuts, r.Intn(len(full)))
		}
		for _, cut := range cuts {
			if cut < 0 || cut >= len(full) {
				continue
			}
			for _, abort := range []bool{false, true} {
				o := run(c, full[:cut], func(w http.ResponseWriter, raw []byte) {
					w.Header().Set("Content-Type", "text/event-stream")
					w.Write(raw)
					if abort {
						w.(http.Flusher).Flush()
						panic(http.ErrAbortHandler)
					}
				})
				var me *agentloop.ModelError
				if o.err == nil || !errors.As(o.err, &me) || me.Kind != agentloop.Unreachable {
					t.Fatalf("seed %d cut %d/%d abort=%v: err = %v resp = %v", seed, cut, len(full), abort, o.err, o.resp != nil)
				}
			}
		}
	}
}

// ------------------------------------------------- 4. error classification

type errCase struct {
	status  int
	typ     string
	msg     string
	want    agentloop.ErrorKind
	headers map[string]string
}

func TestErrorTableExhaustive(t *testing.T) {
	statuses := []int{400, 401, 402, 403, 404, 408, 409, 413, 422, 429, 500, 502, 503, 504, 529}
	types := []string{"", "invalid_request_error", "authentication_error", "billing_error", "permission_error", "not_found_error",
		"request_too_large", "rate_limit_error", "api_error", "overloaded_error", "weird_new_error"}
	// The oracle: status decides, except where the error type is
	// unambiguous about money, limits or load.
	oracle := func(status int, typ, msg string) agentloop.ErrorKind {
		switch {
		case status == 402 || typ == "billing_error":
			return agentloop.Billing
		case status == 400 && strings.Contains(strings.ToLower(msg), "credit balance"):
			return agentloop.Billing
		case status == 429 || typ == "rate_limit_error":
			return agentloop.RateLimited
		case status == 408 || status == 409 || status >= 500 || typ == "overloaded_error" || typ == "api_error":
			return agentloop.Overloaded
		case status == 401 || status == 403 || typ == "authentication_error" || typ == "permission_error":
			return agentloop.Unauthorized
		case status == 404 || typ == "not_found_error":
			return agentloop.NotFound
		}
		return agentloop.InvalidRequest
	}
	c := newCapture(t)
	for _, status := range statuses {
		for _, typ := range types {
			for _, msg := range []string{"boom", "Your credit balance is too low"} {
				before := c.count()
				c.setReply(func(w http.ResponseWriter, r *http.Request) {
					w.Header().Set("Content-Type", "application/json")
					w.Header().Set("x-should-retry", "true")
					w.WriteHeader(status)
					fmt.Fprintf(w, `{"type":"error","error":{"type":%q,"message":%q}}`, typ, msg)
				})
				_, err := c.model().Generate(context.Background(), userReq("a"), nil)
				var me *agentloop.ModelError
				if !errors.As(err, &me) {
					t.Fatalf("%d/%s: not a ModelError: %v", status, typ, err)
				}
				want := oracle(status, typ, msg)
				if me.Kind != want || me.Status != status || me.Message != msg {
					t.Errorf("%d/%q/%q: kind %s status %d msg %q, want kind %s", status, typ, msg, me.Kind, me.Status, me.Message, want)
				}
				if n := c.count() - before; n != 1 {
					t.Errorf("%d/%s: %d requests", status, typ, n)
				}
			}
		}
	}
}

func TestErrorBodyShapes(t *testing.T) {
	c := newCapture(t)
	for name, body := range map[string]string{
		"html": "<html>Bad gateway</html>", "empty": "", "json no error": `{"hello":1}`, "null": `null`,
		"truncated json": `{"type":"error","err`, "array": `[1,2]`, "flat error": `{"error":"flat"}`,
		"typed flat error": `{"type":"error","error":"flat"}`, "wrong field types": `{"type":"error","error":{"type":5,"message":7}}`,
		"string": `"nope"`, "number": `42`,
	} {
		for _, status := range []int{400, 429, 502, 529} {
			c.setReply(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(status); fmt.Fprint(w, body) })
			_, err := c.model().Generate(context.Background(), userReq("a"), nil)
			var me *agentloop.ModelError
			if !errors.As(err, &me) || me.Status != status || me.Message == "" || me.Kind == agentloop.Unreachable {
				t.Errorf("%s/%d: %+v (%v)", name, status, me, err)
			}
		}
	}
}

func TestMidStreamErrorEvents(t *testing.T) {
	c := newCapture(t)
	for typ, want := range map[string]agentloop.ErrorKind{
		"overloaded_error": agentloop.Overloaded, "api_error": agentloop.Overloaded, "rate_limit_error": agentloop.RateLimited,
		"invalid_request_error": agentloop.InvalidRequest, "authentication_error": agentloop.Unauthorized,
		"billing_error": agentloop.Billing, "not_found_error": agentloop.NotFound, "permission_error": agentloop.Unauthorized,
	} {
		c.setReply(func(w http.ResponseWriter, r *http.Request) {
			sse(w, start("claude-x"), ev("error", fmt.Sprintf(`{"type":"error","error":{"type":%q,"message":"m"}}`, typ)))
		})
		_, err := c.model().Generate(context.Background(), userReq("a"), nil)
		var me *agentloop.ModelError
		if !errors.As(err, &me) || me.Kind != want || me.Message != "m" {
			t.Errorf("%s: %v", typ, err)
		}
	}
	// A malformed error event is still an error, not a success.
	c.setReply(func(w http.ResponseWriter, r *http.Request) {
		sse(w, start("claude-x"), ev("error", `not json`))
	})
	_, err := c.model().Generate(context.Background(), userReq("a"), nil)
	if agentloop.AsModelError(err).Kind != agentloop.Unreachable {
		t.Errorf("malformed error event: %v", err)
	}
}

func TestRetryAfterTable(t *testing.T) {
	now := time.Date(2026, 10, 14, 12, 0, 0, 0, time.UTC)
	cases := []struct {
		in   string
		want time.Time
	}{
		{"", time.Time{}},
		{"0", now},
		{"30", now.Add(30 * time.Second)},
		{" 5 ", now.Add(5 * time.Second)},
		{"1.5", now.Add(1500 * time.Millisecond)},
		{"-5", time.Time{}},
		{"abc", time.Time{}},
		{"NaN", time.Time{}},
		{"Inf", now.Add(maxRetryAfter)},
		{"+Inf", now.Add(maxRetryAfter)},
		{"-Inf", time.Time{}},
		{"1e300", now.Add(maxRetryAfter)},
		{"99999999999", now.Add(maxRetryAfter)},
		{"0x10", time.Time{}},
		{now.Add(time.Hour).Format(http.TimeFormat), now.Add(time.Hour)},
		{now.Add(-time.Hour).Format(http.TimeFormat), now},
		{now.Add(time.Hour).Format(time.RFC850), now.Add(time.Hour)},
		{now.Add(time.Hour).Format(time.ANSIC), now.Add(time.Hour)},
		{now.Add(time.Hour).Format(time.RFC3339), time.Time{}},
		{"Wed, 99 Foo 2026 25:61:61 GMT", time.Time{}},
	}
	for _, c := range cases {
		h := http.Header{}
		h.Set("Retry-After", c.in)
		got := retryAfter(h, now)
		if !got.Equal(c.want) {
			t.Errorf("Retry-After %q: got %v, want %v", c.in, got, c.want)
		}
	}
}

func TestResetsAtTable(t *testing.T) {
	now := time.Date(2026, 10, 14, 12, 0, 0, 0, time.UTC)
	f := func(k, v string) http.Header { h := http.Header{}; h.Add(k, v); return h }
	h := http.Header{}
	h.Set("anthropic-ratelimit-requests-reset", now.Add(time.Minute).Format(time.RFC3339))
	h.Set("anthropic-ratelimit-tokens-reset", now.Add(time.Hour).Format(time.RFC3339Nano))
	h.Set("anthropic-ratelimit-input-tokens-reset", now.Add(-time.Hour).Format(time.RFC3339))
	h.Set("anthropic-ratelimit-output-tokens-reset", "garbage")
	h.Set("anthropic-ratelimit-tokens-limit", now.Add(24*time.Hour).Format(time.RFC3339)) // not a reset header
	h.Set("x-ratelimit-reset-tokens", now.Add(48*time.Hour).Format(time.RFC3339))         // someone else's
	if got := resetsAt(h, now); !got.Equal(now.Add(time.Hour)) {
		t.Errorf("got %v", got)
	}
	for _, v := range []string{"", "x", "2026-13-45T99:00:00Z", now.Format(time.RFC3339), now.Add(-time.Second).Format(time.RFC3339), "0", "-1", "NaN"} {
		if got := resetsAt(f("anthropic-ratelimit-requests-reset", v), now); !got.IsZero() {
			t.Errorf("%q: got %v", v, got)
		}
	}
	if got := resetsAt(nil, now); !got.IsZero() {
		t.Errorf("nil header: %v", got)
	}
}

func FuzzHeaderParsing(f *testing.F) {
	for _, s := range []string{"", "0", "30", "-1", "NaN", "Inf", "1e400", "Wed, 21 Oct 2015 07:28:00 GMT", "2026-01-01T00:00:00Z", "9999-12-31T23:59:59Z", "0001-01-01T00:00:00Z", "99999999999999999999"} {
		f.Add(s, s)
	}
	f.Fuzz(func(t *testing.T, a, b string) {
		now := time.Now()
		h := http.Header{}
		h.Set("Retry-After", a)
		h.Set("anthropic-ratelimit-requests-reset", b)
		h.Add("anthropic-ratelimit-tokens-reset", a)
		if ra := retryAfter(h, now); !ra.IsZero() && ra.Before(now) {
			t.Fatalf("Retry-After %q gave a time in the past: %v", a, ra)
		}
		if ra := resetsAt(h, now); !ra.IsZero() && !ra.After(now) {
			t.Fatalf("reset %q/%q gave a time not in the future: %v", a, b, ra)
		}
	})
}

func TestContextCancelNotModelError(t *testing.T) {
	c := newCapture(t)
	hold := make(chan struct{})
	defer close(hold)
	for name, mk := range map[string]func() (context.Context, context.CancelFunc){
		"cancel": func() (context.Context, context.CancelFunc) { return context.WithCancel(context.Background()) },
		"deadline": func() (context.Context, context.CancelFunc) {
			return context.WithTimeout(context.Background(), 30*time.Millisecond)
		},
	} {
		for _, when := range []string{"before", "during-headers", "during-stream"} {
			t.Run(name+"/"+when, func(t *testing.T) {
				c.setReply(func(w http.ResponseWriter, r *http.Request) {
					if when == "during-stream" {
						w.Header().Set("Content-Type", "text/event-stream")
						fmt.Fprint(w, "event: message_start\ndata: "+start("m")[1]+"\n\n")
						w.(http.Flusher).Flush()
					}
					select {
					case <-r.Context().Done():
					case <-hold:
					}
				})
				ctx, cancel := mk()
				defer cancel()
				switch {
				case when == "before" && name == "cancel":
					cancel()
				case when == "before":
					var cancel2 context.CancelFunc
					ctx, cancel2 = context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
					defer cancel2()
				case name == "cancel":
					go func() { time.Sleep(30 * time.Millisecond); cancel() }()
				}
				_, err := c.model().Generate(ctx, userReq("a"), nil)
				var me *agentloop.ModelError
				want := context.Canceled
				if name == "deadline" {
					want = context.DeadlineExceeded
				}
				if err == nil || errors.As(err, &me) || !errors.Is(err, want) {
					t.Fatalf("err = %#v", err)
				}
			})
		}
	}
}

// ------------------------------------------------- 5. isolation

func TestEnvironmentDoesNotInfluenceRequest(t *testing.T) {
	c := newCapture(t)
	if _, err := c.model().Generate(context.Background(), userReq("a"), nil); err != nil {
		t.Fatal(err)
	}
	cleanBody, cleanHdr := c.lastBody(), c.hdr.Clone()

	decoy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { t.Errorf("decoy server got %s", r.URL) }))
	defer decoy.Close()
	for _, k := range []string{
		"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_WEBHOOK_SIGNING_KEY",
		"ANTHROPIC_ORGANIZATION_ID", "ANTHROPIC_IDENTITY_TOKEN", "ANTHROPIC_IDENTITY_TOKEN_FILE", "ANTHROPIC_FEDERATION_RULE_ID",
		"ANTHROPIC_SERVICE_ACCOUNT_ID", "ANTHROPIC_WORKSPACE_ID", "ANTHROPIC_CONFIG_DIR", "ANTHROPIC_LOG", "ANTHROPIC_SCOPE",
		"ANTHROPIC_BETA", "ANTHROPIC_VERSION", "ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_HEADERS", "OPENAI_API_KEY", "OPENAI_BASE_URL",
	} {
		t.Setenv(k, "env-"+k)
	}
	t.Setenv("ANTHROPIC_BASE_URL", decoy.URL)
	t.Setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Env-Leak: yes\nanthropic-beta: env-beta")
	t.Setenv("ANTHROPIC_LOG", "debug")
	if _, err := c.model().Generate(context.Background(), userReq("a"), nil); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(cleanBody, c.lastBody()) {
		t.Errorf("body changed with env:\n%s\n%s", cleanBody, c.lastBody())
	}
	hdr := c.hdr
	if !reflect.DeepEqual(cleanHdr, hdr) {
		t.Errorf("headers changed with env:\nclean %v\n env  %v", cleanHdr, hdr)
	}
	for k, v := range hdr {
		if strings.Contains(strings.Join(v, ","), "env-") {
			t.Errorf("env value in header %s: %v", k, v)
		}
	}
}

func TestExactlyOneRequestPerGenerate(t *testing.T) {
	c := newCapture(t)
	for _, status := range []int{400, 401, 402, 403, 404, 408, 409, 413, 422, 429, 500, 502, 503, 504, 529} {
		for _, hdr := range []map[string]string{nil, {"x-should-retry": "true", "retry-after": "0"}, {"retry-after-ms": "1"}} {
			before := c.count()
			c.setReply(func(w http.ResponseWriter, r *http.Request) {
				for k, v := range hdr {
					w.Header().Set(k, v)
				}
				w.WriteHeader(status)
				fmt.Fprint(w, `{"type":"error","error":{"type":"api_error","message":"x"}}`)
			})
			if _, err := c.model().Generate(context.Background(), userReq("a"), nil); err == nil {
				t.Fatalf("%d: no error", status)
			}
			if n := c.count() - before; n != 1 {
				t.Errorf("status %d hdr %v: %d requests", status, hdr, n)
			}
		}
	}
	// Dropped connections are not retried either.
	before := c.count()
	c.setReply(func(w http.ResponseWriter, r *http.Request) { panic(http.ErrAbortHandler) })
	if _, err := c.model().Generate(context.Background(), userReq("a"), nil); err == nil {
		t.Fatal("expected error")
	}
	if n := c.count() - before; n != 1 {
		t.Errorf("connection reset: %d requests", n)
	}
}

// ------------------------------------------------- 6. effort, tokens, schemas

func TestEffortMapping(t *testing.T) {
	c := newCapture(t)
	for _, e := range []agentloop.Effort{"", agentloop.EffortLow, agentloop.EffortMedium, agentloop.EffortHigh, agentloop.EffortXHigh, agentloop.EffortMax} {
		req := userReq("a")
		req.Effort = e
		if _, err := c.model().Generate(context.Background(), req, nil); err != nil {
			t.Fatal(err)
		}
		var b map[string]any
		json.Unmarshal(c.lastBody(), &b)
		oc, has := b["output_config"].(map[string]any)
		if e == "" {
			if has && oc["effort"] != nil {
				t.Errorf("effort sent for empty: %v", oc)
			}
			continue
		}
		if oc["effort"] != string(e) {
			t.Errorf("%s -> %v", e, oc)
		}
	}
}

func TestMaxTokensProperty(t *testing.T) {
	c := newCapture(t)
	for _, tc := range []struct{ req, def, want int }{
		{0, 0, fallbackMaxTokens}, {-5, 0, fallbackMaxTokens}, {0, 1000, 1000}, {55, 1000, 55}, {1, 0, 1}, {1 << 30, 0, 1 << 30}, {0, -3, fallbackMaxTokens},
	} {
		req := userReq("a")
		req.MaxOutputTokens = tc.req
		m := New(Config{APIKey: "k", BaseURL: c.URL, DefaultMaxTokens: tc.def})
		if _, err := m.Generate(context.Background(), req, nil); err != nil {
			t.Fatal(err)
		}
		var b map[string]any
		json.Unmarshal(c.lastBody(), &b)
		if b["max_tokens"] != float64(tc.want) {
			t.Errorf("%+v -> %v", tc, b["max_tokens"])
		}
	}
}

func genSchema(r *rand.Rand, depth int) map[string]any {
	s := map[string]any{"type": "object"}
	if r.Intn(5) != 0 {
		props := map[string]any{}
		for i, n := 0, r.Intn(4); i < n; i++ {
			var p any
			if depth > 0 && r.Intn(3) == 0 {
				p = genSchema(r, depth-1)
			} else {
				p = map[string]any{"type": []string{"string", "integer", "number", "boolean", "array", "null"}[r.Intn(6)], "description": genText(r)}
			}
			props[genNonBlank(r)] = p
		}
		s["properties"] = props
		if r.Intn(2) == 0 {
			var req []string
			for k := range props {
				req = append(req, k)
			}
			s["required"] = req
		}
	}
	for _, k := range []string{"additionalProperties", "$schema", "$defs", "definitions", "allOf", "anyOf", "oneOf", "not", "if", "then", "else",
		"patternProperties", "minProperties", "maxProperties", "title", "description", "default", "examples", "x-custom", "$id", "propertyNames", "dependentRequired"} {
		if r.Intn(6) == 0 {
			switch k {
			case "additionalProperties":
				s[k] = r.Intn(2) == 0
			default:
				s[k] = genJSON(r, 3)
			}
		}
	}
	return s
}

func checkSchema(t testing.TB, c *capture, schema json.RawMessage, desc string) {
	t.Helper()
	req := userReq("a")
	req.Tools = []agentloop.ToolSpec{{Name: "t", Description: desc, Parameters: schema}}
	if _, err := c.model().Generate(context.Background(), req, nil); err != nil {
		t.Fatalf("Generate: %v\nschema %.500s", err, schema)
	}
	var sent struct {
		Tools []struct {
			Name        string          `json:"name"`
			Description *string         `json:"description"`
			InputSchema json.RawMessage `json:"input_schema"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(c.lastBody(), &sent); err != nil || len(sent.Tools) != 1 {
		t.Fatalf("body: %v %s", err, c.lastBody())
	}
	want, err := decode(schema)
	if err != nil {
		t.Fatal(err)
	}
	if m, ok := want.(map[string]any); ok {
		m["type"] = "object" // the adapter always sends an object schema
	}
	got, _ := decode(sent.Tools[0].InputSchema)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("schema changed:\n got %.1500s\nwant %.1500s", sent.Tools[0].InputSchema, mustJSON(want))
	}
	if (desc == "") != (sent.Tools[0].Description == nil) || (desc != "" && *sent.Tools[0].Description != desc) {
		t.Fatalf("description %q -> %v", desc, sent.Tools[0].Description)
	}
}

func TestToolSchemaFidelity(t *testing.T) {
	c := newCapture(t)
	for seed := int64(0); seed < 300; seed++ {
		r := rand.New(rand.NewSource(seed))
		checkSchema(t, c, mustJSON(genSchema(r, 2)), genText(r))
	}
	for _, s := range []string{
		`{"type":"object"}`, `{"type":"object","properties":{}}`, `{"type":"object","properties":{},"required":[]}`,
		`{}`, `{"properties":{"a":{"type":"string"}}}`, `{"type":"object","additionalProperties":{"type":"string"}}`,
		`{"type":"object","properties":{"n":{"type":"number","minimum":1e400,"maximum":12345678901234567890.123456789}}}`,
		`{"type":"object","properties":{"a":null}}`, `{"type":"object","required":[]}`,
		`{"type":"object","properties":{"<&>":{"description":"</script>"}}}`,
	} {
		checkSchema(t, c, json.RawMessage(s), "d")
	}
}

func FuzzToolSchema(f *testing.F) {
	c := newCapture(f)
	for _, s := range []string{`{"type":"object"}`, `{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}`, `{}`, `{"properties":null}`, `{"required":null}`, `{"type":"object","properties":{},"$defs":{"a":1}}`} {
		f.Add([]byte(s))
	}
	f.Fuzz(func(t *testing.T, schema []byte) {
		var m map[string]json.RawMessage
		if json.Unmarshal(schema, &m) != nil || m == nil {
			return // Parameters is documented to be a JSON object
		}
		if _, empty := m[""]; empty {
			return // covered by TestEmptySchemaKeyIsInvalidRequest
		}
		var req []string
		if v, ok := m["required"]; ok && json.Unmarshal(v, &req) != nil {
			return // adapter reports InvalidRequest for a non-string-array required
		}
		checkSchema(t, c, schema, "")
	})
}

func TestBadToolSchemaIsInvalidRequest(t *testing.T) {
	c := newCapture(t)
	for _, s := range []string{`[]`, `"x"`, `{`, `{"required":"a"}`, `{"required":[1]}`} {
		req := userReq("a")
		req.Tools = []agentloop.ToolSpec{{Name: "t", Parameters: json.RawMessage(s)}}
		before := c.count()
		_, err := c.model().Generate(context.Background(), req, nil)
		var me *agentloop.ModelError
		if !errors.As(err, &me) || me.Kind != agentloop.InvalidRequest || c.count() != before {
			t.Errorf("%s: %v (requests %d)", s, err, c.count()-before)
		}
	}
}

func TestStreamWithoutMessageStartIsUnreachable(t *testing.T) {
	c := newCapture(t)
	evs := [][2]string{
		ev("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
		ev("content_block_stop", `{"type":"content_block_stop","index":0}`),
		ev("message_stop", `{"type":"message_stop"}`),
	}
	c.setReply(func(w http.ResponseWriter, r *http.Request) { sse(w, evs...) })
	resp, err := c.model().Generate(context.Background(), userReq("a"), nil)
	if err == nil || agentloop.AsModelError(err).Kind != agentloop.Unreachable {
		t.Fatalf("resp %+v err %v", resp, err)
	}
}

func FuzzStreamSplitting(f *testing.F) {
	c := newCapture(f)
	for i := int64(0); i < 6; i++ {
		f.Add(i, uint8(1+i*9), i%2 == 0)
	}
	f.Fuzz(func(t *testing.T, seed int64, chunk uint8, noise bool) {
		r := rand.New(rand.NewSource(seed))
		evs := events(r, "claude-served", genBlocks(r), "end_turn")
		whole := run(c, render(r, evs, false), func(w http.ResponseWriter, raw []byte) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Write(raw)
		})
		got := run(c, render(r, evs, noise), func(w http.ResponseWriter, raw []byte) { writeChunked(w, r, raw, 1+int(chunk)) })
		if whole.err != nil || got.err != nil || !reflect.DeepEqual(got.resp, whole.resp) || !reflect.DeepEqual(got.deltas, whole.deltas) {
			t.Fatalf("split stream differs: %v / %v", whole.err, got.err)
		}
	})
}

func TestHugeStrings(t *testing.T) {
	c := newCapture(t)
	huge := strings.Repeat("x<é>&😀\n", 400000) // ~5MB
	msgs := []agentloop.Message{
		agentloop.Text(agentloop.User, huge),
		{Role: agentloop.Assistant, Parts: []agentloop.Part{{Type: agentloop.ToolCallPart, ToolCall: &agentloop.ToolCall{ID: "c1", Name: "t", Input: mustJSON(map[string]any{"a": huge})}}}},
		{Role: agentloop.User, Parts: []agentloop.Part{{Type: agentloop.ToolResultPart, ToolResult: &agentloop.ToolResult{CallID: "c1", Content: huge}}}},
	}
	checkConvo(t, c, msgs)
	// A big streamed answer survives intact.
	r := rand.New(rand.NewSource(1))
	blocks := []blk{{kind: "text", text: huge[:1<<20]}}
	raw := render(r, events(r, "claude-served", blocks, "end_turn"), false)
	o := run(c, raw, func(w http.ResponseWriter, raw []byte) { writeChunked(w, r, raw, 50000) })
	if o.err != nil || o.resp.Message.Text() != blocks[0].text {
		t.Fatalf("err %v", o.err)
	}
}

func TestWhitespaceSystemPromptIsOmitted(t *testing.T) {
	c := newCapture(t)
	for _, sys := range []string{"", " ", "\n\t"} {
		req := userReq("a")
		req.System = sys
		if _, err := c.model().Generate(context.Background(), req, nil); err != nil {
			t.Fatal(err)
		}
		var b map[string]any
		json.Unmarshal(c.lastBody(), &b)
		if _, has := b["system"]; has {
			t.Errorf("system %q sent: %v", sys, b["system"])
		}
	}
}

func TestForeignToolIDsAreSanitisedConsistently(t *testing.T) {
	c := newCapture(t)
	id := "functions.get:0/é"
	msgs := []agentloop.Message{
		agentloop.Text(agentloop.User, "go"),
		{Role: agentloop.Assistant, Parts: []agentloop.Part{{Type: agentloop.ToolCallPart, ToolCall: &agentloop.ToolCall{ID: id, Name: "get"}}}},
		{Role: agentloop.User, Parts: []agentloop.Part{{Type: agentloop.ToolResultPart, ToolResult: &agentloop.ToolResult{CallID: id, Content: "x"}}}},
	}
	checkConvo(t, c, msgs)
}

// Regression: the SDK drops a zero input_schema, and the API requires one.
func TestToolWithoutParametersStillHasInputSchema(t *testing.T) {
	c := newCapture(t)
	for _, p := range []json.RawMessage{nil, {}, json.RawMessage(`{}`), json.RawMessage(`{"type":"object"}`)} {
		req := userReq("a")
		req.Tools = []agentloop.ToolSpec{{Name: "t", Parameters: p}}
		if _, err := c.model().Generate(context.Background(), req, nil); err != nil {
			t.Fatal(err)
		}
		var sent struct {
			Tools []struct {
				InputSchema map[string]any `json:"input_schema"`
			} `json:"tools"`
		}
		json.Unmarshal(c.lastBody(), &sent)
		if len(sent.Tools) != 1 || sent.Tools[0].InputSchema["type"] != "object" {
			t.Errorf("parameters %q: body %s", p, c.lastBody())
		}
	}
}

// Regression, found by FuzzToolSchema: the SDK cannot marshal an empty
// top-level key, and the failure used to surface as a retryable
// Unreachable.
func TestEmptySchemaKeyIsInvalidRequest(t *testing.T) {
	c := newCapture(t)
	req := userReq("a")
	req.Tools = []agentloop.ToolSpec{{Name: "t", Parameters: json.RawMessage(`{"":""}`)}}
	_, err := c.model().Generate(context.Background(), req, nil)
	var me *agentloop.ModelError
	if !errors.As(err, &me) || me.Kind != agentloop.InvalidRequest || me.Transient() || c.count() != 0 {
		t.Fatalf("err = %v, requests %d", err, c.count())
	}
}
