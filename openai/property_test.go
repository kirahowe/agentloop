package openai

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
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/openai/openai-go/v3/responses"

	"github.com/kirahowe/agentloop"
)

// ---------------------------------------------------------------- generators

var textBits = []string{
	"", " ", "\n", "a", "hello world", "<>&", `"quoted"`, `back\slash`, "line\nbreak", "tab\t",
	"  ", "héllo", "日本語", "😀🎉", "‮RTL", "é", "\x00nul", "\x7f", "</script>",
	`{"json":1}`, "\r\n", "  leading", "trailing  ",
}

func genText(r *rand.Rand) string {
	if r.Intn(400) == 0 {
		return strings.Repeat("x<é>&😀", 4000+r.Intn(4000))
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

var idSeq int

func genID(r *rand.Rand, prefix string) string {
	idSeq++
	return fmt.Sprintf("%s_%d_%x", prefix, idSeq, r.Uint32())
}

func genB64(r *rand.Rand) string {
	const al = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
	b := make([]byte, 1+r.Intn(200))
	for i := range b {
		b[i] = al[r.Intn(len(al))]
	}
	return string(b)
}

// item is one output item, as the API would send it.
type item struct {
	kind     string // reasoning, message, function_call, other
	summary  []string
	texts    []string
	refusal  bool
	callID   string
	name     string
	args     string
	extra    map[string]any
	encrypt  string
	id       string
	statusOK bool
}

func (it item) wire() map[string]any {
	m := map[string]any{}
	for k, v := range it.extra {
		m[k] = v
	}
	m["id"] = it.id
	switch it.kind {
	case "reasoning":
		m["type"] = "reasoning"
		sum := make([]any, len(it.summary))
		for i, s := range it.summary {
			sum[i] = map[string]any{"type": "summary_text", "text": s}
		}
		m["summary"] = sum
		m["encrypted_content"] = it.encrypt
	case "message":
		m["type"], m["role"], m["status"] = "message", "assistant", "completed"
		var cs []any
		for _, s := range it.texts {
			if it.refusal {
				cs = append(cs, map[string]any{"type": "refusal", "refusal": s})
			} else {
				cs = append(cs, map[string]any{"type": "output_text", "annotations": []any{}, "text": s})
			}
		}
		m["content"] = cs
	case "function_call":
		m["type"], m["status"] = "function_call", "completed"
		m["call_id"], m["name"], m["arguments"] = it.callID, it.name, it.args
	default:
		m["type"], m["status"] = "web_search_call", "completed"
	}
	return m
}

func genExtra(r *rand.Rand) map[string]any {
	if r.Intn(3) != 0 {
		return nil
	}
	return map[string]any{"x_future_field": genObject(r, 2), "z_last": genText(r)}
}

func genItems(r *rand.Rand) []item {
	var out []item
	for i, n := 0, r.Intn(5); i < n; i++ {
		switch r.Intn(5) {
		case 0:
			it := item{kind: "reasoning", id: genID(r, "rs"), encrypt: "gAAAA" + genB64(r), extra: genExtra(r)}
			for j, k := 0, r.Intn(3); j < k; j++ {
				it.summary = append(it.summary, genNonBlank(r))
			}
			out = append(out, it)
		case 1, 2:
			it := item{kind: "message", id: genID(r, "msg"), extra: genExtra(r), refusal: r.Intn(8) == 0}
			for j, k := 0, 1+r.Intn(2); j < k; j++ {
				it.texts = append(it.texts, genNonBlank(r))
			}
			out = append(out, it)
		case 3:
			out = append(out, item{kind: "function_call", id: genID(r, "fc"), callID: genID(r, "call"), name: "tool_" + fmt.Sprint(r.Intn(4)),
				args: string(mustJSON(genObject(r, 3))), extra: genExtra(r)})
		default:
			out = append(out, item{kind: "other", id: genID(r, "ws"), extra: genExtra(r)})
		}
	}
	return out
}

func respBody(status string, items []item, extra string) string {
	out := make([]any, len(items))
	for i, it := range items {
		out[i] = it.wire()
	}
	return respJSON(status, string(mustJSON(out)), extra)
}

// itemEvents renders the streamed form of items.
func itemEvents(r *rand.Rand, items []item, fallback bool) []string {
	seq := 0
	e := func(format string, a ...any) string {
		seq++
		return fmt.Sprintf(`{"sequence_number":%d,`, seq) + fmt.Sprintf(format, a...)[1:]
	}
	evs := []string{`{"type":"response.created","sequence_number":0,"response":` + respJSON("in_progress", "[]", "") + `}`}
	pieces := func(s string) []string {
		var out []string
		rs := []rune(s)
		for len(rs) > 0 {
			n := 1 + r.Intn(len(rs))
			if n > 30 {
				n = 1 + r.Intn(30)
			}
			out = append(out, string(rs[:n]))
			rs = rs[n:]
		}
		return out
	}
	for i, it := range items {
		w := string(mustJSON(it.wire()))
		evs = append(evs, e(`{"type":"response.output_item.added","output_index":%d,"item":%s}`, i, w))
		switch it.kind {
		case "reasoning":
			for j, s := range it.summary {
				evs = append(evs, e(`{"type":"response.reasoning_summary_part.added","item_id":%q,"output_index":%d,"summary_index":%d,"part":{"type":"summary_text","text":""}}`, it.id, i, j))
				for _, p := range pieces(s) {
					evs = append(evs, e(`{"type":"response.reasoning_summary_text.delta","item_id":%q,"output_index":%d,"summary_index":%d,"delta":%s}`, it.id, i, j, jsonStr(p)))
				}
			}
		case "message":
			for j, s := range it.texts {
				typ := "response.output_text.delta"
				if it.refusal {
					typ = "response.refusal.delta"
				}
				for _, p := range pieces(s) {
					evs = append(evs, e(`{"type":%q,"item_id":%q,"output_index":%d,"content_index":%d,"delta":%s,"logprobs":[]}`, typ, it.id, i, j, jsonStr(p)))
				}
			}
		case "function_call":
			for _, p := range pieces(it.args) {
				evs = append(evs, e(`{"type":"response.function_call_arguments.delta","item_id":%q,"output_index":%d,"delta":%s}`, it.id, i, jsonStr(p)))
			}
		}
		evs = append(evs, e(`{"type":"response.output_item.done","output_index":%d,"item":%s}`, i, w))
	}
	final := items
	if fallback {
		final = nil
	}
	evs = append(evs, e(`{"type":"response.completed","response":%s}`, respBody("completed", final, usageJSON)))
	return evs
}

func jsonStr(s string) string { b, _ := json.Marshal(s); return string(b) }

func render(r *rand.Rand, evs []string, noise bool) []byte {
	var b bytes.Buffer
	for _, e := range evs {
		if noise && r.Intn(3) == 0 {
			b.WriteString(": keepalive\n\n")
		}
		var h struct {
			Type string `json:"type"`
		}
		json.Unmarshal([]byte(e), &h)
		fmt.Fprintf(&b, "event: %s\ndata: %s\n\n", h.Type, e)
	}
	return b.Bytes()
}

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
	c.reply = func(w http.ResponseWriter, r *http.Request) { sse(w, completed("[]")...) }
	c.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var b bytes.Buffer
		b.ReadFrom(r.Body)
		c.mu.Lock()
		c.body, c.hdr = b.Bytes(), r.Header.Clone()
		c.hits++
		f := c.reply
		c.mu.Unlock()
		f(w, r)
	}))
	t.Cleanup(c.Close)
	return c
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

func userReq(text string) *agentloop.Request {
	return &agentloop.Request{Model: "gpt-x", Messages: []agentloop.Message{agentloop.Text(agentloop.User, text)}}
}

// ------------------------------------------------- 1. request validity

// checkBody is a strict structural check of a Responses API request body.
func checkBody(raw []byte, effort bool) error {
	var top map[string]json.RawMessage
	if err := json.Unmarshal(raw, &top); err != nil {
		return fmt.Errorf("not JSON: %w", err)
	}
	allowed := map[string]bool{"model": true, "input": true, "instructions": true, "store": true, "include": true,
		"reasoning": true, "max_output_tokens": true, "tools": true, "stream": true}
	for k := range top {
		if !allowed[k] {
			return fmt.Errorf("unexpected top-level key %q", k)
		}
	}
	if string(top["store"]) != "false" || string(top["stream"]) != "true" {
		return errors.New("store must be false and stream true")
	}
	if _, has := top["reasoning"]; has != effort {
		return fmt.Errorf("reasoning present = %v, effort set = %v", has, effort)
	}
	var items []map[string]any
	if err := json.Unmarshal(top["input"], &items); err != nil {
		return fmt.Errorf("input: %w", err)
	}
	if len(items) == 0 {
		return errors.New("empty input")
	}
	calls := map[string]int{} // call_id -> index of the function_call
	answered := map[string]bool{}
	for i, it := range items {
		typ, _ := it["type"].(string)
		switch typ {
		case "message":
			role, _ := it["role"].(string)
			if role != "user" && role != "assistant" {
				return fmt.Errorf("item %d: role %q", i, role)
			}
			switch c := it["content"].(type) {
			case string:
				if c == "" {
					return fmt.Errorf("item %d: empty message content", i)
				}
			case []any:
				if len(c) == 0 && role == "assistant" {
					// native assistant message with no content parts is the API's own
				}
			default:
				return fmt.Errorf("item %d: content %T", i, c)
			}
		case "function_call":
			id, _ := it["call_id"].(string)
			name, _ := it["name"].(string)
			args, isStr := it["arguments"].(string)
			if id == "" || name == "" || !isStr || !json.Valid([]byte(args)) {
				return fmt.Errorf("item %d: bad function_call %v", i, it)
			}
			if _, dup := calls[id]; dup {
				return fmt.Errorf("item %d: duplicate call_id %q", i, id)
			}
			calls[id] = i
		case "function_call_output":
			id, _ := it["call_id"].(string)
			out, isStr := it["output"].(string)
			if _, ok := calls[id]; !ok || !isStr {
				return fmt.Errorf("item %d: function_call_output %q has no earlier function_call (or bad output %v)", i, id, it["output"])
			}
			if answered[id] {
				return fmt.Errorf("item %d: call_id %q answered twice", i, id)
			}
			answered[id] = true
			_ = out
		case "":
			return fmt.Errorf("item %d has no type: %v", i, it)
		}
	}
	for id := range calls {
		if !answered[id] {
			return fmt.Errorf("function_call %q has no function_call_output", id)
		}
	}
	last := items[len(items)-1]
	if last["type"] == "message" && last["role"] != "user" {
		return errors.New("input ends with an assistant message")
	}
	return nil
}

func nativeTurn(r *rand.Rand, model string, items []item) agentloop.Message {
	out := make([]any, len(items))
	for i, it := range items {
		out[i] = it.wire()
	}
	var resp responses.Response
	if err := json.Unmarshal([]byte(respJSON("completed", string(mustJSON(out)), "")), &resp); err != nil {
		panic(err)
	}
	res, err := buildResponse(&agentloop.Request{Model: model}, resp, nil)
	if err != nil {
		panic(err)
	}
	return res.Message
}

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
		var m agentloop.Message
		switch r.Intn(3) {
		case 0:
			m = nativeTurn(r, "gpt-"+fmt.Sprint(r.Intn(3)), genItems(r))
		default:
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
					m.Parts = append(m.Parts, agentloop.Part{Type: agentloop.ToolCallPart, ToolCall: &agentloop.ToolCall{ID: genID(r, "toolu"), Name: "t", Input: in}})
				}
			}
			switch r.Intn(3) {
			case 0:
				m.Native = &agentloop.Native{Provider: "anthropic", Model: "claude-x", Data: json.RawMessage(`{"content":[{"type":"thinking","thinking":"x","signature":"s"}]}`)}
			case 1:
				m.Native = &agentloop.Native{Provider: "openai", Model: "gpt-x"} // no data: render from parts
			}
		}
		msgs = append(msgs, m)
		pending = m.ToolCalls()
	}
	return msgs
}

func checkConvo(t testing.TB, c *capture, msgs []agentloop.Message, effort agentloop.Effort) {
	t.Helper()
	req := &agentloop.Request{Model: "gpt-x", Messages: msgs, Effort: effort}
	if _, err := c.model().Generate(context.Background(), req, nil); err != nil {
		dump, _ := json.Marshal(msgs)
		t.Fatalf("Generate: %v\nconversation: %.400s", err, dump)
	}
	if err := checkBody(c.lastBody(), effort != ""); err != nil {
		dump, _ := json.Marshal(msgs)
		t.Fatalf("invalid request: %v\nconversation: %.600s\nbody: %.600s", err, dump, c.lastBody())
	}
}

func TestRequestValidityProperty(t *testing.T) {
	c := newCapture(t)
	for seed := int64(0); seed < 400; seed++ {
		r := rand.New(rand.NewSource(seed))
		msgs := genConvo(r)
		eff := []agentloop.Effort{"", "", agentloop.EffortHigh}[r.Intn(3)]
		t.Run(fmt.Sprint(seed), func(t *testing.T) { checkConvo(t, c, msgs, eff) })
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
		r := rand.New(rand.NewSource(seed))
		checkConvo(t, c, genConvo(r), []agentloop.Effort{"", agentloop.EffortLow}[r.Intn(2)])
	})
}

func TestRequestValidityRegressions(t *testing.T) {
	call := func(id string) agentloop.Message {
		return agentloop.Message{Role: agentloop.Assistant, Parts: []agentloop.Part{{Type: agentloop.ToolCallPart, ToolCall: &agentloop.ToolCall{ID: id, Name: "t", Input: json.RawMessage(`{}`)}}}}
	}
	result := func(id, content string) agentloop.Part {
		return agentloop.Part{Type: agentloop.ToolResultPart, ToolResult: &agentloop.ToolResult{CallID: id, Content: content}}
	}
	txt := func(s string) agentloop.Part { return agentloop.Part{Type: agentloop.TextPart, Text: s} }
	cases := map[string][]agentloop.Message{
		"empty user text between": {agentloop.Text(agentloop.User, "go"), agentloop.Text(agentloop.Assistant, "hm"), agentloop.Text(agentloop.User, ""), agentloop.Text(agentloop.User, "real")},
		"empty assistant text":    {agentloop.Text(agentloop.User, "go"), agentloop.Text(agentloop.Assistant, ""), agentloop.Text(agentloop.User, "real")},
		"text before result": {agentloop.Text(agentloop.User, "go"), call("c1"),
			{Role: agentloop.User, Parts: []agentloop.Part{txt("note"), result("c1", "ok")}}},
		"empty native data": {agentloop.Text(agentloop.User, "go"),
			{Role: agentloop.Assistant, Parts: []agentloop.Part{{Type: agentloop.TextPart, Text: "hi"}}, Native: &agentloop.Native{Provider: "openai", Model: "m"}},
			agentloop.Text(agentloop.User, "more")},
		"null tool input": {agentloop.Text(agentloop.User, "go"),
			{Role: agentloop.Assistant, Parts: []agentloop.Part{{Type: agentloop.ToolCallPart, ToolCall: &agentloop.ToolCall{ID: "c1", Name: "t", Input: json.RawMessage(`null`)}}}},
			{Role: agentloop.User, Parts: []agentloop.Part{result("c1", "ok")}}},
	}
	c := newCapture(t)
	for name, msgs := range cases {
		t.Run(name, func(t *testing.T) { checkConvo(t, c, msgs, "") })
	}
}

// ------------------------------------------------- 2. replay fidelity

func TestReplayFidelityProperty(t *testing.T) {
	c := newCapture(t)
	for seed := int64(0); seed < 300; seed++ {
		r := rand.New(rand.NewSource(seed))
		items := genItems(r)
		raw := render(r, itemEvents(r, items, seed%4 == 3), r.Intn(2) == 0)
		c.setReply(func(w http.ResponseWriter, _ *http.Request) { writeChunked(w, r, raw, 1+r.Intn(300)) })
		resp, err := c.model().Generate(context.Background(), userReq("go"), nil)
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		next := []agentloop.Message{agentloop.Text(agentloop.User, "go"), resp.Message}
		var results []agentloop.Part
		for _, tc := range resp.Message.ToolCalls() {
			results = append(results, agentloop.Part{Type: agentloop.ToolResultPart, ToolResult: &agentloop.ToolResult{CallID: tc.ID, Content: "r"}})
		}
		if len(results) == 0 {
			results = []agentloop.Part{{Type: agentloop.TextPart, Text: "more"}}
		}
		next = append(next, agentloop.Message{Role: agentloop.User, Parts: results})
		c.setReply(func(w http.ResponseWriter, _ *http.Request) { sse(w, completed("[]")...) })
		if _, err := c.model().Generate(context.Background(), &agentloop.Request{Model: "gpt-x", Messages: next}, nil); err != nil {
			t.Fatalf("seed %d replay: %v", seed, err)
		}
		var sent struct {
			Input []json.RawMessage `json:"input"`
		}
		if err := json.Unmarshal(c.lastBody(), &sent); err != nil {
			t.Fatal(err)
		}
		if len(sent.Input) < 1+len(items) {
			t.Fatalf("seed %d: %d input items for %d output items", seed, len(sent.Input), len(items))
		}
		for i, it := range items {
			want, _ := json.Marshal(it.wire())
			if !sameJSON(sent.Input[1+i], want) {
				t.Fatalf("seed %d item %d differs after decoding:\n got %s\nwant %s", seed, i, sent.Input[1+i], want)
			}
		}
		if err := checkBody(c.lastBody(), false); err != nil {
			t.Fatalf("seed %d: replay request invalid: %v", seed, err)
		}
		// Neutral reading.
		var wantParts []agentloop.Part
		for _, it := range items {
			switch it.kind {
			case "reasoning":
				wantParts = append(wantParts, agentloop.Part{Type: agentloop.ReasoningPart, Text: strings.Join(it.summary, "\n\n")})
			case "message":
				for _, s := range it.texts {
					wantParts = append(wantParts, agentloop.Part{Type: agentloop.TextPart, Text: s})
				}
			case "function_call":
				wantParts = append(wantParts, agentloop.Part{Type: agentloop.ToolCallPart, ToolCall: &agentloop.ToolCall{ID: it.callID, Name: it.name, Input: json.RawMessage(it.args)}})
			}
		}
		got := resp.Message.Parts
		if len(got) != len(wantParts) {
			t.Fatalf("seed %d: %d parts, want %d", seed, len(got), len(wantParts))
		}
		for i := range got {
			if got[i].Type != wantParts[i].Type || got[i].Text != wantParts[i].Text ||
				(got[i].ToolCall == nil) != (wantParts[i].ToolCall == nil) ||
				(got[i].ToolCall != nil && (got[i].ToolCall.ID != wantParts[i].ToolCall.ID || got[i].ToolCall.Name != wantParts[i].ToolCall.Name || string(got[i].ToolCall.Input) != string(wantParts[i].ToolCall.Input))) {
				t.Fatalf("seed %d part %d: %+v want %+v", seed, i, got[i], wantParts[i])
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
		evs := itemEvents(r, genItems(r), seed%3 == 0)
		plain, noisy := render(r, evs, false), render(r, evs, true)
		whole := run(c, plain, func(w http.ResponseWriter, raw []byte) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Write(raw)
		})
		if whole.err != nil {
			t.Fatalf("seed %d: %v", seed, whole.err)
		}
		for variant, raw := range map[string][]byte{"plain": plain, "noisy": noisy} {
			for _, max := range []int{1, 7, 64, 1000} {
				if (max == 1 && len(raw) > 3000) || (max == 7 && len(raw) > 30000) {
					continue
				}
				got := run(c, raw, func(w http.ResponseWriter, raw []byte) { writeChunked(w, r, raw, max) })
				if got.err != nil || !reflect.DeepEqual(got.resp, whole.resp) || !reflect.DeepEqual(got.deltas, whole.deltas) {
					t.Fatalf("seed %d %s max %d: differs (err %v)", seed, variant, max, got.err)
				}
			}
		}
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

func FuzzStreamSplitting(f *testing.F) {
	c := newCapture(f)
	for i := int64(0); i < 6; i++ {
		f.Add(i, uint8(1+i*9), i%2 == 0)
	}
	f.Fuzz(func(t *testing.T, seed int64, chunk uint8, noise bool) {
		r := rand.New(rand.NewSource(seed))
		evs := itemEvents(r, genItems(r), seed%3 == 0)
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

func TestTruncatedStreamsAreUnreachable(t *testing.T) {
	c := newCapture(t)
	for seed := int64(0); seed < 40; seed++ {
		r := rand.New(rand.NewSource(seed))
		full := render(r, itemEvents(r, genItems(r), false), true)
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

func TestStreamDoneMarkerWithoutResponseIsUnreachable(t *testing.T) {
	c := newCapture(t)
	c.setReply(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: [DONE]\n\n")
	})
	_, err := c.model().Generate(context.Background(), userReq("a"), nil)
	if agentloop.AsModelError(err).Kind != agentloop.Unreachable {
		t.Fatalf("err = %v", err)
	}
}

// ------------------------------------------------- 4. error classification

func TestErrorTableExhaustive(t *testing.T) {
	statuses := []int{400, 401, 402, 403, 404, 408, 409, 413, 422, 429, 500, 502, 503, 504, 529}
	codes := []string{"", "invalid_api_key", "insufficient_quota", "rate_limit_exceeded", "model_not_found", "context_length_exceeded", "server_error", "weird"}
	types := []string{"", "invalid_request_error", "insufficient_quota", "server_error", "requests", "tokens"}
	oracle := func(status int, code, typ string) agentloop.ErrorKind {
		switch {
		case status == 402:
			return agentloop.Billing
		case status == 429 && (code == "insufficient_quota" || typ == "insufficient_quota"):
			return agentloop.Billing
		case status == 429:
			return agentloop.RateLimited
		case status == 401 || status == 403:
			return agentloop.Unauthorized
		case status == 404:
			return agentloop.NotFound
		case status == 408 || status == 409 || status >= 500:
			return agentloop.Overloaded
		}
		return agentloop.InvalidRequest
	}
	c := newCapture(t)
	for _, status := range statuses {
		for _, code := range codes {
			for _, typ := range types {
				before := c.count()
				c.setReply(func(w http.ResponseWriter, r *http.Request) {
					w.Header().Set("Content-Type", "application/json")
					w.Header().Set("x-should-retry", "true")
					w.WriteHeader(status)
					cj := "null"
					if code != "" {
						cj = jsonStr(code)
					}
					fmt.Fprintf(w, `{"error":{"message":"boom","type":%s,"code":%s,"param":null}}`, jsonStr(typ), cj)
				})
				_, err := c.model().Generate(context.Background(), userReq("a"), nil)
				var me *agentloop.ModelError
				if !errors.As(err, &me) {
					t.Fatalf("%d/%s/%s: not a ModelError: %v", status, code, typ, err)
				}
				if want := oracle(status, code, typ); me.Kind != want || me.Status != status || me.Message != "boom" {
					t.Errorf("%d/%q/%q: kind %s status %d msg %q, want %s", status, code, typ, me.Kind, me.Status, me.Message, want)
				}
				if n := c.count() - before; n != 1 {
					t.Errorf("%d: %d requests", status, n)
				}
			}
		}
	}
}

func TestErrorBodyShapes(t *testing.T) {
	c := newCapture(t)
	for name, body := range map[string]string{
		"html": "<html>Bad gateway</html>", "empty": "", "json no error": `{"hello":1}`, "null": `null`,
		"truncated json": `{"error":{"mess`, "array": `[1,2]`, "error string": `{"error":"flat"}`,
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

func TestStreamErrorEvents(t *testing.T) {
	kinds := map[string]agentloop.ErrorKind{
		"insufficient_quota": agentloop.Billing, "rate_limit_exceeded": agentloop.RateLimited, "invalid_api_key": agentloop.Unauthorized,
		"model_not_found": agentloop.NotFound, "invalid_prompt": agentloop.InvalidRequest, "context_length_exceeded": agentloop.InvalidRequest,
		"server_error": agentloop.Overloaded, "vector_store_timeout": agentloop.Overloaded, "something_new": agentloop.Overloaded,
	}
	c := newCapture(t)
	for code, want := range kinds {
		for form, ev := range map[string]string{
			"error event":   fmt.Sprintf(`{"type":"error","sequence_number":1,"code":%q,"message":"m","param":null}`, code),
			"error object":  fmt.Sprintf(`{"error":{"type":"x","code":%q,"message":"m"}}`, code),
			"response.fail": `{"type":"response.failed","sequence_number":1,"response":` + respJSON("failed", "[]", fmt.Sprintf(`,"error":{"code":%q,"message":"m"}`, code)) + `}`,
		} {
			// Output can have started before the failure.
			c.setReply(func(w http.ResponseWriter, r *http.Request) {
				sse(w, `{"type":"response.output_text.delta","sequence_number":0,"item_id":"m","output_index":0,"content_index":0,"delta":"hi","logprobs":[]}`, ev)
			})
			_, err := c.model().Generate(context.Background(), userReq("a"), nil)
			var me *agentloop.ModelError
			if !errors.As(err, &me) || me.Kind != want || me.Message != "m" {
				t.Errorf("%s/%s: %v", code, form, err)
			}
		}
	}
	// A failed response with no error body is still an error, with a message.
	c.setReply(func(w http.ResponseWriter, r *http.Request) {
		sse(w, `{"type":"response.failed","sequence_number":1,"response":`+respJSON("failed", "[]", "")+`}`)
	})
	_, err := c.model().Generate(context.Background(), userReq("a"), nil)
	var me *agentloop.ModelError
	if !errors.As(err, &me) || me.Message == "" {
		t.Errorf("empty failure: %+v %v", me, err)
	}
}

func TestRetryAfterTable(t *testing.T) {
	now := time.Date(2026, 10, 14, 12, 0, 0, 0, time.UTC)
	cases := []struct {
		hdr  map[string]string
		want time.Time
	}{
		{nil, time.Time{}},
		{map[string]string{"retry-after": "0"}, now},
		{map[string]string{"retry-after": "30"}, now.Add(30 * time.Second)},
		{map[string]string{"retry-after": " 5 "}, now.Add(5 * time.Second)},
		{map[string]string{"retry-after": "1.5"}, now.Add(1500 * time.Millisecond)},
		{map[string]string{"retry-after": "-5"}, time.Time{}},
		{map[string]string{"retry-after": "abc"}, time.Time{}},
		{map[string]string{"retry-after": "NaN"}, time.Time{}},
		{map[string]string{"retry-after": "Inf"}, now.Add(maxRetryAfter)},
		{map[string]string{"retry-after": "1e300"}, now.Add(maxRetryAfter)},
		{map[string]string{"retry-after": "99999999999"}, now.Add(maxRetryAfter)},
		{map[string]string{"retry-after": now.Add(time.Hour).Format(http.TimeFormat)}, now.Add(time.Hour)},
		{map[string]string{"retry-after": now.Add(-time.Hour).Format(http.TimeFormat)}, now},
		{map[string]string{"retry-after-ms": "1500"}, now.Add(1500 * time.Millisecond)},
		{map[string]string{"retry-after-ms": "0"}, now},
		{map[string]string{"retry-after-ms": "-1"}, time.Time{}},
		{map[string]string{"retry-after-ms": "Inf"}, now.Add(maxRetryAfter)},
		{map[string]string{"retry-after-ms": "1e300"}, now.Add(maxRetryAfter)},
		{map[string]string{"retry-after-ms": "garbage", "retry-after": "7"}, now.Add(7 * time.Second)},
		{map[string]string{"retry-after-ms": "500", "retry-after": "7"}, now.Add(500 * time.Millisecond)},
	}
	for _, c := range cases {
		h := http.Header{}
		for k, v := range c.hdr {
			h.Set(k, v)
		}
		if got := retryAfter(h, now); !got.Equal(c.want) {
			t.Errorf("%v: got %v, want %v", c.hdr, got, c.want)
		}
	}
}

func TestResetsAtTable(t *testing.T) {
	now := time.Date(2026, 10, 14, 12, 0, 0, 0, time.UTC)
	mk := func(req, tok string) http.Header {
		h := http.Header{}
		if req != "" {
			h.Set("x-ratelimit-reset-requests", req)
		}
		if tok != "" {
			h.Set("x-ratelimit-reset-tokens", tok)
		}
		return h
	}
	for _, c := range []struct {
		req, tok string
		want     time.Time
	}{
		{"1s", "6m0s", now.Add(6 * time.Minute)},
		{"20ms", "", now.Add(20 * time.Millisecond)},
		{"1m30.5s", "2s", now.Add(90500 * time.Millisecond)},
		{"", "", time.Time{}},
		{"garbage", "also", time.Time{}},
		{"-5s", "-1m", time.Time{}},
		{"0s", "0", time.Time{}},
		{"5", "7", time.Time{}}, // bare numbers have no unit
		{"2562047h47m16.854775807s", "", now.Add(maxResetIn)},
		{"9999999h", "", time.Time{}}, // overflows ParseDuration
		{"1s", "garbage", now.Add(time.Second)},
	} {
		if got := resetsAt(mk(c.req, c.tok), now); !got.Equal(c.want) {
			t.Errorf("%q/%q: got %v, want %v", c.req, c.tok, got, c.want)
		}
	}
	if !resetsAt(nil, now).IsZero() {
		t.Error("nil header")
	}
}

func FuzzHeaderParsing(f *testing.F) {
	for _, s := range []string{"", "0", "30", "-1", "NaN", "Inf", "1e400", "Wed, 21 Oct 2015 07:28:00 GMT", "1s", "6m0s", "20ms", "-3h", "9999999h", "99999999999999999999"} {
		f.Add(s, s)
	}
	f.Fuzz(func(t *testing.T, a, b string) {
		now := time.Now()
		h := http.Header{}
		h.Set("retry-after", a)
		h.Set("retry-after-ms", b)
		h.Set("x-ratelimit-reset-requests", a)
		h.Set("x-ratelimit-reset-tokens", b)
		if ra := retryAfter(h, now); !ra.IsZero() && ra.Before(now) {
			t.Fatalf("retry-after %q/%q gave a time in the past: %v", a, b, ra)
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
						fmt.Fprint(w, "event: response.created\ndata: "+completed("[]")[0]+"\n\n")
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
		"OPENAI_API_KEY", "OPENAI_ADMIN_KEY", "OPENAI_ORG_ID", "OPENAI_ORGANIZATION", "OPENAI_PROJECT_ID", "OPENAI_PROJECT",
		"OPENAI_WEBHOOK_SECRET", "OPENAI_LOG", "OPENAI_API_VERSION", "OPENAI_MODEL", "OPENAI_DEFAULT_HEADERS", "ANTHROPIC_API_KEY",
	} {
		t.Setenv(k, "env-"+k)
	}
	t.Setenv("OPENAI_BASE_URL", decoy.URL)
	t.Setenv("OPENAI_CUSTOM_HEADERS", "X-Env-Leak: yes\nOpenAI-Beta: env-beta")
	t.Setenv("OPENAI_LOG", "debug")
	if _, err := c.model().Generate(context.Background(), userReq("a"), nil); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(cleanBody, c.lastBody()) {
		t.Errorf("body changed with env:\n%s\n%s", cleanBody, c.lastBody())
	}
	if !reflect.DeepEqual(cleanHdr, c.hdr) {
		t.Errorf("headers changed with env:\nclean %v\n env  %v", cleanHdr, c.hdr)
	}
	for k, v := range c.hdr {
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
				fmt.Fprint(w, `{"error":{"message":"x","type":"server_error","code":null}}`)
			})
			if _, err := c.model().Generate(context.Background(), userReq("a"), nil); err == nil {
				t.Fatalf("%d: no error", status)
			}
			if n := c.count() - before; n != 1 {
				t.Errorf("status %d hdr %v: %d requests", status, hdr, n)
			}
		}
	}
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

func TestReasoningOnlyWhenEffortIsSet(t *testing.T) {
	c := newCapture(t)
	req := userReq("a")
	if _, err := c.model().Generate(context.Background(), req, nil); err != nil {
		t.Fatal(err)
	}
	var b map[string]any
	json.Unmarshal(c.lastBody(), &b)
	if _, has := b["reasoning"]; has {
		t.Errorf("reasoning sent without effort: %v", b["reasoning"])
	}
	if !reflect.DeepEqual(b["include"], []any{"reasoning.encrypted_content"}) {
		t.Errorf("include = %v", b["include"])
	}
	for _, e := range []agentloop.Effort{agentloop.EffortLow, agentloop.EffortMedium, agentloop.EffortHigh, agentloop.EffortXHigh, agentloop.EffortMax} {
		req.Effort = e
		if _, err := c.model().Generate(context.Background(), req, nil); err != nil {
			t.Fatal(err)
		}
		b = nil
		json.Unmarshal(c.lastBody(), &b)
		if !reflect.DeepEqual(b["reasoning"], map[string]any{"effort": string(e), "summary": "auto"}) {
			t.Errorf("%s -> %v", e, b["reasoning"])
		}
	}
}

func TestMaxTokensProperty(t *testing.T) {
	c := newCapture(t)
	for _, tc := range []struct {
		req, def int
		want     any
	}{
		{0, 0, nil}, {-5, 0, nil}, {0, 1000, float64(1000)}, {55, 1000, float64(55)}, {1, 0, float64(1)}, {0, -3, nil}, {1 << 30, 0, float64(1 << 30)},
	} {
		req := userReq("a")
		req.MaxOutputTokens = tc.req
		m := New(Config{APIKey: "k", BaseURL: c.URL, DefaultMaxTokens: tc.def})
		if _, err := m.Generate(context.Background(), req, nil); err != nil {
			t.Fatal(err)
		}
		var b map[string]any
		json.Unmarshal(c.lastBody(), &b)
		if b["max_output_tokens"] != tc.want {
			t.Errorf("%+v -> %v", tc, b["max_output_tokens"])
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
			if k == "additionalProperties" {
				s[k] = r.Intn(2) == 0
			} else {
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
			Type        string          `json:"type"`
			Name        string          `json:"name"`
			Description *string         `json:"description"`
			Parameters  json.RawMessage `json:"parameters"`
			Strict      *bool           `json:"strict"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(c.lastBody(), &sent); err != nil || len(sent.Tools) != 1 {
		t.Fatalf("body: %v %s", err, c.lastBody())
	}
	tool := sent.Tools[0]
	if tool.Type != "function" || tool.Name != "t" || tool.Strict == nil || *tool.Strict {
		t.Fatalf("tool header: %+v", tool)
	}
	if !sameJSON(tool.Parameters, schema) {
		t.Fatalf("schema changed:\n got %.1500s\nwant %.1500s", tool.Parameters, schema)
	}
	if (desc == "") != (tool.Description == nil) || (desc != "" && *tool.Description != desc) {
		t.Fatalf("description %q -> %v", desc, tool.Description)
	}
}

func TestToolSchemaFidelity(t *testing.T) {
	c := newCapture(t)
	for seed := int64(0); seed < 300; seed++ {
		r := rand.New(rand.NewSource(seed))
		checkSchema(t, c, mustJSON(genSchema(r, 2)), genText(r))
	}
	for _, s := range []string{
		`{"type":"object"}`, `{"type":"object","properties":{}}`, `{}`, `{"type":"object","required":null}`,
		`{"type":"object","properties":{"n":{"type":"number","minimum":1e400,"maximum":12345678901234567890.123456789}}}`,
		`{"type":"object","properties":{"<&>":{"description":"</script>"}}}`,
	} {
		checkSchema(t, c, json.RawMessage(s), "d")
	}
}

func FuzzToolSchema(f *testing.F) {
	c := newCapture(f)
	for _, s := range []string{`{"type":"object"}`, `{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}`, `{}`, `{"properties":null}`, `{"x":"\u0000"}`} {
		f.Add([]byte(s))
	}
	f.Fuzz(func(t *testing.T, schema []byte) {
		var m map[string]json.RawMessage
		if json.Unmarshal(schema, &m) != nil || m == nil {
			return
		}
		checkSchema(t, c, schema, "")
	})
}

func TestEmptyToolParametersGetAnObjectSchema(t *testing.T) {
	c := newCapture(t)
	for _, p := range []json.RawMessage{nil, {}, json.RawMessage("  \n")} {
		req := userReq("a")
		req.Tools = []agentloop.ToolSpec{{Name: "t", Parameters: p}}
		if _, err := c.model().Generate(context.Background(), req, nil); err != nil {
			t.Fatal(err)
		}
		var sent struct {
			Tools []struct {
				Parameters map[string]any `json:"parameters"`
			} `json:"tools"`
		}
		json.Unmarshal(c.lastBody(), &sent)
		if len(sent.Tools) != 1 || sent.Tools[0].Parameters["type"] != "object" {
			t.Errorf("body %s", c.lastBody())
		}
	}
	req := userReq("a")
	req.Tools = []agentloop.ToolSpec{{Name: "t", Parameters: json.RawMessage(`{bad`)}}
	before := c.count()
	_, err := c.model().Generate(context.Background(), req, nil)
	var me *agentloop.ModelError
	if !errors.As(err, &me) || me.Kind != agentloop.InvalidRequest || c.count() != before {
		t.Errorf("invalid JSON schema: %v", err)
	}
}

func TestHugeStrings(t *testing.T) {
	c := newCapture(t)
	huge := strings.Repeat("x<é>&😀\n", 400000) // ~5MB
	msgs := []agentloop.Message{
		agentloop.Text(agentloop.User, huge),
		{Role: agentloop.Assistant, Parts: []agentloop.Part{{Type: agentloop.ToolCallPart, ToolCall: &agentloop.ToolCall{ID: "c1", Name: "t", Input: mustJSON(map[string]any{"a": huge})}}}},
		{Role: agentloop.User, Parts: []agentloop.Part{{Type: agentloop.ToolResultPart, ToolResult: &agentloop.ToolResult{CallID: "c1", Content: huge}}}},
	}
	checkConvo(t, c, msgs, "")
	r := rand.New(rand.NewSource(1))
	items := []item{{kind: "message", id: "msg_1", texts: []string{huge[:1<<20]}}}
	raw := render(r, itemEvents(r, items, false), false)
	o := run(c, raw, func(w http.ResponseWriter, raw []byte) { writeChunked(w, r, raw, 50000) })
	if o.err != nil || o.resp.Message.Text() != items[0].texts[0] {
		t.Fatalf("err %v", o.err)
	}
}

func TestIncompleteResponseOutranksToolCalls(t *testing.T) {
	c := newCapture(t)
	call := `{"id":"fc_1","type":"function_call","status":"incomplete","call_id":"c1","name":"t","arguments":"{\"a\":"}`
	for reason, want := range map[string]agentloop.Stop{"max_output_tokens": agentloop.StopMaxTokens, "content_filter": agentloop.StopRefusal} {
		c.setReply(func(w http.ResponseWriter, r *http.Request) {
			sse(w, `{"type":"response.incomplete","sequence_number":1,"response":`+
				respJSON("incomplete", "["+call+"]", `,"incomplete_details":{"reason":"`+reason+`"}`)+`}`)
		})
		resp, err := c.model().Generate(context.Background(), userReq("a"), nil)
		if err != nil || resp.Stop != want {
			t.Errorf("%s: %v %+v", reason, err, resp)
		}
	}
}
