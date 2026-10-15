// Package openai adapts OpenAI's Responses API to agentloop.Model.
//
// Requests are stateless: nothing is stored on OpenAI's side, and each call
// carries the whole conversation. Reasoning survives across calls because the
// response's encrypted reasoning items are kept in Native and replayed
// verbatim.
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	oai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"

	"github.com/kirahowe/agentloop"
)

// Provider is the name recorded in Native.Provider.
const Provider = "openai"

// Config configures a Model. Nothing is read from the environment.
type Config struct {
	APIKey  string
	BaseURL string // default https://api.openai.com/v1
	// HTTPClient is used for requests. Nil means http.DefaultClient.
	HTTPClient *http.Client
	// DefaultMaxTokens caps a response when the request sets no cap.
	// Zero leaves the cap to the API.
	DefaultMaxTokens int
}

// Model is an agentloop.Model backed by the Responses API.
type Model struct {
	svc       responses.ResponseService
	models    oai.ModelService
	maxTokens int
}

var (
	_ agentloop.Model     = (*Model)(nil)
	_ agentloop.Describer = (*Model)(nil)
)

// New builds a Model. The SDK's own retries are off.
//
// The SDK client would read OPENAI_API_KEY, OPENAI_BASE_URL and friends, so
// New builds the response service directly, which reads nothing.
func New(cfg Config) *Model {
	base := cfg.BaseURL
	if base == "" {
		base = "https://api.openai.com/v1"
	}
	hc := cfg.HTTPClient
	if hc == nil {
		hc = http.DefaultClient
	}
	opts := []option.RequestOption{
		option.WithBaseURL(base),
		option.WithAPIKey(cfg.APIKey),
		option.WithHTTPClient(hc),
		option.WithMaxRetries(0),
	}
	return &Model{
		svc:       responses.NewResponseService(opts...),
		models:    oai.NewModelService(opts...),
		maxTokens: cfg.DefaultMaxTokens,
	}
}

// Provider returns "openai".
func (m *Model) Provider() string { return Provider }

// Generate makes one streaming Responses call.
func (m *Model) Generate(ctx context.Context, req *agentloop.Request, observe func(agentloop.Delta)) (*agentloop.Response, error) {
	params, err := m.buildParams(req)
	if err != nil {
		return nil, &agentloop.ModelError{Kind: agentloop.InvalidRequest, Message: err.Error(), Err: err}
	}
	var seen httpFailure
	stream := m.svc.NewStreaming(ctx, params, option.WithMiddleware(seen.tap))
	defer stream.Close()
	return m.consume(ctx, req, stream, observe, &seen)
}

// Describe asks the Models API about a model. OpenAI's answer holds the
// id, the owner and the creation time, and no display name or context
// window, so those stay unknown.
func (m *Model) Describe(ctx context.Context, model string) (agentloop.ModelInfo, error) {
	var seen httpFailure
	info, err := m.models.Get(ctx, model, option.WithMiddleware(seen.tap))
	if err != nil {
		return agentloop.ModelInfo{}, m.classify(ctx, err, &seen)
	}
	return agentloop.ModelInfo{ID: info.ID}, nil
}

func (m *Model) buildParams(req *agentloop.Request) (responses.ResponseNewParams, error) {
	p := responses.ResponseNewParams{
		Model:   shared.ResponsesModel(req.Model),
		Store:   param.NewOpt(false),
		Include: []responses.ResponseIncludable{responses.ResponseIncludableReasoningEncryptedContent},
	}
	// Models that do not reason reject the reasoning object, so it goes out
	// only when the caller asks for an effort.
	if e := effort(req.Effort); e != "" {
		p.Reasoning = shared.ReasoningParam{Summary: shared.ReasoningSummaryAuto, Effort: e}
	}
	if req.System != "" {
		p.Instructions = param.NewOpt(req.System)
	}
	limit := req.MaxOutputTokens
	if limit == 0 {
		limit = m.maxTokens
	}
	if limit > 0 {
		p.MaxOutputTokens = param.NewOpt(int64(limit))
	}
	items, err := inputItems(req.Messages)
	if err != nil {
		return p, err
	}
	p.Input = responses.ResponseNewParamsInputUnion{OfInputItemList: items}
	for _, t := range req.Tools {
		raw, err := toolJSON(t)
		if err != nil {
			return p, err
		}
		p.Tools = append(p.Tools, param.Override[responses.ToolUnionParam](raw))
	}
	return p, nil
}

// effort maps a neutral effort to the SDK's enum. The enum has both xhigh
// and max, so each maps to itself.
func effort(e agentloop.Effort) shared.ReasoningEffort {
	switch e {
	case agentloop.EffortLow:
		return shared.ReasoningEffortLow
	case agentloop.EffortMedium:
		return shared.ReasoningEffortMedium
	case agentloop.EffortHigh:
		return shared.ReasoningEffortHigh
	case agentloop.EffortXHigh:
		return shared.ReasoningEffortXhigh
	case agentloop.EffortMax:
		return shared.ReasoningEffortMax
	}
	return ""
}

// toolJSON writes a function tool by hand so the parameters schema goes
// out exactly as given. strict is false: strict mode would reject schemas
// that are not closed objects.
func toolJSON(t agentloop.ToolSpec) (json.RawMessage, error) {
	params := t.Parameters
	if len(bytes.TrimSpace(params)) == 0 {
		params = json.RawMessage(`{"type":"object","properties":{}}`)
	}
	return json.Marshal(struct {
		Type        string          `json:"type"`
		Name        string          `json:"name"`
		Description string          `json:"description,omitempty"`
		Parameters  json.RawMessage `json:"parameters"`
		Strict      bool            `json:"strict"`
	}{"function", t.Name, t.Description, params, false})
}

// inputItems renders the conversation as input items.
//
// An assistant message from this provider is replayed from Native.Data, the
// JSON array of the response's output items, each element passed through
// untouched (the SDK only strips insignificant whitespace). Any other
// assistant message, or one whose Native has no data, is rendered from its
// parts, with reasoning dropped.
//
// The API has no error flag on function_call_output, so a tool result with
// IsError is sent with the output prefixed "Error: ". Within a run of user
// messages the function_call_output items go first, so no text separates a
// call from its output.
func inputItems(msgs []agentloop.Message) (responses.ResponseInputParam, error) {
	type item struct {
		raw    json.RawMessage
		result bool
	}
	var out responses.ResponseInputParam
	var user []item // the current run of user messages
	flush := func() {
		for _, it := range user {
			if it.result {
				out = append(out, param.Override[responses.ResponseInputItemUnionParam](it.raw))
			}
		}
		for _, it := range user {
			if !it.result {
				out = append(out, param.Override[responses.ResponseInputItemUnionParam](it.raw))
			}
		}
		user = nil
	}
	for _, msg := range msgs {
		if msg.Role == agentloop.Assistant && msg.Native != nil && msg.Native.Provider == Provider && len(bytes.TrimSpace(msg.Native.Data)) > 0 {
			var items []json.RawMessage
			if err := json.Unmarshal(msg.Native.Data, &items); err != nil {
				return nil, fmt.Errorf("openai: native data is not an array of output items: %w", err)
			}
			flush()
			for _, it := range items {
				out = append(out, param.Override[responses.ResponseInputItemUnionParam](it))
			}
			continue
		}
		var items []item
		for _, part := range msg.Parts {
			var v map[string]any
			result := false
			switch part.Type {
			case agentloop.TextPart:
				if part.Text == "" {
					continue
				}
				v = map[string]any{"type": "message", "role": string(msg.Role), "content": part.Text}
			case agentloop.ToolCallPart:
				if part.ToolCall == nil {
					continue
				}
				args := string(part.ToolCall.Input)
				if a := strings.TrimSpace(args); a == "" || a == "null" {
					args = "{}"
				}
				v = map[string]any{"type": "function_call", "call_id": part.ToolCall.ID, "name": part.ToolCall.Name, "arguments": args}
			case agentloop.ToolResultPart:
				if part.ToolResult == nil {
					continue
				}
				o := part.ToolResult.Content
				if part.ToolResult.IsError {
					o = "Error: " + o
				}
				v, result = map[string]any{"type": "function_call_output", "call_id": part.ToolResult.CallID, "output": o}, true
			default:
				continue
			}
			b, err := json.Marshal(v)
			if err != nil {
				return nil, err
			}
			items = append(items, item{b, result})
		}
		if msg.Role == agentloop.User {
			user = append(user, items...)
			continue
		}
		flush()
		for _, it := range items {
			out = append(out, param.Override[responses.ResponseInputItemUnionParam](it.raw))
		}
	}
	flush()
	return out, nil
}

func (m *Model) consume(ctx context.Context, req *agentloop.Request, stream *ssestream.Stream[responses.ResponseStreamEventUnion], observe func(agentloop.Delta), seen *httpFailure) (*agentloop.Response, error) {
	emit := func(t agentloop.PartType, s string) {
		if observe != nil && s != "" {
			observe(agentloop.Delta{Type: t, Text: s})
		}
	}
	var done []json.RawMessage // output_item.done, a fallback for the final output
	for stream.Next() {
		ev := stream.Current()
		switch ev.Type {
		case "response.output_text.delta", "response.refusal.delta":
			emit(agentloop.TextPart, ev.Delta)
		case "response.reasoning_summary_text.delta":
			emit(agentloop.ReasoningPart, ev.Delta)
		case "response.reasoning_summary_part.added":
			if ev.SummaryIndex > 0 {
				emit(agentloop.ReasoningPart, "\n\n")
			}
		case "response.output_item.done":
			done = append(done, json.RawMessage(ev.Item.RawJSON()))
		case "response.completed", "response.incomplete":
			return buildResponse(req, ev.Response, done)
		case "response.failed":
			return nil, classifyCode(string(ev.Response.Error.Code), ev.Response.Error.Message)
		case "error":
			return nil, classifyCode(ev.Code, ev.Message)
		}
	}
	if err := stream.Err(); err != nil {
		return nil, m.classify(ctx, err, seen)
	}
	if ctx.Err() != nil {
		return nil, fmt.Errorf("openai: %w", ctx.Err())
	}
	err := errors.New("stream ended without a final response")
	return nil, &agentloop.ModelError{Kind: agentloop.Unreachable, Message: err.Error(), Err: err}
}

type outputItem struct {
	Type      string `json:"type"`
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
	Content   []struct {
		Type    string `json:"type"`
		Text    string `json:"text"`
		Refusal string `json:"refusal"`
	} `json:"content"`
	Summary []struct {
		Text string `json:"text"`
	} `json:"summary"`
}

func buildResponse(req *agentloop.Request, r responses.Response, done []json.RawMessage) (*agentloop.Response, error) {
	data := json.RawMessage(r.JSON.Output.Raw())
	var items []json.RawMessage
	if len(data) == 0 || json.Unmarshal(data, &items) != nil || len(items) == 0 {
		items = done
		data, _ = json.Marshal(items)
		if items == nil {
			data = json.RawMessage("[]")
		}
	}
	var parts []agentloop.Part
	var refusals []string
	toolUse := false
	for _, it := range items {
		var o outputItem
		if err := json.Unmarshal(it, &o); err != nil {
			return nil, &agentloop.ModelError{Kind: agentloop.InvalidRequest, Message: "unreadable output item: " + err.Error(), Err: err}
		}
		switch o.Type {
		case "message":
			for _, c := range o.Content {
				switch c.Type {
				case "output_text":
					parts = append(parts, agentloop.Part{Type: agentloop.TextPart, Text: c.Text})
				case "refusal":
					refusals = append(refusals, c.Refusal)
					parts = append(parts, agentloop.Part{Type: agentloop.TextPart, Text: c.Refusal})
				}
			}
		case "reasoning":
			var s []string
			for _, x := range o.Summary {
				s = append(s, x.Text)
			}
			parts = append(parts, agentloop.Part{Type: agentloop.ReasoningPart, Text: strings.Join(s, "\n\n")})
		case "function_call":
			toolUse = true
			args := json.RawMessage(o.Arguments)
			if strings.TrimSpace(o.Arguments) == "" {
				args = json.RawMessage("{}")
			}
			parts = append(parts, agentloop.Part{Type: agentloop.ToolCallPart, ToolCall: &agentloop.ToolCall{ID: o.CallID, Name: o.Name, Input: args}})
		}
	}

	served := string(r.Model)
	if served == "" {
		served = req.Model
	}
	resp := &agentloop.Response{
		Message: agentloop.Message{
			Role:   agentloop.Assistant,
			Parts:  parts,
			Native: &agentloop.Native{Provider: Provider, Model: served, Data: data},
		},
		Model:   served,
		Refusal: strings.Join(refusals, "\n"),
	}

	// InputTokens counts only uncached input: cache reads and writes are
	// subtracted, since OpenAI's input_tokens includes them.
	u := r.Usage
	cached, written := u.InputTokensDetails.CachedTokens, u.InputTokensDetails.CacheWriteTokens
	resp.Usage = agentloop.Usage{
		InputTokens:      max(u.InputTokens-cached-written, 0),
		OutputTokens:     u.OutputTokens,
		CacheReadTokens:  cached,
		CacheWriteTokens: written,
	}

	// A response cut short by the token cap or a filter outranks its tool
	// calls: their arguments may be cut off, and the agent must not run them.
	switch {
	case r.Status == "incomplete" && r.IncompleteDetails.Reason == "max_output_tokens":
		resp.Stop = agentloop.StopMaxTokens
	case r.Status == "incomplete" && r.IncompleteDetails.Reason == "content_filter", len(refusals) > 0:
		resp.Stop = agentloop.StopRefusal
	case toolUse:
		resp.Stop = agentloop.StopToolUse
	default:
		resp.Stop = agentloop.StopEndTurn
	}
	return resp, nil
}

// classify turns a stream or request error into a *ModelError.
func (m *Model) classify(ctx context.Context, err error, seen *httpFailure) error {
	if ctx.Err() != nil {
		return fmt.Errorf("openai: %w", ctx.Err())
	}
	var me *agentloop.ModelError
	if errors.As(err, &me) {
		return err
	}
	// The SDK failed to build the body: nothing was sent and retrying cannot
	// help.
	// The SDK has its own copy of encoding/json, so match on the text too.
	var mje *json.MarshalerError
	if errors.As(err, &mje) || strings.Contains(err.Error(), "error calling MarshalJSON") {
		return &agentloop.ModelError{Kind: agentloop.InvalidRequest, Message: err.Error(), Err: err}
	}
	var api *oai.Error
	if errors.As(err, &api) {
		return classifyAPI(api, time.Now())
	}
	var se *ssestream.StreamError
	if errors.As(err, &se) {
		var body struct {
			Error struct {
				Type    string `json:"type"`
				Code    string `json:"code"`
				Message string `json:"message"`
			} `json:"error"`
		}
		_ = json.Unmarshal(se.Event.Data, &body)
		code := body.Error.Code
		if code == "" {
			code = body.Error.Type
		}
		e := classifyCode(code, body.Error.Message)
		if e.Message == "" {
			e.Message = se.Message
		}
		e.Err = err
		return e
	}
	if seen.status >= 400 {
		// The SDK could not read the error body (a string where it expects an
		// object, say) and dropped the status; the tap kept it.
		return classifyAPI(&oai.Error{StatusCode: seen.status, Message: seen.message(), Response: &http.Response{Header: seen.header}}, time.Now())
	}
	return &agentloop.ModelError{Kind: agentloop.Unreachable, Message: err.Error(), Err: err}
}

// httpFailure remembers the status, headers and start of the body of an
// error response, for when the SDK fails to turn them into an *oai.Error.
type httpFailure struct {
	status int
	header http.Header
	body   string
}

func (f *httpFailure) tap(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
	res, err := next(req)
	if res != nil && res.StatusCode >= 400 && res.Body != nil {
		head, _ := io.ReadAll(io.LimitReader(res.Body, 4096))
		res.Body = struct {
			io.Reader
			io.Closer
		}{io.MultiReader(bytes.NewReader(head), res.Body), res.Body}
		f.status, f.header, f.body = res.StatusCode, res.Header.Clone(), string(head)
	}
	return res, err
}

func (f *httpFailure) message() string {
	if b := strings.TrimSpace(f.body); b != "" {
		return b
	}
	return ""
}

func classifyAPI(e *oai.Error, now time.Time) *agentloop.ModelError {
	me := &agentloop.ModelError{Status: e.StatusCode, Message: e.Message, Err: e}
	if me.Message == "" {
		me.Message = http.StatusText(e.StatusCode)
	}
	if me.Message == "" {
		me.Message = fmt.Sprintf("HTTP %d", e.StatusCode)
	}
	switch {
	case e.StatusCode == 402, e.StatusCode == 429 && (e.Code == "insufficient_quota" || e.Type == "insufficient_quota"):
		me.Kind = agentloop.Billing
	case e.StatusCode == 429:
		me.Kind = agentloop.RateLimited
	case e.StatusCode == 401 || e.StatusCode == 403:
		me.Kind = agentloop.Unauthorized
	case e.StatusCode == 404:
		me.Kind = agentloop.NotFound
	// 408 and 409 are timeouts and lock conflicts; the API invites a retry.
	case e.StatusCode == 408, e.StatusCode == 409, e.StatusCode >= 500:
		me.Kind = agentloop.Overloaded
	default:
		me.Kind = agentloop.InvalidRequest
	}
	if e.Response != nil {
		h := e.Response.Header
		me.RetryAfter = retryAfter(h, now)
		me.ResetsAt = resetsAt(h, now)
	}
	return me
}

// classifyCode sorts an error that arrived inside a 200 stream, where there
// is no status.
func classifyCode(code, msg string) *agentloop.ModelError {
	if msg == "" {
		msg = code
	}
	if msg == "" {
		msg = "the response failed with no reason given"
	}
	me := &agentloop.ModelError{Message: msg, Err: errors.New(code + ": " + msg)}
	switch {
	case code == "insufficient_quota":
		me.Kind = agentloop.Billing
	case code == "rate_limit_exceeded":
		me.Kind = agentloop.RateLimited
	case code == "invalid_api_key", code == "invalid_authentication":
		me.Kind = agentloop.Unauthorized
	case code == "model_not_found":
		me.Kind = agentloop.NotFound
	case strings.HasPrefix(code, "invalid_"), strings.Contains(code, "policy"), strings.HasPrefix(code, "image_"), code == "context_length_exceeded":
		me.Kind = agentloop.InvalidRequest
	default:
		// server_error, vector_store_timeout and anything unknown.
		me.Kind = agentloop.Overloaded
	}
	return me
}

func retryAfter(h http.Header, now time.Time) time.Time {
	if v := strings.TrimSpace(h.Get("retry-after-ms")); v != "" {
		if ms, err := strconv.ParseFloat(v, 64); err == nil && ms >= 0 {
			return now.Add(capped(ms, time.Millisecond, maxRetryAfter))
		}
	}
	if v := strings.TrimSpace(h.Get("retry-after")); v != "" {
		if s, err := strconv.ParseFloat(v, 64); err == nil && s >= 0 {
			return now.Add(capped(s, time.Second, maxRetryAfter))
		}
		if t, err := http.ParseTime(v); err == nil {
			// A date already gone means retry now, not at some past instant.
			if t.Before(now) {
				return now
			}
			return t
		}
	}
	return time.Time{}
}

// capped is n units as a Duration, at most max. Inf and huge values would
// overflow a Duration into the past.
func capped(n float64, unit, max time.Duration) time.Duration {
	if n >= float64(max)/float64(unit) {
		return max
	}
	return time.Duration(n * float64(unit))
}

// resetsAt is the later of the request and token limit resets. OpenAI
// writes them as durations such as "1s", "6m0s" or "20ms". A reset that is
// zero or past has nothing left to wait for.
func resetsAt(h http.Header, now time.Time) time.Time {
	var latest time.Time
	for _, k := range []string{"x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"} {
		d, err := time.ParseDuration(strings.TrimSpace(h.Get(k)))
		if err != nil || d <= 0 {
			continue
		}
		d = min(d, maxResetIn)
		if t := now.Add(d); t.After(latest) {
			latest = t
		}
	}
	return latest
}

// Caps that keep absurd header values from overflowing a Duration.
const (
	maxRetryAfter = 366 * 24 * time.Hour
	maxResetIn    = 366 * 24 * time.Hour
)
