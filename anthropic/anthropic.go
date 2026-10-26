// Package anthropic adapts the Anthropic Messages API to agentloop.Model.
//
// It uses the beta Messages API with streaming. The client reads nothing
// from the environment: the key and base URL come from Config alone.
package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"

	"github.com/kirahowe/agentloop"
	"github.com/kirahowe/agentloop/internal/httpclient"
)

// Provider is the name recorded in Native.Provider.
const Provider = "anthropic"

const fallbackMaxTokens = 64000

// Config holds everything the adapter needs. Nothing is read from the
// environment.
type Config struct {
	APIKey  string
	BaseURL string
	// HTTPClient is used for every request. Nil means a client that ignores the
	// proxy environment variables.
	HTTPClient *http.Client
	// RefusalFallback asks the API to retry a refused request on its
	// default fallback model.
	RefusalFallback bool
	// DefaultMaxTokens applies when a request sets no MaxOutputTokens.
	// Zero means 64000.
	DefaultMaxTokens int
}

// Model is an agentloop.Model backed by the Anthropic API.
type Model struct {
	client sdk.Client
	cfg    Config
}

var (
	_ agentloop.Model     = (*Model)(nil)
	_ agentloop.Describer = (*Model)(nil)
)

// New builds a Model from cfg.
func New(cfg Config) *Model {
	hc := cfg.HTTPClient
	if hc == nil {
		hc = httpclient.New()
	}
	opts := []option.RequestOption{
		option.WithoutEnvironmentDefaults(),
		option.WithHTTPClient(hc),
		option.WithAPIKey(cfg.APIKey),
		option.WithMaxRetries(0),
	}
	if cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(cfg.BaseURL))
	}
	return &Model{client: sdk.NewClient(opts...), cfg: cfg}
}

// Provider returns "anthropic".
func (m *Model) Provider() string { return Provider }

// Generate makes one streaming request.
func (m *Model) Generate(ctx context.Context, req *agentloop.Request, observe func(agentloop.Delta)) (*agentloop.Response, error) {
	params, err := m.params(req)
	if err != nil {
		return nil, &agentloop.ModelError{Kind: agentloop.InvalidRequest, Message: err.Error(), Err: err}
	}
	stream := m.client.Beta.Messages.NewStreaming(ctx, params)
	defer stream.Close()

	var acc sdk.BetaMessage
	started, stopped := false, false
	for stream.Next() {
		ev := stream.Current()
		if err := acc.Accumulate(ev); err != nil {
			return nil, fail(ctx, err)
		}
		switch ev.Type {
		case "message_start":
			started = true
		case "content_block_delta":
			if observe == nil {
				break
			}
			switch ev.Delta.Type {
			case "text_delta":
				observe(agentloop.Delta{Type: agentloop.TextPart, Text: ev.Delta.Text})
			case "thinking_delta":
				observe(agentloop.Delta{Type: agentloop.ReasoningPart, Text: ev.Delta.Thinking})
			}
		case "message_stop":
			stopped = true
		}
	}
	if err := stream.Err(); err != nil {
		return nil, fail(ctx, err)
	}
	if err := ctx.Err(); err != nil {
		return nil, fail(ctx, err)
	}
	if !started || !stopped {
		err := errors.New("stream ended before message_stop")
		return nil, &agentloop.ModelError{Kind: agentloop.Unreachable, Message: err.Error(), Err: err}
	}
	return response(&acc), nil
}

// Describe asks the Models API about a model: its display name and its
// context window, which the API calls max_input_tokens, and the prices
// this package knows for it.
func (m *Model) Describe(ctx context.Context, model string) (agentloop.ModelInfo, error) {
	info, err := m.client.Models.Get(ctx, model, sdk.ModelGetParams{})
	if err != nil {
		return agentloop.ModelInfo{}, fail(ctx, err)
	}
	return describe(info), nil
}

func describe(info *sdk.ModelInfo) agentloop.ModelInfo {
	return agentloop.ModelInfo{ID: info.ID, DisplayName: info.DisplayName, ContextWindow: max(info.MaxInputTokens, 0), Prices: PricesFor(info.ID)}
}

func (m *Model) params(req *agentloop.Request) (sdk.BetaMessageNewParams, error) {
	maxTokens := int64(req.MaxOutputTokens)
	if maxTokens <= 0 {
		maxTokens = int64(m.cfg.DefaultMaxTokens)
	}
	if maxTokens <= 0 {
		maxTokens = fallbackMaxTokens
	}
	msgs, err := messages(req.Messages)
	if err != nil {
		return sdk.BetaMessageNewParams{}, err
	}
	p := sdk.BetaMessageNewParams{
		Model:        sdk.Model(req.Model),
		MaxTokens:    maxTokens,
		Messages:     msgs,
		CacheControl: sdk.BetaCacheControlEphemeralParam{Type: constant.ValueOf[constant.Ephemeral]()},
		Thinking: sdk.BetaThinkingConfigParamUnion{OfAdaptive: &sdk.BetaThinkingConfigAdaptiveParam{
			Display: sdk.BetaThinkingConfigAdaptiveDisplaySummarized,
		}},
	}
	if strings.TrimSpace(req.System) != "" {
		p.System = []sdk.BetaTextBlockParam{{Text: req.System}}
	}
	if req.Effort != "" {
		p.OutputConfig.Effort = sdk.BetaOutputConfigEffort(req.Effort)
	}
	for _, t := range req.Tools {
		tp, err := tool(t)
		if err != nil {
			return p, err
		}
		p.Tools = append(p.Tools, sdk.BetaToolUnionParam{OfTool: &tp})
	}
	if m.cfg.RefusalFallback {
		p.Fallbacks = sdk.BetaFallbacksParamUnion{OfDefault: constant.ValueOf[constant.Default]()}
		p.Betas = append(p.Betas, sdk.AnthropicBetaServerSideFallback2026_07_01)
	}
	return p, nil
}

// tool converts a ToolSpec, keeping every schema keyword.
func tool(t agentloop.ToolSpec) (sdk.BetaToolParam, error) {
	tp := sdk.BetaToolParam{Name: t.Name}
	// input_schema is required, but the SDK omits a zero schema. A non-nil
	// empty ExtraFields makes it marshal as {"type":"object"}.
	tp.InputSchema.ExtraFields = map[string]any{}
	if t.Description != "" {
		tp.Description = param.NewOpt(t.Description)
	}
	if len(t.Parameters) == 0 {
		return tp, nil
	}
	var schema map[string]json.RawMessage
	if err := json.Unmarshal(t.Parameters, &schema); err != nil {
		return tp, fmt.Errorf("tool %s: parameters are not a JSON object: %w", t.Name, err)
	}
	extra := tp.InputSchema.ExtraFields
	for k, v := range schema {
		switch k {
		case "type":
			// Always "object"; the SDK sets it.
		case "properties":
			tp.InputSchema.Properties = v
		case "required":
			if string(bytes.TrimSpace(v)) == "null" {
				extra[k] = v // not a list the SDK can hold; send it as given
				continue
			}
			var req []string
			if err := json.Unmarshal(v, &req); err != nil {
				return tp, fmt.Errorf("tool %s: bad required: %w", t.Name, err)
			}
			tp.InputSchema.Required = req
		default:
			extra[k] = v
		}
	}
	return tp, nil
}

// block is one raw content block. result marks a tool_result, which the
// API wants ahead of everything else in a user message.
type block struct {
	raw    json.RawMessage
	result bool
}

// outMsg is a message being assembled as raw content blocks.
type outMsg struct {
	role   string
	blocks []block
}

func messages(in []agentloop.Message) ([]sdk.BetaMessageParam, error) {
	var out []outMsg
	for _, m := range in {
		role := string(m.Role)
		var blocks []block
		var err error
		if m.Role == agentloop.Assistant && m.Native != nil && m.Native.Provider == Provider && len(m.Native.Data) > 0 {
			blocks, err = nativeBlocks(m.Native.Data)
		} else {
			blocks, err = partBlocks(m)
		}
		if err != nil {
			return nil, err
		}
		if len(blocks) == 0 {
			continue
		}
		if n := len(out); n > 0 && out[n-1].role == role {
			out[n-1].blocks = append(out[n-1].blocks, blocks...)
			continue
		}
		out = append(out, outMsg{role: role, blocks: blocks})
	}
	res := make([]sdk.BetaMessageParam, len(out))
	for i, o := range out {
		// Merging can put text ahead of a tool_result; the API refuses that.
		// A stable partition keeps every other order.
		content := make([]json.RawMessage, 0, len(o.blocks))
		for _, b := range o.blocks {
			if b.result {
				content = append(content, b.raw)
			}
		}
		for _, b := range o.blocks {
			if !b.result {
				content = append(content, b.raw)
			}
		}
		raw, err := json.Marshal(struct {
			Role    string            `json:"role"`
			Content []json.RawMessage `json:"content"`
		}{o.role, content})
		if err != nil {
			return nil, err
		}
		res[i] = param.Override[sdk.BetaMessageParam](json.RawMessage(raw))
	}
	return res, nil
}

// nativeBlocks returns the content blocks of a recorded response exactly
// as the API sent them.
func nativeBlocks(data json.RawMessage) ([]block, error) {
	var msg struct {
		Content []json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(data, &msg); err != nil {
		return nil, fmt.Errorf("native anthropic message: %w", err)
	}
	blocks := make([]block, len(msg.Content))
	for i, raw := range msg.Content {
		blocks[i] = block{raw: raw}
	}
	return blocks, nil
}

func partBlocks(m agentloop.Message) ([]block, error) {
	var blocks []block
	add := func(b sdk.BetaContentBlockParamUnion) error {
		raw, err := json.Marshal(b)
		if err != nil {
			return err
		}
		blocks = append(blocks, block{raw: raw})
		return nil
	}
	for _, p := range m.Parts {
		var err error
		switch p.Type {
		case agentloop.TextPart:
			// The API refuses empty and whitespace-only text blocks.
			if strings.TrimSpace(p.Text) != "" {
				err = add(sdk.NewBetaTextBlock(p.Text))
			}
		case agentloop.ToolCallPart:
			if m.Role != agentloop.Assistant || p.ToolCall == nil {
				continue
			}
			input := p.ToolCall.Input
			if s := bytes.TrimSpace(input); len(s) == 0 || string(s) == "null" {
				input = json.RawMessage(`{}`)
			}
			err = add(sdk.NewBetaToolUseBlock(toolID(p.ToolCall.ID), input, p.ToolCall.Name))
		case agentloop.ToolResultPart:
			if m.Role != agentloop.User || p.ToolResult == nil {
				continue
			}
			var raw json.RawMessage
			if raw, err = toolResult(p.ToolResult); err == nil {
				blocks = append(blocks, block{raw: raw, result: true})
			}
		}
		if err != nil {
			return nil, err
		}
	}
	return blocks, nil
}

// toolID makes a tool call id acceptable to the API, which allows only
// letters, digits, underscores and hyphens. Ids from Anthropic already are;
// ids from other providers or tools may not be. The same mapping is applied
// to a call and its result, so they still match.
func toolID(id string) string {
	if id == "" {
		return "_"
	}
	return strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '_', r == '-':
			return r
		}
		return '_'
	}, id)
}

// toolResult renders a result. A result with no visible text carries no
// content at all, since the API refuses a blank text block inside one.
func toolResult(r *agentloop.ToolResult) (json.RawMessage, error) {
	type text struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	b := struct {
		Type      string `json:"type"`
		ToolUseID string `json:"tool_use_id"`
		Content   []text `json:"content,omitempty"`
		IsError   bool   `json:"is_error,omitempty"`
	}{Type: "tool_result", ToolUseID: toolID(r.CallID), IsError: r.IsError}
	if strings.TrimSpace(r.Content) != "" {
		b.Content = []text{{"text", r.Content}}
	}
	return json.Marshal(b)
}

func response(acc *sdk.BetaMessage) *agentloop.Response {
	msg := agentloop.Message{Role: agentloop.Assistant}
	for _, b := range acc.Content {
		switch b.Type {
		case "text":
			msg.Parts = append(msg.Parts, agentloop.Part{Type: agentloop.TextPart, Text: b.Text})
		case "thinking":
			msg.Parts = append(msg.Parts, agentloop.Part{Type: agentloop.ReasoningPart, Text: b.Thinking})
		case "redacted_thinking":
			msg.Parts = append(msg.Parts, agentloop.Part{Type: agentloop.ReasoningPart})
		case "tool_use":
			input := json.RawMessage(b.Input)
			if len(input) == 0 {
				input = json.RawMessage(`{}`)
			}
			msg.Parts = append(msg.Parts, agentloop.Part{Type: agentloop.ToolCallPart, ToolCall: &agentloop.ToolCall{
				ID: b.ID, Name: b.Name, Input: input,
			}})
		}
	}
	model := string(acc.Model)
	msg.Native = &agentloop.Native{Provider: Provider, Model: model, Data: json.RawMessage(acc.RawJSON())}
	r := &agentloop.Response{
		Message: msg,
		Model:   model,
		Usage: agentloop.Usage{
			InputTokens:      acc.Usage.InputTokens,
			OutputTokens:     acc.Usage.OutputTokens,
			CacheReadTokens:  acc.Usage.CacheReadInputTokens,
			CacheWriteTokens: acc.Usage.CacheCreationInputTokens,
		},
	}
	switch acc.StopReason {
	case sdk.BetaStopReasonToolUse:
		r.Stop = agentloop.StopToolUse
	case sdk.BetaStopReasonMaxTokens, sdk.BetaStopReasonModelContextWindowExceeded:
		r.Stop = agentloop.StopMaxTokens
	case sdk.BetaStopReasonPauseTurn, sdk.BetaStopReasonCompaction:
		r.Stop = agentloop.StopPause
	case sdk.BetaStopReasonRefusal:
		r.Stop = agentloop.StopRefusal
		var parts []string
		if c := string(acc.StopDetails.Category); c != "" {
			parts = append(parts, c)
		}
		if e := acc.StopDetails.Explanation; e != "" {
			parts = append(parts, e)
		}
		r.Refusal = strings.Join(parts, ": ")
	default:
		r.Stop = agentloop.StopEndTurn
	}
	return r
}

// fail turns an error from the SDK into a *agentloop.ModelError. A done
// context is returned wrapped but unclassified.
func fail(ctx context.Context, err error) error {
	if cerr := ctx.Err(); cerr != nil {
		return fmt.Errorf("anthropic: %w", errors.Join(cerr, err))
	}
	// The SDK failed to build the body: nothing was sent and retrying cannot
	// help.
	// The SDK has its own copy of encoding/json, so match on the text too.
	var mje *json.MarshalerError
	if errors.As(err, &mje) || strings.Contains(err.Error(), "error calling MarshalJSON") {
		return &agentloop.ModelError{Kind: agentloop.InvalidRequest, Message: err.Error(), Err: err}
	}
	var ae *sdk.Error
	if !errors.As(err, &ae) {
		return &agentloop.ModelError{Kind: agentloop.Unreachable, Message: err.Error(), Err: err}
	}
	typ := string(ae.Type())
	status := ae.StatusCode
	msg := apiMessage(ae)
	me := &agentloop.ModelError{Kind: classify(status, typ, msg), Status: status, Message: msg, Err: err}
	if ae.Response != nil {
		now := time.Now()
		me.RetryAfter = retryAfter(ae.Response.Header, now)
		me.ResetsAt = resetsAt(ae.Response.Header, now)
	}
	return me
}

func apiMessage(ae *sdk.Error) string {
	var env struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal([]byte(ae.RawJSON()), &env) == nil && env.Error.Message != "" {
		return env.Error.Message
	}
	if raw := strings.TrimSpace(ae.RawJSON()); raw != "" {
		return raw
	}
	if text := http.StatusText(ae.StatusCode); text != "" {
		return text
	}
	return fmt.Sprintf("HTTP %d", ae.StatusCode)
}

func classify(status int, typ, msg string) agentloop.ErrorKind {
	switch {
	case status == 402 || typ == "billing_error" ||
		(status == 400 && strings.Contains(strings.ToLower(msg), "credit balance")):
		return agentloop.Billing
	case status == 429 || typ == "rate_limit_error":
		return agentloop.RateLimited
	// 408 and 409 are timeouts and conflicts the API invites a retry for.
	case status == 408 || status == 409 || status == 529 || status >= 500 || typ == "overloaded_error" || typ == "api_error":
		return agentloop.Overloaded
	case status == 401 || status == 403 || typ == "authentication_error" || typ == "permission_error":
		return agentloop.Unauthorized
	case status == 404 || typ == "not_found_error":
		return agentloop.NotFound
	case status >= 400 || typ == "invalid_request_error" || typ == "request_too_large":
		return agentloop.InvalidRequest
	}
	return agentloop.Unreachable
}

func retryAfter(h http.Header, now time.Time) time.Time {
	v := strings.TrimSpace(h.Get("Retry-After"))
	if v == "" {
		return time.Time{}
	}
	if s, err := strconv.ParseFloat(v, 64); err == nil && s >= 0 {
		// Inf and huge values would overflow a Duration into the past.
		d := maxRetryAfter
		if s < d.Seconds() {
			d = time.Duration(s * float64(time.Second))
		}
		return now.Add(d)
	}
	if t, err := http.ParseTime(v); err == nil {
		// A date already gone means retry now, not at some past instant.
		if t.Before(now) {
			return now
		}
		return t
	}
	return time.Time{}
}

func resetsAt(h http.Header, now time.Time) time.Time {
	var latest time.Time
	for k, vs := range h {
		lk := strings.ToLower(k)
		if !strings.HasPrefix(lk, "anthropic-ratelimit-") || !strings.HasSuffix(lk, "-reset") {
			continue
		}
		for _, v := range vs {
			if t, err := time.Parse(time.RFC3339, strings.TrimSpace(v)); err == nil && t.After(now) && t.After(latest) {
				latest = t
			}
		}
	}
	return latest
}

// maxRetryAfter caps a Retry-After given in seconds, so absurd values
// cannot overflow a Duration.
const maxRetryAfter = 366 * 24 * time.Hour
