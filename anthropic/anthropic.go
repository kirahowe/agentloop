// Package anthropic adapts the Anthropic Messages API to agentloop.Model.
//
// It uses the beta Messages API with streaming. The client reads nothing
// from the environment: the key and base URL come from Config alone.
package anthropic

import (
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

	"agentloop"
)

// Provider is the name recorded in Native.Provider.
const Provider = "anthropic"

const fallbackMaxTokens = 64000

// Config holds everything the adapter needs. Nothing is read from the
// environment.
type Config struct {
	APIKey  string
	BaseURL string
	// HTTPClient is used for every request. Nil means http.DefaultClient.
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

var _ agentloop.Model = (*Model)(nil)

// New builds a Model from cfg.
func New(cfg Config) *Model {
	hc := cfg.HTTPClient
	if hc == nil {
		hc = http.DefaultClient
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
	stopped := false
	for stream.Next() {
		ev := stream.Current()
		if err := acc.Accumulate(ev); err != nil {
			return nil, fail(ctx, err)
		}
		switch ev.Type {
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
	if !stopped {
		err := errors.New("stream ended before message_stop")
		return nil, &agentloop.ModelError{Kind: agentloop.Unreachable, Message: err.Error(), Err: err}
	}
	return response(&acc), nil
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
	if req.System != "" {
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
	extra := map[string]any{}
	for k, v := range schema {
		switch k {
		case "type":
			// Always "object"; the SDK sets it.
		case "properties":
			tp.InputSchema.Properties = v
		case "required":
			var req []string
			if err := json.Unmarshal(v, &req); err != nil {
				return tp, fmt.Errorf("tool %s: bad required: %w", t.Name, err)
			}
			tp.InputSchema.Required = req
		default:
			extra[k] = v
		}
	}
	if len(extra) > 0 {
		tp.InputSchema.ExtraFields = extra
	}
	return tp, nil
}

// outMsg is a message being assembled as raw content blocks.
type outMsg struct {
	role   string
	blocks []json.RawMessage
}

func messages(in []agentloop.Message) ([]sdk.BetaMessageParam, error) {
	var out []outMsg
	for _, m := range in {
		role := string(m.Role)
		var blocks []json.RawMessage
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
		raw, err := json.Marshal(struct {
			Role    string            `json:"role"`
			Content []json.RawMessage `json:"content"`
		}{o.role, o.blocks})
		if err != nil {
			return nil, err
		}
		res[i] = param.Override[sdk.BetaMessageParam](json.RawMessage(raw))
	}
	return res, nil
}

// nativeBlocks returns the content blocks of a recorded response exactly
// as the API sent them.
func nativeBlocks(data json.RawMessage) ([]json.RawMessage, error) {
	var msg struct {
		Content []json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(data, &msg); err != nil {
		return nil, fmt.Errorf("native anthropic message: %w", err)
	}
	return msg.Content, nil
}

func partBlocks(m agentloop.Message) ([]json.RawMessage, error) {
	var blocks []json.RawMessage
	add := func(b sdk.BetaContentBlockParamUnion) error {
		raw, err := json.Marshal(b)
		if err != nil {
			return err
		}
		blocks = append(blocks, raw)
		return nil
	}
	for _, p := range m.Parts {
		var err error
		switch p.Type {
		case agentloop.TextPart:
			if p.Text != "" {
				err = add(sdk.NewBetaTextBlock(p.Text))
			}
		case agentloop.ToolCallPart:
			if m.Role != agentloop.Assistant || p.ToolCall == nil {
				continue
			}
			input := p.ToolCall.Input
			if len(input) == 0 {
				input = json.RawMessage(`{}`)
			}
			err = add(sdk.NewBetaToolUseBlock(p.ToolCall.ID, input, p.ToolCall.Name))
		case agentloop.ToolResultPart:
			if m.Role != agentloop.User || p.ToolResult == nil {
				continue
			}
			err = add(sdk.NewBetaToolResultBlock(p.ToolResult.CallID, p.ToolResult.Content, p.ToolResult.IsError))
		}
		if err != nil {
			return nil, err
		}
	}
	return blocks, nil
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
	return http.StatusText(ae.StatusCode)
}

func classify(status int, typ, msg string) agentloop.ErrorKind {
	switch {
	case status == 402 || typ == "billing_error" ||
		(status == 400 && strings.Contains(strings.ToLower(msg), "credit balance")):
		return agentloop.Billing
	case status == 429 || typ == "rate_limit_error":
		return agentloop.RateLimited
	case status == 529 || status >= 500 || typ == "overloaded_error" || typ == "api_error":
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
		return now.Add(time.Duration(s * float64(time.Second)))
	}
	if t, err := http.ParseTime(v); err == nil {
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
