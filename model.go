package agentloop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

// A Model is one provider's API, adapted. Adapters live in their own
// packages (anthropic, openai) and are the only code that knows a vendor.
//
// Generate makes exactly one request: no retries, which belong to the
// Agent so they are visible and bounded. It returns a *ModelError for
// anything the provider refused or could not do.
type Model interface {
	// Provider names the provider, as recorded in Native.Provider.
	Provider() string
	Generate(ctx context.Context, req *Request, observe func(Delta)) (*Response, error)
}

// A Request is everything one model call is given.
type Request struct {
	Model    string
	System   string
	Messages []Message
	Tools    []ToolSpec
	// Effort is how hard the model should think. Empty means the
	// provider's default.
	Effort Effort
	// MaxOutputTokens caps one response. Zero means the adapter's default.
	MaxOutputTokens int
}

// Effort is a provider-neutral reasoning level. An adapter maps it to the
// nearest level its provider has.
type Effort string

const (
	EffortLow    Effort = "low"
	EffortMedium Effort = "medium"
	EffortHigh   Effort = "high"
	EffortXHigh  Effort = "xhigh"
	EffortMax    Effort = "max"
)

// A ToolSpec is what the model is told about a tool.
type ToolSpec struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	// Parameters is the JSON Schema of the tool's input, an object.
	Parameters json.RawMessage `json:"parameters"`
}

// A Response is one model call's answer.
type Response struct {
	// Message is the assistant turn, with Native set.
	Message Message
	Stop    Stop
	// Model is the model that served the request, which can differ from
	// the one asked for when a provider falls back.
	Model string
	Usage Usage
	// Refusal explains a Stop of StopRefusal, when the provider said why.
	Refusal string
}

// Stop is why a response ended.
type Stop string

const (
	StopEndTurn   Stop = "end_turn"
	StopToolUse   Stop = "tool_use"
	StopMaxTokens Stop = "max_tokens"
	StopRefusal   Stop = "refusal"
	// StopPause means the provider paused a long turn and the same
	// request should be sent again to continue it.
	StopPause Stop = "pause"
)

// Usage counts tokens.
type Usage struct {
	InputTokens      int64 `json:"input_tokens"`
	OutputTokens     int64 `json:"output_tokens"`
	CacheReadTokens  int64 `json:"cache_read_tokens,omitempty"`
	CacheWriteTokens int64 `json:"cache_write_tokens,omitempty"`
}

func (u *Usage) add(v Usage) {
	u.InputTokens += v.InputTokens
	u.OutputTokens += v.OutputTokens
	u.CacheReadTokens += v.CacheReadTokens
	u.CacheWriteTokens += v.CacheWriteTokens
}

// Total is every token the provider counted.
func (u Usage) Total() int64 {
	return u.InputTokens + u.OutputTokens + u.CacheReadTokens + u.CacheWriteTokens
}

// A Delta is a piece of a response as it streams in, for display.
type Delta struct {
	Type PartType `json:"type"`
	Text string   `json:"text"`
}

// ErrorKind sorts what can go wrong with a model call by what to do
// about it.
type ErrorKind string

const (
	// RateLimited: wait until RetryAfter, or ResetsAt for the whole limit.
	RateLimited ErrorKind = "rate_limited"
	// Overloaded: the provider failed on its side. Retrying may work.
	Overloaded ErrorKind = "overloaded"
	// Unreachable: the request never got an answer.
	Unreachable ErrorKind = "unreachable"
	// Unauthorized: the credentials were refused. Someone has to act.
	Unauthorized ErrorKind = "unauthorized"
	// Billing: the account cannot pay. Someone has to act.
	Billing ErrorKind = "billing"
	// NotFound: the model or endpoint does not exist. Fix the settings.
	NotFound ErrorKind = "not_found"
	// InvalidRequest: the provider rejected the request itself.
	InvalidRequest ErrorKind = "invalid_request"
)

// A ModelError is a model call that failed, sorted by kind.
type ModelError struct {
	Kind    ErrorKind `json:"kind"`
	Status  int       `json:"status,omitempty"`
	Message string    `json:"message"`
	// RetryAfter is the earliest time the provider asked to be retried.
	RetryAfter time.Time `json:"retry_after,omitzero"`
	// ResetsAt is when the exhausted limit fully resets, if the provider
	// said, which can be well after RetryAfter.
	ResetsAt time.Time `json:"resets_at,omitzero"`
	Err      error     `json:"-"`
}

func (e *ModelError) Error() string {
	if e.Status != 0 {
		return fmt.Sprintf("%s (%d): %s", e.Kind, e.Status, e.Message)
	}
	return fmt.Sprintf("%s: %s", e.Kind, e.Message)
}

func (e *ModelError) Unwrap() error { return e.Err }

// Transient reports whether the same request could succeed later.
func (e *ModelError) Transient() bool {
	return e.Kind == RateLimited || e.Kind == Overloaded || e.Kind == Unreachable
}

// AsModelError finds the ModelError in err, wrapping any other error as
// Unreachable.
func AsModelError(err error) *ModelError {
	var me *ModelError
	if errors.As(err, &me) {
		return me
	}
	return &ModelError{Kind: Unreachable, Message: err.Error(), Err: err}
}
