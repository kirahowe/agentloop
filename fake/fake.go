// Package fake is a scripted agentloop.Model for tests. It calls no
// network and records every request it is given.
package fake

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"sync"

	"github.com/kirahowe/agentloop"
)

// Model answers each request with a function, or from a script.
type Model struct {
	// Respond answers one request. A request that arrives after the
	// context is done gets the context's error instead.
	Respond func(req *agentloop.Request) (*agentloop.Response, error)

	mu       sync.Mutex
	requests []*agentloop.Request
}

// New makes a Model that answers with respond.
func New(respond func(req *agentloop.Request) (*agentloop.Response, error)) *Model {
	return &Model{Respond: respond}
}

// Script makes a Model that answers the n-th request with the n-th item,
// each a *agentloop.Response or an error. A request past the end of the
// script fails.
func Script(items ...any) *Model {
	var mu sync.Mutex
	next := 0
	return New(func(*agentloop.Request) (*agentloop.Response, error) {
		mu.Lock()
		defer mu.Unlock()
		if next >= len(items) {
			return nil, fmt.Errorf("fake: the script has no answer for request %d", next+1)
		}
		item := items[next]
		next++
		switch v := item.(type) {
		case *agentloop.Response:
			return v, nil
		case error:
			return nil, v
		}
		return nil, fmt.Errorf("fake: script item %d is a %T, not a response or an error", next, item)
	})
}

// Provider is "fake".
func (m *Model) Provider() string { return "fake" }

// Generate records the request and answers it. Text in the response is
// streamed to observe as one delta.
func (m *Model) Generate(ctx context.Context, req *agentloop.Request, observe func(agentloop.Delta)) (*agentloop.Response, error) {
	cp := *req
	cp.Messages = slices.Clone(req.Messages)
	cp.Tools = slices.Clone(req.Tools)
	m.mu.Lock()
	m.requests = append(m.requests, &cp)
	m.mu.Unlock()

	if err := ctx.Err(); err != nil {
		return nil, err
	}
	resp, err := m.Respond(req)
	if err != nil || resp == nil {
		return resp, err
	}
	if observe != nil {
		for _, p := range resp.Message.Parts {
			if p.Type == agentloop.TextPart || p.Type == agentloop.ReasoningPart {
				observe(agentloop.Delta{Type: p.Type, Text: p.Text})
			}
		}
	}
	return resp, nil
}

// Requests lists the requests received so far.
func (m *Model) Requests() []*agentloop.Request {
	m.mu.Lock()
	defer m.mu.Unlock()
	return slices.Clone(m.requests)
}

// ModelID is the model the fake reports serving.
const ModelID = "fake-model"

// DefaultUsage is the usage of every response built here.
var DefaultUsage = agentloop.Usage{InputTokens: 10, OutputTokens: 5}

// Text builds a response that ends the turn with text.
func Text(text string) *agentloop.Response {
	return build(agentloop.StopEndTurn, text)
}

// ToolUse builds a response that calls tools, with optional text before
// the calls.
func ToolUse(text string, calls ...agentloop.ToolCall) *agentloop.Response {
	return build(agentloop.StopToolUse, text, calls...)
}

// Call builds a tool call. input is JSON; empty means {}.
func Call(id, name, input string) agentloop.ToolCall {
	if input == "" {
		input = "{}"
	}
	return agentloop.ToolCall{ID: id, Name: name, Input: json.RawMessage(input)}
}

// WithStop returns resp with a different stop, such as a cut-off turn.
func WithStop(resp *agentloop.Response, stop agentloop.Stop) *agentloop.Response {
	cp := *resp
	cp.Stop = stop
	return &cp
}

// WithUsage returns resp with a different usage.
func WithUsage(resp *agentloop.Response, u agentloop.Usage) *agentloop.Response {
	cp := *resp
	cp.Usage = u
	return &cp
}

func build(stop agentloop.Stop, text string, calls ...agentloop.ToolCall) *agentloop.Response {
	msg := agentloop.Message{Role: agentloop.Assistant}
	if text != "" {
		msg.Parts = append(msg.Parts, agentloop.Part{Type: agentloop.TextPart, Text: text})
	}
	for _, c := range calls {
		c := c
		msg.Parts = append(msg.Parts, agentloop.Part{Type: agentloop.ToolCallPart, ToolCall: &c})
	}
	// The native record is the neutral parts again; enough to check that
	// an agent keeps it with the turn.
	data, _ := json.Marshal(msg.Parts)
	msg.Native = &agentloop.Native{Provider: "fake", Model: ModelID, Data: data}
	return &agentloop.Response{Message: msg, Stop: stop, Model: ModelID, Usage: DefaultUsage}
}
