package agentloop

import (
	"encoding/json"
	"strings"
)

// Role is who wrote a message.
type Role string

const (
	User      Role = "user"
	Assistant Role = "assistant"
)

// A Message is one turn of a conversation in a form no provider owns.
//
// Parts is the neutral reading of the turn: what policies, logs and other
// providers see. Native, set only on assistant turns, is the provider's own
// record of the same turn, kept byte for byte. A provider replays its own
// turns from Native, so reasoning and anything else the neutral form cannot
// express goes back to it unchanged; history is never rewritten.
type Message struct {
	Role   Role    `json:"role"`
	Parts  []Part  `json:"parts"`
	Native *Native `json:"native,omitempty"`
}

// PartType says which field of a Part is set.
type PartType string

const (
	TextPart       PartType = "text"
	ReasoningPart  PartType = "reasoning"
	ToolCallPart   PartType = "tool_call"
	ToolResultPart PartType = "tool_result"
)

// A Part is one piece of a message.
type Part struct {
	Type PartType `json:"type"`
	// Text is the text of a text part, or the readable summary of a
	// reasoning part, which may be empty.
	Text       string      `json:"text,omitempty"`
	ToolCall   *ToolCall   `json:"tool_call,omitempty"`
	ToolResult *ToolResult `json:"tool_result,omitempty"`
}

// A ToolCall is the model asking to run a tool.
type ToolCall struct {
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input"`
}

// A ToolResult answers one ToolCall.
type ToolResult struct {
	CallID  string `json:"call_id"`
	Content string `json:"content"`
	IsError bool   `json:"is_error,omitempty"`
}

// Native is a provider's own record of an assistant turn.
type Native struct {
	Provider string          `json:"provider"`
	Model    string          `json:"model"`
	Data     json.RawMessage `json:"data"`
}

// Text builds a message holding only text.
func Text(role Role, text string) Message {
	return Message{Role: role, Parts: []Part{{Type: TextPart, Text: text}}}
}

// Text joins the message's text parts.
func (m Message) Text() string {
	var b strings.Builder
	for _, p := range m.Parts {
		if p.Type == TextPart {
			if b.Len() > 0 {
				b.WriteString("\n")
			}
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

// ToolCalls lists the message's tool calls in order.
func (m Message) ToolCalls() []ToolCall {
	var calls []ToolCall
	for _, p := range m.Parts {
		if p.Type == ToolCallPart && p.ToolCall != nil {
			calls = append(calls, *p.ToolCall)
		}
	}
	return calls
}
